from pathlib import Path
FILE=Path(__file__).resolve()
ROOT=FILE.parents[1]
import sys
sys.path.append(str(ROOT))
from utils.add_version import add_version
from utils.seconds_counter import SecondsCounter
from utils.Gaussian_Dithering import Gaussian_Dithering
# from utils.set_value import mk_surface
from geomdl import utilities
from geomdl.NURBS import Surface,Curve,Volume
from geomdl.visualization import VisMPL
from typing import List,Tuple,Union,Optional
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import lru_cache


@add_version
class NURBS_Surface(Surface):
    """
        The rational shapes have some minor differences between the non-rational ones. This class is designed to operate
    with weighted control points (Pw) as described in *The NURBS Book* by Piegl and Tiller. Therefore, it provides
    a different set of properties (i.e. getters and setters):

        * ``ctrlptsw``: 1-dimensional array of weighted control points
        * ``ctrlpts2d``: 2-dimensional array of weighted control points
        * ``ctrlpts``: 1-dimensional array of control points
        * ``weights``: 1-dimensional array of weights
        * //TODO: add more if needed

    You may also use ``set_ctrlpts()`` function which is designed to work with all types of control points.
    
    This class provides the following properties:

    * :py:attr:`order_u`
    * :py:attr:`order_v`
    * :py:attr:`degree_u`
    * :py:attr:`degree_v`
    * :py:attr:`knotvector_u`
    * :py:attr:`knotvector_v`
    * :py:attr:`ctrlptsw`
    * :py:attr:`ctrlpts`
    * :py:attr:`weights`
    * :py:attr:`ctrlpts_size_u`
    * :py:attr:`ctrlpts_size_v`
    * :py:attr:`ctrlpts2d`
    * :py:attr:`delta`
    * :py:attr:`delta_u`
    * :py:attr:`delta_v`
    * :py:attr:`sample_size`
    * :py:attr:`sample_size_u`
    * :py:attr:`sample_size_v`
    * :py:attr:`bbox`
    * :py:attr:`name`
    * :py:attr:`dimension`
    * :py:attr:`vis`
    * :py:attr:`evaluator`
    * :py:attr:`tessellator`
    * :py:attr:`rational`
    * :py:attr:`trims`

    """
    
    _save_path=None
    def __init__(self,
                 control_points: list[list[float]] | np.ndarray | None = None,
                 u_degree: int | None = None,
                 v_degree: int | None = None,
                 delta: float = 1e-2,
                 knots_vector_u: list[float] | None = None,
                 knots_vector_v: list[float] | None = None,
                 generate: Union[bool, str] | None = True,
                 base_set: Optional[list[float]] = None,
                 site_range: Optional[list[float]] = None,
                 weight_range: Optional[tuple] = (0.5, 2.0),
                 noise_avg: Optional[float] = 0.0,
                 noise_var: Optional[float] = 0.02,
                 save_path: Optional[str] = None,
                 disorder_vector: Optional[list[float]] = None,
                 visualisation: Optional[bool] = None
                 ):
        super().__init__()

        # 参数验证与默认值
        if u_degree is None or v_degree is None:
            raise ValueError("u_degree and v_degree must be provided")
        if knots_vector_u is None or knots_vector_v is None:
            raise ValueError("knots_vector_u and knots_vector_v must be provided")

        if base_set is None:
            base_set = [0.0, 1.0]
        if site_range is None:
            site_range = [0.0, 10.0]
        if disorder_vector is None:
            disorder_vector = [1.0, 1.0, 1.0]

        _auto = True
        _save_path = save_path
        self.site_range, self.base_set, self.delta, self.weight_range, self.noise_avg, self.noise_var, self.disorder_vector = \
            site_range, base_set, delta, weight_range, noise_avg, noise_var, disorder_vector

        if isinstance(generate, str):
            if generate == "auto":
                _auto = True
            elif generate == "manual":
                _auto = False
            else:
                raise ValueError("please choose mode from \"auto\" and \"manual\"")
        elif isinstance(generate, bool):
            _auto = generate
        else:
            print("you may need to set generate mode, default is auto.")

        # 按 geomdl 要求的顺序设置：degree → ctrlpts → knotvector
        # 1. 先设置 degree
        self.degree_u = u_degree
        self.degree_v = v_degree

        # 2. 生成并设置控制点
        # 修复：直接从控制点数量推导 num_u 和 num_v
        if control_points is not None:
            num_ctrlpts = len(control_points)
            num_u = int(np.sqrt(num_ctrlpts))  # 假设控制点是正方形网格
            num_v = num_ctrlpts // num_u
        else:
            num_u = len(knots_vector_u) - u_degree - 1
            num_v = len(knots_vector_v) - v_degree - 1

        if control_points is None:
            # 没有传入控制点，进行随机生成与权重混合
            ctrl_xyz = self._init_control_points(None)
            ctrl_xyz = self._apply_random_weights(ctrl_xyz, weight_range)
        else:
            # 传入了控制点（可能是 齐次 或 XYZ），直接规范化
            arr = np.asarray(control_points, dtype=float)
            if arr.shape[1] == 4:
                # 齐次坐标，提取 XYZ 并确保权重 w 不为零
                ctrl_xyz = arr[:, :3]
                weights = arr[:, 3]
                weights[weights == 0] = 1.0  # 修复权重为零的问题
                ctrl_xyz = np.column_stack((ctrl_xyz, weights))
            elif arr.shape[1] == 3:
                # 如果没有权重，默认添加权重为 1.0
                ctrl_xyz = np.column_stack((arr, np.ones((arr.shape[0], 1))))
            else:
                raise ValueError("control_points must have shape (N,3) or (N,4)")

        # 验证控制点数量是否匹配 num_u * num_v
        if len(ctrl_xyz) != num_u * num_v:
            raise ValueError(f"Number of control points ({len(ctrl_xyz)}) does not match the expected grid size ({num_u} x {num_v} = {num_u * num_v})")

        # set_ctrlpts 期望 (N,4) 的齐次坐标列表
        self.set_ctrlpts(ctrl_xyz.tolist(), num_u, num_v)

        # 3. 最后设置 knotvector（此时 degree 和 ctrlpts 已设置）
        self.knotvector_u = knots_vector_u
        self.knotvector_v = knots_vector_v

        self.delta = delta
        self.visualization = visualisation
        if self.visualization:
            try:
                self.visualize("default")
            except Exception:
                pass
    def _init_control_points(self, control_points):
        # 统一把输入规范为 (N,3) 的 XYZ 数组（如果传入的是 (N,4) 齐次点，则丢弃原权重列）
        if control_points is None:
            num_pts = (self.degree_u + 1) * (self.degree_v + 1)
            xyz = np.random.uniform(self.site_range[0], self.site_range[1], size=(num_pts, 3))
        else:
            arr = np.asarray(control_points, dtype=float)
            if arr.ndim != 2:
                raise TypeError("control_points must be a 2D sequence/array")
            if arr.shape[1] == 4:
                # 输入为齐次坐标 [x,y,z,w]，只取 xyz 做后续混合
                xyz = arr[:, :3]
            elif arr.shape[1] == 3:
                xyz = arr
            else:
                raise TypeError("control_points must have shape (N,3) or (N,4)")

        # 保证后续生成函数能访问到正确的点数与形状
        self.control_points = xyz

        weight1, weight2, weight3 = self.disorder_vector
        # 生成辅助矩阵（generate_* 函数依赖 self.control_points.shape）
        reg = self.generate_regular_control_points(self.site_range[0], self.site_range[1])
        flat = self.generate_flat_control_points()

        # 逐元素加权合成，结果仍为 (N,3)
        self.control_points = weight1 * xyz + weight2 * reg + weight3 * flat

        return Gaussian_Dithering(self.control_points, self.noise_avg, self.noise_var)
    
    def generate_regular_control_points(self, loc, scale, low=-1, high=1):
        print("control points: ", self.control_points.shape)
        control_points = np.random.normal(loc, scale, size=self.control_points.shape)
        control_points = np.clip(control_points, low, high)
        print("generating regular control points is done")
        return control_points
    
    def generate_flat_control_points(self, value=0.0):
        print("control points: ", self.control_points.shape)
        # 直接返回与 self.control_points 相同形状的平面
        num_pts = self.control_points.shape[0]
        control_points = np.column_stack((
            np.linspace(self.site_range[0], self.site_range[1], num_pts),
            np.linspace(self.site_range[0], self.site_range[1], num_pts),
            np.full(num_pts, value)
        ))
        print("generating flat control points is done")
        return control_points
    def _apply_random_weights(self, ctrl_xyz, weight_range):
        lo, hi = weight_range
        ctrl_xyz_arr = np.asarray(ctrl_xyz, dtype=float)
        print("the shape of ctrl_xyz_arr is ", ctrl_xyz_arr.shape)
        print("the first control point is ", ctrl_xyz_arr[0])
        # 只返回 XYZ，不要添加权重列（set_ctrlpts 会处理权重）
        return ctrl_xyz_arr
    def visualize(self, mode):
        if mode == "default":
            # 修复：将 self.surface 替换为 self
            self.vis = VisMPL.VisSurface()
            self.render()
        elif mode == "clear_color":
            surf_pts = np.array(self.evalpts) if len(self.evalpts) > 0 else None
            created = False
            if ax is None:
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection="3d")
                created = True

            if surf_pts is not None:
                try:
                    n = len(surf_pts)
                    k = int(n ** 0.5)
                    xs = surf_pts[:, 0].reshape(k, k)
                    ys = surf_pts[:, 1].reshape(k, k)
                    zs = surf_pts[:, 2].reshape(k, k)
                    surf = ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=True, alpha=0.85)
                    if created:
                        ax.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=6)
                except Exception:
                    ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], s=1, alpha=0.8)

                ctrl = np.array(self.ctrlpts)  
                if ctrl.shape[1] >= 3:
                    ax.scatter(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], s=30, label='Control Points', depthshade=True)

            ax.set_title(f"U{self.degree_u}-V{self.degree_v} NURBS Surface", fontsize=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="best")
            plt.show()

if __name__ == "__main__":
    surface=Surface()
    surface.degree_u=2
    surface.degree_v=2
    
    surface.set_ctrlpts([[x,y,z,1.0] for x,y,z in [[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[0.,1.,0.],[1.,1.,1.],[2.,1.,0.],[0.,2.,0.],[1.,2.,0.],[2.,2.,0.]] ],3,3)
    surface.knotvector_u=[0,0,0,1,1,1]    # 
    surface.knotvector_v=[0,0,0,1,1,1]
    surface.vis=VisMPL.VisCurve3D()
    surface.render()
    print(surface.knotvector_u,surface.knotvector_v,surface.ctrlpts_size_u,surface.ctrlpts_size_v,surface.ctrlpts)

    a=NURBS_Surface(control_points=surface._control_points,u_degree=surface.degree_u,v_degree=surface.degree_v,knots_vector_u=surface.knotvector_u,knots_vector_v=surface.knotvector_v
                    ,visualisation=True,delta=0.01)

    a.visualize("default")
    