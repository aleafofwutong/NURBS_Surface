from geomdl import utilities
from geomdl.NURBS import Surface,Curve,Volume
from geomdl.visualization import VisMPL
import numpy as np
from pathlib import Path
FILE=Path(__file__).resolve()
ROOT=FILE.parents[1]
import sys
sys.path.append(str(ROOT))
from surface.NURBS_Surface import NURBS_Surface
from functools import lru_cache
@lru_cache(maxsize=None)
def mk_surface(degree_u:int,degree_v:int,
                ctrlpts:list[list],
                num_of_pts_in_u:int,num_of_pts_in_v:int,
                knotvector_u:list[int] | list[float],knotvector_v:list[int] | list[float],
                vis=False,
                auto=True,
                **kwargs)->NURBS_Surface:
    assert (degree_u<num_of_pts_in_u and degree_v<num_of_pts_in_v),"Number of control points should be at least degree + 1"
    assert (len(knotvector_u)== degree_u + num_of_pts_in_u + 1 and len(knotvector_v)== degree_v+num_of_pts_in_v+1) ,"You should make the length of the knot vector = number of control points + degree + 1"
    _ctrlpts =np.array(ctrlpts)
    assert _ctrlpts.shape==(num_of_pts_in_u,num_of_pts_in_v),"The shape of control points of surface is wrong!\nExpected Shape:(num_of_pts_in_u,num_of_pts_in_v)"
    _surface=NURBS_Surface(control_points=ctrlpts,u_degree=degree_u,v_degree=degree_v,auto_generate=auto)

if __name__=="__main__":
    # print(type(np.array([[1,2],[2,3]]).shape))
    print()