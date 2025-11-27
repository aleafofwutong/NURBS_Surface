#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> adaptive_tessellation_wrapper(py::array_t<float> points, 
                                                float threshold, 
                                                std::vector<float> bbox, 
                                                int res) {
    // 转换Python数组为C++向量
    auto buf = points.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::vector<float3> cpp_points;
    for (size_t i = 0; i < buf.size; i += 3) {
        cpp_points.push_back({ptr[i], ptr[i+1], ptr[i+2]});
    }
    
    // 调用GPU函数
    std::vector<float3> output;
    gpu_adaptive_tessellation(cpp_points, output, threshold, 
                             {bbox[0], bbox[1], bbox[2]}, res);
    
    // 转换回NumPy数组
    py::array_t<float> result({output.size(), 3});
    auto r = result.mutable_unchecked<2>();
    for (size_t i = 0; i < output.size(); i++) {
        r(i, 0) = output[i].x;
        r(i, 1) = output[i].y;
        r(i, 2) = output[i].z;
    }
    return result;
}

PYBIND11_MODULE(tessellation, m) {
    m.def("adaptive_tessellation", &adaptive_tessellation_wrapper, 
         "GPU-accelerated adaptive tessellation with hash grid",
         py::arg("points"), py::arg("threshold"), py::arg("bbox"), py::arg("res"));
}