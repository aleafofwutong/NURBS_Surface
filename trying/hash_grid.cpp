#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA 内核函数：计算哈希值
__global__ void hash_kernel(const int* indices, const int* hash_params, int* hashes, int num_points, int hash_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int x = indices[idx * 3];
        int y = indices[idx * 3 + 1];
        int z = indices[idx * 3 + 2];

        hashes[idx] = (x * hash_params[0] ^ y * hash_params[1] ^ z * hash_params[2]) % hash_size;
    }
}

// 哈希网格类
class HashGrid {
public:
    HashGrid(std::vector<float> bounding_box, int resolution, int hash_size)
        : bbox(bounding_box), res(resolution), hash_size(hash_size) {
        cell_size = {(bbox[1] - bbox[0]) / resolution,
                     (bbox[3] - bbox[2]) / resolution,
                     (bbox[5] - bbox[4]) / resolution};

        hash_params = torch::randint(0, 1 << 20, {3}, torch::kInt32).to(torch::kCUDA);
    }

    std::tuple<torch::Tensor, torch::Tensor> build(torch::Tensor points) {
        // 计算每个点的网格索引
        auto indices = ((points - bbox_tensor.index({torch::indexing::Slice(None, None, 2)})) / cell_size_tensor).floor().to(torch::kInt32);

        // 过滤超出边界的点
        auto mask = (indices >= 0).all(1) & (indices < res).all(1);
        auto valid_indices = indices.index({mask});
        auto valid_points = points.index({mask});

        // 计算哈希值
        auto hashes = torch::empty({valid_indices.size(0)}, torch::kInt32).to(torch::kCUDA);
        int threads = 256;
        int blocks = (valid_indices.size(0) + threads - 1) / threads;
        hash_kernel<<<blocks, threads>>>(valid_indices.data_ptr<int>(), hash_params.data_ptr<int>(), hashes.data_ptr<int>(), valid_indices.size(0), hash_size);

        return std::make_tuple(valid_points, valid_indices);
    }

private:
    std::vector<float> bbox;
    std::vector<float> cell_size;
    int res;
    int hash_size;
    torch::Tensor hash_params;
};

// Pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<HashGrid>(m, "HashGrid")
        .def(py::init<std::vector<float>, int, int>())
        .def("build", &HashGrid::build);
}