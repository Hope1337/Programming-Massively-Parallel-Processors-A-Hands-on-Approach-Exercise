#include <torch/extension.h>

void row_matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out);

torch::Tensor row_matmul_cpp(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");

    auto out = torch::zeros({a.size(0), b.size(1)}, torch::TensorOptions().device(torch::kCUDA).dtype(a.dtype()));
    row_matmul(a, b, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_matmul", &row_matmul_cpp, "Add two vectors on CUDA");
}