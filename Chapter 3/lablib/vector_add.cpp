#include <torch/extension.h>

void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out);

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensors must be float32");

    auto out = torch::empty_like(a);
    vector_add_cuda(a, b, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Add two vectors on CUDA");
}