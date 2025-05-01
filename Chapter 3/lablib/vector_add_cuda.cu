#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel for vector addition
__global__ void vector_add_kernel(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Wrapper function to launch the kernel
void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    int n = a.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
}