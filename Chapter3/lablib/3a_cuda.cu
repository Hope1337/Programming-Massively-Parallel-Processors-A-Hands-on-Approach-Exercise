#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// matmul ik kj -> ij
// gridDim: 1D
// blockDim: 1D
__global__ void row_matmul_cuda(double* a, double* b, double* out, int row, int com, int col) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= row) return;

    for (int j = 0; j < col; ++j) {
        double total = 0.0;
        for (int k = 0; k < com; ++k) {
            total += a[idx * com + k] * b[k * col + j];
        }
        out[idx * col + j] = total;
    }
}

void row_matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kDouble, "a must be float64");
    TORCH_CHECK(b.dtype() == torch::kDouble, "b must be float64");
    TORCH_CHECK(out.dtype() == torch::kDouble, "out must be float64");
    TORCH_CHECK(a.size(1) == b.size(0), "Mismatch shape");

    a = a.contiguous();
    b = b.contiguous();
    out = out.contiguous();

    int row = a.size(0);
    int com = a.size(1);
    int col = b.size(1);

    int threads = 256;
    int blocks = (row + threads - 1) / threads;

    row_matmul_cuda<<<blocks, threads>>>(
        a.data_ptr<double>(),
        b.data_ptr<double>(),
        out.data_ptr<double>(),
        row, com, col
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA execution error: ", cudaGetErrorString(err));
}