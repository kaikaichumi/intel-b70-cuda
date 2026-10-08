// Device side of the minimal CUDA extension, in the style of most real ones:
// ATen dispatch macros, the current CUDA stream, launch checks, warp shuffles.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void affine_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ y,
                              int64_t n, scalar_t a, scalar_t b) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x)
    y[i] = a * x[i] + b;
}

template <typename scalar_t>
__global__ void row_sum_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ out,
                               int64_t rows, int64_t cols) {
  // one warp per row
  int64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  if (row >= rows) return;
  scalar_t v = 0;
  for (int64_t c = lane; c < cols; c += warpSize) v += x[row * cols + c];
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) out[row] = v;
}

at::Tensor affine_cuda(at::Tensor x, double a, double b) {
  auto y = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return y;
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "affine_cuda", [&] {
    affine_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n, (scalar_t)a, (scalar_t)b);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor row_sum_cuda(at::Tensor x) {
  const int64_t rows = x.size(0), cols = x.size(1);
  auto out = at::empty({rows}, x.options());
  const int threads = 256, warps_per_block = threads / at::cuda::warp_size();
  const int blocks = (int)((rows + warps_per_block - 1) / warps_per_block);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // the other common idiom
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "row_sum_cuda", [&] {
    row_sum_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, cols);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
