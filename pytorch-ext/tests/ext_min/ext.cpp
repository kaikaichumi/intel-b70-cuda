// Host side of the minimal CUDA extension: the usual checks, guards and bindings.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

at::Tensor affine_cuda(at::Tensor x, double a, double b);
at::Tensor row_sum_cuda(at::Tensor x);

at::Tensor affine(at::Tensor x, double a, double b) {
  TORCH_CHECK(x.is_cuda(), "affine: expected a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "affine: expected a contiguous tensor");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  return affine_cuda(x, a, b);
}

at::Tensor row_sum(at::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "row_sum: expected a contiguous 2-D CUDA tensor");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  return row_sum_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine", &affine, "y = a * x + b");
  m.def("row_sum", &row_sum, "sum over the last dim with warp shuffles");
}
