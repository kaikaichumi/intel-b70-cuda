// b70torch shim for <c10/cuda/CUDAException.h>
#pragma once
#include <cuda_runtime_api.h>
#include <c10/util/Exception.h>

#define C10_CUDA_CHECK(EXPR)                                                   \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    TORCH_CHECK(__err == cudaSuccess, "CUDA error (b70): ", cudaGetErrorString(__err)); \
  } while (0)

#define C10_CUDA_CHECK_WARN(EXPR)                                              \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    if (__err != cudaSuccess)                                                  \
      TORCH_WARN("CUDA warning (b70): ", cudaGetErrorString(__err));           \
  } while (0)

#define C10_CUDA_ERROR_HANDLED(EXPR) EXPR
#define C10_CUDA_IGNORE_ERROR(EXPR) (void)(EXPR)
#define C10_CUDA_CLEAR_ERROR() (void)cudaGetLastError()
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace c10 {
namespace cuda {
inline void c10_cuda_check_implementation(const int32_t err, const char *, const char *, const int, const bool) {
  TORCH_CHECK(err == cudaSuccess, "CUDA error (b70): ", cudaGetErrorString((cudaError_t)err));
}
}  // namespace cuda
}  // namespace c10
