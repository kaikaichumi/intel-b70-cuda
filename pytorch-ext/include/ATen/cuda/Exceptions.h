// b70torch shim for <ATen/cuda/Exceptions.h>
#pragma once
#include <c10/cuda/CUDAException.h>

#define AT_CUDA_CHECK(EXPR) C10_CUDA_CHECK(EXPR)
#define AT_CUDA_DRIVER_CHECK(EXPR) C10_CUDA_CHECK(EXPR)
// cuBLAS/cuSPARSE/cuFFT are not available through the b70 toolchain yet; make
// a stray use fail loudly at compile time instead of linking to nothing.
#define TORCH_CUDABLAS_CHECK(EXPR) static_assert(false, "cuBLAS is not available on the B70 toolchain yet")
#define TORCH_CUDASPARSE_CHECK(EXPR) static_assert(false, "cuSPARSE is not available on the B70 toolchain yet")
#define AT_CUFFT_CHECK(EXPR) static_assert(false, "cuFFT is not available on the B70 toolchain yet")
