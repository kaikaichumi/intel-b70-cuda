// b70 toolchain: CUB over hipCUB (see cub/config.cuh).
// cub::detail::uninitialized_copy / uninitialized_copy_single, used by code
// that builds CUB-style shared-memory temporaries (Mamba's reverse_scan.cuh).
#pragma once
#include <cub/config.cuh>
#include <new>
#include <utility>

namespace hipcub {
namespace detail {
template <typename T, typename U>
__host__ __device__ inline void uninitialized_copy_single(T *ptr, U &&val) {
  ::new (static_cast<void *>(ptr)) T(std::forward<U>(val));
}
template <typename T, typename U>
__host__ __device__ inline void uninitialized_copy(T *ptr, U &&val) {
  uninitialized_copy_single(ptr, std::forward<U>(val));
}
}  // namespace detail
}  // namespace hipcub
