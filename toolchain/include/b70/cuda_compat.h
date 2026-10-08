// b70 toolchain: CUDA runtime API that chipStar's cuspv headers leave out.
// Force-included by b70cc into every translation unit (after cuda_runtime.h),
// and by the PyTorch extension prelude.
#pragma once

// rocPRIM (behind the CUB shims) takes the warp size for lane masks and warp
// dispatch from this macro, which only hipcc defines; without it rocPRIM
// defaults to 64 lanes. chipStar runs CUDA code with warpSize == 32 on Xe2.
#ifndef __AMDGCN_WAVEFRONT_SIZE
#define __AMDGCN_WAVEFRONT_SIZE 32
#endif

// cudaFuncSetAttribute: CUDA needs it to opt a kernel into >48 KB of dynamic
// shared memory or to pick a carveout. Xe2 hands every kernel the full SLM
// (128 KB per work-group) and has no carveout, so success is the right answer.
#ifndef B70_CUDA_FUNC_ATTRIBUTE
#define B70_CUDA_FUNC_ATTRIBUTE
enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
  cudaFuncAttributePreferredSharedMemoryCarveout = 9,
  cudaFuncAttributeRequiredClusterWidth = 10,
  cudaFuncAttributeRequiredClusterHeight = 11,
  cudaFuncAttributeRequiredClusterDepth = 12,
  cudaFuncAttributeNonPortableClusterSizeAllowed = 13,
  cudaFuncAttributeClusterSchedulingPolicyPreference = 14,
  cudaFuncAttributeMax
};
template <typename T>
inline cudaError_t cudaFuncSetAttribute(T *, cudaFuncAttribute, int) { return cudaSuccess; }
inline cudaError_t cudaFuncSetAttribute(const void *, cudaFuncAttribute, int) { return cudaSuccess; }
#endif
