// Force-included (after chipStar's cuda_runtime.h) into every extension TU.
// torch's Half.h / BFloat16.h only pull in <cuda_fp16.h> under __CUDACC__,
// which cucc does not define; without these includes the __half/__nv_bfloat16
// types and their __ldg overloads are missing when torch's headers need them.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// chipStar's __ldg(const __half*) sits in a namespace torch's Half.h cannot
// see; the bfloat16 one is already global. A plain load is correct on Xe2.
__device__ inline __half __ldg(const __half *p) { return *p; }

// cudaFuncSetAttribute and other runtime API chipStar leaves out: shared with
// b70cc (toolchain/include/b70/cuda_compat.h, on the include path).
#include <b70/cuda_compat.h>
