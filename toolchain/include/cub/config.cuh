// b70 toolchain: CUB over hipCUB (chipStar's rocPRIM/hipCUB branches).
// Packages that reach for CUB internals (Mamba's reverse_scan.cuh) include this.
#pragma once
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;

// Warp geometry macros from CUB's util_arch.cuh. The B70 runs CUDA code with
// warpSize == 32 (chipStar fixes the sub-group size), so these are constants.
#ifndef CUB_PTX_WARP_THREADS
#define CUB_PTX_WARP_THREADS 32
#endif
#ifndef CUB_PTX_LOG_WARP_THREADS
#define CUB_PTX_LOG_WARP_THREADS 5
#endif
#ifndef CUB_WARP_THREADS
#define CUB_WARP_THREADS(arch) CUB_PTX_WARP_THREADS
#endif
#ifndef CUB_LOG_WARP_THREADS
#define CUB_LOG_WARP_THREADS(arch) CUB_PTX_LOG_WARP_THREADS
#endif
#ifndef CUB_PTX_ARCH
#define CUB_PTX_ARCH 800
#endif
#ifndef CUB_RUNTIME_FUNCTION
#define CUB_RUNTIME_FUNCTION
#endif
