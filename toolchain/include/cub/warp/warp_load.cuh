// b70torch: CUB stand-in. The CUB API is served by hipCUB (chipStar branch,
// installed with the toolchain); every cub::X resolves to hipcub::X.
#pragma once
#include <hipcub/hipcub.hpp>
#ifndef B70_CUB_ALIAS
#define B70_CUB_ALIAS
namespace cub = hipcub;
#endif
