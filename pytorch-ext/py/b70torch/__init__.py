"""b70torch: build and run PyTorch CUDA extensions on the Intel Arc Pro B70.

Importing this package (or the b70torch.pth hook) makes torch.utils.cpp_extension
build `CUDAExtension` modules with the b70 toolchain:

  * nvcc        -> b70cc (chipStar cucc), through a flag-filtering wrapper
  * cudart etc. -> libCHIP (the CUDA runtime on Level Zero) + libb70torch
  * <ATen/cuda/*.h>, <c10/cuda/*.h> -> the shims in pytorch-ext/include, which
    put extension kernels on torch's XPU stream (see src/interop.cpp)
  * x.is_cuda() in extension code -> x.is_xpu()

Settings: B70_CUDA_INSTALL (chipStar install dir), B70TORCH_CACHE (built
libb70torch and the fake CUDA_HOME, default ~/.cache/b70torch),
B70TORCH_SYNC=host (debug: host-side sync instead of events).
"""
import os

os.environ.setdefault("CHIP_BE", "level0")
os.environ.setdefault("CHIP_LOGLEVEL", "err")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pytorch-ext/
INCLUDE = os.path.join(ROOT, "include")
TOOLCHAIN_INCLUDE = os.path.join(os.path.dirname(ROOT), "toolchain", "include")  # cub/, thrust/
SRC = os.path.join(ROOT, "src")


def chipstar_install():
    p = os.environ.get("B70_CUDA_INSTALL")
    if p:
        return p
    for base in (os.environ.get("B70_DATA"), os.path.expanduser("~/.local/share/b70")):
        if base and os.path.exists(os.path.join(base, "cuda", "install", "bin", "cucc")):
            return os.path.join(base, "cuda", "install")
    raise RuntimeError("chipStar install not found: set B70_CUDA_INSTALL or run `b70cuda build`")


def cache_dir():
    d = os.environ.get("B70TORCH_CACHE") or os.path.expanduser("~/.cache/b70torch")
    os.makedirs(d, exist_ok=True)
    return d


def ensure():
    """Build libb70torch and the fake CUDA_HOME if needed; return (cuda_home, lib_dir)."""
    from . import _build
    return _build.ensure()


def patch_cpp_extension(ce=None):
    from . import _hook
    return _hook.patch_cpp_extension(ce)
