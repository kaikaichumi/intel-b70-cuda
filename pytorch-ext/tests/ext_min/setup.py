# A CUDA extension written exactly as for an NVIDIA box. Nothing here knows
# about the B70: the b70torch hook redirects CUDAExtension to the b70 toolchain.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="b70_ext_min",
    version="0.1",
    ext_modules=[CUDAExtension("b70_ext_min", ["ext.cpp", "ext_kernel.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
