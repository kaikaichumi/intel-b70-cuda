"""Build libb70torch.so and lay out a fake CUDA_HOME for torch.utils.cpp_extension."""
import os
import stat
import subprocess
import sys

from . import INCLUDE, SRC, cache_dir, chipstar_install

NVCC_WRAPPER = r'''#!/usr/bin/env python3
# nvcc look-alike: drops NVIDIA-only flags, then runs chipStar's cucc.
import os, shlex, subprocess, sys
CUCC = %(cucc)r
PRELUDE = %(prelude)r
args = sys.argv[1:]
if args in (["--version"], ["-V"]):  # setup.py scripts probe both spellings
    print("nvcc: NVIDIA (R) Cuda compiler driver\nb70cc (chipStar cucc) for the Intel Arc Pro B70\n"
          "Cuda compilation tools, release 12.4, V12.4.0")
    sys.exit(0)
DROP_WITH_VALUE = {"-gencode", "-arch", "-code", "--generate-code", "--gpu-architecture", "--gpu-code",
                   "-ccbin", "--compiler-bindir", "-Xcudafe", "--diag-suppress", "-Xptxas", "-Xfatbin",
                   "-Xnvlink", "--default-stream", "-maxrregcount", "--maxrregcount", "--threads", "-t"}
DROP = {"--expt-relaxed-constexpr", "--expt-extended-lambda", "--extended-lambda", "-lineinfo",
        "--generate-line-info", "-G", "--use-local-env", "-rdc=true", "--relocatable-device-code=true",
        "--use_fast_math", "-use_fast_math", "--ptxas-options=-v", "-dc"}
DROP_PREFIX = ("-gencode=", "-arch=", "--gpu-architecture=", "--gpu-code=", "--expt-", "-Xptxas", "-Xcudafe",
               "--diag-suppress=", "--default-stream=", "-maxrregcount=", "--threads=", "--generate-code=")
out = []
it = iter(args)
for a in it:
    if a in DROP_WITH_VALUE:
        next(it, None)
    elif a in DROP or a.startswith(DROP_PREFIX):
        pass
    elif a in ("-Xcompiler", "--compiler-options"):
        out += shlex.split(next(it, ""))
    elif a.startswith(("-Xcompiler=", "--compiler-options=")):
        out += shlex.split(a.split("=", 1)[1])
    else:
        out.append(a)
if "-c" in out:
    # __CUDA__: torch's Macros.h then skips its __host__ __device__ __assert_fail
    # declaration, which clashes with chipStar's __device__ definition.
    out = ["-D__CUDA__", "-include", PRELUDE] + out
env = dict(os.environ)
env.setdefault("CUCC_VERSION_STRING", "nvcc: NVIDIA (R) Cuda compiler driver")
if env.get("B70TORCH_VERBOSE"):
    print("[b70 nvcc] " + " ".join(shlex.quote(x) for x in [CUCC] + out), file=sys.stderr)
sys.exit(subprocess.call([CUCC] + out, env=env))
'''

# c++ stand-in for extension builds: host .cpp files are compiled as HIP by cucc
# (torch's headers need the HIP/CUDA-mode definitions once <ATen/cuda/*> is in
# play); linking is left to the system c++.
CXX_WRAPPER = r'''#!/usr/bin/env python3
import os, sys
NVCC = %(nvcc)r
args = sys.argv[1:]
if "-c" in args:
    if "-x" not in args:
        args = ["-x", "cu"] + args
    os.execv(NVCC, [NVCC] + args)
os.execvp("c++", ["c++"] + args)
'''


def _write_exec(path, content):
    if not os.path.exists(path) or open(path).read() != content:
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def ensure_cuda_home(install):
    home = os.path.join(cache_dir(), "cuda_home")
    os.makedirs(os.path.join(home, "bin"), exist_ok=True)
    nvcc = os.path.join(home, "bin", "nvcc")
    _write_exec(nvcc, NVCC_WRAPPER % {"cucc": os.path.join(install, "bin", "cucc"),
                                      "prelude": os.path.join(INCLUDE, "b70torch", "prelude.h")})
    _write_exec(os.path.join(home, "bin", "b70-c++"), CXX_WRAPPER % {"nvcc": nvcc})
    for link, target in (("include", os.path.join(install, "include")),
                         ("lib64", os.path.join(install, "lib")),
                         ("lib", os.path.join(install, "lib"))):
        p = os.path.join(home, link)
        if os.path.islink(p) and os.readlink(p) != target:
            os.remove(p)
        if not os.path.exists(p):
            os.symlink(target, p)
    return home


def ensure_interop(install):
    import torch
    lib_dir = os.path.join(cache_dir(), "lib")
    os.makedirs(lib_dir, exist_ok=True)
    lib = os.path.join(lib_dir, "libb70torch.so")
    src = os.path.join(SRC, "interop.cpp")
    stamp = os.path.join(lib_dir, "libb70torch.stamp")
    key = "%s %s %s" % (torch.__version__, os.path.getmtime(src), install)
    if os.path.exists(lib) and os.path.exists(stamp) and open(stamp).read() == key:
        return lib_dir
    tdir = os.path.dirname(torch.__file__)
    venv = sys.prefix
    cmd = ["g++", "-std=c++20", "-O2", "-fPIC", "-shared", "-w", src, "-o", lib,
           "-I" + os.path.join(tdir, "include"),
           "-I" + os.path.join(tdir, "include", "torch", "csrc", "api", "include"),
           "-I" + os.path.join(venv, "include"), "-I" + os.path.join(venv, "include", "sycl"),
           "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
           "-L" + os.path.join(tdir, "lib"), "-L" + os.path.join(venv, "lib"), "-L" + os.path.join(install, "lib"),
           "-lc10_xpu", "-lc10", "-ltorch_xpu", "-ltorch", "-ltorch_cpu", "-lsycl", "-lze_loader", "-lCHIP",
           "-Wl,-rpath," + os.path.join(tdir, "lib"), "-Wl,-rpath," + os.path.join(venv, "lib"),
           "-Wl,-rpath," + os.path.join(install, "lib")]
    print("[b70torch] building libb70torch.so", file=sys.stderr)
    subprocess.check_call(cmd)
    with open(stamp, "w") as f:
        f.write(key)
    return lib_dir


def ensure():
    install = chipstar_install()
    return ensure_cuda_home(install), ensure_interop(install)
