"""Redirect torch.utils.cpp_extension's CUDA build path to the b70 toolchain.

Loaded by b70torch.pth at interpreter start-up (cheap: nothing is imported
until torch.utils.cpp_extension itself is), or explicitly via
b70torch.patch_cpp_extension().
"""
import importlib.abc
import importlib.util
import os
import sys

DROP_LIBS = {"cudart", "cudart_static", "cudadevrt", "c10_cuda", "torch_cuda", "torch_cuda_cu",
             "torch_cuda_cpp", "cublas", "cublasLt", "cusparse", "cufft", "curand", "cusolver",
             "cudnn", "nvToolsExt", "nvrtc", "cuda"}
ADD_LIBS = ["c10_xpu", "torch_xpu", "CHIP", "b70torch"]
_PATCHED = False


def patch_cpp_extension(ce=None):
    global _PATCHED
    if ce is None:
        import torch.utils.cpp_extension as ce
    if _PATCHED or getattr(ce, "__b70torch__", False):
        return ce
    from . import INCLUDE, TOOLCHAIN_INCLUDE, chipstar_install, ensure
    install = chipstar_install()
    cuda_home, lib_dir = ensure()
    verbose = bool(os.environ.get("B70TORCH_VERBOSE"))

    ce.CUDA_HOME = cuda_home
    ce._check_cuda_version = lambda *a, **k: None
    ce._get_cuda_arch_flags = lambda cflags=None: []
    # host .cpp files of a CUDA extension are compiled as HIP too (see _build.CXX_WRAPPER)
    os.environ.setdefault("CXX", os.path.join(cuda_home, "bin", "b70-c++"))

    def shadow_sources(sources, include_dirs):
        """Build from a rewritten copy of the package tree (see rewrite.py)."""
        from .rewrite import rewrite_tree
        root = os.path.abspath(os.getcwd())
        shadow = os.path.join(root, "build", "b70_src")
        n = rewrite_tree(root, shadow, verbose=verbose)
        if n and verbose:
            print("[b70torch] %d file(s) rewritten for XPU in %s" % (n, shadow), file=sys.stderr)

        def map_path(p):
            ap = os.path.abspath(p)
            if ap == root:
                return shadow
            if ap.startswith(root + os.sep) and not ap.startswith(shadow):
                return os.path.join(shadow, os.path.relpath(ap, root))
            return p

        return [map_path(s) for s in sources], [map_path(d) for d in include_dirs] + [shadow]

    def fix_ext(ext):
        ext.libraries = [l for l in ext.libraries if l not in DROP_LIBS]
        for l in ADD_LIBS:
            if l not in ext.libraries:
                ext.libraries.append(l)
        for d in (os.path.join(install, "lib"), lib_dir):
            if d not in ext.library_dirs:
                ext.library_dirs.append(d)
            ext.extra_link_args.append("-Wl,-rpath," + d)
        return ext

    orig_cuda_ext = ce.CUDAExtension

    def CUDAExtension(name, sources, *args, **kwargs):
        sources, inc = shadow_sources(list(sources), list(kwargs.get("include_dirs", [])))
        kwargs["include_dirs"] = [INCLUDE, TOOLCHAIN_INCLUDE, os.path.join(install, "include", "cuspv")] + inc
        kwargs["define_macros"] = list(kwargs.get("define_macros", [])) + [("B70_TORCH", "1")]
        return fix_ext(orig_cuda_ext(name, sources, *args, **kwargs))

    CUDAExtension.__doc__ = orig_cuda_ext.__doc__
    ce.CUDAExtension = CUDAExtension

    # JIT path (torch.utils.cpp_extension.load / load_inline)
    orig_ldflags = ce._prepare_ldflags

    def _prepare_ldflags(extra_ldflags, with_cuda, with_sycl, verbose_, is_standalone):
        flags = orig_ldflags(extra_ldflags, with_cuda, with_sycl, verbose_, is_standalone)
        if with_cuda:
            flags = [f for f in flags if not (f.startswith("-l") and f[2:] in DROP_LIBS)]
            flags += ["-L" + os.path.join(install, "lib"), "-L" + lib_dir,
                      "-Wl,-rpath," + os.path.join(install, "lib"), "-Wl,-rpath," + lib_dir]
            flags += ["-l" + l for l in ADD_LIBS]
        return flags

    ce._prepare_ldflags = _prepare_ldflags

    # JIT path: load() / load_inline() end up in _jit_compile(name, sources, ...).
    # load_inline writes its sources into the build directory (rewrite in place);
    # load() gets user files (shadow their common directory like CUDAExtension).
    import inspect
    from .rewrite import rewrite_text, rewrite_tree
    orig_jit_compile = ce._jit_compile
    sig = inspect.signature(orig_jit_compile)

    def _jit_compile(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        sources = [os.path.abspath(str(s)) for s in a["sources"]]
        build_dir = os.path.abspath(a["build_directory"])
        inc = list(a.get("extra_include_paths") or [])
        if all(s.startswith(build_dir + os.sep) for s in sources):
            for s in sources:
                text = open(s, encoding="utf-8").read()
                new = rewrite_text(text)
                if new != text:
                    with open(s, "w", encoding="utf-8") as f:
                        f.write(new)
        else:
            root = os.path.commonpath([os.path.dirname(s) for s in sources] + [os.path.abspath(d) for d in inc]
                                      if inc else [os.path.dirname(s) for s in sources])
            shadow = os.path.join(build_dir, "b70_src")
            rewrite_tree(root, shadow, verbose=verbose)

            def map_path(p):
                ap = os.path.abspath(p)
                if ap == root or ap.startswith(root + os.sep):
                    return os.path.join(shadow, os.path.relpath(ap, root))
                return p

            sources = [map_path(s) for s in sources]
            inc = [map_path(d) for d in inc] + [shadow]
        a["sources"] = sources
        a["extra_include_paths"] = inc
        return orig_jit_compile(*bound.args, **bound.kwargs)

    ce._jit_compile = _jit_compile
    orig_include_paths = ce.include_paths

    def include_paths(device_type="cpu"):
        paths = orig_include_paths(device_type)
        if device_type == "cuda":
            paths = [INCLUDE, TOOLCHAIN_INCLUDE, os.path.join(install, "include", "cuspv")] + paths
        return paths

    ce.include_paths = include_paths
    ce.__b70torch__ = True
    _PATCHED = True
    return ce


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch.utils.cpp_extension":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                patch_cpp_extension(module)
            except Exception as e:  # toolchain not built on this machine: leave torch alone
                if os.environ.get("B70TORCH_VERBOSE"):
                    print("[b70torch] not active: %r" % (e,), file=sys.stderr)

        spec.loader.exec_module = exec_and_patch
        return spec


def install():
    if os.environ.get("B70TORCH", "1").lower() in ("0", "off", "false", "no"):
        return
    # chipStar reads these when libCHIP loads; its default backend is OpenCL,
    # which cannot take the Level Zero handles b70torch hands it.
    os.environ.setdefault("CHIP_BE", "level0")
    os.environ.setdefault("CHIP_LOGLEVEL", "err")
    if "torch.utils.cpp_extension" in sys.modules:
        patch_cpp_extension(sys.modules["torch.utils.cpp_extension"])
        return
    sys.meta_path.insert(0, _Finder())


install()
