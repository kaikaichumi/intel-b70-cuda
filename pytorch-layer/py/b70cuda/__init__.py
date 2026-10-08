"""b70cuda: run CUDA-flavoured PyTorch code on an Intel XPU (Arc Pro B70) unchanged.

Loaded automatically by b70cuda.pth (see _hook.py) right after `import torch`.
Set B70_CUDA=0 to turn it off for one run.

What it does
  * every place a device can be named -- tensor factories, Tensor.to/.cuda,
    Module.to/.cuda, torch.load, torch.Generator, autocast, GradScaler,
    set_default_device -- maps "cuda", "cuda:N", torch.device("cuda") and bare
    integers to the XPU
  * torch.cuda.* (memory, sync, streams, events, RNG, properties ...) answers
    from torch.xpu; Tensor.is_cuda is True for XPU tensors (outside torch/triton)
  * torch.cuda.is_available() / device_count() tell the truth (False / 0) to
    libraries that already support XPU natively -- torch itself, transformers,
    diffusers, accelerate, ComfyUI ... -- so they keep their tested XPU code
    paths, and say True / N to everybody else (your script, a model repo)

What it cannot do: run real CUDA kernels. Custom CUDA extensions (.cu files,
cupy, PTX, CUDA graphs) still fail; see README.md for what to use instead.
"""
import functools
import os
import sys

__all__ = ["patch", "is_patched"]

_PATCHED = False

# Callers inside these top-level packages see the real (CUDA-less) torch.cuda
# gate, because they have their own XPU support and behave better through it.
_DEFAULT_NATIVE = (
    "torch triton transformers diffusers accelerate peft optimum comfy "
    "comfy_extras comfy_api comfy_execution vllm bitsandbytes "
    "intel_extension_for_pytorch huggingface_hub safetensors kernels"
)


def is_patched():
    return _PATCHED


def _native_callers():
    extra = os.environ.get("B70_CUDA_NATIVE", "")
    names = set(_DEFAULT_NATIVE.split()) | set(extra.replace(",", " ").split())
    names.discard("")
    return frozenset(names)


def patch(torch=None):
    """Install the CUDA -> XPU mapping. Safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return True
    if torch is None:
        import torch
    try:
        if torch.cuda.is_available():  # a real NVIDIA card: leave it alone
            return False
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return False
    except Exception:
        return False
    _install(torch)
    _PATCHED = True
    if os.environ.get("B70_CUDA_VERBOSE"):
        print("[b70cuda] torch.cuda -> torch.xpu mapping active", file=sys.stderr)
    return True


def _install(torch):
    xpu = torch.xpu
    cuda = torch.cuda
    native = _native_callers()
    # B70_CUDA_SKIP=part,part turns pieces off (for working around a problem):
    # factories, tensor, is_cuda, module, load, generator, default_device, amp, cuda_api
    skip = set(os.environ.get("B70_CUDA_SKIP", "").replace(",", " ").split())
    Device = torch.device
    RealGenerator = torch.Generator

    # ---- device-name mapping ---------------------------------------------
    def m(d):
        t = type(d)
        if t is str:
            if d.startswith("cuda"):
                return "xpu" + d[4:]
            return d
        if t is Device:
            if d.type == "cuda":
                return Device("xpu", d.index)
            return d
        if t is int:
            return Device("xpu", d)
        return d

    def m_kw(kwargs):
        d = kwargs.get("device")
        if d is not None:
            kwargs["device"] = m(d)

    def from_native_lib(depth=2, within=native):
        try:
            f = sys._getframe(depth)
        except ValueError:
            return False
        mod = f.f_globals.get("__name__") or ""
        return mod.partition(".")[0] in within

    torch_itself = frozenset(("torch", "triton"))

    # ---- tensor factories --------------------------------------------------
    def wrap_device_kw(fn):
        @functools.wraps(fn)
        def w(*args, **kwargs):
            if "device" in kwargs:
                m_kw(kwargs)
            return fn(*args, **kwargs)
        w.__b70_orig__ = fn
        return w

    factories = (
        "tensor as_tensor asarray zeros ones empty full rand randn randint "
        "randperm arange range linspace logspace eye empty_strided zeros_like "
        "ones_like empty_like full_like rand_like randn_like randint_like normal "
        "scalar_tensor tril_indices triu_indices sparse_coo_tensor "
        "sparse_csr_tensor sparse_compressed_tensor from_file hann_window "
        "hamming_window blackman_window bartlett_window kaiser_window"
    ).split()
    # TorchScript resolves torch.zeros & co. by object identity in its builtin
    # table; register each wrapper under the same aten op as the original, or
    # scripting any function that uses it fails (kornia scripts some on import).
    try:
        from torch.jit._builtins import _get_builtin_table, _register_builtin
        builtin_table = _get_builtin_table()
    except Exception:
        builtin_table = None

    def replace(mod, name):
        fn = getattr(mod, name, None)
        if fn is None or hasattr(fn, "__b70_orig__") or name in skip:
            return
        w = wrap_device_kw(fn)
        setattr(mod, name, w)
        if (builtin_table is not None and id(fn) in builtin_table
                and "jitreg" not in skip and "jitreg:" + name not in skip):
            _register_builtin(w, builtin_table[id(fn)])

    def fix_device_context():
        # `with torch.device(...)` / set_default_device work through
        # DeviceContext, which only touches functions found in
        # _device_constructors() -- a set of function objects. The C++ side
        # hands it the original builtins while the set (built lazily) may
        # hold our wrappers, so include both. Libraries rely on this:
        # transformers probes torch.tensor([]).device to detect a meta context.
        import torch.utils._device as dev
        base = dev._device_constructors.__wrapped__()
        both = set(base)
        for f in base:
            orig = getattr(f, "__b70_orig__", None)
            if orig is not None:
                both.add(orig)
            else:
                wrapped = getattr(torch, getattr(f, "__name__", ""), None)
                if getattr(wrapped, "__b70_orig__", None) is f:
                    both.add(wrapped)
        both = frozenset(both)
        dev._device_constructors = functools.lru_cache(1)(lambda: both)

        real_init = dev.DeviceContext.__init__

        def init(self, device):  # with torch.device("cuda"): ...
            real_init(self, m(device))

        if "default_device" not in skip:
            dev.DeviceContext.__init__ = init

    T = torch.Tensor
    if "factories" not in skip:
        for name in factories:
            replace(torch, name)
        for name in ("fftfreq", "rfftfreq"):
            replace(torch.fft, name)
        for name in ("new_tensor", "new_zeros", "new_ones", "new_empty", "new_full",
                     "new_empty_strided"):
            fn = getattr(T, name, None)
            if fn is not None:
                setattr(T, name, wrap_device_kw(fn))
    fix_device_context()

    # ---- Tensor.is_cuda -----------------------------------------------------
    # Kernel packages guard their launches with `assert x.is_cuda` (sageattention,
    # Liger, flash-attn look-alikes, CUDA extensions built by b70torch). XPU
    # tensors answer True to them; torch and triton themselves keep seeing the
    # real value so their own device dispatch is untouched. is_xpu is unchanged.
    if "is_cuda" not in skip:
        real_is_cuda = T.is_cuda  # getset descriptor on torch._C.TensorBase

        def is_cuda(self):
            if real_is_cuda.__get__(self, T):
                return True
            return self.is_xpu and not from_native_lib(within=torch_itself)

        T.is_cuda = property(is_cuda, doc=real_is_cuda.__doc__)

    # ---- Tensor.to / .cuda --------------------------------------------------
    orig_tensor_to = T.to

    def tensor_to(self, *args, **kwargs):
        if args:
            a0 = args[0]
            t0 = type(a0)
            if t0 is str or t0 is Device or t0 is int:
                args = (m(a0),) + args[1:]
        if "device" in kwargs:
            m_kw(kwargs)
        return orig_tensor_to(self, *args, **kwargs)

    tensor_to.__b70_orig__ = orig_tensor_to
    if "tensor" not in skip:
        T.to = tensor_to

    def tensor_cuda(self, device=None, non_blocking=False,
                    memory_format=torch.preserve_format):
        dev = Device("xpu", xpu.current_device()) if device is None else m(device)
        return orig_tensor_to(self, dev, non_blocking=non_blocking,
                              memory_format=memory_format)

    if "tensor" not in skip:
        T.cuda = tensor_cuda

    # ---- nn.Module.to / .cuda / .to_empty ----------------------------------
    Module = torch.nn.Module
    orig_module_to = Module.to

    def module_to(self, *args, **kwargs):
        if args:
            a0 = args[0]
            t0 = type(a0)
            if t0 is str or t0 is Device or t0 is int:
                args = (m(a0),) + args[1:]
        if "device" in kwargs:
            m_kw(kwargs)
        return orig_module_to(self, *args, **kwargs)

    if "module" not in skip:
        Module.to = module_to

    def module_cuda(self, device=None):
        dev = Device("xpu", xpu.current_device()) if device is None else m(device)
        return orig_module_to(self, dev)

    if "module" not in skip:
        Module.cuda = module_cuda

    orig_to_empty = Module.to_empty

    def module_to_empty(self, *, device, recurse=True):
        return orig_to_empty(self, device=m(device), recurse=recurse)

    if "module" not in skip:
        Module.to_empty = module_to_empty

    # ---- torch.load: map_location, and checkpoints saved on a CUDA card ------
    from torch.serialization import default_restore_location

    def restore_cuda_as_xpu(storage, location):
        if location.startswith("cuda"):
            location = "xpu" + location[4:]
        return default_restore_location(storage, location)

    orig_load = torch.load

    @functools.wraps(orig_load)
    def load(f, map_location=None, *args, **kwargs):
        ml = map_location
        if ml is None:
            ml = restore_cuda_as_xpu
        elif isinstance(ml, (str, Device)):
            ml = m(ml)
        elif isinstance(ml, dict):
            ml = {k: m(v) for k, v in ml.items()}
        return orig_load(f, ml, *args, **kwargs)

    load.__b70_orig__ = orig_load
    if "load" not in skip:
        torch.load = load

    # ---- torch.Generator(device="cuda") ------------------------------------
    class _GeneratorMeta(type):
        def __call__(cls, device="cpu"):
            return RealGenerator(m(device))

        def __instancecheck__(cls, obj):
            return isinstance(obj, RealGenerator)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, RealGenerator)

        def __getattr__(cls, name):  # unbound methods, __doc__, ...
            return getattr(RealGenerator, name)

    class Generator(metaclass=_GeneratorMeta):
        pass

    Generator.__module__ = "torch"
    Generator.__qualname__ = Generator.__name__ = "Generator"
    if "generator" not in skip:
        torch.Generator = Generator

    # ---- set_default_device --------------------------------------------------
    orig_sdd = torch.set_default_device

    @functools.wraps(orig_sdd)
    def set_default_device(device):
        return orig_sdd(m(device))

    if "default_device" not in skip:
        torch.set_default_device = set_default_device

    # ---- autocast / GradScaler -------------------------------------------------
    RealAutocast = torch.amp.autocast_mode.autocast

    class autocast(RealAutocast):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            if device_type == "cuda":
                device_type = "xpu"
                if dtype is None:  # CUDA's autocast default is fp16; keep it
                    dtype = torch.float16
            super().__init__(device_type, dtype=dtype, enabled=enabled,
                             cache_enabled=cache_enabled)

    autocast.__module__ = "torch.amp.autocast_mode"
    if "amp" not in skip:
        torch.autocast = autocast
        torch.amp.autocast = autocast

    RealGradScaler = torch.amp.GradScaler

    class GradScaler(RealGradScaler):
        def __init__(self, device="cuda", *args, **kwargs):
            super().__init__(m(device), *args, **kwargs)

    if "amp" not in skip:
        torch.amp.GradScaler = GradScaler

    # ---- torch.cuda.* answered by torch.xpu ------------------------------------
    real_is_available = cuda.is_available
    real_device_count = cuda.device_count

    def is_available():
        if from_native_lib():
            return real_is_available()
        return True

    def device_count():
        if from_native_lib():
            return real_device_count()
        return xpu.device_count()

    def is_initialized():
        if from_native_lib():
            return False
        return True

    def get_device_capability(device=None):
        return (8, 0)  # an Ampere-class answer: enables bf16/SDPA, no FP8/Hopper paths

    class _Props:
        _defaults = {"major": 8, "minor": 0, "is_integrated": False,
                     "is_multi_gpu_board": False, "multi_processor_count": None,
                     "max_threads_per_multi_processor": 2048,
                     "regs_per_multiprocessor": 65536, "warp_size": 32,
                     "L2_cache_size": 18 * 1024 * 1024, "uuid": None,
                     "gcnArchName": ""}

        def __init__(self, p):
            self._p = p

        def __getattr__(self, name):
            try:
                return getattr(self._p, name)
            except AttributeError:
                if name == "multi_processor_count":
                    return getattr(self._p, "gpu_subslice_count", None) or \
                        getattr(self._p, "max_compute_units", 32)
                if name in self._defaults:
                    return self._defaults[name]
                raise

        def __repr__(self):
            return repr(self._p)

    def get_device_properties(device=None):
        return _Props(xpu.get_device_properties(_xpu_index(device)))

    def _xpu_index(device):
        if device is None:
            return xpu.current_device()
        d = m(device)
        if isinstance(d, Device):
            return d.index if d.index is not None else xpu.current_device()
        if isinstance(d, str):
            d = Device(d)
            return d.index if d.index is not None else xpu.current_device()
        return d

    def set_device(device):
        xpu.set_device(_xpu_index(device))

    def get_device_name(device=None):
        return xpu.get_device_name(_xpu_index(device))

    def mem_get_info(device=None):
        return xpu.mem_get_info(_xpu_index(device))

    def _noop(*a, **k):
        return None

    def _zero(*a, **k):
        return 0

    def _with_index(fn):
        def w(device=None, *a, **k):
            return fn(_xpu_index(device), *a, **k)
        return w

    def custom_fwd(fwd=None, *, cast_inputs=None):
        return torch.amp.custom_fwd(fwd, device_type="xpu", cast_inputs=cast_inputs)

    def custom_bwd(bwd=None):
        return torch.amp.custom_bwd(bwd, device_type="xpu")

    def cuda_autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        return RealAutocast("xpu", dtype=dtype, enabled=enabled,
                            cache_enabled=cache_enabled)

    def cuda_grad_scaler(*args, **kwargs):
        return RealGradScaler("xpu", *args, **kwargs)

    direct = {
        "is_available": is_available,
        "device_count": device_count,
        "is_initialized": is_initialized,
        "get_device_capability": get_device_capability,
        "get_device_properties": get_device_properties,
        "get_device_name": get_device_name,
        "set_device": set_device,
        "mem_get_info": mem_get_info,
        "current_device": xpu.current_device,
        "synchronize": xpu.synchronize,
        "empty_cache": xpu.empty_cache,
        "init": getattr(xpu, "init", _noop),
        "_lazy_init": getattr(xpu, "_lazy_init", _noop),
        "is_bf16_supported": lambda *a, **k: True,
        "ipc_collect": _noop,
        "can_device_access_peer": lambda *a, **k: False,
        "utilization": _zero,
        "Stream": xpu.Stream,
        "Event": xpu.Event,
        "stream": xpu.stream,
        "device": xpu.device,
        "device_of": getattr(xpu, "device_of", xpu.device),
        "current_stream": xpu.current_stream,
        "default_stream": getattr(xpu, "default_stream", xpu.current_stream),
        "set_stream": getattr(xpu, "set_stream", _noop),
    }
    for name in ("memory_allocated", "max_memory_allocated", "memory_reserved",
                 "max_memory_reserved", "reset_peak_memory_stats",
                 "reset_accumulated_memory_stats", "memory_stats"):
        fn = getattr(xpu, name, None)
        if fn is not None:
            direct[name] = _with_index(fn)
    direct["reset_max_memory_allocated"] = direct.get("reset_peak_memory_stats", _noop)
    direct["reset_max_memory_cached"] = direct.get("reset_peak_memory_stats", _noop)
    direct["memory_cached"] = direct.get("memory_reserved", _zero)
    direct["max_memory_cached"] = direct.get("max_memory_reserved", _zero)
    direct["set_per_process_memory_fraction"] = getattr(
        xpu, "set_per_process_memory_fraction", _noop)
    if hasattr(xpu, "memory_summary"):
        direct["memory_summary"] = xpu.memory_summary
    else:
        direct["memory_summary"] = lambda *a, **k: (
            "xpu: %.2f GiB allocated" % (xpu.memory_allocated() / 2**30))
    for name in ("manual_seed", "manual_seed_all", "seed", "seed_all",
                 "initial_seed", "get_rng_state", "set_rng_state",
                 "get_rng_state_all", "set_rng_state_all"):
        fn = getattr(xpu, name, None)
        if fn is not None:
            direct[name] = fn
    # Each torch.cuda.X becomes a fresh wrapper that
    #  * answers from torch.xpu for your code and for libraries (when you hand
    #    transformers device_map="cuda", it calls torch.cuda.* itself);
    #  * behaves exactly like the original (no CUDA) for torch itself -- e.g.
    #    inductor expects get_device_properties() to raise and catches that;
    #  * is never the same object as torch.xpu.X: dynamo keys its handlers by
    #    function object and asserts on duplicates.
    # (is_available / device_count above stay truthful for every XPU-aware
    # library, so by default they still pick their own XPU code paths.)
    def gated(ours, orig, name):
        if isinstance(ours, type):
            return type(ours.__name__, (ours,), {})
        if orig is None:
            def w(*a, **k):
                return ours(*a, **k)
        else:
            def w(*a, **k):
                if from_native_lib(within=torch_itself):
                    return orig(*a, **k)
                return ours(*a, **k)
        w.__name__ = w.__qualname__ = name
        w.__doc__ = getattr(orig, "__doc__", None) or getattr(ours, "__doc__", None)
        return w

    if "cuda_api" in skip:
        direct = {}
    for name, fn in direct.items():
        if name in ("is_available", "device_count", "is_initialized"):
            setattr(cuda, name, fn)  # already gated
            continue
        setattr(cuda, name, gated(fn, getattr(cuda, name, None), name))

    # torch.cuda.amp (deprecated but everywhere in older repos)
    amp = cuda.amp
    amp.autocast = cuda_autocast
    amp.GradScaler = cuda_grad_scaler
    amp.custom_fwd = custom_fwd
    amp.custom_bwd = custom_bwd

    # NVTX ranges are profiling annotations: make them harmless no-ops
    import contextlib

    nvtx = cuda.nvtx
    nvtx.range_push = _zero
    nvtx.range_pop = _zero
    nvtx.mark = _noop
    nvtx.range_start = _zero
    nvtx.range_end = _noop
    nvtx.range = lambda *a, **k: contextlib.nullcontext()
