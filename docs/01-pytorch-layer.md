English | [繁體中文](01-pytorch-layer.zh-TW.md)

# 01 PyTorch layer: running CUDA-style Python code directly on the B70

Usage instructions are in [pytorch-layer/README.md](../pytorch-layer/README.md). This document covers the design rationale, test results and pitfalls.

## Components

| Part | Purpose |
|---|---|
| `b70-ai` image | Layered on top of `vllm/vllm-openai-xpu` (torch 2.13+xpu, triton-xpu 3.7.2, oneAPI runtime), adding only about 0.6 GB: diffusers, peft, gguf, gradio, bitsandbytes 0.50.2, av, kornia, spandrel, etc. |
| `b70cuda` | CUDA-to-XPU compatibility layer, enabled automatically on `import torch` (`.pth` + import hook; programs that never import torch are unaffected) |
| `flash_attn` stand-in | The flash-attn 2 interface, backed by PyTorch SDPA (which is itself a flash kernel on the B70); ships with dist-info so `pip` treats it as installed |
| `b70` command | run / python / shell / install / bg / gpu / comfyui / build / test |
| `b70 install` | Filters requirements: keeps the XPU builds of the torch family, drops CUDA-only packages and CUDA wheel indexes, and installs into each folder's own `.b70venv` |

How `.b70venv` works: `python -m venv --without-pip`, then drop in a `.pth` containing
`import site; site.addsitedir("/opt/venv/lib/python3.12/site-packages")`.
The venv therefore inherits every package in the image, and `pip` treats them as installed; when a project needs a different version, whatever is installed into the venv takes precedence.
`PIP_CONSTRAINT` pins torch / torchvision / torchaudio / triton / intel-*; any package that tries to replace them fails outright
instead of silently swapping in a CUDA build.

## The compatibility layer's three-way decision

`torch.cuda.*` answers differently depending on who is calling:

| Caller | `is_available()` / `device_count()` | Other `torch.cuda.*` |
|---|---|---|
| PyTorch itself (torch, triton) | Answers truthfully (no CUDA) | Original behavior |
| Libraries with native XPU support (transformers, diffusers, accelerate, ComfyUI...) | Answers truthfully, so they default to their own XPU path | Redirected to `torch.xpu` (must work when you explicitly pass `"cuda"`) |
| User code | Reports that CUDA devices exist | Redirected to `torch.xpu` |

Device-name translation (`"cuda"`, `"cuda:N"`, `torch.device("cuda")`, integers) applies to every caller, because it is always correct to do so.

Activation timing (fixed 2026-10-08): the import hook installed by `b70cuda.pth` waits until torch has **actually finished executing** before applying the layer and removing itself. It used to retire "the first time anyone asks about torch",
but libraries such as transformers probe with `importlib.util.find_spec("torch")` first; once the probe triggered the hook, it missed the real import,
so a program that did "import transformers, then import torch" got no compatibility layer at all (caught by the upstream tests using mamba-ssm).

`Tensor.is_cuda` (added 2026-10-08): kernel packages routinely gate startup with `assert x.is_cuda` (sageattention, Liger,
assorted CUDA extensions), so XPU tensors answer True to those callers; torch and triton themselves still see the real value (they rely on it for device dispatch),
and `is_xpu` is left untouched. Disable with `B70_CUDA_SKIP=is_cuda`.

## Test results

**Self-tests** (`b70 test`, using only about 1 GB of VRAM): 14/14 passed.

- Hard-coded CUDA code: device strings, `with torch.device`, `torch.cuda.amp` training, Generator, checkpoint
- flash_attn stand-in compared against the reference implementation: max error 0.0077; GQA, sliding window and varlen all correct
- Others: `torch.compile`, diffusers, transformers `device_map="cuda"`, safetensors, bitsandbytes NF4

**Real models**:

| Test | Result |
|---|---|
| SDXL-Turbo, model-card code verbatim (`pipe.to("cuda")`), diffusers | Load 17.5 s; **1024x1024, 4 steps 2.9 s**; peak VRAM 10.5 GB |
| ComfyUI 0.39.0 (auto-installed by `b70 comfyui`), SDXL-Turbo via the API | Auto-detects `xpu:0 Arc Pro B70`; after model load **512x512, 4 steps 1.0 s** |

ComfyUI reads diffusers-format models straight from the HF cache: link the snapshot directory to `ComfyUI/models/diffusers/<name>`
and use the `DiffusersLoader` node in the workflow (`pytorch-layer/tests/comfyui_api_test.py`); no need to download a single-file checkpoint again.

## Pitfalls

1. **`torch.cuda.X` cannot simply be `torch.xpu.X`**. dynamo registers handlers keyed by function object, and registering the same object twice trips an assert
   ("Handler already registered for current_stream"). Each one needs to be wrapped in a fresh function.
2. **PyTorch internals must see the original `torch.cuda`**. inductor calls `torch.cuda.get_device_properties()`
   expecting it to raise and then catches the exception; if it returns the B70's information instead, inductor goes on to read `gcnArchName` and crashes.
3. **TorchScript identifies builtins by object id**. Wrapped functions such as `torch.zeros` must be registered as the same aten op via `torch.jit._builtins._register_builtin`,
   otherwise kornia fails the moment it is imported (it scripts functions at import time), and diffusers fails along with it.
4. **`torch.utils._device._device_constructors()` compares by object** (`with torch.device(...)` and `set_default_device` both depend on it).
   C++ hands it the native functions, but the set contains the wrapped versions, so nothing matches and the device setting has no effect.
   transformers 5 uses `torch.tensor([]).device` to detect meta initialization; when that detection fails, non-persistent buffers (CLIP's `position_ids`)
   end up uninitialized, which on the GPU shows up as an embedding "index out of bounds" and floods the output with millions of assert lines, looking like a hang.
   Fix: put both the native functions and the wrapped versions in the set.
5. **The Dockerfile must not `chmod -R` on `/opt/venv`**. It copies the whole venv into the build cache, eating 10 GB in one go.

## Known limitations

| Item | Reason / alternative |
|---|---|
| Packages with `.cu` files that need nvcc | Handled by the CUDA toolchain: `b70 install` automatically switches to compiling with `b70cc` ([pytorch-ext](../pytorch-ext/); causal-conv1d and mamba-ssm tested and passing, see the [roadmap](03-roadmap.md) M4) |
| xformers | Just remove it; diffusers/ComfyUI fall back to SDPA |
| CUDA Graphs | Not supported |
| cupy, TensorRT, `onnxruntime-gpu` | Use numpy/torch, OpenVINO, `onnxruntime` instead |
