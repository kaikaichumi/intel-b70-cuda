English | [繁體中文](README.zh-TW.md)

# b70: Using the Arc Pro B70 as a CUDA Card

Goal: AI programs written for NVIDIA that you find online (image, video, speech, LLM, ...) run on the B70 **without changing the code**.
Whatever cannot be done fails with an error instead of silently falling back to the CPU.

## Three-minute quick start

```bash
b70 gpu free                           # large models need the whole card: stop vLLM first (the API on :8000 pauses)
cd ~/some-model-repo
b70 install -r requirements.txt        # automatically filters out CUDA-only packages
b70 python app.py                      # "cuda" in the program is automatically redirected to the B70
b70 gpu llm                            # bring vLLM back when you are done (loading takes a few minutes)
```

ComfyUI: `b70 comfyui`, then open `http://<host IP>:8188`. Put models in `$B70_DATA/ComfyUI/models/`.
Diffusers-format models downloaded from HF do not need a second copy: link them into `models/diffusers/<name>` and load them with the `DiffusersLoader` node.

Measured (with vLLM stopped): SDXL-Turbo 1024×1024, 4 steps in 2.9 s (diffusers, model-card code verbatim);
ComfyUI 512×512, 4 steps in 1.0 s.

## Commands

| Command | What it does |
|---|---|
| `b70 python x.py` / `b70 run CMD` | Run in the current directory on the B70 |
| `b70 shell` | Open a shell in the B70 environment |
| `b70 install ARGS` | `pip install` into this directory's `.b70venv` (created automatically the first time) |
| `b70 bg NAME CMD` | Run in the background (e.g. a gradio web UI); `b70 ps`, `b70 logs NAME`, `b70 stop NAME` |
| `b70 comfyui` / `b70 comfyui stop` | ComfyUI (installed automatically the first time) |
| `b70 gpu` | Who is using the card and how much VRAM is left |
| `b70 gpu free` / `b70 gpu llm` | Stop / start vLLM |
| `b70 build` | Rebuild the image (worth running once after the vLLM image is updated) |
| `b70 test` | Self-tests (small models; can run while vLLM is up) |

To turn the CUDA compatibility layer off temporarily: `B70_CUDA=0 b70 python x.py`.
To turn off only part of it (useful when hitting compatibility problems): `B70_CUDA_SKIP=load,amp b70 python x.py`;
the available names are `factories tensor is_cuda module load generator default_device amp cuda_api jitreg`,
and a single function name also works, e.g. `B70_CUDA_SKIP=arange`.

## What it does

1. **CUDA → XPU compatibility layer** (`b70cuda`, enabled automatically on `import torch`)
   - Everything that names a device is redirected: `.cuda()`, `.to("cuda")`, `device="cuda"`, `torch.device("cuda:0")`,
     `torch.Generator("cuda")`, `torch.load` (including checkpoints saved on an NVIDIA card), `autocast("cuda")`,
     `torch.cuda.amp.*`, `GradScaler`
   - `with torch.device("cuda")` and `torch.set_default_device("cuda")` are redirected to the B70 as well
   - `torch.cuda.*` (VRAM, synchronization, streams, events, random numbers, device info) is answered by `torch.xpu`
   - `torch.cuda.is_available()` returns True to your program; to libraries that already support XPU themselves (torch,
     transformers, diffusers, accelerate, ComfyUI, ...) it answers truthfully, so they default to their own tested XPU paths.
     Passing `"cuda"` to them explicitly (e.g. `device_map="cuda"`) works just as well
   - PyTorch's own internals (`torch.compile` etc.) still see the original `torch.cuda` and are unaffected
2. **Fake `flash_attn` and `sageattention`**: same interface as flash-attn 2 and SageAttention 2, implemented on top of PyTorch SDPA (which is already a flash kernel on the B70).
   flash_attn supports GQA, causal, sliding window, varlen and `bert_padding`; sageattention supports HND/NHD layouts, causal, GQA and the 2.x kernel entry points
3. **Install filtering** (`b70 install`):
   - torch / torchvision / torchaudio / triton are never reinstalled (so they cannot be swapped for CUDA builds)
   - CUDA-only packages are dropped: xformers, flash-attn, sageattention (the latter two have SDPA stand-ins in the image), nvidia-\*, cupy, tensorrt, apex, ...
   - Packages that ship `.cu` source code are compiled with `b70cc` instead (see [pytorch-ext](../pytorch-ext/))
   - `onnxruntime-gpu` is replaced with `onnxruntime`; `--index-url` entries pointing at CUDA wheels are dropped
   - Each directory gets its own `.b70venv`, so projects that need different transformers versions do not interfere with each other
4. **The `b70-ai` image**: reuses the torch 2.13+xpu already validated in the vLLM image,
   plus common packages such as diffusers, peft, gguf, gradio, bitsandbytes, av, kornia and spandrel

## What still does not run

| Item | Reason / alternative |
|---|---|
| Packages that ship their own `.cu` files (extensions that need `nvcc`) | **Now works**: `b70 install` compiles them with `b70cc` from the [CUDA toolchain](../toolchain/) instead (see [pytorch-ext](../pytorch-ext/); causal-conv1d and mamba-ssm verified). Extensions that use cuBLAS/cuDNN/tensor cores (`mma`) are the exception |
| Attention packages written in CUDA (sageattention, flash-attn) | Both are replaced by SDPA-based stand-ins inside the image (`import sageattention` / `import flash_attn` keep working). SDPA is already a fused kernel on the B70; the Triton version of sageattention actually measured 17× slower |
| xformers | Just remove it; diffusers / ComfyUI fall back to SDPA automatically |
| CUDA Graphs, `torch.cuda.graph` | Not supported |
| cupy, TensorRT, `onnxruntime-gpu` | Use numpy / torch, OpenVINO, `onnxruntime` |
| Multiple cards (NCCL) | There is only one card; not applicable |

## Where things live

| | |
|---|---|
| The tool itself | This directory (the `b70` command is symlinked to `~/.local/bin/b70`, see [docs/04-install.md](../docs/04-install.md)) |
| Models and caches (HF, pip, kernel build cache) | `$B70_DATA`, default `~/.local/share/b70` |
| ComfyUI | `$B70_DATA/ComfyUI` |

The container runs as your user account, so the files it creates are owned by you; your home directory, `$B70_DATA` and the current directory have the same paths inside the container.

## Configuration (`~/.config/b70/config`)

Per-machine settings go here in `KEY=VALUE` format; environment variables take precedence:

```bash
B70_DATA=/mnt/bigdisk/b70        # where models and caches go (a large disk is recommended)
B70_MOUNTS="/mnt/bigdisk"         # extra directories to make visible inside the container, space-separated
B70_GPU_PCI=0000:03:00.0          # set this when there are several Intel cards; default: first card using the xe driver
B70_VLLM=vllm                     # name of the vLLM container that shares the GPU with B70 (used by b70 gpu free/llm)
```
