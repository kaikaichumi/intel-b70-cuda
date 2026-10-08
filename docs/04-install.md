English | [繁體中文](04-install.zh-TW.md)

# 04 Installing from scratch and everyday use

Everything needed to take an Ubuntu host with an Arc Pro B70 (or another Intel GPU using the `xe` driver) from a clean system to running CUDA-style programs is here.
The two layers can be installed separately: to run Python/PyTorch AI tools only, the PyTorch layer is enough; the toolchain is only needed to compile `.cu` source code (including PyTorch CUDA extensions).

## Host requirements

| | Requirement | Notes |
|---|---|---|
| OS | Ubuntu 24.04 or later, kernel 6.8 or later (tested on 26.04, kernel 7.0) | The GPU must be bound to the `xe` driver: `ls /sys/class/drm/renderD*/device/driver` shows `xe` |
| Docker | Any recent version; the user is in the `docker` group | Everything runs inside containers; the host gets no oneAPI, no compiler, and the system Python is left untouched |
| Disk | About 20 GB for the PyTorch layer (image + model cache); about another 10 GB for the toolchain | Both can live on a large disk (`B70_DATA`) |
| Network | The first run downloads the images and the chipStar source code (several GB) | Works offline afterwards |

The host does **not** need Intel's compute runtime or Level Zero installed; the container images bring their own. Only the kernel's `xe` driver is required (built into Ubuntu 24.04 and later).

## Installation

```bash
git clone https://github.com/kaikaichumi/intel-b70-cuda ~/b70-cuda
mkdir -p ~/.local/bin
ln -s ~/b70-cuda/pytorch-layer/b70 ~/.local/bin/b70
ln -s ~/b70-cuda/toolchain/b70cuda ~/.local/bin/b70cuda
# make sure ~/.local/bin is in PATH (Ubuntu's default login shell adds it)

# optional: keep data and caches on a big disk
mkdir -p ~/.config/b70
echo 'B70_DATA=/mnt/bigdisk/b70' >> ~/.config/b70/config
```

### PyTorch layer (running AI tools)

```bash
b70 build          # build the b70-ai image: torch 2.13+xpu from vllm/vllm-openai-xpu plus the compatibility layer and common packages (~10 min)
b70 test           # self-tests: 15 checks, runs even while vLLM is up
```

### CUDA toolchain (compiling .cu source code and PyTorch CUDA extensions)

```bash
b70cuda setup      # download chipStar's patched LLVM 22 and the chipStar source (~5 GB, streamed straight into $B70_DATA/cuda)
b70cuda build      # build the dev image, apply toolchain/patches/, build and install chipStar, install rocPRIM/hipCUB (~10 min on 6 cores)
b70cuda test       # acceptance tests: 13 programs, 76 checks
```

When `b70 install` compiles CUDA extensions it uses this same toolchain installed in `$B70_DATA/cuda/install`, so b70-ai does not need to be rebuilt.
Four commands in total, about half an hour; after updating the repo later, only `b70 build` needs to be rerun (`b70cuda build` only when the toolchain patches have changed).

## Everyday use

### Running an existing AI project

```bash
cd ~/some-project                       # any repo written for NVIDIA
b70 install -r requirements.txt         # pip install into this folder's .b70venv; CUDA-only packages filtered out or replaced
b70 python app.py                       # "cuda" in the program becomes the B70
b70 shell                               # a shell inside the container; paths are the same as on the host
```

Packages that ship `.cu` source code (e.g. `pip install mamba-ssm`) also go through `b70 install`; it automatically switches to `b70cc` for compilation, and the first time takes a bit longer.

### Compiling CUDA programs

```bash
b70cuda cc -O3 main.cu -o main          # same usage as nvcc (--use_fast_math, -arch=sm_xx are accepted)
b70cuda run ./main                      # run on the B70
b70cuda shell                           # dev environment with b70cc on PATH; for CMake add -DCMAKE_CUDA_COMPILER=$(which b70cc)
```

The first time a program runs, IGC has to compile the SPIR-V into machine code (about 0.1 seconds to a few seconds); the result is cached in `$B70_DATA/cache`, so there is no wait afterwards.

### Sharing the GPU with long-running services such as vLLM

The B70 is a single card; if vLLM is running at the same time, it holds the VRAM. `b70 gpu` shows who is using it and how much is left; `b70 gpu free`/`b70 gpu llm` stop and restart vLLM
(the container name is set with `B70_VLLM`). Small models and the self-tests do not require stopping it.

## Things you will run into

| Symptom | Cause/fix |
|---|---|
| `b70 install` dropped a package | It only exists as a CUDA build (xformers, flash-attn, cupy, ...). diffusers, ComfyUI and the like automatically fall back to SDPA; the list is in `pytorch-layer/py/b70cuda/pipfilter.py` |
| The program says `Torch not compiled with CUDA enabled` | The compatibility layer is not active. Make sure it was run with `b70 python`; use `B70_CUDA_SKIP`/`B70_CUDA=0` to rule out whether a particular part of the layer is causing the problem |
| Extension behavior unchanged after a toolchain update | pip used an old wheel from its cache: `b70 install --no-build-isolation --no-deps --force-reinstall --no-cache-dir <package>` |
| The `.cu` uses `cooperative_groups`, `mma`/`wmma`, inline PTX, cuBLAS/cuDNN | Not supported by chipStar; for now the only option is to change the source code (see M5 in the [03 roadmap](03-roadmap.md)) |
| A kernel is very slow the first time it runs | IGC just-in-time compilation; from the second run on it comes from the cache |

## Where everything lives

| | Location |
|---|---|
| The tools themselves | `~/b70-cuda` (the two commands are symbolic links) |
| Models, pip and kernel caches | `$B70_DATA` (default `~/.local/share/b70`) |
| chipStar source code, build and install | `$B70_DATA/cuda` |
| Each project's own venv | `.b70venv` inside the project folder |
| Per-machine settings | `~/.config/b70/config` (`B70_DATA`, `B70_MOUNTS`, `B70_GPU_PCI`, `B70_VLLM`) |

Containers run as your own account, so the files they create are owned by you; your home directory, `$B70_DATA` and the current folder have the same paths inside the container.
