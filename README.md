English | [繁體中文](README.zh-TW.md)

# b70-cuda

**CUDA-style computing on the Intel Arc Pro B70 (Battlemage / Xe2).**
Two layers: a PyTorch layer that lets CUDA-flavoured Python code run unchanged on the B70,
and a source-level CUDA toolchain that compiles `.cu` code (including PyTorch CUDA extensions)
for the B70 through SPIR-V and Level Zero.

Makes the Intel Arc Pro B70 feel like a CUDA card. Two layers:

| Layer | What it solves | Status |
|---|---|---|
| [PyTorch layer](pytorch-layer/README.md) (`b70` command) | AI tools written in Python/PyTorch: `"cuda"`, `.cuda()`, `torch.cuda.*` and `flash_attn` in the code run on the B70 without changes; `pip install` automatically filters out CUDA-only packages and swaps in Triton versions | Usable; self-tests 15/15, SDXL-Turbo, ComfyUI and Liger-Kernel verified in practice; flash_attn and sageattention replaced by an SDPA stand-in |
| [CUDA toolchain](toolchain/README.md) (`b70cc`) | `.cu` programs with source code: compile them the same way as with nvcc and run them on the B70 | Usable; all 76 acceptance tests pass, 97% of the 61 HeCBench benchmarks compile with 0 wrong results, kernel time on par with the SYCL versions (geometric mean 0.89, see [roadmap](docs/03-roadmap.md)) |
| [PyTorch CUDA extensions](pytorch-ext/README.md) | pip packages that ship `.cu` files (`CUDAExtension`) are compiled automatically by `b70 install` using `b70cc`; kernels read and write torch XPU tensors directly | Usable; causal-conv1d and mamba-ssm build with unmodified source code and pass the upstream tests |

## How to use it in three minutes

```bash
# after installing (see docs/04-install.md, four commands)
cd ~/some-nvidia-project
b70 install -r requirements.txt     # pip; CUDA-only packages are handled automatically
b70 python app.py                   # "cuda" becomes the B70 automatically

b70cuda cc -O3 main.cu -o main      # same usage as nvcc
b70cuda run ./main
```

## What it does not do

No binary compatibility, i.e. it does not run programs or PTX built by the NVIDIA toolchain. NVIDIA's CUDA license terms forbid
running its compiled output on non-NVIDIA hardware, so this project only deals with source code: Python code and `.cu` source code.
Closed-source software that ships only as executables is out of scope. The reasons and a comparison with other approaches are in [docs/00-background.md](docs/00-background.md).

Not possible yet (chipStar limitations): cooperative groups, tensor cores (`mma`/`wmma`), inline PTX, cuBLAS/cuDNN/cuFFT, CUDA Graphs.

## Documentation

| Document | Contents |
|---|---|
| [00 Research background](docs/00-background.md) | Comparison of existing approaches (ZLUDA, chipStar, SCALE, SYCLomatic, PyTorch XPU) and why this route was chosen |
| [01 PyTorch layer](docs/01-pytorch-layer.md) | Design of the `b70cuda` compatibility layer, test results, pitfalls |
| [02 B70 hardware and driver](docs/02-b70-hardware-driver.md) | Xe2 architecture, driver stack, and how CUDA concepts map onto the B70 |
| [03 Roadmap](docs/03-roadmap.md) | Milestones, acceptance criteria, all measured numbers, optimization and bug-hunting records (IGC shared memory bug, fast math, torch XPU conv1d bug) |
| [04 Install and usage](docs/04-install.md) | From a clean Ubuntu to a working setup, plus everyday commands and FAQ |

## Test environment

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB, BMG-G31 (32 Xe-cores) |
| OS | Ubuntu 26.04, kernel 7.0, `xe` driver |
| Driver | Intel compute-runtime 26.05, IGC 2.28/2.38, Level Zero loader 1.28 |
| PyTorch | 2.13.0+xpu, Triton 3.7 (XPU) |
| Toolchain | chipStar (pinned commit, see `toolchain/b70cuda`) + 4 patches, LLVM 22 |

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, modify and distribute, **not for commercial purposes**.
Exception: `toolchain/patches/` contains modifications to chipStar (MIT); to make upstreaming easy, these patches are licensed under [MIT](toolchain/patches/LICENSE).
Licenses of dependencies: chipStar, rocPRIM and hipCUB are MIT, LLVM is Apache-2.0 with LLVM exception, PyTorch is BSD-3.
