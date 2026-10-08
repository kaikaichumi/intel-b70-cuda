English | [繁體中文](README.zh-TW.md)

# toolchain: CUDA Source Code Toolchain for the B70

Compiles `.cu` source code for the Arc Pro B70. Underneath is [chipStar](https://github.com/CHIP-SPV/chipStar)
(CUDA/HIP → SPIR-V → Level Zero); this directory builds it on the B70 and wraps it in convenient commands.

## Requirements

- Linux, Intel GPU using the `xe` driver
- Docker (the user must be in the `docker` group)
- About 10 GB of disk space for LLVM and chipStar (under `$B70_DATA/cuda`, which can be placed on a large disk)

No compiler or oneAPI needs to be installed on the host; everything lives in the `b70-cuda-dev` container (see `Dockerfile.dev`).

## Installation

```bash
ln -s "$PWD/b70cuda" ~/.local/bin/b70cuda
b70cuda setup     # download chipStar's patched LLVM 22 and the chipStar source code (~5 GB, streamed straight into $B70_DATA/cuda)
b70cuda build     # build the dev image (first time), compile and install chipStar (about 10 minutes on 6 cores),
                  # then install rocPRIM and hipCUB (the CUB counterpart, header-only; redo on its own with b70cuda libs)
b70cuda test      # acceptance tests: tests/run_cuda_tests.sh
```

`build` first applies the fixes in `patches/` (the chipStar version is pinned by `CHIPSTAR_COMMIT`):

| Patch | What it does |
|---|---|
| `0001-level0-fast-event-record` | `cudaEventRecord` 480 µs → 9 µs; external (SYCL) events can be waited on with `cudaStreamWaitEvent` |
| `0002-cucc-cxx20` | `cucc` accepts `-std=c++20`/`c++23` (PyTorch 2.13 builds extensions with c++20) |
| `0003-hipdynmem-opaque-base` | Works around an Intel IGC bug: when dynamic shared memory (`extern __shared__`) is accessed at a constant offset of 32–64 KB, sub-group-uniform reads and writes return 0 (hipCUB's BlockReduce breaks when placed in the tail of a large buffer; this is how the Mamba backward pass broke). The access now goes through an offset that is only known to be 0 at run time, keeping the address in a register |
| `0004-fast-intrinsics-always` | CUDA's fast intrinsics (`__expf`, `__logf`, `__fdividef`, `__powf`, `__tanf`, ...) are **always** the fast version, as with nvcc. chipStar originally used OpenCL's `native_*` only under `-DCHIP_FAST_MATH`, and its default was actually slower than the precise version (`__expf·__sinf+__fdividef+__powf` 1.33 ms vs `expf...` 0.92 ms; 0.26 ms after the fix). The exceptions are `__sinf`/`__cosf`/`__sincosf`: Intel's `native_sin`/`native_cos` have an error of 3e-5 while CUDA guarantees 2^-21.4 (4e-7), so these three stay on the precise version and only use native under `--use_fast_math`. Add `-DCHIP_PRECISE_INTRINSICS` to restore the original behavior entirely |

## Floating-point flags (handled for you by `b70cc`)

| What you write | What b70cc actually does | Why |
|---|---|---|
| (nothing) | Adds `-fno-math-errno` on the device side | CUDA device code never sets errno anyway; chipStar keeps errno by default, which stops clang from turning `expf`/`sqrtf` into IGC's hardware versions, making precise transcendentals 2-3× slower (`bench/fast_math.cu`: 0.92 ms → 0.31-0.41 ms, same precision) |
| `--use_fast_math` | `-DCHIP_FAST_MATH` + `-fapprox-func -freciprocal-math -ffp-contract=fast -fgpu-flush-denormals-to-zero` on the device side | Matches nvcc's definition (approximate transcendentals, approximate division and square root, FMA, FTZ). chipStar's `cucc` originally ignored this flag outright |
| `-ffast-math` (including `-Xcompiler -ffast-math`) | The set above, plus `-funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros` etc. on both the device and host side | chipStar's `hipcc` swallows this flag whole and keeps only `-DCHIP_FAST_MATH`, so it is split into its individual sub-flags and passed down |

Measured native version of each function on inputs in (0.01, 3) (`toolchain/probe/intrinsics_probe.cu`; relative error and compute-bound speedup):
exp 2.9e-7/2.0×, log 1.0e-7/4.7×, log2 1.0e-7/3.1×, log10 1.5e-7/3.2×, exp10 2.6e-7/2.3×, divide 9.7e-8/2.3×,
powr 2.4e-7/5.6×, tan 7.3e-7/2.6×; **sin 3.4e-5/2.3×, cos 3.1e-5/2.6×** (which is why sin/cos are not used by default); exp2 and sqrt are already just as fast.
`--use_fast_math` overall: `expf·sinf+cosf·logf+powf` maximum relative error 1.4e-5 (default 2.4e-7).

`setup` does not use `docker pull`; it streams and unpacks the official image's contents directly with `crane`, so the system disk only grows by the dev image's layer.

## Usage

```bash
b70cuda cc -O3 my.cu -o my     # same usage as nvcc
b70cuda run ./my               # run on the B70
b70cuda shell                  # enter the dev environment; b70cc and icpx are available directly inside
```

CMake projects: inside `b70cuda shell`, add `-DCMAKE_CUDA_COMPILER=$(which b70cc)`.

## Configuration

Shares `~/.config/b70/config` with the `b70` command:

| Variable | Default | Purpose |
|---|---|---|
| `B70_DATA` | `~/.local/share/b70` | Data and caches |
| `B70_CUDA_HOME` | `$B70_DATA/cuda` | LLVM, chipStar source code, build and install directories |
| `B70_GPU_PCI` | auto-detected | Set this when there are several Intel cards |
| `CHIP_LOGLEVEL` | `err` | chipStar debug messages (`trace`/`debug`/`info`) |

## Current support level

See the measured results in [docs/03-roadmap.md](../docs/03-roadmap.md). Known chipStar limitations: no inline PTX,
tensor cores (`wmma`/`mma`), NVIDIA libraries such as cuBLAS/cuFFT, or CUDA Graphs.
