English | [繁體中文](03-roadmap.zh-TW.md)

# 03 Roadmap: Milestones, Acceptance Criteria, Tests and Performance Targets

Scope: **source-code level**. Programs that come with `.cu` source code are compiled with `b70cc` (same usage as nvcc) and run on the B70.
No binary compatibility (see [00](00-background.md) for why).

## Architecture

```
.cu source code
  │  b70cc (nvcc-compatible compiler entry point)
  ▼
clang (chipStar-patched LLVM 22) + CUDA header mapping + chipStar's LLVM passes
  │
  ▼
SPIR-V (embedded in the executable)
  │  at run time: b70 runtime (CUDA Runtime API → chipStar → Level Zero)
  ▼
IGC compiles SPIR-V to Xe2 instructions → B70
```

The toolchain is based on [chipStar](https://github.com/CHIP-SPV/chipStar); this project is responsible for:

- Building and validating the toolchain on the B70
- Filling in the features missing on the B70 and tuning for Xe2
- Hooking into PyTorch XPU so that CUDA extensions operate directly on torch tensors

## Milestones

| | Content | Acceptance criteria | Status |
|---|---|---|---|
| **M0** Project skeleton | Project folders, documentation, personal information removed | Can be `git init`-ed and pushed to GitHub as is | Done |
| **M1** Toolchain online | chipStar toolchain usable on the B70; `b70cc` compiler entry point; hardware and driver investigation ([02](02-b70-hardware-driver.md)) | ① chipStar unit-test pass rate on the B70 (Level Zero) matches its known-failure list for Intel GPUs<br>② Standard CUDA samples (vectorAdd, matmul, reduction, scan, transpose, histogram, etc.) compile with `b70cc` without source changes and produce correct results<br>③ `warpSize == 32` inside kernels | ②③ passed (`tests/run_cuda_tests.sh`: 13 programs, 76 checks all pass, including CUB block primitives, dynamic shared memory, fast intrinsics); ① not run yet |
| **M2** Correctness | Run the CUDA versions of HeCBench | ① Pick 50+ benchmarks; 80%+ compile and verify correct<br>② Every failure classified (missing API, missing device function, driver issue, performance timeout) and recorded | ① 61 benchmarks: 97% compile, 85% run to completion; all with self-checks pass; ② classified (see below) |
| **M3** Performance | Micro-benchmarks and application benchmarks | ① HeCBench: CUDA version (b70cc) vs SYCL version (icpx) of the same benchmark, geometric mean ≥ 0.8x<br>② Memory bandwidth (STREAM-like copy, triad) ≥ 90% of the SYCL measurement on the same card<br>③ Kernel launch latency: measure, compare with SYCL, record the gap<br>④ Every optimization records before/after numbers | ① wall-clock time **0.88x** (34 benchmarks), kernel time **0.89x** (32 benchmarks, see "M3 round 2"; most of the wall-clock gap is the CPU side under icpx's default fast-math); ② passed (about 100%); ③ CUDA 1.95 µs vs SYCL 2.33 µs; ④ optimization 1 (events) and optimization 2 (math flags) both have before/after numbers |
| **M4** PyTorch CUDA extensions | `b70 install` switches to compiling with `b70cc` when it meets a package with `.cu` files; extension kernels read and write torch XPU tensors directly (shared Level Zero context, no copies). The three hard parts are in "The actual M4 work" below | ① A self-written sample extension (`CUDAExtension` style, using `ATen/cuda` headers) produces the same results on XPU tensors as on CPU<br>② At least one real open-source CUDA extension package installs and passes its own tests<br>③ Packages that have a Triton version are automatically switched to the Triton version by `b70 install` | ① passed (6/6); ② passed (causal-conv1d 1.7.0 and mamba-ssm 2.2.5 install with unmodified source code and pass correctness checks, see the log below); ③ passed (Liger-Kernel runs directly; sageattention automatically swapped for the Triton version; `test_triton*.py`) |
| **M5** Broader support | `_sync` warp functions, float atomics (B70 hardware support), tensor cores mapped to XMX, cuBLAS→oneMKL | Ordered by the gaps found in M2–M4, tests added item by item | |

## M2/M3 HeCBench results (2026-10-08, `tests/hecbench/results-2026-10-08.txt`)

61 benchmarks (`tests/hecbench/list.txt`), CUDA version with `b70cc`, SYCL version with `icpx`, each run once for wall-clock time (other work was on the GPU at the same time; the numbers are only meaningful relative to each other).

| | Round 1 | Rerun of 4 after adding CUB and libomp (`results-2026-10-08-rerun.txt`) |
|---|---|---|
| CUDA version compiles | 57/61 (93%) | **59/61 (97%)** |
| Runs to completion | 48 (79%) | **52 (85%)** |
| Self-check prints PASS | 32; another 16 programs have no self-check at all | **36** |
| Prints FAIL (wrong result) | **0** | **0** |
| Both sides verified, performance comparable | 30 benchmarks, geometric mean SYCL/CUDA time = **0.88** (CUDA version 12% slower) | 34 benchmarks; the 4 rerun ratios are 0.80/0.98/0.90/1.08 |

Performance outliers: CUDA version clearly slower on `tsa` 0.34, `hausdorff` 0.37, `bitonic-sort` 0.65; clearly faster on `gaussian` 2.30, `iso2dfd` 1.68.
These are the optimization targets for the next M3 round.

Classification of the 13 failures:

| Cause | benchmark | Nature |
|---|---|---|
| Needs `cub/cub.cuh` | all-pairs-distance, laplace | The toolchain had no hipCUB installed at the time; `b70cc` now ships the `cub/` under `toolchain/include` by default, **both pass after rerun** |
| Needs `cooperative_groups.h` | layernorm, softmax | **A genuine chipStar gap** (cooperative groups unsupported) |
| Missing input data files (HeCBench's data directory must be downloaded separately) | bfs, cfd, hotspot, hotspot3D, kmeans, urng | Test environment, not the toolchain |
| `libomp.so` not found at runtime | bilateral, heat2d | Environment: the program uses OpenMP and the container's library path did not include LLVM's libomp; **passes after rerun** once `b70cuda` added the path |
| Timeout (600 s; the first half had already printed PASS) | convolution1D | Performance, filed under M3 |

Conclusion: not a single "wrong answer"; the only genuine toolchain gap is cooperative groups (2 benchmarks); the rest that did not finish are test data files (6) and one timeout.

## Reordering (2026-10-08)

M4 moved ahead of M2/M3. Reason: the user's goal is "install an AI tool and it just runs"; what gets stuck is packages with `.cu` files,
not HeCBench's scientific computing programs. M2/M3 run in the background; their results serve as a reference for toolchain correctness and performance and do not block M4.

### The actual M4 work

The hard part is not the compiler but these three things:

| Hard part | Explanation | Approach |
|---|---|---|
| **torch's `ATen/cuda` headers** | Almost every extension includes `ATen/cuda/CUDAContext.h` and uses `getCurrentCUDAStream()`, `CUDAGuard`, `at::Half`. The torch XPU build does not have these files | Write a set of compatibility headers with the same names, wired underneath to torch XPU's stream and device guard |
| **Sharing the GPU context** | chipStar has its own Level Zero context, torch XPU has its own SYCL queue; without sharing, tensor pointers are invalid inside kernels | Use chipStar's `hipInitFromNativeHandles` to hand torch's driver/device/context/queue over to it; synchronize by chaining events |
| **`setup.py` must not change** | Packages all write `CUDAExtension(...)`, which calls nvcc | Intercept `torch.utils.cpp_extension`: `CUDAExtension` compiles with `b70cc` instead and links the chipStar runtime |

One more cheap bonus: `triton-xpu` is already in the image, so kernels written in Triton run as is (Liger-Kernel is all Triton,
flash-attn also has a Triton version). Packages with a Triton alternative are swapped to it directly, without going through the compiler.

Steps: ① a minimal self-made extension (to force out the actual shape of the three hard parts) → ② a real package, candidates `causal-conv1d` (simple kernels, no inline PTX)
or Mamba's selective scan (uses CUB; chipStar has hipCUB to hook up).

### M4 ① measurements (2026-10-08)

`pytorch-ext/tests/ext_min`: an extension written the NVIDIA way (`CUDAExtension`, `ATen/cuda/CUDAContext.h`,
`AT_DISPATCH_FLOATING_TYPES`, `__shfl_down_sync`, `OptionalCUDAGuard`, `x.is_cuda()` checks), **source code untouched**.

| Check | Result |
|---|---|
| float/double correctness | error 0 |
| warp shuffle row sum (`warpSize` 32) | error 2e-5 |
| torch op → extension → torch op, interleaved 100 times, no synchronization added | error 0 |
| torch immediately reduces the extension's output | correct |
| Overhead per call (1024 elements) | **44 µs** (pure torch op 13 µs; host-side synchronous mode 300 µs) |

Actual solutions to the three hard parts (all under `pytorch-ext/`):

| Hard part | Solution |
|---|---|
| `ATen/cuda`, `c10/cuda` headers | Same-named compatibility headers in `include/`, placed ahead of torch's includes. The torch XPU wheel actually ships `ATen/cuda/*.h`, but they pull in cuBLAS/cuSPARSE headers and cannot be used directly |
| Context sharing and synchronization | `src/interop.cpp` (`libb70torch.so`): pulls the Level Zero driver/device/context out of torch's SYCL queue and hands them to `hipInitFromNativeHandles`. Synchronization goes through GPU events in both directions: the torch queue's barrier event is converted to a hip event that the CUDA stream waits on; the event recorded on the CUDA stream is converted to a SYCL event that the torch queue waits on before the next ATen op starts (`RecordFunction` callback) |
| `setup.py` unchanged | `py/b70torch`: intercepts `torch.utils.cpp_extension`, points `CUDA_HOME` at a fake directory (`bin/nvcc` is a wrapper that filters flags and calls cucc; `.cpp` files are also compiled in HIP mode, otherwise torch's Half.h breaks), and swaps out the cudart/c10_cuda etc. libraries |
| `is_cuda()`, `kCUDA`, `DeviceType::CUDA` | Cannot be changed with macros (would clash with torch's own declarations). Changed to a **shadow source tree**: the package directory is copied to `build/b70_src`, and only the package's own files are rewritten (the rule table in `rewrite.py`); torch headers are left alone |

**Stress test**: 20,000 extension calls interleaved with torch ops (`pytorch-ext/tests/ext_min`), both event mode and host mode pass, error 0.
(The first run segfaulted; gdb showed the test script had not set `CHIP_BE=level0`, so chipStar picked the OpenCL backend but was handed Level Zero handles;
the `b70` command and b70torch now both set `CHIP_BE=level0` as the default.)

**Synchronization overhead breakdown** (`B70TORCH_PROFILE=1`, one extension call + torch op pair 68 µs, two pure torch ops 25 µs; HeCBench was running at the same time, so the numbers are on the high side):

| Stage | Average |
|---|---|
| torch → CUDA: SYCL barrier event, convert to hip event, `hipStreamWaitEvent` | 2.6 + 0.8 + 6.6 µs |
| CUDA → torch: `hipEventRecord`, convert to SYCL event, queue barrier | **17.2** + 1.1 + 5.2 µs |

What can still be saved next: `hipEventRecord` is still the big one (chipStar writes a timestamp on every record, even with `cudaEventDisableTiming`);
the fundamental fix is to have chipStar's stream use torch's immediate command list directly, so that a single in-order queue needs no events at all.

chipStar patches added for this (all in `toolchain/patches/0001`): events wrapped from external handles are treated as already recorded and do not own the handle;
`hipStreamWaitEvent` on such an event waits on it directly; `cucc` accepts `-std=c++20` (patch 0002).
Also added a few small things torch headers need that chipStar lacks: `thrust/complex.h` (types and math functions only), `__ldg(const __half*)`,
`cudaFuncSetAttribute` (Xe2's SLM needs no allocation request; returns success directly).

**CUB**: real packages (causal-conv1d, Mamba) make heavy use of `cub::BlockLoad`/`BlockStore`/`BlockReduce`. chipStar does not ship CUB,
but chipStar maintains forks of rocPRIM and hipCUB (the HIP counterpart of CUB, header-only). `b70cuda build` now installs them into the toolchain as well,
and `toolchain/include/cub/*.cuh` maps `cub::` to `hipcub::` (`b70cc` carries this include by default, shared by HeCBench and PyTorch extensions).

**JIT path** (`torch.utils.cpp_extension.load_inline`/`load`): also goes through the rewrite (in place inside the build directory;
`load()`'s user files get a shadow tree). `pytorch-ext/tests/test_load_inline.py` passes (including the `is_cuda()` check, build 44 s).

### M4 ② measurements (2026-10-08): causal-conv1d 1.7.0

Dao-AILab's [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) (a Mamba dependency), PyPI source code **untouched**,
installed with `pip install --no-build-isolation` inside the `b70 install` virtual environment (`setup.py` goes through `CUDAExtension`;
the kernels use `cub::BlockLoad`/`BlockStore`, `c10::cuda::CUDAStream`, `__ldg`, `cudaFuncSetAttribute`, with fp32/fp16/bf16 dispatch).

`pytorch-ext/tests/test_causal_conv1d_pkg.py`: compared against the package's own `causal_conv1d_ref` (pure torch).

| Check | Result |
|---|---|
| Forward: 3 dtypes × width 2/4 × none/silu × channel-first/channel-last, 24 combinations | 24/24 PASS |
| Backward (dx, dw, db), same 24 combinations | 24/24 PASS |
| Max error | fp32 6e-6, fp16 4e-3, bf16 1.6e-2 (all within the dtype's rounding range) |
| Throughput (8×2048×4096 bf16, silu, Mamba size) | **kernel 1.07 ms, torch reference implementation 11.70 ms** (11x faster) |

Problems forced out by the install and already fixed: the `nvcc -V` version query, `-std=c++20`, `cub/block/block_load.cuh`, `cudaFuncSetAttribute`,
`__ldg(const __half*)`, `thrust/complex.h`, `hipStreamWaitEvent` handling of external events.
The most important point: **synchronization correctness** does not rely on `cudaDeviceSynchronize` but on chained GPU events, so the throughput numbers are not dragged down by host-side synchronization.

**The package's own tests** (the literal wording of acceptance criterion ②; upstream `tests/test_causal_conv1d.py`, v1.7.0, 13,373 parameter combinations):

| | Count | Notes |
|---|---|---|
| Passed | 9,005 | |
| Skipped | 3,888 | Skipped by upstream itself (channel-first does not support initial/final states) |
| Failed | 480 | **All of them are the reference implementation computing wrong, not the kernel**: 204 are all `seqlen=1` + channel-last; 276 varlen cases randomly cut out segments of length 1 |

Investigation result (reproducible with `pytorch-ext/tests/torch_xpu_conv1d_len1_bug.py`): **the weight gradient of `F.conv1d` in torch 2.13.0+xpu** is wrong when the input length is 1
and the stride of the length dimension is not 1 (`randn(B,1,C).transpose(1,2)`, stride `(C,1,C)`): torch treats it as a contiguous tensor (strides of size-1 dimensions are ignored),
but the XPU convolution backward passes the original strides to oneDNN. Forward is correct, groups=1 is also wrong, CPU is correct.
The kernel's answer matches hand calculation (at length 1 the weight gradient for the "past" position must be 0; the kernel gives 0, the torch XPU reference gives a non-zero value). This is a torch XPU bug worth reporting upstream.

**mamba-ssm 2.2.5** (same route, `selective_scan_cuda`: CUB's `BlockRakingLayout`, reverse scan, complex dtype, 10 `.cu` files):
the first compile was missing `cub/config.cuh`, `cub/detail/uninitialized_copy.cuh`, `cuda/std/type_traits` (libcu++);
after adding small headers mapping to hipCUB/the standard library, the whole package compiled and installed.
Side finding: 2.2.5 does not match the image's transformers version (it imports two classes that have been renamed); unrelated to the GPU, aliases were added in the test.

`pytorch-ext/tests/test_mamba_pkg.py` (compared against the package's own `selective_scan_ref`, tolerances as in upstream `tests/ops/test_selective_scan.py`,
but upstream only enables fp32; fp16/bf16 are tested in addition here):

| Check | Result |
|---|---|
| `selective_scan_fn` forward, fp32/fp16/bf16 × with/without z + softplus | 6/6 PASS |
| Backward (du, ddelta, dA, dB, dC, dD, dz, ddelta_bias) | 6/6 PASS |
| Whole `Mamba` block (fused path vs non-fused path with the same weights), forward and backward | error 0, 1e-6 |
| Throughput (8×2048×768 bf16, one layer forward) | fused 9.2 ms, non-fused 9.3 ms (at this size the convolution and scan are not the bottleneck) |

There was an episode with the half-precision backward: with bf16 plus z gating, the absolute error of dA reached 1e2, which looked like a wrong result; but dA itself is 1e5 (summed over batch×seqlen),
and recomputing the same reference with the inputs "before rounding to bf16" gives the same 1e2 gap: the error comes from the input precision itself, and the kernel is within 0.1% of the reference.
The test now lists this "input rounding floor" as a pass condition too.

**mamba-ssm upstream tests** (`tests/ops/test_selective_scan.py`, 20 cases): first run 16 passed, 4 failed, all failures being `seqlen ≥ 2048` combined with
B/C varying along the sequence: the `dA` gradient was **all zeros**, `dD` and `ddelta_bias` wrong, the other gradients right. A whole round of investigation (recorded below) ended at a bug in the Intel GPU compiler;
after adding patch 0003 to chipStar and recompiling mamba (`--no-cache-dir`, otherwise pip picks up the old wheel from its cache), **20/20 pass**,
with `seqlen` 2048/4096, various batch/dim/dstate, and gradients with and without z all within 1e-6 relative error.

### Investigation log: IGC constant offset bug (2026-10-08)

How the symptom was narrowed down (every step has a standalone `.cu` reproducer; the final minimal reproducer is `toolchain/probe/igc_slm_const_offset.cu`):

1. When `seqlen > 1024`, the mamba backward kernel switches to the 128-thread version; but BlockReduce itself is correct at 32–1024 threads.
2. The 128-thread version with B/C not varying along the sequence (non-variable) is completely correct; the only difference is the shared memory layout: the variable version adds two BlockExchange scratch areas,
   pushing BlockReduce's scratch to 33,792 B and the `dA` accumulation area to above 36 KB.
3. Reproduced independently with dynamic shared memory: `BlockReduce<float,128>` scratch placed at constant offset 50,688 B → result all zeros; placed at 25,344 B → correct.
   Offset changed to a runtime parameter → correct. `-O0` is also wrong → not an LLVM optimization problem; the IR chipStar generates is just a `gep i8, %dyn_local_mem, 32832` plus ordinary loads/stores, nothing wrong with it.
4. Sweeping offsets: **constant offsets between 32 KB and 64 KB that are not "round"** (32,832, 33,024, 34,816, 36,864, 50,688, 51,200) are wrong; 32,768, 40,960, 49,152, and 65,536 and above are all right.
   Only **sub-group-uniform (scalar) accesses** of the form "one lane per sub-group writes, thread 0 reads" are wrong; the vector path where each lane accesses on its own is correct.
5. The IGC in both images (2.28, 2.38) is affected.

The fix: chipStar's `HipDynMem` pass replaces `extern __shared__` with a hidden kernel argument; there we make all accesses go through
`arg + (ptrtoint(arg) >> 40)`, which is always 0 at runtime but the compiler cannot fold, so the address stays in a register (`toolchain/patches/0003`).
The cost is one extra offset addition per kernel. The first version used `>> 24`, and that broke `scan_dynamic_shared` in the acceptance tests instead:
the raw value of a local pointer is not an offset starting from 0 but `0x10000000 + offset` (IGC's SLM window), so shifting right by 24 gives 16, the whole dynamic buffer is pushed back by 16 bytes, and the last 4 elements of every block fall off the end.
After changing to a right shift by 40 both are right; `tests/cuda/cub_block.cu` adds the CUB block primitives (32–1024 threads, dynamic shared memory, large offsets) to the acceptance tests,
and `scan_dynamic_shared` guards that "the offset really is 0".

Two other things fixed along the way: `b70cc` now force-includes `b70/cuda_compat.h` (`cudaFuncSetAttribute`, which chipStar lacks)
and defines `__AMDGCN_WAVEFRONT_SIZE=32`; when rocPRIM does not see this hipcc macro it defaults to 64 and the lane mask type is wrong (it did not directly cause an error this time, but it is a time bomb).

**Compatibility layer bug** (found while running the upstream pytest): transformers probes whether torch exists with `importlib.util.find_spec("torch")`;
`b70cuda`'s original import hook was one-shot, and once triggered by this probe it retired, so the real `import torch` was never intercepted: any program that "imports transformers before torch"
ended up with no compatibility layer. Changed to retire only after torch has actually finished executing (all three import orders verified).

### M4 ③ measurements (2026-10-08): Triton packages

Prerequisite: `triton-xpu` 3.7.2 is in the image alongside the torch XPU wheel. `pytorch-ext/tests/test_triton.py`: Triton kernels written following the NVIDIA tutorials
with `device="cuda"` (vector add, softmax) run directly on the B70, 4/4 correct; 1M-element add Triton 39 µs, torch 33 µs.

Real packages (`pytorch-ext/tests/test_triton_pkgs.py`, all installed untouched with `b70 install`):

| Package | Approach | Result |
|---|---|---|
| **Liger-Kernel 0.8.4** (pure Triton, fused kernels for training) | Install directly, run directly | RMSNorm forward/backward, SwiGLU, fused linear cross-entropy all correct (fp32 error 1e-6, bf16 3e-2); fused linear CE 4.67 ms vs torch 3.89 ms |
| **sageattention** (2.x needs nvcc + sm80 PTX, was previously dropped entirely) | `pipfilter` gains a `TRITON_ALT` table: `sageattention>=2.0` → `sageattention==1.0.6` (the last pure-Triton version, same `sageattn` interface) | 4 shapes, causal/non-causal all correct (INT8 QK<sup>T</sup> error within 4e-2); but speed 31 ms vs torch SDPA 1.8 ms |

Two things fixed along the way: `Tensor.is_cuda` now returns True for XPU tensors (sageattention's very first line is `assert q.is_cuda`; see [01](01-pytorch-layer.md) for details);
Liger's RMSNorm backward rewrites the incoming gradient **in place** by default, so the test has to `clone()` first or the reference value gets clobbered (the first version of the test misdiagnosed this as a GPU computation error).

Performance observations: triton-xpu is within 20% of native torch on memory-bandwidth-bound kernels (RMSNorm, cross-entropy);
matrix-multiply-type Triton kernels (sageattention's attention) reach only about 4 TFLOPS, far from XMX, and torch's SDPA (oneDNN) is 17x faster.
Conclusion: the Triton alternative suits packages that "would not run at all otherwise"; it is not a performance route. Attention-type packages should be routed to SDPA instead (same as the `flash_attn` stand-in); filed as follow-up.

**Follow-up (2026-10-09)**: since sageattention's Triton version is 17x slower than SDPA, `b70 install` now drops sageattention outright,
and the image provides a stand-in implemented with SDPA (`pytorch-layer/py/sageattention/`, same approach as the `flash_attn` stand-in): `sageattn()` with HND/NHD layouts,
`is_causal`, `sm_scale`, GQA, and all the 2.x kernel entry point names are there; `return_lse` and `sageattn_varlen` are unsupported. The self-test `sageattention_shim`
compares against a reference attention and times 4k tokens.

## M1/M3 measurements

2026-10-08, vLLM idle on the B70 (occupying VRAM, not compute).

**Acceptance tests** (`tests/run_cuda_tests.sh`): 11/11 passed, none needed source changes. Covers `__shared__` (static and dynamic),
`__syncthreads`, warp shuffle (legacy and `_sync` versions), integer and float/double `atomicAdd`, streams, events,
pinned memory, `__constant__`, managed memory, device `printf`, `warpSize == 32`.

**Micro-benchmarks** (`bench/micro.cu` vs `bench/micro_sycl.cpp`, same sizes and work-groups, best wall-clock time of 10 runs):

| Item | CUDA (b70cc) | SYCL (icpx) | CUDA/SYCL |
|---|---|---|---|
| stream copy | 526–529 GB/s | 527–531 GB/s | 1.00 |
| stream triad | 535 GB/s | 533–535 GB/s | 1.00 |
| kernel launch latency | 1.91–1.95 µs | 2.22–2.46 µs | CUDA about 16% faster |
| float atomicAdd (1024 locations) | 7.03–7.10 G ops/s | 6.72–6.78 G ops/s | 1.04 |

**Why reduction showed only 14 GB/s**: the kernel is not slow (measured in isolation it takes 0.057 ms, `bench/reduction_probe.cu`);
the test's first kernel launch was inside the timed region, so what was measured was IGC compiling the module for the first time (about 90 ms). The test now warms up first.

### Optimization 1: `cudaEventRecord` 480 µs → 9 µs

While looking into the problem above, `bench/api_overhead.cu` showed that `cudaEventRecord` took 480 µs per call (normally single-digit microseconds),
and many CUDA programs record events every iteration. Measuring the Level Zero calls step by step with `toolchain/probe/ze_record_probe.c` found two causes:

| Cause | Cost | Fix |
|---|---|---|
| The GPU copies the timestamp into the event object, and the event object lives in ordinary pageable host memory | ~300 µs (copying into USM host memory takes only ~10 µs) | Timestamps moved to slots in a USM host memory pool |
| Every re-record creates a new event pool (the workaround for chipStar issue #1258) | ~300 µs per `zeEventPoolCreate` | Keep up to 8 old events that have already signaled completion for reuse (the safety condition is the same as the original destruction point) |
| Every record calls `zeDeviceGetGlobalTimestamps`, but the value is only used for ordering | ~15 µs | Use the host monotonic clock instead |

Tried but did not help: moving the timestamp write to the compute engine (to avoid crossing engines) was actually slower, reverted.

| | Before | After |
|---|---|---|
| `cudaEventRecord` (GPU idle) | 480 µs | **9 µs** |
| kernel + record, per iteration | 292 µs | **28 µs** (kernel alone 22 µs) |

The change lives in `toolchain/patches/0001-level0-fast-event-record.patch`; `b70cuda build` applies it automatically.
Acceptance tests still pass 11/11 (now 70/70).

## M3 round 2 (2026-10-08): the "slow benchmarks" are actually slow on the CPU side, plus fast math

### 1. The truth about tsa, hausdorff, bitonic-sort

Round 1 compared wall-clock time from `make run`; the three where the CUDA version was slower were tsa 0.34, hausdorff 0.37, bitonic-sort 0.65.
Pulling out the **kernel times the programs print themselves** on both sides (`results-m2/*.run.log`):

| benchmark | CUDA (b70cc) kernel | SYCL (icpx) kernel | Wall-clock time CUDA/SYCL |
|---|---|---|---|
| hausdorff | 22.05 ms | 22.02 ms | 20.4 s/7.5 s |
| bitonic-sort | 120 ms | 120 ms | 15.7 s/10.2 s |
| tsa (float/double) | 68/263 µs | 82/317 µs | 6.8 s/2.1 s |

The GPU side is equally fast (the CUDA version of tsa is even slightly faster). All of the difference is in the **CPU reference computation**: hausdorff's CPU side is 10¹⁰ distance computations,
and with `repeat=1` the CUDA version takes 18.2 s, the SYCL version 5.1 s. Chased down to the end, this is a compiler matter, not a toolchain matter:

| How the CPU reference program is compiled | Time |
|---|---|
| clang 22 `-O3` (b70cc's host side) | 17.8 s |
| clang 22 `-O3 -ffast-math` | 17.8 s (`-Rpass-missed` says the `float2` struct load in the inner loop "return type cannot be vectorized") |
| clang 22 `-O3 -ffast-math -march=native` | 13.4 s |
| icpx 2026.0 `-O3` (default `-fp-model=fast`) | **5.2 s** |
| icpx 2026.0 `-O3 -fp-model=precise` | 17.8 s |

icpx needs both its default fast-math (allowing the `min` reduction to be reordered) and its own vectorizer (which splits the struct load) to get the 3.5x;
open-source clang satisfies neither condition, and nvcc with a gcc host side would not vectorize this loop either, so b70cc's behavior is correct;
it is just that the M3 ① wall-clock comparison counted this part in. Also, round 1's 1393 µs for tsa float was **the first-launch IGC compile** inside the timed region
(130 ms amortized over a 100-run average); once the caches (`$B70_DATA/cache/neo`, `cache/chipstar`) are warm it is 68 µs.

Action taken: added `tests/hecbench/kernel_times.sh` + `summarize_kernel.py`, which compare only the kernel times the programs report themselves,
running each program twice and taking the second run (to avoid the cold-start compile). Results in "kernel time comparison" below.

### 2. The real optimization: device-side math flags

While looking into the above, `-###` was used to see the flags b70cc actually passes to clang, revealing three things (`bench/fast_math.cu`, 16M elements, vLLM idle on the B70):

| kernel | Default (before fix) | Default after fix | `--use_fast_math` (after fix) | SYCL default/precise |
|---|---|---|---|---|
| Precise transcendentals `expf·sinf+cosf·logf+powf` | 0.924 ms | **0.31–0.41 ms** (multiple measurements) | 0.259 ms (error 1.4e-5) | 0.564/0.506 ms |
| Division + square root ×8 | 0.499 ms | 0.511 ms | **0.278 ms** | 0.452/0.758 ms |
| CUDA intrinsics `__expf·__sinf+__fdividef+__powf` | **1.331 ms** | **0.255 ms** | 0.255 ms | 0.241 ms (`native::`) |
| 32nd-order polynomial (FMA) | 0.258 ms | 0.256 ms | 0.256 ms | 0.240 ms |

1. **`-fmath-errno` is on for the device side**. CUDA device code never sets errno, but chipStar did not turn it off, so clang can only treat `expf`, `sqrtf` as
   function calls that may set errno, and IGC never gets its own hardware version. `b70cc` now always adds `-Xarch_device -fno-math-errno`:
   precise transcendentals 0.924 → 0.31–0.41 ms, precision completely unchanged (2.41e-7), on par with or faster than SYCL's precise mode.
2. **Fast intrinsics like `__expf` default to the precise version**, and are even slower than `expf` (1.33 ms vs 0.92 ms; `__powf` went through `pow`).
   nvcc's semantics are that these are always the fast version. First, each of Intel's `native_*` functions was measured for accuracy and speedup (`toolchain/probe/intrinsics_probe.cu`, inputs (0.01, 3), compute-bound):

   | Function | Precise version error/time | native error/time | Speedup |
   |---|---|---|---|
   | exp | 1.8e-7/0.518 ms | 2.9e-7/0.260 ms | 2.0× |
   | log | 8.1e-8/0.843 ms | 1.0e-7/0.179 ms | 4.7× |
   | **sin** | 1.0e-7/0.450 ms | **3.4e-5**/0.200 ms | 2.3× |
   | **cos** | 1.0e-7/0.482 ms | **3.1e-5**/0.186 ms | 2.6× |
   | tan | 3.4e-7/1.064 ms | 7.3e-7/0.407 ms | 2.6× |
   | log2/log10/exp10 | ~1e-7/0.44–0.57 ms | 1.0–2.6e-7/0.18–0.19 ms | 2.3–3.2× |
   | divide | 9.7e-8/0.458 ms | 9.7e-8/0.196 ms | 2.3× |
   | powr | 2.2e-7/1.547 ms | 2.4e-7/0.277 ms | 5.6× |
   | exp2, sqrt | same | same | 1.0× |

   The CUDA documentation guarantees an absolute error of 2^-21.4 (3.6e-7) within [-π, π] for `__sinf`/`__cosf`; Intel's native versions are 100x worse, while all the others are within CUDA's bounds.
   Patch `0004-fast-intrinsics-always`: in `sp_intrinsics.hh`, the eight intrinsics exp/exp10/log/log2/log10/powf/fdividef/tanf
   now default to native (`-DCHIP_PRECISE_INTRINSICS` turns it off), while `__sinf` (upstream was unconditionally native), `__cosf`, `__sincosf` stay precise
   and use native only under `--use_fast_math`. Effect: `__intrinsics` kernel 1.33 → 0.26 ms.
   Acceptance tests gain `tests/cuda/fast_intrinsics.cu` (compiled once with each flag): precision within bounds, and the compute-bound exp/log/pow/div/tan intrinsic versions are **3.9x** faster than the precise versions.
3. **`--use_fast_math` was a no-op**: `cucc` simply ignores it, and `-ffast-math` to `hipcc` only becomes `-DCHIP_FAST_MATH`; neither reaches the device compiler.
   `b70cc` now expands `--use_fast_math` into nvcc's definition (`-fapprox-func -freciprocal-math -ffp-contract=fast -fgpu-flush-denormals-to-zero`
   + `-DCHIP_FAST_MATH`), and `-ffast-math` additionally adds unsafe/finite/no-signed-zeros on both the device and host sides
   (`hipcc` swallows a literal `-ffast-math`, so it has to be passed split into sub-flags). In the HeCBench list, adam, perplexity, softmax, stddev use `--use_fast_math`.
   FMA contraction was already on (the polynomial kernel is the same before and after the fix).

Tried with no difference: `-fgpu-flush-denormals-to-zero` on its own shows no timing difference (kept anyway, to match nvcc semantics).

### 3. Kernel time comparison (`tests/hecbench/kernel_times.sh`, 2026-10-09, `tests/hecbench/kernel-times-2026-10-09.txt`)

The 34 benchmarks verified on both sides (`list-verified.txt`), CUDA versions rebuilt with the current toolchain (including `-fno-math-errno` and patch 0004),
each run twice taking the second run, comparing only the kernel times the programs print themselves. heat2d and nbody print only bandwidth/rate and no time, so they are skipped; for the remaining 32:

**Geometric mean SYCL/CUDA kernel time = 0.89** (CUDA version 11% slower). All three deemed slow in round 1 have vanished: tsa 1.00, hausdorff 0.99, bitonic-sort 0.98.

| Slowest five | CUDA | SYCL | Ratio | | Fastest five | CUDA | SYCL | Ratio |
|---|---|---|---|---|---|---|---|---|
| adam | 0.213 ms | 0.081 ms | 0.38 | | iso2dfd | 3.12 s | 5.52 s | 1.77 |
| fdtd3d | 0.332 ms | 0.188 ms | 0.57 | | gaussian | 0.97 s | 1.04 s | 1.07 |
| bilateral | 10.0 ms | 6.6 ms | 0.66 | | nlll | 0.079 ms | 0.084 ms | 1.06 |
| laplace | 1.81 s | 1.27 s | 0.70 | | lombscargle | 0.535 ms | 0.559 ms | 1.05 |
| jacobi | 881 ms | 630 ms | 0.72 | | all-pairs-distance | 23.5 ms | 24.4 ms | 1.04 |

The middle 22 are between 0.72 and 1.01, 17 of them between 0.95 and 1.01.

**Why adam is 0.38** (fast-math again, but this time on the GPU side): the kernel computes `powf(beta, t)` twice per time step; the SYCL version uses the approximate `pow` because icpx defaults to `-fp-model=fast`,
while the CUDA version's Makefile has `--use_fast_math` commented out, so it is the precise version (nvcc would be precise too). Comparison (re-measured with the GPU idle):

| | Precise | fast-math |
|---|---|---|
| CUDA (b70cc) | 0.152 ms | **0.116 ms** (`--use_fast_math`, via b70cc's new flag expansion) |
| SYCL (icpx) | 0.833 ms (`-fp-model=precise`) | 0.081 ms (default) |

Both precise, the b70cc version is 5x faster than the SYCL version; both fast-math, it is 1.4x behind (possibly the `-fapprox-func` expansion of `pow` differs from icpx's; to be investigated).
Conclusion: the M3 ① comparison baseline itself is biased toward SYCL (which defaults to fast-math); actual kernel performance is on par.

**The genuinely slow ones** (targets for the next round): fdtd3d 0.57 (stencil, shared memory; `--use_fast_math` does not help: 0.367 ms), bilateral 0.66, laplace 0.70, jacobi 0.72.
These need a look at the differences between the code IGC generates (registers, SIMD width, shared memory access patterns) and the SYCL version.

### 4. Overhead of each layer (2026-10-09, vLLM online)

**Toolchain** (`bench/micro.cu` vs `bench/micro_sycl.cpp`, the same program): stream copy/scale/add/triad 518/520/533/531 GB/s vs SYCL 529/529/533/531 GB/s (0–2% apart);
kernel launch 1.88 µs vs 2.32 µs (CUDA build faster); float atomics 7.09 vs 6.72 G ops/s (CUDA build faster). Together with the HeCBench kernel geometric mean of 0.89 in section 3,
the same CUDA source run through b70cc is on average about 10% slower than a hand-written native SYCL version, with no difference on bandwidth-bound work.

**PyTorch layer** (the same torch code, `"cuda"` through the layer vs `"xpu"` native with `B70_CUDA=0`; `bench/layer_overhead.py`): matmul 4096² bf16 0.83 vs 1.02 ms (run-to-run jitter, same oneDNN kernel),
SDPA 4k tokens 0.83 vs 0.83 ms, conv2d 0.58 vs 0.58 ms, TransformerEncoderLayer forward 0.99 vs 0.97 ms, 1000 tiny ops 9.99 vs 10.45 ms,
`.to(device)` 0.017 vs 0.015 ms. The layer only changes how device names are resolved and never touches the computation; its overhead is within measurement noise (0%).

## Approach to tests and performance

- All acceptance tests live in `tests/`, performance tests in `bench/`; all must be re-runnable with a single command
- Performance numbers always record: test date, driver version, whether other work was running on the B70
- When comparing with SYCL, use the same card and the same driver, run each 5 times and take the median
