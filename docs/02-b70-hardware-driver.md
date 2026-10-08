English | [繁體中文](02-b70-hardware-driver.zh-TW.md)

# 02 B70 hardware and driver, and how CUDA concepts map onto it

Markers: **[measured]** means queried on the B70 with `toolchain/probe/ze_probe.c` (Level Zero API);
**[docs]** means taken from Intel's public documentation and not yet verified on this card.

## Driver stack

```
CUDA source ──b70cc/clang──▶ SPIR-V 1.5
                               │
                     Level Zero loader 1.28 (API 1.14)
                               │
              Intel compute-runtime (NEO) 26.05 ── IGC 2.28: SPIR-V → Xe2 instructions
                               │
                     Linux kernel 7.0 xe driver
                               │
                      Arc Pro B70 (BMG-G31)
```

| Layer | Version [measured] | Role |
|---|---|---|
| Kernel driver | `xe` (Linux 7.0) | Memory management, scheduling, firmware interface |
| compute-runtime | 26.05.37020.3 (Level Zero driver 0x103909c) | Implements Level Zero and OpenCL |
| IGC | 2.28.4 | Compiles SPIR-V to Xe2 machine code (JIT at run time) |
| Level Zero loader | 1.28.2, API 1.14 | The API entry point applications call |

## Hardware specifications [measured]

| Item | Value |
|---|---|
| Device ID / IP version | 0xe223 / 0x5008000 (Xe2) |
| Clock | 2800 MHz |
| Structure | 8 slices x 4 Xe-cores x 8 EUs x 8 threads = **32 Xe-cores, 256 EUs, 2048 hardware threads** |
| EU physical SIMD width | 16 |
| sub-group widths | **16, 32** |
| work-group limit | 1024 (1024 in each dimension) |
| group count limit | 2³²−1 per dimension |
| Shared memory (SLM) | Up to **128 KiB** per work-group |
| Cache | 24 MiB (last-level cache as reported by the driver) |
| Memory | 30.3 GiB allocatable; single-allocation limit 30.3 GiB |
| SPIR-V | 1.5 |
| Module capabilities | fp16, fp64, int64 atomics, dp4a |
| float atomics | fp32, fp64: global/local add and min/max **are all native in hardware**; fp16 has only load/store and min/max |
| Kernel argument limit | 2048 bytes |
| printf buffer | 4 MiB |
| Command queues | Group 0: compute + copy + **cooperative**, 1 engine; group 1: copy, 1 engine |
| USM | host/device/shared (single device) all support read/write and atomics; **system allocations are not supported** (memory from plain `malloc` cannot be handed to the GPU directly) |
| Timer resolution | 52 ns |

## Measured performance baselines (SYCL/icpx 2026.0, `bench/micro_sycl.cpp`)

| Item | Result | Notes |
|---|---|---|
| Memory bandwidth (STREAM copy/scale/add/triad) | **527-534 GB/s** | About 87% of the theoretical 256-bit GDDR6 figure (~608 GB/s) |
| Kernel launch latency (2000 empty kernels submitted back to back) | 2.2-2.5 µs each | in-order queue, including host-side overhead |
| float atomic add (1024 locations under contention) | 6.7 G ops/s | |

**Bandwidth must be measured with incompressible data**: Xe2 compresses VRAM contents. With arrays filled entirely with zeros, the measured "bandwidth" is
898-1318 GB/s, far beyond the physical limit of GDDR6. Only after filling them with hashed random values do you get the numbers in the table above.
Keep this in mind for any memory performance test on the B70.

## CUDA concept mapping

| CUDA | B70/Xe2 | Notes |
|---|---|---|
| SM (streaming multiprocessor) | Xe-core (32 of them) | 8 EUs per Xe-core, 8 hardware threads per EU |
| warp (32 threads) | sub-group 32 | Hardware SIMD is 16; SIMD32 is assembled by the compiler from two SIMD16 groups. `warpSize` is fixed at 32 |
| thread block | work-group | Same limit of 1024 |
| grid | ND-range | Dimension limits are wider than CUDA's (CUDA's y/z only go to 65535) |
| `__shared__` | SLM | Up to 128 KiB, larger than most NVIDIA consumer cards (about 100 KB) |
| warp shuffle/vote (`__shfl_*`, `__ballot`) | sub-group operations | chipStar currently supports only the non-`_sync` versions → **to be filled in for M5** |
| `atomicAdd(float/double)` | Native float atomics | chipStar emulates with a CAS loop by default → **the B70 can switch to the native instruction (M3 performance item)** |
| `atomicAdd(__half)` | CAS emulation only | Hardware has no fp16 atomic add |
| `__dp4a` | dp4a | Supported in hardware |
| tensor core (`wmma`, `mma.sync`) | XMX (DPAS) | DPAS requires sub-group 16 [earlier XMX experiments], and the data layout differs from warp 32 → no one-to-one mapping, needs a separate design (M5) |
| cooperative launch, grid sync | Level Zero cooperative kernel | Supported on the compute queue → cooperative groups are feasible (chipStar does not have them yet) |
| stream | command list/queue | Only 1 compute engine + 1 copy engine, so kernels on multiple streams do not truly run concurrently |
| `cudaMemcpyAsync` | copy engine | There is 1 independent copy engine that can overlap with compute |
| unified memory (`cudaMallocManaged`) | shared USM | Supported; but system allocations are not, so there is no HMM-style "any pointer works" |
| Kernel arguments (CUDA limit 4 KB, 32 KB in newer versions) | 2048 bytes | When exceeded, chipStar moves the arguments into an extra buffer |
| `printf` | Available, 4 MiB buffer | |
| `clock64()` | Timer | 52 ns resolution |
