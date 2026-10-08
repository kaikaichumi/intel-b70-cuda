English | [繁體中文](00-background.zh-TW.md)

# 00 Research background: ways to run CUDA on Intel GPUs

Surveyed: 2026-10. Target hardware: Intel Arc Pro B70 (Battlemage/Xe2, BMG-G31, 32 GB).

## Four kinds of approach

| Type | How it works | What it can run | Examples |
|---|---|---|---|
| Binary compatibility | Translate NVIDIA-compiled code into the target GPU's instructions at run time | Already-compiled CUDA programs | ZLUDA |
| Source retargeting | Compile CUDA/HIP source code to a portable intermediate representation (SPIR-V), which the driver then turns into GPU instructions | Programs with source code | chipStar, AdaptiveCpp |
| Native reimplementation | Compile CUDA source code directly into the target GPU's instructions | Programs with source code | SCALE (currently AMD and NVIDIA only) |
| Source translation | Convert CUDA source code into SYCL/HIP source code and maintain the converted version from then on | Programs with source code | SYCLomatic, HIPIFY |

## State of each approach

**ZLUDA**: originally (2020-2021) built for Intel GPUs, later pivoted to AMD. The official FAQ states that Intel is currently not supported.
This project **does not do binary compatibility**: NVIDIA's CUDA license terms forbid running its compiled output on non-NVIDIA hardware.

**chipStar** ([CHIP-SPV/chipStar](https://github.com/CHIP-SPV/chipStar), MIT): compiles HIP/CUDA source code to SPIR-V via LLVM,
runs on OpenCL or Level Zero, and provides `cucc`, used the same way as nvcc. Key numbers (IWOCL 2026 presentation):

- 49 HeCBench tests on the Arc A770, geometric mean **1.28x faster** than the SYCL versions
- 70% of the HIP API implemented (224/319)
- Not yet supported: warp matrix operations (tensor cores), inline PTX, cooperative groups, warp functions ending in `_sync`
- Battlemage is not on the support list

This project's toolchain is based on chipStar, with features and tuning added for the B70.

**SYCLomatic**: Intel's official CUDA-to-SYCL conversion tool. Example: the Intel build of the Strata inference engine was made by converting 86 compilation units with it
and then hand-fixing 28 kernel files. Good for a one-off port, not for tracking upstream updates.

**PyTorch XPU**: PyTorch has supported Intel GPUs natively since 2.5, and `torch.xpu` maps almost one-to-one onto `torch.cuda`.
Most Python-level AI tools merely hardcode the device as `"cuda"`, which a compatibility layer can solve (see [01](01-pytorch-layer.md)).

## Conclusion: two layers

1. **Python layer**: most AI tools (diffusers, transformers, ComfyUI, assorted demos) only use PyTorch.
   Mapping CUDA to XPU at the PyTorch layer lets these tools run without code changes.
2. **`.cu` source code layer**: what most often remains stuck is packages that ship CUDA source code (PyTorch CUDA extensions, the llama.cpp CUDA backend, research code).
   Recompile them for the B70 with a source-retargeting toolchain.

Closed-source software that ships only as executables is out of scope; for such software, look for the official Intel, oneAPI, OpenCL or Vulkan versions.

## References

- [ZLUDA FAQ](https://zluda.readthedocs.io/latest/faq.html)
- [chipStar](https://github.com/CHIP-SPV/chipStar), [IWOCL 2026 chipStar presentation](https://www.iwocl.org/wp-content/uploads/IWOCL-2026-Velesko-ChipStar.pdf)
- [Phoronix: chipStar 1.3](https://www.phoronix.com/news/chipStar-1.3-Released)
- [Strata Intel B70 port (PR #423)](https://github.com/Niko1221/Strata/pull/423)
- [PyTorch: Getting Started on Intel GPU](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)
