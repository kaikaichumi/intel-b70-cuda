English | [繁體中文](README.zh-TW.md)

# pytorch-ext: Building PyTorch CUDA Extensions on the B70

Lets packages written with `torch.utils.cpp_extension.CUDAExtension` (`.cu` source code, a `setup.py` that calls nvcc)
compile and load on the Intel Arc Pro B70 **without changing a single file**, operating directly on torch XPU tensors.

## Usage

Inside the `b70-ai` image (`b70 build` installs this directory into it), everything is automatic:

```bash
b70 install --no-build-isolation causal-conv1d     # the CUDAExtension in setup.py goes through the b70 toolchain
# After a toolchain update (b70cuda build), add --no-cache-dir when rebuilding already-installed extensions,
# otherwise pip reuses the cached wheel built with the old toolchain:
#   b70 install --no-build-isolation --no-deps --force-reinstall --no-cache-dir mamba-ssm
b70 python my_script.py                            # the extension's kernels run on the B70
```

Requires the chipStar toolchain to be built first with `b70cuda build` (`$B70_DATA/cuda/install`).

Manual use (without the image): put `py/` on `PYTHONPATH`; `sitecustomize.py` loads the hook.

## What it does

| Component | Role |
|---|---|
| `py/b70torch/_hook.py` | Intercepts `torch.utils.cpp_extension`: points `CUDA_HOME` at a fake directory, swaps out libraries such as cudart/c10_cuda, adds compatibility headers, and compiles `.cpp` files in HIP mode too |
| `py/b70torch/_build.py` | Generates the fake `nvcc` (strips flags like `-gencode` and `--expt-*`, then calls chipStar's cucc) and `b70-c++`; compiles `libb70torch.so` |
| `py/b70torch/rewrite.py` | Shadow source tree: copies the package directory to `build/b70_src` and rewrites `x.is_cuda()`, `kCUDA`, `DeviceType::CUDA`, `TORCH_LIBRARY_IMPL(…, CUDA, …)` etc.; torch's headers are left untouched |
| `include/` | Compatible versions of `ATen/cuda/*.h` and `c10/cuda/*.h` (stream, guard, event, exception macros), `b70torch/prelude.h`; CUB (mapped to hipCUB) and `thrust/complex.h` live in `../toolchain/include` |
| `src/interop.cpp` | `libb70torch.so`: hands torch's Level Zero context to chipStar; the extension kernels' stream and torch's queue are synchronized in both directions with GPU events |

How synchronization works: when the extension calls `getCurrentCUDAStream()`, the CUDA stream waits on the torch queue; work on the CUDA side makes the torch queue wait on it **before the next ATen op starts**
(via a `RecordFunction` callback). So "torch op → extension → torch op" needs no manual synchronization at all.
`torch.cuda.synchronize()` (mapped by the PyTorch layer) also waits for the CUDA side.

## Environment variables

| Variable | Purpose |
|---|---|
| `B70_CUDA_INSTALL` | chipStar install directory (default `$B70_DATA/cuda/install`) |
| `B70TORCH_CACHE` | Location of `libb70torch.so` and the fake `CUDA_HOME` (default `~/.cache/b70torch`) |
| `B70TORCH_VERBOSE=1` | Print the actual compile commands and the rewritten files |
| `B70TORCH_PROFILE=1` | Print the average time of each synchronization stage at exit |
| `B70TORCH_SYNC=host` | Use host-side synchronization instead (for debugging; slow) |
| `B70TORCH=0` | Disable the hook |

## Tests

```bash
cd tests/ext_min && b70 run python setup.py build_ext --inplace && b70 python test.py
```

Measured numbers are in the M4 notes of [docs/03-roadmap.md](../docs/03-roadmap.md).

## Known limitations

- No cuBLAS/cuFFT/cuDNN/cuRAND/Thrust algorithms (`thrust/complex.h` provides only the type and its math functions); CUB's block/warp/device layers are provided through hipCUB
- Inline PTX, tensor cores (`wmma`/`mma`), cooperative groups and CUDA Graphs are not supported (chipStar limitations)
- `torch.utils.cpp_extension.load()` (the JIT path) is not tested yet
