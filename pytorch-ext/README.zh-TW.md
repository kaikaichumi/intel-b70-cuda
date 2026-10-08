[English](README.md) | 繁體中文

# pytorch-ext：在 B70 上建置 PyTorch 的 CUDA 擴充套件

讓用 `torch.utils.cpp_extension.CUDAExtension` 寫的套件（`.cu` 原始碼、`setup.py` 呼叫 nvcc）
**不改任何檔案**就能在 Intel Arc Pro B70 上編譯、載入，並直接操作 torch XPU 的張量。

## 用法

在 `b70-ai` 映像檔裡（`b70 build` 會把這個目錄裝進去），一切自動：

```bash
b70 install --no-build-isolation causal-conv1d     # setup.py 裡的 CUDAExtension 走 b70 工具鏈
# 工具鏈更新（b70cuda build）之後要重編已裝的擴充套件時，記得加 --no-cache-dir，
# 否則 pip 會直接拿快取裡舊工具鏈編出來的 wheel：
#   b70 install --no-build-isolation --no-deps --force-reinstall --no-cache-dir mamba-ssm
b70 python my_script.py                            # 擴充套件的 kernel 跑在 B70 上
```

需要 `b70cuda build` 先建好 chipStar 工具鏈（`$B70_DATA/cuda/install`）。

手動用（不經映像檔）：把 `py/` 放進 `PYTHONPATH`，`sitecustomize.py` 會載入 hook。

## 它做了什麼

| 元件 | 作用 |
|---|---|
| `py/b70torch/_hook.py` | 攔截 `torch.utils.cpp_extension`：`CUDA_HOME` 指向假的目錄、換掉 cudart／c10_cuda 等函式庫、加入相容標頭、`.cpp` 也用 HIP 模式編譯 |
| `py/b70torch/_build.py` | 產生假的 `nvcc`（濾掉 `-gencode`、`--expt-*` 這類旗標後呼叫 chipStar 的 cucc）和 `b70-c++`；編譯 `libb70torch.so` |
| `py/b70torch/rewrite.py` | 影子原始碼樹：把套件目錄複製到 `build/b70_src`，改寫 `x.is_cuda()`、`kCUDA`、`DeviceType::CUDA`、`TORCH_LIBRARY_IMPL(…, CUDA, …)` 等；torch 的標頭不動 |
| `include/` | `ATen/cuda/*.h`、`c10/cuda/*.h` 的相容版本（stream、guard、event、例外巨集）、`b70torch/prelude.h`；CUB（對應 hipCUB）和 `thrust/complex.h` 在 `../toolchain/include` |
| `src/interop.cpp` | `libb70torch.so`：把 torch 的 Level Zero context 交給 chipStar；擴充 kernel 的 stream 與 torch 的 queue 用 GPU 事件雙向同步 |

同步的方式：擴充呼叫 `getCurrentCUDAStream()` 時讓 CUDA stream 等 torch queue；CUDA 端的工作則在**下一個 ATen 運算開始前**
（`RecordFunction` 回呼）讓 torch queue 等它。所以「torch 運算 → 擴充 → torch 運算」不需要任何手動同步。
`torch.cuda.synchronize()`（經 PyTorch 層對應）也會等 CUDA 端。

## 環境變數

| 變數 | 用途 |
|---|---|
| `B70_CUDA_INSTALL` | chipStar 安裝目錄（預設 `$B70_DATA/cuda/install`） |
| `B70TORCH_CACHE` | `libb70torch.so` 和假 `CUDA_HOME` 的位置（預設 `~/.cache/b70torch`） |
| `B70TORCH_VERBOSE=1` | 印出實際的編譯指令和被改寫的檔案 |
| `B70TORCH_PROFILE=1` | 結束時印出同步各階段的平均時間 |
| `B70TORCH_SYNC=host` | 改用主機端同步（除錯用，慢） |
| `B70TORCH=0` | 關掉 hook |

## 測試

```bash
cd tests/ext_min && b70 run python setup.py build_ext --inplace && b70 python test.py
```

實測數字見 [docs/03-roadmap.md](../docs/03-roadmap.zh-TW.md) 的 M4 紀錄。

## 已知限制

- 沒有 cuBLAS／cuFFT／cuDNN／cuRAND／Thrust 演算法（`thrust/complex.h` 只有型別和數學函式）；CUB 的 block／warp／device 層透過 hipCUB 提供
- inline PTX、tensor core（`wmma`／`mma`）、cooperative groups、CUDA Graphs 不支援（chipStar 的限制）
- `torch.utils.cpp_extension.load()`（JIT 路徑）還沒測
