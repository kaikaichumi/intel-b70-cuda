# b70-cuda

**CUDA-style computing on the Intel Arc Pro B70 (Battlemage / Xe2).**
Two layers: a PyTorch layer that lets CUDA-flavoured Python code run unchanged on the B70,
and a source-level CUDA toolchain that compiles `.cu` code (including PyTorch CUDA extensions)
for the B70 through SPIR-V and Level Zero. Docs are in Traditional Chinese.

讓 Intel Arc Pro B70 用起來像 CUDA 卡。分兩層：

| 層 | 解決什麼 | 狀態 |
|---|---|---|
| [PyTorch 層](pytorch-layer/)（`b70` 指令） | Python／PyTorch 寫的 AI 工具：程式裡的 `"cuda"`、`.cuda()`、`torch.cuda.*`、`flash_attn` 不用改就能在 B70 上跑；`pip install` 自動濾掉 CUDA 專用套件、換成 Triton 版 | 可用；自我測試 15/15，SDXL-Turbo、ComfyUI、Liger-Kernel 實測通過；flash_attn、sageattention 由 SDPA 替身取代 |
| [CUDA 工具鏈](toolchain/)（`b70cc`） | 有原始碼的 `.cu` 程式：用跟 nvcc 一樣的方式編譯，在 B70 上執行 | 可用；驗收測試 76 項全過，HeCBench 61 個 benchmark 97% 編得過、0 個算錯，kernel 時間與 SYCL 版相當（幾何平均 0.89，見[路線圖](docs/03-roadmap.md)） |
| [PyTorch CUDA 擴充](pytorch-ext/) | 帶 `.cu` 的 pip 套件（`CUDAExtension`）由 `b70 install` 自動用 `b70cc` 編譯，kernel 直接讀寫 torch XPU 張量 | 可用；causal-conv1d、mamba-ssm 原始碼不改，上游測試通過 |

## 三分鐘看懂怎麼用

```bash
# 裝好之後（見 docs/04-install.md，四個指令）
cd ~/some-nvidia-project
b70 install -r requirements.txt     # pip，CUDA 專用套件自動處理
b70 python app.py                   # "cuda" 自動變成 B70

b70cuda cc -O3 main.cu -o main      # 跟 nvcc 一樣
b70cuda run ./main
```

## 不做什麼

不做二進位相容，也就是不執行 NVIDIA 工具鏈編好的程式或 PTX。NVIDIA 的 CUDA 授權條款禁止把它的編譯產物
轉到非 NVIDIA 的硬體上執行，所以本專案只處理原始碼：Python 程式碼，以及 `.cu` 原始碼。
只有執行檔的閉源軟體不在範圍內。原因和其他方案的比較見 [docs/00-background.md](docs/00-background.md)。

目前做不到的（chipStar 的限制）：cooperative groups、tensor core（`mma`／`wmma`）、inline PTX、cuBLAS／cuDNN／cuFFT、CUDA Graphs。

## 文件

| 文件 | 內容 |
|---|---|
| [00 研究背景](docs/00-background.md) | 現有方案比較（ZLUDA、chipStar、SCALE、SYCLomatic、PyTorch XPU），為什麼選這條路 |
| [01 PyTorch 層](docs/01-pytorch-layer.md) | `b70cuda` 轉接層的設計、測試結果、踩過的坑 |
| [02 B70 硬體與驅動](docs/02-b70-hardware-driver.md) | Xe2 架構、驅動堆疊，以及 CUDA 概念在 B70 上的對應 |
| [03 路線圖](docs/03-roadmap.md) | 里程碑、驗收條件、所有實測數字、優化與 bug 追查紀錄（IGC 共享記憶體 bug、fast math、torch XPU conv1d bug） |
| [04 安裝與使用](docs/04-install.md) | 從乾淨的 Ubuntu 到能用，以及日常指令、常見問題 |

## 測試環境

| | |
|---|---|
| GPU | Intel Arc Pro B70，32 GB，BMG-G31（32 Xe-core） |
| OS | Ubuntu 26.04，kernel 7.0，`xe` 驅動 |
| 驅動 | Intel compute-runtime 26.05、IGC 2.28／2.38、Level Zero loader 1.28 |
| PyTorch | 2.13.0+xpu，Triton 3.7（XPU） |
| 工具鏈 | chipStar（固定 commit，見 `toolchain/b70cuda`）＋ 4 個補丁，LLVM 22 |

## 授權

[PolyForm Noncommercial 1.0.0](LICENSE)：可以自由使用、修改、散布，**不得用於商業目的**。
例外：`toolchain/patches/` 是對 chipStar（MIT）的修改，為了方便回饋上游，這些補丁用 [MIT](toolchain/patches/LICENSE)。
相依專案的授權：chipStar、rocPRIM、hipCUB 是 MIT，LLVM 是 Apache-2.0 with LLVM exception，PyTorch 是 BSD-3。
