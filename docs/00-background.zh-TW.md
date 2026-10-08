[English](00-background.md) | 繁體中文

# 00 研究背景：在 Intel GPU 上跑 CUDA 的各種方案

調查時間：2026-10。目標硬體：Intel Arc Pro B70（Battlemage／Xe2，BMG-G31，32 GB）。

## 四類做法

| 類型 | 做法 | 能跑什麼 | 代表 |
|---|---|---|---|
| 二進位相容 | 執行時把 NVIDIA 編好的程式碼轉成目標 GPU 的指令 | 已編譯的 CUDA 程式 | ZLUDA |
| 原始碼重定向 | 把 CUDA／HIP 原始碼編成可攜的中間碼（SPIR-V），再由驅動轉成 GPU 指令 | 有原始碼的程式 | chipStar、AdaptiveCpp |
| 原生重新實作 | 把 CUDA 原始碼直接編成目標 GPU 的指令 | 有原始碼的程式 | SCALE（目前只有 AMD、NVIDIA） |
| 原始碼轉換 | 把 CUDA 原始碼轉成 SYCL／HIP 原始碼，之後維護轉出來的版本 | 有原始碼的程式 | SYCLomatic、HIPIFY |

## 各方案現況

**ZLUDA**：最早（2020–2021）是為 Intel GPU 做的，後來轉向 AMD。官方 FAQ 寫明目前不支援 Intel。
本專案**不走二進位相容**：NVIDIA 的 CUDA 授權條款禁止把它的編譯產物轉到非 NVIDIA 硬體上執行。

**chipStar**（[CHIP-SPV/chipStar](https://github.com/CHIP-SPV/chipStar)，MIT）：把 HIP／CUDA 原始碼經 LLVM 編成 SPIR-V，
跑在 OpenCL 或 Level Zero 上，提供跟 nvcc 用法相同的 `cucc`。重點數字（IWOCL 2026 簡報）：

- Arc A770 上 49 個 HeCBench 測試，幾何平均比 SYCL 版**快 1.28 倍**
- HIP API 實作了 70%（224／319）
- 還不支援：warp 矩陣運算（tensor core）、inline PTX、cooperative groups、`_sync` 結尾的 warp 函式
- 支援清單沒有 Battlemage

本專案的工具鏈以 chipStar 為基礎，再針對 B70 補功能和調校。

**SYCLomatic**：Intel 官方的 CUDA→SYCL 轉碼工具。實例：Strata 推論引擎的 Intel 版就是用它轉 86 個編譯單元，
再手修 28 個 kernel 檔做出來的。適合一次性移植，不適合跟著上游一起更新。

**PyTorch XPU**：PyTorch 2.5 起原生支援 Intel GPU，`torch.xpu` 跟 `torch.cuda` 幾乎一一對應。
Python 層的 AI 工具大部分只是把裝置寫死成 `"cuda"`，這部分用轉接層就能解決（見 [01](01-pytorch-layer.zh-TW.md)）。

## 結論：分兩層

1. **Python 層**：大部分 AI 工具（diffusers、transformers、ComfyUI、各種 demo）只用到 PyTorch。
   在 PyTorch 層把 CUDA 對應到 XPU，就能讓這些工具不改程式碼直接跑。
2. **`.cu` 原始碼層**：剩下最常卡住的是帶 CUDA 原始碼的套件（PyTorch CUDA 擴充、llama.cpp CUDA 後端、研究用程式碼）。
   用原始碼重定向的工具鏈重新編譯給 B70。

只有執行檔的閉源軟體不在範圍內；這類軟體請找官方的 Intel、oneAPI、OpenCL 或 Vulkan 版本。

## 參考資料

- [ZLUDA FAQ](https://zluda.readthedocs.io/latest/faq.html)
- [chipStar](https://github.com/CHIP-SPV/chipStar)、[IWOCL 2026 chipStar 簡報](https://www.iwocl.org/wp-content/uploads/IWOCL-2026-Velesko-ChipStar.pdf)
- [Phoronix：chipStar 1.3](https://www.phoronix.com/news/chipStar-1.3-Released)
- [Strata Intel B70 移植（PR #423）](https://github.com/Niko1221/Strata/pull/423)
- [PyTorch：Getting Started on Intel GPU](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)
