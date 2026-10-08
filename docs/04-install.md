# 04 從零安裝與日常使用

一台裝了 Arc Pro B70（或其他用 `xe` 驅動的 Intel 顯卡）的 Ubuntu 主機，從乾淨系統到能跑 CUDA 風格的程式，要做的事全部在這裡。
兩層可以分開裝：只跑 Python／PyTorch 的 AI 工具，裝 PyTorch 層就夠；要編 `.cu` 原始碼（含 PyTorch 的 CUDA 擴充套件）才需要工具鏈。

## 主機需求

| | 需求 | 說明 |
|---|---|---|
| OS | Ubuntu 24.04 以上，kernel 6.8 以上（實測 26.04、kernel 7.0） | 顯卡要由 `xe` 驅動接管：`ls /sys/class/drm/renderD*/device/driver` 看得到 `xe` |
| Docker | 任何近期版本；使用者在 `docker` 群組 | 所有東西都在容器裡跑，主機不裝 oneAPI、不裝編譯器、不碰系統 Python |
| 硬碟 | PyTorch 層約 20 GB（映像檔＋模型快取）；工具鏈另外約 10 GB | 都可以放大硬碟（`B70_DATA`） |
| 網路 | 第一次要下載映像檔與 chipStar 原始碼（數 GB） | 之後離線可用 |

主機上**不需要**裝 Intel 的 compute runtime 或 Level Zero，容器映像檔自帶；只需要 kernel 的 `xe` 驅動（Ubuntu 24.04 以後內建）。

## 安裝

```bash
git clone <這個 repo> ~/b70-cuda
mkdir -p ~/.local/bin
ln -s ~/b70-cuda/pytorch-layer/b70 ~/.local/bin/b70
ln -s ~/b70-cuda/toolchain/b70cuda ~/.local/bin/b70cuda
# 確認 ~/.local/bin 在 PATH 裡（Ubuntu 預設登入 shell 會自動加）

# 選擇性：資料和快取放大硬碟
mkdir -p ~/.config/b70
echo 'B70_DATA=/mnt/bigdisk/b70' >> ~/.config/b70/config
```

### PyTorch 層（跑 AI 工具）

```bash
b70 build          # 建 b70-ai 映像檔：以 vllm/vllm-openai-xpu 的 torch 2.13+xpu 為底，加上轉接層與常用套件（約 10 分鐘）
b70 test           # 自我測試：15 項，vLLM 開著也能跑
```

### CUDA 工具鏈（編 .cu 原始碼、PyTorch CUDA 擴充）

```bash
b70cuda setup      # 下載 chipStar 補丁版 LLVM 22 與 chipStar 原始碼（約 5 GB，直接串流到 $B70_DATA/cuda）
b70cuda build      # 建開發映像檔、套用 toolchain/patches/、編譯安裝 chipStar、裝 rocPRIM／hipCUB（6 核心約 10 分鐘）
b70cuda test       # 驗收測試：13 支程式、76 項檢查
```

`b70 install` 編 CUDA 擴充套件時用的就是這套安裝在 `$B70_DATA/cuda/install` 的工具鏈，不必重建 b70-ai。
一共四個指令、約半小時，之後更新 repo 只要重跑 `b70 build`（工具鏈補丁有變才要 `b70cuda build`）。

## 日常使用

### 跑現成的 AI 專案

```bash
cd ~/some-project                       # 任何寫給 NVIDIA 的 repo
b70 install -r requirements.txt         # pip install 進這個資料夾的 .b70venv；CUDA 專用套件自動濾掉或換成 Triton 版
b70 python app.py                       # 程式裡的 "cuda" 自動變成 B70
b70 shell                               # 要手動操作就進容器 shell，路徑跟主機一樣
```

帶 `.cu` 原始碼的套件（例如 `pip install mamba-ssm`）也是 `b70 install`，它會自動改用 `b70cc` 編譯，第一次會久一點。

### 編 CUDA 程式

```bash
b70cuda cc -O3 main.cu -o main          # 跟 nvcc 一樣的用法（--use_fast_math、-arch=sm_xx 都接受）
b70cuda run ./main                      # 在 B70 上執行
b70cuda shell                           # 進開發環境，裡面直接用 b70cc；CMake 專案加 -DCMAKE_CUDA_COMPILER=$(which b70cc)
```

第一次執行某支程式時 IGC 要把 SPIR-V 編成機器碼（約 0.1 秒到幾秒），結果快取在 `$B70_DATA/cache`，之後不用再等。

### 跟 vLLM 這類常駐服務共用顯卡

B70 只有一張卡；如果同時跑著 vLLM，顯存會被它佔住。`b70 gpu` 看誰在用、剩多少；`b70 gpu free`／`b70 gpu llm` 停掉與重開 vLLM
（容器名稱用 `B70_VLLM` 設定）。小模型和自我測試不必停。

## 會遇到的事

| 現象 | 原因／處理 |
|---|---|
| `b70 install` 把某個套件丟掉了 | 它只有 CUDA 版（xformers、flash-attn、cupy…）。diffusers／ComfyUI 等會自動改用 SDPA；清單在 `pytorch-layer/py/b70cuda/pipfilter.py` |
| 程式說 `Torch not compiled with CUDA enabled` | 轉接層沒啟動。確認是用 `b70 python` 跑的；可用 `B70_CUDA_SKIP`／`B70_CUDA=0` 排除是不是某一段轉接造成問題 |
| 工具鏈更新後擴充套件行為沒變 | pip 用了快取裡舊的 wheel：`b70 install --no-build-isolation --no-deps --force-reinstall --no-cache-dir <套件>` |
| `.cu` 用到 `cooperative_groups`、`mma`／`wmma`、inline PTX、cuBLAS／cuDNN | chipStar 不支援，目前只能改原始碼（見 [03 路線圖](03-roadmap.md) 的 M5） |
| 第一次跑某個 kernel 很慢 | IGC 即時編譯，第二次起走快取 |

## 檔案都放在哪

| | 位置 |
|---|---|
| 工具本身 | `~/b70-cuda`（兩個指令是符號連結） |
| 模型、pip、kernel 快取 | `$B70_DATA`（預設 `~/.local/share/b70`） |
| chipStar 原始碼、建置與安裝 | `$B70_DATA/cuda` |
| 每個專案自己的 venv | 專案資料夾裡的 `.b70venv` |
| 每台機器的設定 | `~/.config/b70/config`（`B70_DATA`、`B70_MOUNTS`、`B70_GPU_PCI`、`B70_VLLM`） |

容器以你的帳號執行，產生的檔案擁有者是你；家目錄、`$B70_DATA` 和目前資料夾在容器裡路徑相同。
