# b70：把 Arc Pro B70 當成 CUDA 卡來用

目標：網路上寫給 NVIDIA 的 AI 程式（影像、影片、語音、LLM……），**不改程式碼**就能在 B70 上跑。
做不到的部分會直接報錯，不會安靜地退回 CPU。

## 三分鐘上手

```bash
b70 gpu free                           # 大模型要整張卡：先停 vLLM（API :8000 會暫停）
cd ~/某個模型的 repo
b70 install -r requirements.txt        # 自動濾掉只有 CUDA 版的套件
b70 python app.py                      # 程式裡的 "cuda" 會自動改走 B70
b70 gpu llm                            # 用完把 vLLM 開回來（載入要幾分鐘）
```

ComfyUI：`b70 comfyui`，開 `http://<主機 IP>:8188`。模型放 `$B70_DATA/ComfyUI/models/`。
用 HF 下載的 diffusers 格式模型不必再下載一份：把它連結到 `models/diffusers/<名字>`，用 `DiffusersLoader` 節點載入。

實測（vLLM 停掉時）：SDXL-Turbo 1024×1024 跑 4 步 2.9 秒（diffusers，模型卡程式碼原文）；
ComfyUI 512×512 跑 4 步 1.0 秒。

## 指令

| 指令 | 作用 |
|---|---|
| `b70 python x.py` / `b70 run CMD` | 在目前資料夾、用 B70 執行 |
| `b70 shell` | 進 B70 環境的 shell |
| `b70 install ARGS` | `pip install`，裝進這個資料夾的 `.b70venv`（第一次自動建立） |
| `b70 bg NAME CMD` | 背景執行（例如 gradio 網頁）；`b70 ps`、`b70 logs NAME`、`b70 stop NAME` |
| `b70 comfyui` / `b70 comfyui stop` | ComfyUI（第一次會自動安裝） |
| `b70 gpu` | 誰在用卡、剩多少顯存 |
| `b70 gpu free` / `b70 gpu llm` | 停 / 開 vLLM |
| `b70 build` | 重建映像檔（vLLM 映像檔更新後可以跑一次） |
| `b70 test` | 自我測試（小模型，vLLM 開著也能跑） |

暫時關掉 CUDA 轉接：`B70_CUDA=0 b70 python x.py`。
只關掉其中一部分（遇到相容性問題時用）：`B70_CUDA_SKIP=load,amp b70 python x.py`，
可用的名稱有 `factories tensor is_cuda module load generator default_device amp cuda_api jitreg`，
也可以寫單一函式名稱，例如 `B70_CUDA_SKIP=arange`。

## 做了哪些事

1. **CUDA → XPU 轉接層**（`b70cuda`，`import torch` 時自動啟用）
   - 指定裝置的地方都會轉：`.cuda()`、`.to("cuda")`、`device="cuda"`、`torch.device("cuda:0")`、
     `torch.Generator("cuda")`、`torch.load`（包括在 NVIDIA 卡上存的 checkpoint）、`autocast("cuda")`、
     `torch.cuda.amp.*`、`GradScaler`
   - `with torch.device("cuda")`、`torch.set_default_device("cuda")` 也會轉到 B70
   - `torch.cuda.*`（顯存、同步、stream、event、亂數、裝置資訊）改由 `torch.xpu` 回答
   - `torch.cuda.is_available()` 對你的程式回答 True；對本來就支援 XPU 的函式庫（torch、
     transformers、diffusers、accelerate、ComfyUI…）照實回答，讓它們預設走自己測過的 XPU 路徑。
     你明確傳 `"cuda"` 給它們（例如 `device_map="cuda"`）也一樣能用
   - PyTorch 自己內部（`torch.compile` 等）看到的是原本的 `torch.cuda`，不受影響
2. **假的 `flash_attn` 與 `sageattention`**：介面跟 flash-attn 2、SageAttention 2 一樣，底層用 PyTorch SDPA（B70 上本身就是 flash kernel）。
   flash_attn 支援 GQA、causal、sliding window、varlen、`bert_padding`；sageattention 支援 HND／NHD 版面、causal、GQA、2.x 的各 kernel 入口
3. **安裝過濾**（`b70 install`）：
   - torch / torchvision / torchaudio / triton 永遠不重裝（避免被換成 CUDA 版）
   - 丟掉只有 CUDA 版的套件：xformers、flash-attn、sageattention（後兩個映像檔有 SDPA 替身）、nvidia-\*、cupy、tensorrt、apex…
   - 帶 `.cu` 原始碼的套件改用 `b70cc` 編譯（見 [pytorch-ext](../pytorch-ext/)）
   - `onnxruntime-gpu` 換成 `onnxruntime`；丟掉指向 CUDA wheel 的 `--index-url`
   - 每個資料夾有自己的 `.b70venv`，要不同版本的 transformers 也不會互相影響
4. **映像檔 `b70-ai`**：直接沿用 vLLM 映像檔裡已經驗證過的 torch 2.13+xpu，
   另外加上 diffusers、peft、gguf、gradio、bitsandbytes、av、kornia、spandrel 等常用套件

## 還是不能跑的

| 東西 | 原因 / 替代 |
|---|---|
| 自己帶 `.cu` 檔的套件（要 `nvcc` 編譯的 extension） | **可以跑了**：`b70 install` 會改用 [CUDA 工具鏈](../toolchain/)的 `b70cc` 編譯（見 [pytorch-ext](../pytorch-ext/)；causal-conv1d、mamba-ssm 實測通過）。用到 cuBLAS／cuDNN／tensor core（`mma`）的擴充例外 |
| 用 CUDA 寫的 attention 套件（sageattention、flash-attn） | 兩個都由映像檔內用 SDPA 實作的替身取代（`import sageattention`／`import flash_attn` 照常可用）。B70 上 SDPA 本身就是 fused kernel，sageattention 的 Triton 版實測反而慢 17 倍 |
| xformers | 拿掉就好，diffusers / ComfyUI 會自動改用 SDPA |
| CUDA Graphs、`torch.cuda.graph` | 不支援 |
| cupy、TensorRT、`onnxruntime-gpu` | 用 numpy / torch、OpenVINO、`onnxruntime` |
| 多張卡（NCCL） | 只有一張卡，不適用 |

## 檔案位置

| | |
|---|---|
| 工具本身 | 這個資料夾（`b70` 指令用符號連結放到 `~/.local/bin/b70`，見 [docs/04-install.md](../docs/04-install.md)） |
| 模型、快取（HF、pip、kernel 編譯快取） | `$B70_DATA`，預設 `~/.local/share/b70` |
| ComfyUI | `$B70_DATA/ComfyUI` |

容器以你的帳號執行，產生的檔案擁有者是你；家目錄、`$B70_DATA` 和目前資料夾在容器裡路徑相同。

## 設定（`~/.config/b70/config`）

每台機器不同的設定寫在這裡，格式是 `KEY=VALUE`，環境變數優先：

```bash
B70_DATA=/mnt/bigdisk/b70        # 模型和快取放哪（建議放大硬碟）
B70_MOUNTS="/mnt/bigdisk"         # 另外要在容器裡看得到的資料夾，空白分隔
B70_GPU_PCI=0000:03:00.0          # 有多張 Intel 卡時指定；預設自動找第一張用 xe 驅動的卡
B70_VLLM=vllm                     # 跟 B70 共用顯卡的 vLLM 容器名稱（b70 gpu free/llm 用）
```
