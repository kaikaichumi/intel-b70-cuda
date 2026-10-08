[English](01-pytorch-layer.md) | 繁體中文

# 01 PyTorch 層：讓 CUDA 風格的 Python 程式碼直接跑在 B70 上

使用說明見 [pytorch-layer/README.md](../pytorch-layer/README.zh-TW.md)。這份文件寫設計理由、測試結果和踩過的坑。

## 組成

| 部分 | 作用 |
|---|---|
| 映像檔 `b70-ai` | 疊在 `vllm/vllm-openai-xpu` 上（torch 2.13+xpu、triton-xpu 3.7.2、oneAPI runtime），另外只多約 0.6 GB：diffusers、peft、gguf、gradio、bitsandbytes 0.50.2、av、kornia、spandrel 等 |
| `b70cuda` | `import torch` 時自動啟用的 CUDA→XPU 轉接層（`.pth` + import hook，沒 import torch 的程式不受影響） |
| `flash_attn` 替身 | flash-attn 2 的介面，底層是 PyTorch SDPA（B70 上本身就是 flash kernel）；附 dist-info，`pip` 視為已安裝 |
| `b70` 指令 | run / python / shell / install / bg / gpu / comfyui / build / test |
| `b70 install` | 過濾 requirements：torch 家族保留 XPU 版、丟掉只有 CUDA 版的套件和 CUDA wheel index，裝進每個資料夾自己的 `.b70venv` |

`.b70venv` 的做法：`python -m venv --without-pip`，再放一個
`import site; site.addsitedir("/opt/venv/lib/python3.12/site-packages")` 的 `.pth`。
venv 因此繼承映像檔的所有套件，`pip` 也把它們當成已安裝；專案要不同版本時，裝進 venv 的會優先。
`PIP_CONSTRAINT` 鎖住 torch / torchvision / torchaudio / triton / intel-*，有套件要換掉它們會直接報錯，
不會悄悄換成 CUDA 版。

## 轉接層的三層判斷

`torch.cuda.*` 依呼叫者給不同答案：

| 呼叫者 | `is_available()` / `device_count()` | 其他 `torch.cuda.*` |
|---|---|---|
| PyTorch 本身（torch、triton） | 照實回答（沒有 CUDA） | 原本的行為 |
| 原生支援 XPU 的函式庫（transformers、diffusers、accelerate、ComfyUI…） | 照實回答，讓它們預設走自己的 XPU 路徑 | 轉到 `torch.xpu`（你明確傳 `"cuda"` 時要能用） |
| 使用者的程式 | 回答有 CUDA 裝置 | 轉到 `torch.xpu` |

裝置名稱的轉換（`"cuda"`、`"cuda:N"`、`torch.device("cuda")`、整數）對所有呼叫者都套用，因為這樣做一定對。

啟用時機（2026-10-08 修正）：`b70cuda.pth` 掛的 import hook 等到 torch **真的執行完**才套用並退場。原本是「第一次有人問 torch 就退場」，
而 transformers 之類的函式庫會先用 `importlib.util.find_spec("torch")` 探測，hook 被探測觸發後就錯過了真正的 import，
導致「先 import transformers 再 import torch」的程式完全沒有轉接（用 mamba-ssm 的上游測試抓到）。

`Tensor.is_cuda`（2026-10-08 加入）：kernel 套件慣用 `assert x.is_cuda` 擋住啟動（sageattention、Liger、
各種 CUDA 擴充），XPU 張量對這些呼叫者回答 True；torch 和 triton 自己看到的仍是真實值（它們靠這個分派裝置），
`is_xpu` 不動。可用 `B70_CUDA_SKIP=is_cuda` 關掉。

## 測試結果

**自我測試**（`b70 test`，只用約 1 GB 顯存）：14／14 通過。

- 寫死 CUDA 的程式碼：裝置字串、`with torch.device`、`torch.cuda.amp` 訓練、Generator、checkpoint
- flash_attn 替身跟參考實作比對：最大誤差 0.0077；GQA、sliding window、varlen 都對
- 其他：`torch.compile`、diffusers、transformers `device_map="cuda"`、safetensors、bitsandbytes NF4

**真實模型**：

| 測試 | 結果 |
|---|---|
| SDXL-Turbo，模型卡程式碼原文（`pipe.to("cuda")`），diffusers | 載入 17.5 s；**1024×1024、4 步 2.9 s**；顯存峰值 10.5 GB |
| ComfyUI 0.39.0（`b70 comfyui` 自動安裝），API 跑 SDXL-Turbo | 自動認到 `xpu:0 Arc Pro B70`；模型載入後 **512×512、4 步 1.0 s** |

ComfyUI 直接讀 HF 快取裡的 diffusers 格式模型：把 snapshot 目錄連結到 `ComfyUI/models/diffusers/<名字>`，
工作流程用 `DiffusersLoader` 節點（`pytorch-layer/tests/comfyui_api_test.py`），不必再下載單檔 checkpoint。

## 踩過的坑

1. **`torch.cuda.X` 不能直接等於 `torch.xpu.X`**。dynamo 用函式物件當 key 註冊 handler，同一個物件註冊兩次會 assert
   （"Handler already registered for current_stream"）。每個都要包一層新函式。
2. **PyTorch 內部要看到原本的 `torch.cuda`**。inductor 會呼叫 `torch.cuda.get_device_properties()`，
   預期它拋出例外再接住；如果回傳 B70 的資訊，它會接著讀 `gcnArchName` 然後掛掉。
3. **TorchScript 用物件 id 認內建函式**。包過的 `torch.zeros` 等要用 `torch.jit._builtins._register_builtin`
   登記成同一個 aten op，否則 kornia 一 import（它在 import 時就 script 函式）就失敗，連帶 diffusers 也失敗。
4. **`torch.utils._device._device_constructors()` 用物件比對**（`with torch.device(...)`、`set_default_device` 都靠它）。
   C++ 交給它的是原生函式，集合裡卻是包裝版，結果比對不到，裝置設定就不生效。
   transformers 5 用 `torch.tensor([]).device` 偵測 meta 初始化，偵測失敗後非持久 buffer（CLIP 的 `position_ids`）
   會是未初始化的值，在 GPU 上的症狀是 embedding「index out of bounds」，還會洗出好幾百萬行 assert，看起來像當掉。
   修法：集合裡同時放原生函式和包裝版。
5. **Dockerfile 不能對 `/opt/venv` 做 `chmod -R`**。會把整個 venv 複製一份進 build cache，一次吃掉 10 GB。

## 已知限制

| 東西 | 原因／替代 |
|---|---|
| 帶 `.cu` 檔、要 nvcc 編譯的套件 | 由 CUDA 工具鏈處理：`b70 install` 自動改用 `b70cc` 編譯（[pytorch-ext](../pytorch-ext/)；causal-conv1d、mamba-ssm 實測通過，見[路線圖](03-roadmap.zh-TW.md) M4） |
| xformers | 拿掉即可，diffusers／ComfyUI 會改用 SDPA |
| CUDA Graphs | 不支援 |
| cupy、TensorRT、`onnxruntime-gpu` | 改用 numpy／torch、OpenVINO、`onnxruntime` |
