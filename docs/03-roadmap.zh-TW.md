[English](03-roadmap.md) | 繁體中文

# 03 路線圖：里程碑、驗收條件、測試與效能目標

範圍：**原始碼層級**。有 `.cu` 原始碼的程式，用 `b70cc`（用法跟 nvcc 相同）編譯後在 B70 上執行。
不做二進位相容（原因見 [00](00-background.zh-TW.md)）。

## 架構

```
.cu 原始碼
  │  b70cc（nvcc 相容的編譯器入口）
  ▼
clang（chipStar 補丁版 LLVM 22）＋ CUDA 標頭對應 ＋ chipStar 的 LLVM passes
  │
  ▼
SPIR-V（嵌在執行檔裡）
  │  執行時：b70 runtime（CUDA Runtime API → chipStar → Level Zero）
  ▼
IGC 把 SPIR-V 編成 Xe2 指令 → B70
```

工具鏈以 [chipStar](https://github.com/CHIP-SPV/chipStar) 為基礎，本專案負責：

- 在 B70 上建好、驗證工具鏈
- 補 B70 上缺的功能，針對 Xe2 調校
- 接上 PyTorch XPU，讓 CUDA 擴充套件直接操作 torch 的張量

## 里程碑

| | 內容 | 驗收條件 | 狀態 |
|---|---|---|---|
| **M0** 專案骨架 | 專案資料夾、文件、拿掉個人資訊 | 可直接 `git init` 推上 GitHub | 完成 |
| **M1** 工具鏈上線 | chipStar 工具鏈在 B70 上可用；`b70cc` 編譯入口；硬體與驅動調查（[02](02-b70-hardware-driver.zh-TW.md)） | ① chipStar 單元測試在 B70（Level Zero）上的通過率，跟它對 Intel GPU 的已知失敗清單一致<br>② 標準 CUDA 範例（vectorAdd、matmul、reduction、scan、transpose、histogram 等）不改原始碼，用 `b70cc` 編譯後算出正確結果<br>③ kernel 裡 `warpSize == 32` | ②③ 通過（`tests/run_cuda_tests.sh`：13 個程式、76 項檢查全過，含 CUB block 原語、動態共享記憶體、快速內建函式）；① 待跑 |
| **M2** 正確性 | 跑 HeCBench 的 CUDA 版本 | ① 挑 50 個以上的 benchmark，八成以上編得過、結果驗證正確<br>② 每個失敗都分類（缺 API、缺 device 函式、驅動問題、效能逾時）並記錄 | ① 61 個：97% 編得過、85% 跑完；有自我檢查的全部通過；② 已分類（見下） |
| **M3** 效能 | 微基準和應用基準 | ① HeCBench：同一個 benchmark 的 CUDA 版（b70cc）對 SYCL 版（icpx），幾何平均 ≥ 0.8 倍<br>② 記憶體頻寬（類 STREAM 的複製、三元組運算）≥ 同一張卡上 SYCL 實測值的 90%<br>③ kernel 啟動延遲：量測並跟 SYCL 比較，記錄差距<br>④ 每一項優化都記錄改動前後的數字 | ① 牆上時間 **0.88 倍**（34 個），kernel 時間 **0.89 倍**（32 個，見「M3 第二輪」；牆上時間的差距多半是 icpx 預設 fast-math 的 CPU 端）；② 通過（約 100%）；③ CUDA 1.95 µs 對 SYCL 2.33 µs；④ 優化 1（事件）、優化 2（數學旗標）都有前後數字 |
| **M4** PyTorch CUDA 擴充 | `b70 install` 遇到帶 `.cu` 的套件時改用 `b70cc` 編譯；擴充 kernel 直接讀寫 torch XPU 張量（共用 Level Zero context，不複製）。三個難點見下方「M4 的實際工作」 | ① 自寫的範例擴充（`CUDAExtension` 寫法、用 `ATen/cuda` 標頭）在 XPU 張量上算出跟 CPU 一樣的結果<br>② 至少一個真實的開源 CUDA 擴充套件裝起來並通過它自己的測試<br>③ 有 Triton 版本的套件由 `b70 install` 自動改用 Triton 版 | ① 通過（6／6）；② 通過（causal-conv1d 1.7.0 與 mamba-ssm 2.2.5 原始碼不改、裝起來、通過正確性檢查，見下方紀錄）；③ 通過（Liger-Kernel 直接跑；sageattention 自動換成 Triton 版；`test_triton*.py`） |
| **M5** 擴大支援 | `_sync` warp 函式、float atomic（B70 硬體支援）、tensor core 對應 XMX、cuBLAS→oneMKL | 依 M2–M4 找到的缺口排序，逐項加測試 | |

## M2／M3 HeCBench 結果（2026-10-08，`tests/hecbench/results-2026-10-08.txt`）

61 個 benchmark（`tests/hecbench/list.txt`），CUDA 版用 `b70cc`，SYCL 版用 `icpx`，各跑一次牆上時間（同時有其他工作在 GPU 上，數字只看相對）。

| | 第一輪 | 補上 CUB、libomp 後重跑 4 個（`results-2026-10-08-rerun.txt`） |
|---|---|---|
| CUDA 版編得過 | 57／61（93%） | **59／61（97%）** |
| 跑完 | 48（79%） | **52（85%）** |
| 自我檢查印出 PASS | 32；另外 16 個程式本身沒有自我檢查 | **36** |
| 印出 FAIL（結果錯） | **0** | **0** |
| 雙邊都驗證通過、可比效能 | 30 個，幾何平均 SYCL／CUDA 時間 = **0.88**（CUDA 版慢 12%） | 34 個，補跑的 4 個比值 0.80／0.98／0.90／1.08 |

效能極端值：CUDA 版明顯慢的 `tsa` 0.34、`hausdorff` 0.37、`bitonic-sort` 0.65；明顯快的 `gaussian` 2.30、`iso2dfd` 1.68。
這幾個是 M3 下一輪的優化對象。

13 個失敗的分類：

| 原因 | benchmark | 性質 |
|---|---|---|
| 要 `cub/cub.cuh` | all-pairs-distance、laplace | 當時工具鏈還沒裝 hipCUB；`b70cc` 現在預設帶 `toolchain/include` 的 `cub/`，**重跑後兩個都通過** |
| 要 `cooperative_groups.h` | layernorm、softmax | **chipStar 的真實缺口**（cooperative groups 不支援） |
| 缺輸入資料檔（HeCBench 的 data 目錄要另外下載） | bfs、cfd、hotspot、hotspot3D、kmeans、urng | 測試環境，不是工具鏈 |
| 執行時找不到 `libomp.so` | bilateral、heat2d | 環境：程式用 OpenMP，容器的函式庫路徑沒含 LLVM 的 libomp；`b70cuda` 補上路徑後**重跑通過** |
| 逾時（600 s；前半段已印 PASS） | convolution1D | 效能，列入 M3 |

結論：沒有任何一個「算錯」，真正的工具鏈缺口只有 cooperative groups（2 個）；其餘未跑完的是測試資料檔（6 個）和一個逾時。

## 順序調整（2026-10-08）

M4 提前到 M2／M3 之前。理由：使用者的目標是「AI 工具裝了就能跑」，卡住的是帶 `.cu` 的套件，
不是 HeCBench 的科學計算程式。M2／M3 在背景跑，結果當作工具鏈正確性與效能的參考，不擋 M4。

### M4 的實際工作

難點不在編譯器，而在這三件事：

| 難點 | 說明 | 做法 |
|---|---|---|
| **torch 的 `ATen/cuda` 標頭檔** | 幾乎每個擴充套件都 include `ATen/cuda/CUDAContext.h`，用 `getCurrentCUDAStream()`、`CUDAGuard`、`at::Half`。torch XPU 版沒有這些檔案 | 寫一組同名的相容標頭，底下接到 torch XPU 的 stream 和 device guard |
| **GPU context 共用** | chipStar 有自己的 Level Zero context，torch XPU 有自己的 SYCL queue；不共用的話 tensor 指標在 kernel 裡無效 | 用 chipStar 的 `hipInitFromNativeHandles` 把 torch 的 driver／device／context／queue 交給它；同步用事件串接 |
| **`setup.py` 不用改** | 套件都寫 `CUDAExtension(...)`，呼叫 nvcc | 攔截 `torch.utils.cpp_extension`：`CUDAExtension` 改用 `b70cc` 編譯、連結 chipStar runtime |

另外一個便宜的加分：`triton-xpu` 已在映像檔裡，Triton 寫的 kernel 本來就能跑（Liger-Kernel 全是 Triton，
flash-attn 也有 Triton 版）。有 Triton 替代版本的套件直接換用，不必走編譯器。

步驟：① 最小的自製擴充套件（逼出三個難點的實際形狀）→ ② 真實套件，候選 `causal-conv1d`（kernel 簡單、沒有 inline PTX）
或 Mamba 的 selective scan（用到 CUB，chipStar 有 hipCUB 可接）。

### M4 ① 實測紀錄（2026-10-08）

`pytorch-ext/tests/ext_min`：一個照 NVIDIA 寫法寫的擴充（`CUDAExtension`、`ATen/cuda/CUDAContext.h`、
`AT_DISPATCH_FLOATING_TYPES`、`__shfl_down_sync`、`OptionalCUDAGuard`、`x.is_cuda()` 檢查），**原始碼一字不改**。

| 檢查 | 結果 |
|---|---|
| float／double 正確性 | 誤差 0 |
| warp shuffle 列加總（`warpSize` 32） | 誤差 2e-5 |
| torch 運算 → 擴充 → torch 運算，交錯 100 次，不加任何同步 | 誤差 0 |
| torch 立刻對擴充輸出做 reduction | 正確 |
| 每次呼叫的額外開銷（1024 個元素） | **44 µs**（純 torch 運算 13 µs；主機端同步模式 300 µs） |

三個難點的實際解法（都在 `pytorch-ext/`）：

| 難點 | 解法 |
|---|---|
| `ATen/cuda`、`c10/cuda` 標頭 | `include/` 裡的同名相容標頭，排在 torch 的 include 之前。torch XPU wheel 其實附了 `ATen/cuda/*.h`，但會拉進 cuBLAS／cuSPARSE 標頭，不能直接用 |
| context 共用與同步 | `src/interop.cpp`（`libb70torch.so`）：從 torch 的 SYCL queue 取出 Level Zero driver／device／context 交給 `hipInitFromNativeHandles`。同步兩個方向都走 GPU 事件：torch queue 的 barrier 事件轉成 hip 事件讓 CUDA stream 等；CUDA stream 記錄的事件轉成 SYCL 事件，在下一個 ATen 運算開始前（`RecordFunction` 回呼）讓 torch queue 等 |
| `setup.py` 不改 | `py/b70torch`：攔截 `torch.utils.cpp_extension`，`CUDA_HOME` 指向假的目錄（`bin/nvcc` 是過濾旗標後呼叫 cucc 的包裝；`.cpp` 也用 HIP 模式編，否則 torch 的 Half.h 會壞）、換掉 cudart／c10_cuda 等函式庫 |
| `is_cuda()`、`kCUDA`、`DeviceType::CUDA` | 無法用巨集改（會跟 torch 自己的宣告重複）。改成**影子原始碼樹**：套件目錄複製到 `build/b70_src`，只改寫套件自己的檔案（`rewrite.py` 的規則表），torch 標頭不動 |

**壓力測試**：2 萬次擴充呼叫與 torch 運算交錯（`pytorch-ext/tests/ext_min`），事件模式與主機模式都通過，誤差 0。
（第一次跑時段錯誤，gdb 顯示是測試腳本沒設 `CHIP_BE=level0`，chipStar 選了 OpenCL 後端卻拿到 Level Zero 的 handle；
現在 `b70` 指令和 b70torch 都把 `CHIP_BE=level0` 設成預設。）

**同步開銷明細**（`B70TORCH_PROFILE=1`，擴充呼叫＋torch 運算一對 68 µs，純 torch 兩個運算 25 µs；HeCBench 同時在跑，數字偏高）：

| 階段 | 平均 |
|---|---|
| torch → CUDA：SYCL barrier 事件、轉成 hip 事件、`hipStreamWaitEvent` | 2.6 ＋ 0.8 ＋ 6.6 µs |
| CUDA → torch：`hipEventRecord`、轉成 SYCL 事件、queue barrier | **17.2** ＋ 1.1 ＋ 5.2 µs |

下一步能省的：`hipEventRecord` 仍是大頭（chipStar 每次記錄都寫時間戳，即使 `cudaEventDisableTiming`）；
根本解法是讓 chipStar 的 stream 直接用 torch 的 immediate command list，同一條 in-order 佇列就完全不需要事件。

為此追加的 chipStar 補丁（都在 `toolchain/patches/0001`）：外部 handle 包成的事件視為已記錄且不擁有 handle；
`hipStreamWaitEvent` 對這種事件直接等它本身；`cucc` 接受 `-std=c++20`（patch 0002）。
另外補了 torch 標頭需要、chipStar 沒有的小東西：`thrust/complex.h`（只有型別和數學函式）、`__ldg(const __half*)`、
`cudaFuncSetAttribute`（Xe2 的 SLM 不需要申請，直接回傳成功）。

**CUB**：真實套件（causal-conv1d、Mamba）大量用 `cub::BlockLoad`／`BlockStore`／`BlockReduce`。chipStar 沒附 CUB，
但 chipStar 維護 rocPRIM 和 hipCUB 的分支（CUB 的 HIP 對應版，純標頭）。`b70cuda build` 現在會一併裝進工具鏈，
`toolchain/include/cub/*.cuh` 把 `cub::` 對應到 `hipcub::`（`b70cc` 預設帶這個 include，HeCBench 和 PyTorch 擴充共用）。

**JIT 路徑**（`torch.utils.cpp_extension.load_inline`／`load`）：也會經過改寫（在 build 目錄裡就地改；
`load()` 的使用者檔案則做影子樹）。`pytorch-ext/tests/test_load_inline.py` 通過（含 `is_cuda()` 檢查，建置 44 s）。

### M4 ② 實測紀錄（2026-10-08）：causal-conv1d 1.7.0

Dao-AILab 的 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)（Mamba 的依賴），PyPI 原始碼**一字不改**，
在 `b70 install` 的虛擬環境裡 `pip install --no-build-isolation` 裝起來（`setup.py` 走 `CUDAExtension`、
kernel 用 `cub::BlockLoad`／`BlockStore`、`c10::cuda::CUDAStream`、`__ldg`、`cudaFuncSetAttribute`，fp32／fp16／bf16 三種 dispatch）。

`pytorch-ext/tests/test_causal_conv1d_pkg.py`：對套件自己的 `causal_conv1d_ref`（純 torch）比對。

| 檢查 | 結果 |
|---|---|
| 前向：3 種 dtype × 寬度 2／4 × 無／silu × channel-first／channel-last，共 24 組 | 24／24 PASS |
| 反向（dx、dw、db），同 24 組 | 24／24 PASS |
| 最大誤差 | fp32 6e-6、fp16 4e-3、bf16 1.6e-2（都在 dtype 的捨入範圍內） |
| 吞吐量（8×2048×4096 bf16、silu，Mamba 尺寸） | **kernel 1.07 ms，torch 參考實作 11.70 ms**（快 11 倍） |

安裝過程逼出來、已修掉的問題：`nvcc -V` 版本查詢、`-std=c++20`、`cub/block/block_load.cuh`、`cudaFuncSetAttribute`、
`__ldg(const __half*)`、`thrust/complex.h`、`hipStreamWaitEvent` 對外部事件的處理。
最重要的一點：**同步正確性**不是靠 `cudaDeviceSynchronize`，是 GPU 事件串接，所以吞吐量數字沒有被主機端同步拖累。

**套件自己的測試**（驗收條件 ② 的原文；上游 `tests/test_causal_conv1d.py`，v1.7.0，13,373 個參數組合）：

| | 數量 | 說明 |
|---|---|---|
| 通過 | 9,005 | |
| 跳過 | 3,888 | 上游自己跳（channel-first 不支援 initial／final states） |
| 失敗 | 480 | **全部是參考實作算錯，不是 kernel**：204 個全是 `seqlen=1`＋channel-last；276 個 varlen 是隨機切出長度 1 的片段 |

追查結果（`pytorch-ext/tests/torch_xpu_conv1d_len1_bug.py` 可重現）：**torch 2.13.0+xpu 的 `F.conv1d` 權重梯度**在輸入長度 1、
而長度那一維的 stride 不是 1 時（`randn(B,1,C).transpose(1,2)`，stride `(C,1,C)`）算錯——torch 把它當成連續張量（size 1 的維度不看 stride），
XPU 的卷積反向卻把原始 stride 交給 oneDNN。前向正確、groups=1 也錯、CPU 正確。
kernel 的答案跟手算一致（長度 1 時「過去」那一格的權重梯度一定是 0，kernel 給 0，torch XPU 參考給非零值）。這是 torch XPU 的 bug，值得回報上游。

**mamba-ssm 2.2.5**（同一條路，`selective_scan_cuda`：CUB 的 `BlockRakingLayout`、反向 scan、complex dtype，10 個 `.cu`）：
第一次編譯缺 `cub/config.cuh`、`cub/detail/uninitialized_copy.cuh`、`cuda/std/type_traits`（libcu++），
補了對應 hipCUB／標準庫的小標頭後整包編過、裝起來。
附帶發現：2.2.5 跟映像檔的 transformers 版本不合（它 import 兩個已改名的類別），跟 GPU 無關，測試裡加了別名。

`pytorch-ext/tests/test_mamba_pkg.py`（對套件自己的 `selective_scan_ref` 比，容忍度照上游 `tests/ops/test_selective_scan.py`，
但上游只開 fp32，這裡加測 fp16／bf16）：

| 檢查 | 結果 |
|---|---|
| `selective_scan_fn` 前向，fp32／fp16／bf16 × 有無 z＋softplus | 6／6 PASS |
| 反向（du、ddelta、dA、dB、dC、dD、dz、ddelta_bias） | 6／6 PASS |
| 整個 `Mamba` block（fused 路徑 vs 同權重的非 fused 路徑）前向、反向 | 誤差 0、1e-6 |
| 吞吐（8×2048×768 bf16 一層前向） | fused 9.2 ms，非 fused 9.3 ms（這個尺寸下卷積與 scan 不是瓶頸） |

半精度的反向有一段插曲：bf16 加 z 門控時 dA 的絕對誤差到 1e2，看起來像算錯；但 dA 本身是 1e5（對 batch×seqlen 加總），
把同一份參考改用「捨入成 bf16 之前」的輸入去算，差距一樣是 1e2——誤差來自輸入精度本身，kernel 跟參考在 0.1% 內。
測試現在把這個「輸入捨入地板」也列為通過條件。

**mamba-ssm 上游測試**（`tests/ops/test_selective_scan.py`，20 個案例）：第一次 16 通過、4 失敗，失敗的全是 `seqlen ≥ 2048`、
B／C 隨序列變動的組合：`dA` 梯度**全為 0**，`dD`、`ddelta_bias` 錯，其餘梯度對。追了一整輪（記在下面），最後是 Intel GPU 編譯器的 bug；
chipStar 加上補丁 0003、mamba 重新編譯（`--no-cache-dir`，不然 pip 會拿快取裡舊的 wheel）後 **20／20 通過**，
`seqlen` 2048／4096、各種 batch／dim／dstate、有無 z 的梯度都在 1e-6 相對誤差內。

### 追查紀錄：IGC 常數位移 bug（2026-10-08）

症狀縮小的過程（每一步都有獨立的 `.cu` 重現；最後的最小重現在 `toolchain/probe/igc_slm_const_offset.cu`）：

1. `seqlen > 1024` 時 mamba 反向 kernel 改用 128 執行緒的版本——但 BlockReduce 本身在 32～1024 執行緒都對。
2. B／C 不隨序列變動（non-variable）的 128 執行緒版本完全正確，差別只在共享記憶體的配置：variable 版多了兩塊 BlockExchange 的暫存，
   把 BlockReduce 的暫存推到 33,792 B、`dA` 的累加區推到 36 KB 以上。
3. 用動態共享記憶體自己重現：`BlockReduce<float,128>` 的暫存放在常數位移 50,688 B → 結果全為 0；放在 25,344 B → 正確。
   位移改成執行期參數 → 正確。`-O0` 也錯 → 不是 LLVM 最佳化的問題；chipStar 產生的 IR 就是一個 `gep i8, %dyn_local_mem, 32832` 加普通的 load／store，沒問題。
4. 掃位移：**32 KB～64 KB 之間、不是「整數」的常數位移**（32,832、33,024、34,816、36,864、50,688、51,200）會錯；32,768、40,960、49,152、65,536 以上都對。
   只有「每個 sub-group 一條 lane 寫、thread 0 讀」這種 **sub-group 一致（scalar）的存取**會錯；每條 lane 各自存取的向量路徑正確。
5. 兩個映像檔的 IGC（2.28、2.38）都中。

修法：chipStar 的 `HipDynMem` pass 把 `extern __shared__` 換成隱藏的 kernel 參數，我們在那裡讓所有存取經過
`arg + (ptrtoint(arg) >> 40)`——執行期一定是 0，編譯器卻不能折疊，位址就留在暫存器裡（`toolchain/patches/0003`）。
代價是每個 kernel 多一次位移加法。第一版用 `>> 24`，結果驗收測試的 `scan_dynamic_shared` 反而壞了：
local 指標的原始值不是 0 起算的位移，是 `0x10000000 + 位移`（IGC 的 SLM 視窗），右移 24 得到 16，整個動態緩衝區被推後 16 bytes，每個 block 最後 4 個元素掉出去。
改成右移 40 後兩邊都對；`tests/cuda/cub_block.cu` 把 CUB block 原語（32～1024 執行緒、動態共享記憶體、大位移）加進驗收測試，
`scan_dynamic_shared` 則守住「位移真的是 0」。

順帶修掉的另外兩件事：`b70cc` 現在強制包含 `b70/cuda_compat.h`（`cudaFuncSetAttribute`，chipStar 沒有），
並定義 `__AMDGCN_WAVEFRONT_SIZE=32`——rocPRIM 沒看到 hipcc 的這個巨集時預設 64，lane mask 型別會錯（這次沒直接造成錯誤，但是定時炸彈）。

**轉接層的 bug**（跑上游 pytest 時發現）：transformers 用 `importlib.util.find_spec("torch")` 探測 torch 存不存在，
`b70cuda` 原本的 import hook 是一次性的，被這個探測觸發後就退場，真正的 `import torch` 沒被攔到——任何「先 import transformers 再 import torch」
的程式都會變成沒有轉接。已改成 torch 真的執行完才退場（三種 import 順序都驗過）。

### M4 ③ 實測紀錄（2026-10-08）：Triton 套件

前提：`triton-xpu` 3.7.2 隨 torch XPU wheel 一起在映像檔裡。`pytorch-ext/tests/test_triton.py`：照 NVIDIA 教學寫、
用 `device="cuda"` 的 Triton kernel（向量加、softmax）直接跑在 B70，4／4 正確；1M 元素加法 Triton 39 µs、torch 33 µs。

真實套件（`pytorch-ext/tests/test_triton_pkgs.py`，都用 `b70 install` 原封不動裝）：

| 套件 | 做法 | 結果 |
|---|---|---|
| **Liger-Kernel 0.8.4**（純 Triton，訓練用的融合 kernel） | 直接裝、直接跑 | RMSNorm 前向／反向、SwiGLU、fused linear cross-entropy 都對（fp32 誤差 1e-6、bf16 3e-2）；fused linear CE 4.67 ms 對 torch 3.89 ms |
| **sageattention**（2.x 要 nvcc＋sm80 PTX，原本被整個丟掉） | `pipfilter` 新增 `TRITON_ALT` 表：`sageattention>=2.0` → `sageattention==1.0.6`（最後一個純 Triton 版，同樣的 `sageattn` 介面） | 4 種形狀、causal／非 causal 全對（INT8 QK<sup>T</sup> 誤差 4e-2 內）；但速度 31 ms 對 torch SDPA 1.8 ms |

兩個順帶修掉的東西：`Tensor.is_cuda` 現在對 XPU 張量回 True（sageattention 第一行就 `assert q.is_cuda`；詳見 [01](01-pytorch-layer.zh-TW.md)）；
Liger 的 RMSNorm 反向預設**就地**改寫傳入的梯度，測試要先 `clone()`，不然參考值被弄壞（第一版測試誤判成 GPU 算錯）。

效能上的觀察：triton-xpu 對記憶體頻寬型 kernel（RMSNorm、cross-entropy）跟 torch 原生差 20% 內；
矩陣乘法型的 Triton kernel（sageattention 的 attention）只有約 4 TFLOPS，離 XMX 差很遠，torch 的 SDPA（oneDNN）快 17 倍。
結論：Triton 替代適合「本來跑不起來」的套件，不是效能路線；attention 這類應該改接 SDPA（跟 `flash_attn` 替身一樣）—— 列為後續。

**後續（2026-10-09）**：既然 sageattention 的 Triton 版比 SDPA 慢 17 倍，`b70 install` 改成直接丟掉 sageattention，
映像檔提供用 SDPA 實作的替身（`pytorch-layer/py/sageattention/`，跟 `flash_attn` 替身同一套做法）：`sageattn()` 的 HND／NHD 版面、
`is_causal`、`sm_scale`、GQA，以及 2.x 的各 kernel 入口名稱都有；`return_lse`、`sageattn_varlen` 不支援。自我測試 `sageattention_shim`
對照參考 attention 並量 4k token 的時間。

## M1／M3 實測紀錄

2026-10-08，B70 上 vLLM 閒置（佔顯存，不佔運算）。

**驗收測試**（`tests/run_cuda_tests.sh`）：11／11 通過，沒有一支要改原始碼。涵蓋 `__shared__`（靜態與動態）、
`__syncthreads`、warp shuffle（舊版與 `_sync` 版）、整數與 float／double `atomicAdd`、stream、event、
pinned 記憶體、`__constant__`、managed 記憶體、device `printf`、`warpSize == 32`。

**微基準**（`bench/micro.cu` 對 `bench/micro_sycl.cpp`，同樣的大小與 work-group，牆上時間取 10 次最佳）：

| 項目 | CUDA（b70cc） | SYCL（icpx） | CUDA／SYCL |
|---|---|---|---|
| stream copy | 526～529 GB/s | 527～531 GB/s | 1.00 |
| stream triad | 535 GB/s | 533～535 GB/s | 1.00 |
| kernel 啟動延遲 | 1.91～1.95 µs | 2.22～2.46 µs | CUDA 快約 16% |
| float atomicAdd（1024 位置） | 7.03～7.10 G ops/s | 6.72～6.78 G ops/s | 1.04 |

**reduction 只有 14 GB/s 的原因**：不是 kernel 慢（拆開量只要 0.057 ms，`bench/reduction_probe.cu`），
是測試第一次啟動 kernel 就在計時範圍內，量到的是 IGC 第一次編譯模組（約 90 ms）。測試已改成先暖機。

### 優化 1：`cudaEventRecord` 480 µs → 9 µs

在找上面的問題時，用 `bench/api_overhead.cu` 發現 `cudaEventRecord` 每次要 480 µs（正常是個位數微秒），
很多 CUDA 程式每一輪都會記錄事件。用 `toolchain/probe/ze_record_probe.c` 逐步量 Level Zero 呼叫，找到兩個原因：

| 原因 | 成本 | 修法 |
|---|---|---|
| GPU 把時間戳複製進事件物件，而事件物件在一般的 pageable 主機記憶體 | ~300 µs（複製到 USM 主機記憶體只要 ~10 µs） | 時間戳改放在 USM 主機記憶體池的位置 |
| 每次重新記錄都建一個新的事件池（chipStar issue #1258 的繞法） | ~300 µs／次 `zeEventPoolCreate` | 已觸發完成的舊事件最多留 8 個重複使用（安全條件跟原本銷毀的時機相同） |
| 每次記錄呼叫 `zeDeviceGetGlobalTimestamps`，但值只拿來比先後 | ~15 µs | 改用主機單調時鐘 |

試過但沒用的：把時間戳寫入改到計算引擎（避免跨引擎），反而比較慢，已撤回。

| | 修改前 | 修改後 |
|---|---|---|
| `cudaEventRecord`（GPU 閒置） | 480 µs | **9 µs** |
| kernel ＋ 記錄，每輪 | 292 µs | **28 µs**（單跑 kernel 22 µs） |

修改放在 `toolchain/patches/0001-level0-fast-event-record.patch`，`b70cuda build` 會自動套用。
驗收測試 11／11 仍然通過（現在是 70／70）。

## M3 第二輪（2026-10-08）：「慢的 benchmark」其實慢在 CPU 端，以及 fast math

### 1. tsa、hausdorff、bitonic-sort 的真相

第一輪用 `make run` 的牆上時間比較，CUDA 版慢的三個是 tsa 0.34、hausdorff 0.37、bitonic-sort 0.65。
把兩邊程式**自己印出來的 kernel 時間**抓出來看（`results-m2/*.run.log`）：

| benchmark | CUDA（b70cc）kernel | SYCL（icpx）kernel | 牆上時間 CUDA／SYCL |
|---|---|---|---|
| hausdorff | 22.05 ms | 22.02 ms | 20.4 s／7.5 s |
| bitonic-sort | 120 ms | 120 ms | 15.7 s／10.2 s |
| tsa（float／double） | 68／263 µs | 82／317 µs | 6.8 s／2.1 s |

GPU 端一樣快（tsa 的 CUDA 版還快一點）。差的全是 **CPU 參考計算**：hausdorff 的 CPU 端是 10¹⁰ 次距離計算，
`repeat=1` 時 CUDA 版 18.2 s、SYCL 版 5.1 s。追到最後是編譯器的事，不是工具鏈的事：

| 編譯 CPU 參考程式的方式 | 時間 |
|---|---|
| clang 22 `-O3`（b70cc 的 host 端） | 17.8 s |
| clang 22 `-O3 -ffast-math` | 17.8 s（`-Rpass-missed` 說內層迴圈的 `float2` struct 載入「return type cannot be vectorized」） |
| clang 22 `-O3 -ffast-math -march=native` | 13.4 s |
| icpx 2026.0 `-O3`（預設 `-fp-model=fast`） | **5.2 s** |
| icpx 2026.0 `-O3 -fp-model=precise` | 17.8 s |

icpx 要同時靠預設的 fast-math（允許 `min` 歸約重排）和它自己的向量化器（會拆 struct 載入）才拿到 3.5 倍；
開源 clang 兩個條件都不滿足，而 nvcc 配 gcc 的 host 端也一樣不會向量化這個迴圈，所以 b70cc 的行為是正確的，
只是 M3 ① 用牆上時間比較把這部分算進去了。另外第一輪 tsa float 的 1393 µs 是**第一次啟動的 IGC 編譯**算進計時區間
（100 次平均攤了 130 ms），快取（`$B70_DATA/cache/neo`、`cache/chipstar`）暖了之後是 68 µs。

處置：新增 `tests/hecbench/kernel_times.sh`＋`summarize_kernel.py`，只比較程式自己報的 kernel 時間，
每支程式跑兩次取第二次（避開冷啟動編譯）。結果見下方「kernel 時間比較」。

### 2. 真正的優化：device 端的數學旗標

查上面的事情時用 `-###` 看 b70cc 實際傳給 clang 的旗標，發現三件事（`bench/fast_math.cu`，16M 個元素，B70 上 vLLM 閒置）：

| kernel | 預設（修前） | 修後預設 | `--use_fast_math`（修後） | SYCL 預設／precise |
|---|---|---|---|---|
| 精確超越函式 `expf·sinf＋cosf·logf＋powf` | 0.924 ms | **0.31～0.41 ms**（多次量測） | 0.259 ms（誤差 1.4e-5） | 0.564／0.506 ms |
| 除法＋開根號 ×8 | 0.499 ms | 0.511 ms | **0.278 ms** | 0.452／0.758 ms |
| CUDA 內建 `__expf·__sinf＋__fdividef＋__powf` | **1.331 ms** | **0.255 ms** | 0.255 ms | 0.241 ms（`native::`） |
| 32 階多項式（FMA） | 0.258 ms | 0.256 ms | 0.256 ms | 0.240 ms |

1. **device 端開著 `-fmath-errno`**。CUDA device 程式不會設 errno，但 chipStar 沒關，clang 就只能把 `expf`、`sqrtf` 當成
   會設 errno 的函式呼叫，IGC 拿不到它自己的硬體版本。`b70cc` 現在一律加 `-Xarch_device -fno-math-errno`：
   精確超越函式 0.924 → 0.31～0.41 ms，精度完全不變（2.41e-7），跟 SYCL 的 precise 模式相當或更快。
2. **`__expf` 這類快速內建函式預設是精確版**，而且比 `expf` 還慢（1.33 ms 對 0.92 ms；`__powf` 走了 `pow`）。
   nvcc 的語意是這些永遠是快速版。先逐一量 Intel 的 `native_*` 準不準、快多少（`toolchain/probe/intrinsics_probe.cu`，輸入 (0.01, 3)，compute-bound）：

   | 函式 | 精確版誤差／時間 | native 誤差／時間 | 加速 |
   |---|---|---|---|
   | exp | 1.8e-7／0.518 ms | 2.9e-7／0.260 ms | 2.0× |
   | log | 8.1e-8／0.843 ms | 1.0e-7／0.179 ms | 4.7× |
   | **sin** | 1.0e-7／0.450 ms | **3.4e-5**／0.200 ms | 2.3× |
   | **cos** | 1.0e-7／0.482 ms | **3.1e-5**／0.186 ms | 2.6× |
   | tan | 3.4e-7／1.064 ms | 7.3e-7／0.407 ms | 2.6× |
   | log2／log10／exp10 | ~1e-7／0.44～0.57 ms | 1.0～2.6e-7／0.18～0.19 ms | 2.3～3.2× |
   | divide | 9.7e-8／0.458 ms | 9.7e-8／0.196 ms | 2.3× |
   | powr | 2.2e-7／1.547 ms | 2.4e-7／0.277 ms | 5.6× |
   | exp2、sqrt | 一樣 | 一樣 | 1.0× |

   CUDA 文件對 `__sinf`／`__cosf` 保證 [-π, π] 內絕對誤差 2^-21.4（3.6e-7），Intel 的 native 版差 100 倍，其他都在 CUDA 的界限內。
   補丁 `0004-fast-intrinsics-always`：`sp_intrinsics.hh` 裡 exp／exp10／log／log2／log10／powf／fdividef／tanf 八個內建函式
   改成預設走 native（`-DCHIP_PRECISE_INTRINSICS` 可關），`__sinf`（上游原本無條件 native）、`__cosf`、`__sincosf` 維持精確版，
   `--use_fast_math` 時才用 native。效果：`__intrinsics` kernel 1.33 → 0.26 ms。
   驗收測試新增 `tests/cuda/fast_intrinsics.cu`（兩種旗標各編一次）：精度在界限內，compute-bound 的 exp/log/pow/div/tan 內建版比精確版快 **3.9 倍**。
3. **`--use_fast_math` 是空的**：`cucc` 直接忽略，`-ffast-math` 到 `hipcc` 只變成 `-DCHIP_FAST_MATH`，兩者都到不了 device 編譯器。
   `b70cc` 現在把 `--use_fast_math` 展開成 nvcc 的定義（`-fapprox-func -freciprocal-math -ffp-contract=fast -fgpu-flush-denormals-to-zero`
   ＋ `-DCHIP_FAST_MATH`），`-ffast-math` 再多加 unsafe／finite／no-signed-zeros 到 device 與 host 兩邊
   （`hipcc` 會把字面上的 `-ffast-math` 吃掉，所以要拆成子旗標傳）。HeCBench 清單裡 adam、perplexity、softmax、stddev 用了 `--use_fast_math`。
   FMA 收縮本來就是開的（多項式 kernel 修前修後一樣）。

試過沒差的：`-fgpu-flush-denormals-to-zero` 單獨看不出時間差（仍保留，對應 nvcc 語意）。

### 3. kernel 時間比較（`tests/hecbench/kernel_times.sh`，2026-10-09，`tests/hecbench/kernel-times-2026-10-09.txt`）

34 個雙邊驗證通過的 benchmark（`list-verified.txt`），CUDA 版用現在的工具鏈（含 `-fno-math-errno`、補丁 0004）重編，
每支跑兩次取第二次，只比程式自己印的 kernel 時間。heat2d、nbody 只印頻寬／速率不印時間，略過；其餘 32 個：

**幾何平均 SYCL／CUDA kernel 時間 = 0.89**（CUDA 版慢 11%）。第一輪認定慢的三個全部消失：tsa 1.00、hausdorff 0.99、bitonic-sort 0.98。

| 最慢的五個 | CUDA | SYCL | 比值 | | 最快的五個 | CUDA | SYCL | 比值 |
|---|---|---|---|---|---|---|---|---|
| adam | 0.213 ms | 0.081 ms | 0.38 | | iso2dfd | 3.12 s | 5.52 s | 1.77 |
| fdtd3d | 0.332 ms | 0.188 ms | 0.57 | | gaussian | 0.97 s | 1.04 s | 1.07 |
| bilateral | 10.0 ms | 6.6 ms | 0.66 | | nlll | 0.079 ms | 0.084 ms | 1.06 |
| laplace | 1.81 s | 1.27 s | 0.70 | | lombscargle | 0.535 ms | 0.559 ms | 1.05 |
| jacobi | 881 ms | 630 ms | 0.72 | | all-pairs-distance | 23.5 ms | 24.4 ms | 1.04 |

中間 22 個在 0.72～1.01 之間，其中 17 個在 0.95～1.01。

**adam 0.38 的原因**（又是 fast-math，但這次在 GPU 端）：kernel 每個時間步算兩次 `powf(beta, t)`，SYCL 版因為 icpx 預設 `-fp-model=fast`
用了近似版 `pow`；CUDA 版的 Makefile 把 `--use_fast_math` 註解掉了，所以是精確版（nvcc 也會是精確版）。對照（GPU 閒置時重量）：

| | 精確 | fast-math |
|---|---|---|
| CUDA（b70cc） | 0.152 ms | **0.116 ms**（`--use_fast_math`，靠 b70cc 新的旗標展開） |
| SYCL（icpx） | 0.833 ms（`-fp-model=precise`） | 0.081 ms（預設） |

同樣是精確版，b70cc 版比 SYCL 版快 5 倍；同樣是 fast-math 差 1.4 倍（可能是 `pow` 的 `-fapprox-func` 展開跟 icpx 不同，待查）。
結論：M3 ① 的比較基準本身偏向 SYCL（它預設 fast-math），實際的 kernel 效能相當。

**還真的慢的**（下一輪目標）：fdtd3d 0.57（stencil、共享記憶體，`--use_fast_math` 沒有幫助：0.367 ms）、bilateral 0.66、laplace 0.70、jacobi 0.72。
這幾個要看 IGC 產生的程式碼（暫存器、SIMD 寬度、共享記憶體存取方式）跟 SYCL 版的差異。

### 4. 兩層各自的開銷（2026-10-09，vLLM 在線）

**工具鏈**（`bench/micro.cu` 對 `bench/micro_sycl.cpp`，同一支程式）：stream copy／scale／add／triad 518／520／533／531 GB/s 對 SYCL 529／529／533／531 GB/s（差 0～2%）；
kernel 啟動 1.88 µs 對 2.32 µs（CUDA 版快）；float atomic 7.09 對 6.72 G ops/s（CUDA 版快）。加上第 3 節的 HeCBench kernel 幾何平均 0.89，
整體來說同一段 CUDA 原始碼經 b70cc 跑，比手寫 SYCL 原生版平均慢約 10%，頻寬類的工作沒有差別。

**PyTorch 層**（同一段 torch 程式，`"cuda"` 經轉接層對 `"xpu"` 原生，`B70_CUDA=0`）：matmul 4096² bf16 0.83 對 1.02 ms（兩次量測的抖動，同一個 oneDNN kernel）、
SDPA 4k token 0.83 對 0.83 ms、conv2d 0.58 對 0.58 ms、TransformerEncoderLayer 前向 0.99 對 0.97 ms、1000 個小運算 9.99 對 10.45 ms、
`.to(device)` 0.017 對 0.015 ms。轉接層只改裝置名稱的解析，不碰運算本身，開銷在量測誤差內（0%）。

## 測試與效能的做法

- 所有驗收測試放在 `tests/`，效能測試放在 `bench/`，都要能用一個指令重跑
- 效能數字一律記錄：測試日期、驅動版本、B70 上有沒有其他工作在跑
- 和 SYCL 比較時，用同一張卡、同一套驅動，各跑 5 次取中位數
