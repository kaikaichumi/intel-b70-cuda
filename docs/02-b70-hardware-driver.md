# 02 B70 硬體與驅動，以及 CUDA 概念的對應

標記：**〔實測〕** 是用 `toolchain/probe/ze_probe.c`（Level Zero API）在 B70 上查到的；
**〔文件〕** 是 Intel 公開資料的說法，還沒在這張卡上驗證。

## 驅動堆疊

```
CUDA 原始碼 ──b70cc/clang──▶ SPIR-V 1.5
                               │
                     Level Zero loader 1.28（API 1.14）
                               │
              Intel compute-runtime（NEO）26.05 ── IGC 2.28：SPIR-V → Xe2 指令
                               │
                     Linux kernel 7.0 的 xe 驅動
                               │
                      Arc Pro B70（BMG-G31）
```

| 層 | 版本〔實測〕 | 角色 |
|---|---|---|
| kernel 驅動 | `xe`（Linux 7.0） | 記憶體管理、排程、韌體介面 |
| compute-runtime | 26.05.37020.3（Level Zero driver 0x103909c） | 實作 Level Zero 與 OpenCL |
| IGC | 2.28.4 | 把 SPIR-V 編成 Xe2 機器碼（執行時 JIT） |
| Level Zero loader | 1.28.2，API 1.14 | 應用程式呼叫的 API 入口 |

## 硬體規格〔實測〕

| 項目 | 數值 |
|---|---|
| 裝置 ID／IP 版本 | 0xe223／0x5008000（Xe2） |
| 時脈 | 2800 MHz |
| 結構 | 8 slice × 4 Xe-core × 8 EU × 8 執行緒 = **32 Xe-core、256 EU、2048 條硬體執行緒** |
| EU 實體 SIMD 寬度 | 16 |
| sub-group 寬度 | **16、32** |
| work-group 上限 | 1024（各維度都 1024） |
| group 數量上限 | 每個維度 2³²−1 |
| 共享記憶體（SLM） | 每個 work-group 最多 **128 KiB** |
| 快取 | 24 MiB（driver 回報的最後一層快取） |
| 記憶體 | 可配置 30.3 GiB；單次配置上限 30.3 GiB |
| SPIR-V | 1.5 |
| 模組功能 | fp16、fp64、int64 atomics、dp4a |
| float 原子運算 | fp32、fp64：global／local 的 add、min/max **都是硬體原生**；fp16 只有 load/store、min/max |
| kernel 參數上限 | 2048 bytes |
| printf 緩衝 | 4 MiB |
| 命令佇列 | 群組 0：compute + copy + **cooperative**，1 個引擎；群組 1：copy，1 個引擎 |
| USM | host／device／shared（單裝置）都可讀寫和原子運算；**不支援 system 配置**（一般 `malloc` 的記憶體不能直接給 GPU 用） |
| 計時器解析度 | 52 ns |

## 實測效能基準（SYCL／icpx 2026.0，`bench/micro_sycl.cpp`）

| 項目 | 結果 | 說明 |
|---|---|---|
| 記憶體頻寬（STREAM copy／scale／add／triad） | **527～534 GB/s** | 約為 256-bit GDDR6 理論值（~608 GB/s）的 87% |
| kernel 啟動延遲（連續送 2000 個空 kernel） | 2.2～2.5 µs／個 | in-order queue，含主機端開銷 |
| float 原子加法（1024 個位置互搶） | 6.7 G ops/s | |

**量頻寬一定要用不可壓縮的資料**：Xe2 會壓縮顯示記憶體。陣列全填 0 時，量到的「頻寬」是
898～1318 GB/s，遠超 GDDR6 的物理上限。改填雜湊亂數後才是上表的數字。
任何在 B70 上的記憶體效能測試都要注意這點。

## CUDA 概念對應

| CUDA | B70／Xe2 | 備註 |
|---|---|---|
| SM（streaming multiprocessor） | Xe-core（32 個） | 每個 Xe-core 8 個 EU，每個 EU 8 條硬體執行緒 |
| warp（32 執行緒） | sub-group 32 | 硬體 SIMD 是 16，SIMD32 由編譯器用兩組 SIMD16 組成。`warpSize` 固定為 32 |
| thread block | work-group | 上限同為 1024 |
| grid | ND-range | 維度上限比 CUDA 寬（CUDA 的 y／z 只到 65535） |
| `__shared__` | SLM | 最多 128 KiB，比多數 NVIDIA 消費級卡（約 100 KB）大 |
| warp shuffle／vote（`__shfl_*`、`__ballot`） | sub-group 運算 | chipStar 目前只支援非 `_sync` 版本 → **M5 要補** |
| `atomicAdd(float/double)` | 原生 float atomic | chipStar 預設用 CAS 迴圈模擬 → **B70 可以改走原生指令（M3 效能項目）** |
| `atomicAdd(__half)` | 只有 CAS 模擬 | 硬體不支援 fp16 原子加法 |
| `__dp4a` | dp4a | 硬體支援 |
| tensor core（`wmma`、`mma.sync`） | XMX（DPAS） | DPAS 要求 sub-group 16〔先前的 XMX 實驗〕，跟 warp 32 的資料排列不同 → 不能逐條對應，要另外設計（M5） |
| cooperative launch、grid sync | Level Zero cooperative kernel | compute 佇列支援 → cooperative groups 有機會做（chipStar 目前沒有） |
| stream | command list／queue | 只有 1 個 compute 引擎 + 1 個 copy 引擎，多個 stream 的 kernel 不會真的同時跑 |
| `cudaMemcpyAsync` | copy 引擎 | 有 1 個獨立 copy 引擎，可以跟計算重疊 |
| unified memory（`cudaMallocManaged`） | shared USM | 支援；但不支援 system 配置，所以沒有 HMM 那種「任何指標都能用」 |
| kernel 參數（CUDA 上限 4 KB，新版可到 32 KB） | 2048 bytes | 超過時 chipStar 會把參數搬到額外的緩衝區 |
| `printf` | 有，4 MiB 緩衝 | |
| `clock64()` | 計時器 | 解析度 52 ns |
