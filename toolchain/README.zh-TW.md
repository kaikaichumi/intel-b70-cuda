[English](README.md) | 繁體中文

# toolchain：B70 的 CUDA 原始碼工具鏈

把 `.cu` 原始碼編譯給 Arc Pro B70。底層是 [chipStar](https://github.com/CHIP-SPV/chipStar)
（CUDA／HIP → SPIR-V → Level Zero），本資料夾負責在 B70 上建好、包裝成好用的指令。

## 需求

- Linux，Intel GPU 用 `xe` 驅動
- Docker（使用者要在 `docker` 群組）
- 約 10 GB 硬碟空間給 LLVM 和 chipStar（放在 `$B70_DATA/cuda`，可以設在大硬碟上）

主機上不需要裝任何編譯器或 oneAPI，全部在 `b70-cuda-dev` 容器裡（見 `Dockerfile.dev`）。

## 安裝

```bash
ln -s "$PWD/b70cuda" ~/.local/bin/b70cuda
b70cuda setup     # 下載 chipStar 補丁版 LLVM 22 和 chipStar 原始碼（~5 GB，直接串流到 $B70_DATA/cuda）
b70cuda build     # 建開發映像檔（第一次）、編譯並安裝 chipStar（6 核心約 10 分鐘），
                  # 再裝 rocPRIM、hipCUB（CUB 的對應版，純標頭；單獨重做用 b70cuda libs）
b70cuda test      # 驗收測試：tests/run_cuda_tests.sh
```

`build` 會先套用 `patches/` 裡的修正（chipStar 版本固定在 `CHIPSTAR_COMMIT`）：

| 補丁 | 內容 |
|---|---|
| `0001-level0-fast-event-record` | `cudaEventRecord` 480 µs → 9 µs；外部（SYCL）事件可以被 `cudaStreamWaitEvent` 等待 |
| `0002-cucc-cxx20` | `cucc` 接受 `-std=c++20`／`c++23`（PyTorch 2.13 用 c++20 建擴充套件） |
| `0003-hipdynmem-opaque-base` | 繞過 Intel IGC 的 bug：動態共享記憶體（`extern __shared__`）以常數位移 32–64 KB 存取時，sub-group 一致的讀寫會得到 0（hipCUB 的 BlockReduce 放在大緩衝區後段就會中；Mamba 反向就是這樣壞的）。改成經過一個執行期才知道是 0 的位移，位址留在暫存器裡 |
| `0004-fast-intrinsics-always` | CUDA 的快速內建函式（`__expf`、`__logf`、`__fdividef`、`__powf`、`__tanf`…）跟 nvcc 一樣**永遠**是快速版。chipStar 原本只在 `-DCHIP_FAST_MATH` 時才用 OpenCL 的 `native_*`，預設反而比精確版慢（`__expf·__sinf＋__fdividef＋__powf` 1.33 ms 對 `expf…` 0.92 ms；修後 0.26 ms）。例外是 `__sinf`／`__cosf`／`__sincosf`：Intel 的 `native_sin`／`native_cos` 誤差 3e-5，CUDA 保證 2^-21.4（4e-7），所以這三個維持精確版，`--use_fast_math` 時才用 native。要全部回到原本行為可加 `-DCHIP_PRECISE_INTRINSICS` |

## 浮點數旗標（`b70cc` 幫你處理的）

| 你寫的 | b70cc 實際做的 | 為什麼 |
|---|---|---|
| （什麼都不加） | device 端加 `-fno-math-errno` | CUDA device 程式本來就不會設 errno；chipStar 預設留著 errno，clang 就不能把 `expf`／`sqrtf` 變成 IGC 的硬體版，精確超越函式慢 2～3 倍（`bench/fast_math.cu`：0.92 ms → 0.31～0.41 ms，精度不變） |
| `--use_fast_math` | `-DCHIP_FAST_MATH` ＋ device 端 `-fapprox-func -freciprocal-math -ffp-contract=fast -fgpu-flush-denormals-to-zero` | 對應 nvcc 的定義（近似超越函式、近似除法與開根號、FMA、FTZ）。chipStar 的 `cucc` 原本直接忽略這個旗標 |
| `-ffast-math`（含 `-Xcompiler -ffast-math`） | 上面那組，再加 device 與 host 端的 `-funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros` 等 | chipStar 的 `hipcc` 會把這個旗標整個吃掉只留 `-DCHIP_FAST_MATH`，所以拆成各個子旗標傳下去 |

各函式的 native 版在 (0.01, 3) 的輸入上實測（`toolchain/probe/intrinsics_probe.cu`，相對誤差與 compute-bound 的加速倍數）：
exp 2.9e-7／2.0×、log 1.0e-7／4.7×、log2 1.0e-7／3.1×、log10 1.5e-7／3.2×、exp10 2.6e-7／2.3×、divide 9.7e-8／2.3×、
powr 2.4e-7／5.6×、tan 7.3e-7／2.6×；**sin 3.4e-5／2.3×、cos 3.1e-5／2.6×**（所以 sin／cos 不預設用）；exp2、sqrt 本來就一樣快。
`--use_fast_math` 整體：`expf·sinf＋cosf·logf＋powf` 最大相對誤差 1.4e-5（預設 2.4e-7）。

`setup` 不用 `docker pull`，用 `crane` 把官方映像檔的內容直接串流解壓，所以系統碟只會多出開發映像檔那一層。

## 使用

```bash
b70cuda cc -O3 my.cu -o my     # 跟 nvcc 一樣的用法
b70cuda run ./my               # 在 B70 上執行
b70cuda shell                  # 進開發環境；裡面可以直接用 b70cc、icpx
```

CMake 專案：在 `b70cuda shell` 裡加上 `-DCMAKE_CUDA_COMPILER=$(which b70cc)`。

## 設定

跟 `b70` 指令共用 `~/.config/b70/config`：

| 變數 | 預設 | 用途 |
|---|---|---|
| `B70_DATA` | `~/.local/share/b70` | 資料和快取 |
| `B70_CUDA_HOME` | `$B70_DATA/cuda` | LLVM、chipStar 原始碼、建置與安裝目錄 |
| `B70_GPU_PCI` | 自動偵測 | 有多張 Intel 卡時指定 |
| `CHIP_LOGLEVEL` | `err` | chipStar 的除錯訊息（`trace`／`debug`／`info`） |

## 目前支援程度

見 [docs/03-roadmap.md](../docs/03-roadmap.zh-TW.md) 的實測紀錄。已知 chipStar 的限制：不支援 inline PTX、
tensor core（`wmma`／`mma`）、cuBLAS／cuFFT 等 NVIDIA 函式庫、CUDA Graphs。
