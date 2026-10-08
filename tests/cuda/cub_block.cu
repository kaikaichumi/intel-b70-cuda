// CUB block primitives through the toolchain's cub -> hipCUB -> rocPRIM shims,
// at every block size real extensions use (mamba's selective scan picks 32, 64
// or 128 threads from seqlen; CUB code commonly goes to 256/512/1024).
// Part 1: static shared storage. Part 2: the pattern mamba uses -- TempStorage
// objects carved out of dynamic (extern) shared memory at computed offsets,
// BlockReduce<float> aliasing BlockReduce<float2>'s storage, __launch_bounds__
// with a min-blocks hint, several reductions reusing one storage.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static int fails = 0;
static void check(const char *name, bool ok, const char *detail = "") {
  printf("%s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
  if (!ok) fails++;
}

// ---------------------------------------------------------------- part 1
template <int NT, int NI>
__global__ void k_block(const float *in, float *reduce_out, float *scan_out, float *copy_out, int *warp_out) {
  using Reduce = cub::BlockReduce<float, NT>;
  using Scan = cub::BlockScan<float, NT>;
  using Load = cub::BlockLoad<float, NT, NI, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Store = cub::BlockStore<float, NT, NI, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using WReduce = cub::WarpReduce<int>;
  __shared__ union {
    typename Reduce::TempStorage reduce;
    typename Scan::TempStorage scan;
    typename Load::TempStorage load;
    typename Store::TempStorage store;
    typename WReduce::TempStorage wreduce[NT / 32];
  } smem;
  const float *blk_in = in + blockIdx.x * NT * NI;
  float items[NI];
  Load(smem.load).Load(blk_in, items);
  __syncthreads();
  float local = 0;  // before the store: BlockStore WARP_TRANSPOSE permutes `items` in place (CUB does too)
  for (int i = 0; i < NI; ++i) local += items[i];
  Store(smem.store).Store(copy_out + blockIdx.x * NT * NI, items);  // load/store round trip
  __syncthreads();
  float total = Reduce(smem.reduce).Sum(local);
  if (threadIdx.x == 0) reduce_out[blockIdx.x] = total;
  __syncthreads();
  float t2 = Reduce(smem.reduce).Sum(local * 2.0f);  // reuse of the same storage
  if (threadIdx.x == 0) reduce_out[gridDim.x + blockIdx.x] = t2;
  __syncthreads();
  float incl;
  Scan(smem.scan).InclusiveSum(local, incl);
  scan_out[blockIdx.x * NT + threadIdx.x] = incl;
  __syncthreads();
  int w = WReduce(smem.wreduce[threadIdx.x / 32]).Sum(1);
  if (threadIdx.x % 32 == 0) warp_out[blockIdx.x * (NT / 32) + threadIdx.x / 32] = w;
}

template <int NT, int NI>
void run(const char *tag) {
  const int blocks = 8, n = blocks * NT * NI;
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = float((i * 7919) % 97) - 48.0f;  // small integers: exact in fp32
  float *d_in, *d_red, *d_scan, *d_copy;
  int *d_warp;
  CK(cudaMalloc(&d_in, n * 4)); CK(cudaMalloc(&d_red, 2 * blocks * 4));
  CK(cudaMalloc(&d_scan, blocks * NT * 4)); CK(cudaMalloc(&d_copy, n * 4));
  CK(cudaMalloc(&d_warp, blocks * (NT / 32) * 4));
  CK(cudaMemcpy(d_in, h.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_red, 0, 2 * blocks * 4));
  k_block<NT, NI><<<blocks, NT>>>(d_in, d_red, d_scan, d_copy, d_warp);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> red(2 * blocks), scan(blocks * NT), copy(n);
  std::vector<int> warp(blocks * (NT / 32));
  CK(cudaMemcpy(red.data(), d_red, 2 * blocks * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(scan.data(), d_scan, blocks * NT * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(copy.data(), d_copy, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(warp.data(), d_warp, blocks * (NT / 32) * 4, cudaMemcpyDeviceToHost));
  bool ok_red = true, ok_red2 = true, ok_scan = true, ok_copy = true, ok_warp = true;
  char d_red_s[128] = "", d_scan_s[160] = "";
  for (int b = 0; b < blocks; ++b) {
    double sum = 0, run_sum = 0;
    for (int t = 0; t < NT; ++t) {
      double per_thread = 0;
      for (int i = 0; i < NI; ++i) per_thread += h[b * NT * NI + t * NI + i];
      sum += per_thread;
      run_sum += per_thread;
      if (scan[b * NT + t] != float(run_sum) && ok_scan) {
        ok_scan = false;
        snprintf(d_scan_s, sizeof d_scan_s, "block %d thread %d got %g want %g (thread value %g, prev got %g)", b, t,
                 scan[b * NT + t], run_sum, per_thread, t ? scan[b * NT + t - 1] : 0.f);
      }
    }
    if (red[b] != float(sum)) { ok_red = false; snprintf(d_red_s, sizeof d_red_s, "block %d got %g want %g", b, red[b], sum); }
    if (red[blocks + b] != float(2 * sum)) ok_red2 = false;
    for (int t = 0; t < NT / 32; ++t) if (warp[b * (NT / 32) + t] != 32) ok_warp = false;
  }
  for (int i = 0; i < n; ++i) if (copy[i] != h[i]) ok_copy = false;
  char name[64];
  snprintf(name, sizeof name, "BlockReduce<%s>", tag); check(name, ok_red, d_red_s);
  snprintf(name, sizeof name, "BlockReduce_reuse<%s>", tag); check(name, ok_red2);
  snprintf(name, sizeof name, "BlockScan<%s>", tag); check(name, ok_scan, d_scan_s);
  snprintf(name, sizeof name, "BlockLoadStore_warp_transpose<%s>", tag); check(name, ok_copy);
  snprintf(name, sizeof name, "WarpReduce_32lanes<%s>", tag); check(name, ok_warp);
  CK(cudaFree(d_in)); CK(cudaFree(d_red)); CK(cudaFree(d_scan)); CK(cudaFree(d_copy)); CK(cudaFree(d_warp));
}

// ---------------------------------------------------------------- part 2
// mamba's selective_scan_bwd layout: everything in extern shared memory.
template <int NT, int NI>
struct DynTraits {
  using LoadT = cub::BlockLoad<float, NT, NI, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using ExchangeT = cub::BlockExchange<float, NT, NI>;
  using ReduceT = cub::BlockReduce<float2, NT>;
  using ReduceFloatT = cub::BlockReduce<float, NT>;
  using ScanT = cub::BlockScan<float2, NT>;
  static constexpr int kSmemIOSize = sizeof(typename LoadT::TempStorage);
  static constexpr int kSmemExchangeSize = 2 * sizeof(typename ExchangeT::TempStorage);
  static constexpr int kSmemReduceSize = sizeof(typename ReduceT::TempStorage);
  static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename ScanT::TempStorage);
  static constexpr int kTail = 256;  // floats after the TempStorage area (mamba: smem_da etc.)
  static constexpr int kTotal = kSmemSize + kTail * 4;
};

struct __align__(8) f2sum { __device__ float2 operator()(const float2 &a, const float2 &b) const { return make_float2(a.x + b.x, a.y + b.y); } };

template <int NT, int NI, int MINB>
__global__ __launch_bounds__(NT, MINB)
void k_dyn(const float *in, float *out, int n_chunks) {
  using T = DynTraits<NT, NI>;
  extern __shared__ char smem_[];
  auto &smem_load = *reinterpret_cast<typename T::LoadT::TempStorage *>(smem_);
  auto &smem_exchange = *reinterpret_cast<typename T::ExchangeT::TempStorage *>(smem_ + T::kSmemIOSize);
  auto &smem_reduce = *reinterpret_cast<typename T::ReduceT::TempStorage *>(reinterpret_cast<char *>(&smem_exchange) + T::kSmemExchangeSize);
  auto &smem_reduce_float = *reinterpret_cast<typename T::ReduceFloatT::TempStorage *>(&smem_reduce);
  float *smem_tail = reinterpret_cast<float *>(smem_ + T::kSmemSize);
  float acc_f = 0;
  for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {  // mamba walks chunks backwards
    float items[NI];
    typename T::LoadT(smem_load).Load(in + (blockIdx.x * n_chunks + chunk) * NT * NI, items);
    __syncthreads();
    float local = 0;
    for (int i = 0; i < NI; ++i) local += items[i];
    // float2 reduce into the shared reduce storage, then float reduces aliasing it (dA path)
    float2 r2 = typename T::ReduceT(smem_reduce).Reduce(make_float2(local, 2.0f * local), f2sum());
    if (threadIdx.x == 0) smem_tail[0] = chunk == n_chunks - 1 ? r2.x : r2.x + smem_tail[0];
    __syncthreads();
    for (int s = 0; s < 4; ++s) {  // several states, each one reduce + thread-0 write (dA path)
      float r = typename T::ReduceFloatT(smem_reduce_float).Sum(local * (s + 1));
      if (threadIdx.x == 0) smem_tail[8 + s] = chunk == n_chunks - 1 ? r : r + smem_tail[8 + s];
      __syncthreads();
    }
    acc_f += local;
  }
  float dd = typename T::ReduceFloatT(smem_reduce_float).Sum(acc_f);  // dD path
  if (threadIdx.x == 0) {
    atomicAdd(&out[blockIdx.x * 8 + 0], dd);
    atomicAdd(&out[blockIdx.x * 8 + 1], smem_tail[0]);
  }
  if (threadIdx.x < 4) atomicAdd(&out[blockIdx.x * 8 + 2 + threadIdx.x], smem_tail[8 + threadIdx.x]);
}

template <int NT, int NI, int MINB>
void run_dyn(const char *tag) {
  using T = DynTraits<NT, NI>;
  const int blocks = 4, n_chunks = 3, n = blocks * n_chunks * NT * NI;
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = float((i * 7919) % 97) - 48.0f;
  float *d_in, *d_out;
  CK(cudaMalloc(&d_in, n * 4)); CK(cudaMalloc(&d_out, blocks * 8 * 4));
  CK(cudaMemcpy(d_in, h.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_out, 0, blocks * 8 * 4));
  if (T::kTotal >= 48 * 1024) CK(cudaFuncSetAttribute((const void *)k_dyn<NT, NI, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::kTotal));
  k_dyn<NT, NI, MINB><<<blocks, NT, T::kTotal>>>(d_in, d_out, n_chunks);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> out(blocks * 8);
  CK(cudaMemcpy(out.data(), d_out, blocks * 8 * 4, cudaMemcpyDeviceToHost));
  bool ok_dd = true, ok_r2 = true, ok_states = true;
  char detail[200] = "";
  for (int b = 0; b < blocks; ++b) {
    double total = 0;
    for (int i = 0; i < n_chunks * NT * NI; ++i) total += h[b * n_chunks * NT * NI + i];
    if (out[b * 8 + 0] != float(total)) ok_dd = false;
    if (out[b * 8 + 1] != float(total)) ok_r2 = false;
    for (int s = 0; s < 4; ++s)
      if (out[b * 8 + 2 + s] != float(total * (s + 1))) {
        if (ok_states) snprintf(detail, sizeof detail, "block %d state %d got %g want %g", b, s, out[b * 8 + 2 + s], total * (s + 1));
        ok_states = false;
      }
  }
  char name[96];
  snprintf(name, sizeof name, "dyn_smem_reduce_float_aliasing_float2<%s> (smem %d B)", tag, T::kTotal); check(name, ok_states, detail);
  snprintf(name, sizeof name, "dyn_smem_reduce_float2<%s>", tag); check(name, ok_r2);
  snprintf(name, sizeof name, "dyn_smem_reduce_after_loop<%s>", tag); check(name, ok_dd);
  CK(cudaFree(d_in)); CK(cudaFree(d_out));
}

int main() {
  run<32, 4>("32x4");
  run<64, 4>("64x4");
  run<128, 4>("128x4");
  run<128, 16>("128x16");
  run<256, 4>("256x4");
  run<512, 2>("512x2");
  run<1024, 1>("1024x1");
  run_dyn<32, 16, 2>("32x16,minblocks2");
  run_dyn<64, 16, 2>("64x16,minblocks2");
  run_dyn<128, 16, 3>("128x16,minblocks3");
  run_dyn<128, 16, 1>("128x16,minblocks1");
  run_dyn<256, 16, 1>("256x16,minblocks1");
  run_dyn<256, 8, 1>("256x8,minblocks1");
  run_dyn<512, 4, 1>("512x4,minblocks1");
  run_dyn<128, 32, 1>("128x32,minblocks1");
  printf("%d failed\n", fails);
  return fails ? 1 : 0;
}
