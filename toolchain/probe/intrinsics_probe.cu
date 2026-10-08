// Per-function: accuracy (single evaluation vs double, inputs in (0.01, 3)) and compute-bound
// speed (64 dependent evaluations per thread) of OpenCL native_* vs the precise function.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
extern "C++" __device__ float native_exp(float); extern "C++" __device__ float native_log(float);
extern "C++" __device__ float native_sin(float); extern "C++" __device__ float native_cos(float);
extern "C++" __device__ float native_tan(float); extern "C++" __device__ float native_exp2(float);
extern "C++" __device__ float native_log2(float); extern "C++" __device__ float native_exp10(float);
extern "C++" __device__ float native_log10(float); extern "C++" __device__ float native_divide(float, float);
extern "C++" __device__ float native_sqrt(float); extern "C++" __device__ float native_rsqrt(float);
extern "C++" __device__ float native_powr(float, float);
#define NF 12
__device__ __forceinline__ float fn(int f, bool fast, float v) {
  switch (f) {
    case 0: return fast ? native_exp(v) : expf(v);
    case 1: return fast ? native_log(v) : logf(v);
    case 2: return fast ? native_sin(v) : sinf(v);
    case 3: return fast ? native_cos(v) : cosf(v);
    case 4: return fast ? native_tan(v * 0.3f) : tanf(v * 0.3f);
    case 5: return fast ? native_exp2(v) : exp2f(v);
    case 6: return fast ? native_log2(v) : log2f(v);
    case 7: return fast ? native_exp10(v * 0.3f) : exp10f(v * 0.3f);
    case 8: return fast ? native_log10(v) : log10f(v);
    case 9: return fast ? native_divide(v, v + 1.f) : v / (v + 1.f);
    case 10: return fast ? native_sqrt(v) : sqrtf(v);
    case 11: return fast ? native_powr(v, 1.5f) : powf(v, 1.5f);
  }
  return 0.f;
}
static double ref(int f, double v) {
  switch (f) {
    case 0: return exp(v); case 1: return log(v); case 2: return sin(v); case 3: return cos(v);
    case 4: return tan(v * 0.3); case 5: return exp2(v); case 6: return log2(v); case 7: return pow(10.0, v * 0.3);
    case 8: return log10(v); case 9: return v / (v + 1); case 10: return sqrt(v); default: return pow(v, 1.5);
  }
}
template <int F, bool FAST> __global__ void k_acc(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  y[i] = fn(F, FAST, x[i]);
}
template <int F, bool FAST> __global__ void k_speed(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float v = x[i], acc = 0.f;
  #pragma unroll 4
  for (int k = 0; k < 64; k++) acc += fn(F, FAST, v + k * 1e-3f);
  y[i] = acc;
}
#define N (4 << 20)
template <int F> void one(const char* name, const float* dx, float* dy, const std::vector<float>& hx, std::vector<float>& hy) {
  double err[2]; float ms[2];
  for (int fast = 0; fast < 2; fast++) {
    if (fast) k_acc<F, true><<<N / 256, 256>>>(dx, dy, N); else k_acc<F, false><<<N / 256, 256>>>(dx, dy, N);
    cudaMemcpy(hy.data(), dy, N * 4, cudaMemcpyDeviceToHost);
    double w = 0; for (int i = 0; i < N; i += 7) { double r = ref(F, hx[i]); double e = fabs(hy[i] - r) / fmax(fabs(r), 1.0); if (e > w) w = e; }
    err[fast] = w;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto L = [&]() { if (fast) k_speed<F, true><<<N / 256, 256>>>(dx, dy, N); else k_speed<F, false><<<N / 256, 256>>>(dx, dy, N); };
    L(); cudaDeviceSynchronize(); cudaEventRecord(e0); for (int r = 0; r < 5; r++) L(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms[fast], e0, e1); ms[fast] /= 5;
  }
  printf("%-10s precise: err %.1e %7.3f ms | native: err %.1e %7.3f ms | native speedup %.2fx\n", name, err[0], ms[0], err[1], ms[1], ms[0] / ms[1]);
}
int main() {
  std::vector<float> hx(N), hy(N);
  for (int i = 0; i < N; i++) hx[i] = 0.01f + 2.99f * ((i * 2654435761u) % 1000003) / 1000003.f;
  float *dx, *dy; cudaMalloc(&dx, N * 4); cudaMalloc(&dy, N * 4); cudaMemcpy(dx, hx.data(), N * 4, cudaMemcpyHostToDevice);
  one<0>("exp", dx, dy, hx, hy); one<1>("log", dx, dy, hx, hy); one<2>("sin", dx, dy, hx, hy); one<3>("cos", dx, dy, hx, hy);
  one<4>("tan", dx, dy, hx, hy); one<5>("exp2", dx, dy, hx, hy); one<6>("log2", dx, dy, hx, hy); one<7>("exp10", dx, dy, hx, hy);
  one<8>("log10", dx, dy, hx, hy); one<9>("divide", dx, dy, hx, hy); one<10>("sqrt", dx, dy, hx, hy); one<11>("powr", dx, dy, hx, hy);
}
