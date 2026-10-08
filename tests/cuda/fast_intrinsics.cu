// CUDA fast intrinsics (__expf, __logf, __fdividef, __powf, ...) must be the *fast* versions
// (always, as in nvcc) and still be accurate to CUDA's documented bounds; the precise
// functions (expf, sinf, ...) must stay precise.  __sinf/__cosf are the exception on the
// B70: Intel's native_sin/native_cos are only accurate to ~3e-5, so toolchain patch 0004
// keeps them precise unless fast math is requested (see toolchain/README.md).
// The speed check uses compute-bound kernels (exp/log/pow/div/tan, 32 dependent rounds).
//
// Built twice by run_cuda_tests.sh: plain, and with --use_fast_math (then the precise
// functions are allowed to be approximate too, as with nvcc).
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define N (4 << 20)

__global__ void k_fast(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  float s, c;
  __sincosf(v, &s, &c);
  out[0 * n + i] = __expf(v);
  out[1 * n + i] = __logf(v);
  out[2 * n + i] = __sinf(v);
  out[3 * n + i] = __cosf(v);
  out[4 * n + i] = __tanf(v * 0.3f);
  out[5 * n + i] = __powf(v, 1.5f);
  out[6 * n + i] = __fdividef(v, v + 1.f);
  out[7 * n + i] = __log2f(v) + __log10f(v) + __exp10f(v * 0.25f) + s + c;
}

__global__ void k_precise(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  out[0 * n + i] = expf(v);
  out[1 * n + i] = logf(v);
  out[2 * n + i] = sinf(v);
  out[3 * n + i] = cosf(v);
  out[4 * n + i] = tanf(v * 0.3f);
  out[5 * n + i] = powf(v, 1.5f);
  out[6 * n + i] = v / (v + 1.f);
  out[7 * n + i] = log2f(v) + log10f(v) + exp10f(v * 0.25f) + sinf(v) + cosf(v);
}

static double ref(int f, double v) {
  switch (f) {
    case 0: return exp(v);
    case 1: return log(v);
    case 2: return sin(v);
    case 3: return cos(v);
    case 4: return tan(v * 0.3);
    case 5: return pow(v, 1.5);
    case 6: return v / (v + 1);
    default: return log2(v) + log10(v) + pow(10.0, v * 0.25) + sin(v) + cos(v);
  }
}

__global__ void k_fast_heavy(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i], acc = 0.f;
  #pragma unroll 4
  for (int k = 0; k < 32; k++) {
    float u = v + k * 1e-3f;
    acc += __expf(u) + __logf(u) + __powf(u, 1.5f) + __fdividef(u, u + 1.f) + __tanf(u * 0.3f);
  }
  out[i] = acc;
}

__global__ void k_precise_heavy(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i], acc = 0.f;
  #pragma unroll 4
  for (int k = 0; k < 32; k++) {
    float u = v + k * 1e-3f;
    acc += expf(u) + logf(u) + powf(u, 1.5f) + u / (u + 1.f) + tanf(u * 0.3f);
  }
  out[i] = acc;
}

static void launch_heavy(int which, const float* x, float* out) {
  if (which) k_precise_heavy<<<(N + 255) / 256, 256>>>(x, out, N);
  else       k_fast_heavy<<<(N + 255) / 256, 256>>>(x, out, N);
}

static void launch(int which, const float* x, float* out) {
  if (which) k_precise<<<(N + 255) / 256, 256>>>(x, out, N);
  else       k_fast<<<(N + 255) / 256, 256>>>(x, out, N);
}

static float time_kernel(int which, const float* x, float* out) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  launch_heavy(which, x, out);
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int r = 0; r < 5; r++) launch_heavy(which, x, out);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms / 5;
}

int main() {
  // inputs in (0.01, 3): inside the ranges CUDA documents the intrinsic error bounds for
  std::vector<float> hx(N), hy(8 * (size_t)N);
  for (int i = 0; i < N; i++) hx[i] = 0.01f + 2.99f * ((i * 2654435761u) % 1000003) / 1000003.f;
  float *dx, *dy;
  cudaMalloc(&dx, N * sizeof(float));
  cudaMalloc(&dy, 8 * (size_t)N * sizeof(float));
  cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  const char* names[8] = {"exp", "log", "sin", "cos", "tan", "pow", "div", "log2+log10+exp10+sincos"};
  // CUDA's bounds for __expf etc. are a few ulp (absolute 2^-21.4 for sin/cos/log near 1);
  // 1e-5 relative is comfortably loose for them and far tighter than "precise accidentally
  // became garbage".  The precise functions get 1e-6 (~8 ulp) unless --use_fast_math.
  // With --use_fast_math everything may use Intel's native_* (sin/cos ~3e-5), so both sides
  // get 1e-4 then, as nvcc also degrades sinf to __sinf in that mode.
#if defined(__USE_FAST_MATH__) || defined(CHIP_FAST_MATH)
  const double tol_fast = 1e-4, tol_precise = 1e-4;
  const char* mode = "--use_fast_math";
#else
  const double tol_fast = 1e-5, tol_precise = 1e-6;
  const char* mode = "default";
#endif

  int bad_total = 0;
  for (int which = 0; which < 2; which++) {
    launch(which, dx, dy);
    cudaMemcpy(hy.data(), dy, 8 * (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
    double tol = which ? tol_precise : tol_fast;
    double worst = 0; int bad = 0; const char* worst_name = "";
    for (int f = 0; f < 8; f++) {
      for (int i = 0; i < N; i += 13) {
        double r = ref(f, hx[i]);
        double e = fabs((double)hy[f * (size_t)N + i] - r) / fmax(fabs(r), 1.0);  // relative, absolute below 1
        if (e > worst) { worst = e; worst_name = names[f]; }
        if (e > tol) bad++;
      }
    }
    bad_total += bad;
    printf("%s %s functions (%s): worst error %.2e (%s), tolerance %.0e, %d bad samples\n",
           bad ? "FAIL" : "PASS", which ? "precise" : "__intrinsic", mode, worst, worst_name, tol, bad);
  }

  float t_fast = time_kernel(0, dx, dy), t_precise = time_kernel(1, dx, dy);
  printf("device: compute-bound exp/log/pow/div/tan: __intrinsics %.3f ms, precise %.3f ms (%.2fx)\n",
         t_fast, t_precise, t_precise / t_fast);
  // The whole point of the intrinsics.  In default mode they must be clearly faster (measured
  // 2-5x per function on the B70); with --use_fast_math both sides are fast: only "not slower".
#if defined(__USE_FAST_MATH__) || defined(CHIP_FAST_MATH)
  const float need = 1.0f / 1.05f;
#else
  const float need = 1.3f;
#endif
  printf("%s __intrinsics are %.2fx faster than the precise functions (need >= %.2f)\n",
         t_precise / t_fast >= need ? "PASS" : "FAIL", t_precise / t_fast, need);
  return bad_total ? 1 : 0;
}
