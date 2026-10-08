// bench/fast_math.cu — does fast math matter on the B70?  Build variants (see docs/03-roadmap.md):
//   b70cc -O3 fast_math.cu                    default (precise functions, fast __intrinsics)
//   b70cc -O3 --use_fast_math fast_math.cu    nvcc-style fast math
// SYCL twin: fast_math_sycl.cpp (icpx -O3 -fsycl [-fp-model=precise]).
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#define N (16 << 20)
__global__ void transc(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float v = x[i]; y[i] = expf(v) * sinf(v) + cosf(v) * logf(v + 2.f) + powf(v + 1.f, 1.5f);
}
__global__ void divsqrt(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float v = x[i], a = v;
  #pragma unroll
  for (int k = 0; k < 8; k++) a = sqrtf(a + 1.f) / (a + 0.5f) + v / (a + 1.f);
  y[i] = a;
}
__global__ void intrins(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float v = x[i]; y[i] = __expf(v) * __sinf(v) + __fdividef(v, __cosf(v) + 2.f) + __powf(v + 1.f, 1.5f);
}
__global__ void poly(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float v = x[i], a = 1.f;
  #pragma unroll
  for (int k = 0; k < 32; k++) a = a * v + 0.37f;
  y[i] = a;
}
static double ref(int which, double v) {
  if (which == 0 || which == 2) return which == 0 ? exp(v) * sin(v) + cos(v) * log(v + 2) + pow(v + 1, 1.5)
                                                   : exp(v) * sin(v) + v / (cos(v) + 2) + pow(v + 1, 1.5);
  if (which == 1) { double a = v; for (int k = 0; k < 8; k++) a = sqrt(a + 1) / (a + 0.5) + v / (a + 1); return a; }
  double a = 1; for (int k = 0; k < 32; k++) a = a * v + 0.37; return a;
}
int main() {
  std::vector<float> hx(N), hy(N);
  for (int i = 0; i < N; i++) hx[i] = 0.001f + 0.999f * ((i * 2654435761u) % 1000003) / 1000003.f;
  float *dx, *dy; cudaMalloc(&dx, N * 4); cudaMalloc(&dy, N * 4);
  cudaMemcpy(dx, hx.data(), N * 4, cudaMemcpyHostToDevice);
  const char* names[] = {"transcendental", "div+sqrt", "__intrinsics", "polynomial(fma)"};
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int w = 0; w < 4; w++) {
    auto launch = [&]() {
      int g = (N + 255) / 256;
      if (w == 0) transc<<<g, 256>>>(dx, dy, N); else if (w == 1) divsqrt<<<g, 256>>>(dx, dy, N);
      else if (w == 2) intrins<<<g, 256>>>(dx, dy, N); else poly<<<g, 256>>>(dx, dy, N);
    };
    launch(); cudaDeviceSynchronize();
    cudaEventRecord(e0); for (int r = 0; r < 10; r++) launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
    cudaMemcpy(hy.data(), dy, N * 4, cudaMemcpyDeviceToHost);
    double maxrel = 0; for (int i = 0; i < N; i += 97) { double r = ref(w, hx[i]); double e = fabs(hy[i] - r) / fmax(fabs(r), 1e-30); if (e > maxrel) maxrel = e; }
    printf("%-16s %8.3f ms   max rel err %.2e\n", names[w], ms, maxrel);
  }
  return 0;
}
