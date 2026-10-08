// bench/fast_math_sycl.cpp — SYCL twin of fast_math.cu (icpx default is -fp-model=fast; compare with -fp-model=precise).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#define N (16 << 20)
static double ref(int which, double v) {
  if (which == 0) return exp(v) * sin(v) + cos(v) * log(v + 2) + pow(v + 1, 1.5);
  if (which == 1) { double a = v; for (int k = 0; k < 8; k++) a = sqrt(a + 1) / (a + 0.5) + v / (a + 1); return a; }
  if (which == 2) return exp(v) * sin(v) + v / (cos(v) + 2) + pow(v + 1, 1.5);
  double a = 1; for (int k = 0; k < 32; k++) a = a * v + 0.37; return a;
}
int main() {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  std::vector<float> hx(N), hy(N);
  for (int i = 0; i < N; i++) hx[i] = 0.001f + 0.999f * ((i * 2654435761u) % 1000003) / 1000003.f;
  float* dx = sycl::malloc_device<float>(N, q); float* dy = sycl::malloc_device<float>(N, q);
  q.memcpy(dx, hx.data(), N * 4).wait();
  const char* names[] = {"transcendental", "div+sqrt", "native::", "polynomial(fma)"};
  for (int w = 0; w < 4; w++) {
    auto launch = [&]() {
      q.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
        int i = it.get_global_id(0); float v = dx[i];
        if (w == 0) dy[i] = sycl::exp(v) * sycl::sin(v) + sycl::cos(v) * sycl::log(v + 2.f) + sycl::pow(v + 1.f, 1.5f);
        else if (w == 1) { float a = v; for (int k = 0; k < 8; k++) a = sycl::sqrt(a + 1.f) / (a + 0.5f) + v / (a + 1.f); dy[i] = a; }
        else if (w == 2) dy[i] = sycl::native::exp(v) * sycl::native::sin(v) + sycl::native::divide(v, sycl::native::cos(v) + 2.f) + sycl::native::powr(v + 1.f, 1.5f);
        else { float a = 1.f; for (int k = 0; k < 32; k++) a = a * v + 0.37f; dy[i] = a; }
      });
    };
    launch(); q.wait();
    auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < 10; r++) launch(); q.wait();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10;
    q.memcpy(hy.data(), dy, N * 4).wait();
    double maxrel = 0; for (int i = 0; i < N; i += 97) { double r = ref(w, hx[i]); double e = fabs(hy[i] - r) / fmax(fabs(r), 1e-30); if (e > maxrel) maxrel = e; }
    printf("%-16s %8.3f ms   max rel err %.2e\n", names[w], ms, maxrel);
  }
}
