// M3 micro-benchmarks, CUDA side (build with b70cc; bench/micro_sycl.cpp is the
// SYCL twin built with icpx). Reports the best of 10 runs for each test.
//   stream: copy / scale / add / triad over 3 x 256 MiB arrays -> GB/s
//   launch: back-to-back empty kernels -> microseconds per launch
//   atomic: contended float atomicAdd into 1024 slots -> G atomic ops/s
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("error %s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

__global__ void k_copy(const float *a, float *c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) c[i] = a[i]; }
__global__ void k_scale(float *b, const float *c, float s, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) b[i] = s * c[i]; }
__global__ void k_add(const float *a, const float *b, float *c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
__global__ void k_triad(float *a, const float *b, const float *c, float s, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) a[i] = b[i] + s * c[i]; }
__global__ void k_empty() {}
// Xe2 compresses device memory: zero-filled arrays would read back at far more than
// the GDDR6 can deliver. Fill with hash noise so the numbers are real bandwidth.
__global__ void k_fill(float *x, size_t n, unsigned seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned h = (unsigned)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        x[i] = (float)(h & 0xffffff) * 5.96e-8f + 0.5f;
    }
}
__global__ void k_atomic(float *slots, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < iters; k++) atomicAdd(&slots[(i + k) & 1023], 1.0f);
}

// Wall clock around launch + synchronize, exactly like the SYCL twin, so the two
// numbers include the same host-side costs.
template <class F> static double best_ms(F f, int reps = 10) {
    f(); CK(cudaDeviceSynchronize());  // warm-up / JIT
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f(); CK(cudaDeviceSynchronize());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    const size_t n = 64ull << 20;  // 64 Mi floats = 256 MiB per array
    const int T = 256; const unsigned G = (unsigned)((n + T - 1) / T);
    float *a, *b, *c;
    CK(cudaMalloc(&a, n * 4)); CK(cudaMalloc(&b, n * 4)); CK(cudaMalloc(&c, n * 4));
    k_fill<<<G, T>>>(a, n, 1u); k_fill<<<G, T>>>(b, n, 2u); k_fill<<<G, T>>>(c, n, 3u);
    CK(cudaDeviceSynchronize());
    double gb2 = 2.0 * n * 4 / 1e6, gb3 = 3.0 * n * 4 / 1e6;  // bytes / ms -> GB/s
    std::printf("stream_copy   %8.1f GB/s\n", gb2 / best_ms([&] { k_copy<<<G, T>>>(a, c, n); }));
    std::printf("stream_scale  %8.1f GB/s\n", gb2 / best_ms([&] { k_scale<<<G, T>>>(b, c, 3.0f, n); }));
    std::printf("stream_add    %8.1f GB/s\n", gb3 / best_ms([&] { k_add<<<G, T>>>(a, b, c, n); }));
    std::printf("stream_triad  %8.1f GB/s\n", gb3 / best_ms([&] { k_triad<<<G, T>>>(a, b, c, 3.0f, n); }));

    const int L = 2000;
    double ms = best_ms([&] { for (int i = 0; i < L; i++) k_empty<<<1, 32>>>(); }, 5);
    std::printf("launch        %8.2f us/kernel\n", ms * 1000.0 / L);

    float *slots; CK(cudaMalloc(&slots, 1024 * 4)); CK(cudaMemset(slots, 0, 1024 * 4));
    const int AB = 1024, AT = 256, IT = 64;
    ms = best_ms([&] { k_atomic<<<AB, AT>>>(slots, IT); });
    std::printf("atomic_f32    %8.2f G ops/s\n", (double)AB * AT * IT / ms / 1e6);
    return 0;
}
