// Host-API overhead probe: how long do the small runtime calls take that sit
// between kernels in real programs? (Found while chasing a slow reduction test.)
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

__global__ void touch(float *p) { if (threadIdx.x == 0) p[0] += 1.0f; }

template <class F> static double us_per_call(F f, int n = 200) {
    f(); cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
}

int main() {
    float *d, h = 0, *pinned;
    cudaMalloc(&d, 1 << 20);
    cudaMallocHost(&pinned, 1 << 20);
    std::printf("kernel launch           %8.2f us\n", us_per_call([&] { touch<<<1, 32>>>(d); }));
    std::printf("cudaMemset 4 B          %8.2f us\n", us_per_call([&] { cudaMemset(d, 0, 4); }));
    std::printf("cudaMemsetAsync 4 B     %8.2f us\n", us_per_call([&] { cudaMemsetAsync(d, 0, 4); }));
    std::printf("cudaMemset 1 MiB        %8.2f us\n", us_per_call([&] { cudaMemset(d, 0, 1 << 20); }));
    std::printf("memcpy D2H 4 B          %8.2f us\n", us_per_call([&] { cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); }));
    std::printf("memcpy H2D 4 B          %8.2f us\n", us_per_call([&] { cudaMemcpy(d, &h, 4, cudaMemcpyHostToDevice); }));
    std::printf("memcpy H2D 1 MiB pinned %8.2f us\n", us_per_call([&] { cudaMemcpy(d, pinned, 1 << 20, cudaMemcpyHostToDevice); }));
    std::printf("memset+launch           %8.2f us\n", us_per_call([&] { cudaMemset(d, 0, 4); touch<<<1, 32>>>(d); }));
    std::printf("cudaDeviceSynchronize   %8.2f us\n", us_per_call([&] { cudaDeviceSynchronize(); }));
    std::printf("cudaEventRecord         %8.2f us\n", us_per_call([&] {
        static cudaEvent_t e = nullptr; if (!e) cudaEventCreate(&e); cudaEventRecord(e); }));
    return 0;
}
