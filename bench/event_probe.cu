// Do CUDA events report the right time, and what does cudaEventRecord cost?
// Mirrors the gpu_ms() helper of tests/cuda/common.h around memset + kernel.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

__global__ void work(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.0001f + 1.0f;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    const int n = 1 << 22, reps = 20;
    float *d;
    cudaMalloc(&d, n * 4);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    work<<<n / 256, 256>>>(d, n);
    cudaDeviceSynchronize();

    for (int variant = 0; variant < 2; variant++) {
        double t0 = now_ms();
        cudaEventRecord(a);
        for (int r = 0; r < reps; r++) {
            if (variant) cudaMemset(d, 0, 4);
            work<<<n / 256, 256>>>(d, n);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        double wall = now_ms() - t0;
        float ev = 0;
        cudaEventElapsedTime(&ev, a, b);
        std::printf("%-22s wall %8.3f ms   events %8.3f ms   (per rep: wall %.3f, events %.3f)\n",
                    variant ? "memset + kernel" : "kernel only", wall, ev, wall / reps, ev / reps);
    }

    // cost of cudaEventRecord itself
    const int m = 200;
    double t0 = now_ms();
    for (int i = 0; i < m; i++) cudaEventRecord(a);
    cudaDeviceSynchronize();
    std::printf("cudaEventRecord (same event, idle GPU)  %.2f us each\n", (now_ms() - t0) * 1000 / m);
    t0 = now_ms();
    for (int i = 0; i < m; i++) { work<<<n / 256, 256>>>(d, n); cudaEventRecord(a); }
    cudaDeviceSynchronize();
    double with = now_ms() - t0;
    t0 = now_ms();
    for (int i = 0; i < m; i++) work<<<n / 256, 256>>>(d, n);
    cudaDeviceSynchronize();
    double without = now_ms() - t0;
    std::printf("kernel+record vs kernel alone           %.2f vs %.2f us per iteration\n",
                with * 1000 / m, without * 1000 / m);
    return 0;
}
