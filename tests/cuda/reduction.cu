// M1/M5: block reduction in shared memory, finished with warp shuffles.
// Built twice by the runner: plain (legacy __shfl_down) and with -DUSE_SYNC
// (__shfl_down_sync, what all modern CUDA code uses).
#include "common.h"
#include <vector>

__inline__ __device__ float warp_sum(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
#ifdef USE_SYNC
        v += __shfl_down_sync(0xffffffffu, v, off);
#else
        v += __shfl_down(v, off);
#endif
    }
    return v;
}

__global__ void reduce(const float *in, float *out, int n) {
    __shared__ float warp_part[32];
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        v += in[i];
    v = warp_sum(v);
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) warp_part[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lane < blockDim.x / warpSize ? warp_part[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

int main() {
    const int n = 1 << 24;
    std::vector<float> h(n);
    double ref = 0;
    for (int i = 0; i < n; i++) { h[i] = ((i * 37) % 101) * 0.01f; ref += h[i]; }
    float *din, *dout;
    CK(cudaMalloc(&din, n * sizeof(float)));
    CK(cudaMalloc(&dout, sizeof(float)));
    CK(cudaMemcpy(din, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int threads = 256, blocks = 32 * 8;
    // warm-up: the first launch JIT-compiles the module, keep that out of the timing
    reduce<<<blocks, threads>>>(din, dout, n);
    CK(cudaDeviceSynchronize());
    float ms = gpu_ms([&] {
        cudaMemset(dout, 0, sizeof(float));
        reduce<<<blocks, threads>>>(din, dout, n);
    }, 20);
    CK(cudaGetLastError());
    float got;
    CK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
    double rel = std::fabs(got - ref) / ref;
    char d[128];
    std::snprintf(d, sizeof d, "warpSize-based, %.3f ms, %.0f GB/s, rel err %.1e", ms,
                  n * sizeof(float) / ms / 1e6, rel);
#ifdef USE_SYNC
    const char *name = "reduction_sync";
#else
    const char *name = "reduction";
#endif
    CK(cudaFree(din)); CK(cudaFree(dout));
    return finish(name, rel < 1e-4, d);
}
