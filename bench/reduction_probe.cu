// Why is tests/cuda/reduction.cu slow? Time the same kernel built up one feature
// at a time (wall clock, best of 10, 16 Mi floats = 64 MiB).
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

__inline__ __device__ float warp_sum(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off);
    return v;
}

template <int STAGE>
__global__ void k(const float *in, float *out, int n) {
    __shared__ float part[32];
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += in[i];
    if (STAGE >= 1) v = warp_sum(v);
    if (STAGE >= 2) {
        int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
        if (lane == 0) part[wid] = v;
        __syncthreads();
        if (wid == 0) v = warp_sum(lane < blockDim.x / warpSize ? part[lane] : 0.0f);
        if (STAGE >= 3) { if (threadIdx.x == 0) atomicAdd(out, v); return; }
    }
    out[1 + blockIdx.x * blockDim.x + threadIdx.x] = v;  // keep the work alive
}

// the same loop with a fixed stride (no gridDim/blockDim products in the loop)
__global__ void k_fixed(const float *in, float *out, int n, int stride) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) v += in[i];
    out[1 + blockIdx.x * blockDim.x + threadIdx.x] = v;
}

template <class F> static double best_ms(F f) {
    f(); cudaDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f(); cudaDeviceSynchronize();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    const int n = 1 << 24;
    float *in, *out;
    cudaMalloc(&in, n * 4);
    cudaMalloc(&out, (n + 1) * 4);
    cudaMemset(in, 0x3c, n * 4);
    const char *names[] = {"loop only", "+ warp shuffle", "+ shared + syncthreads", "+ atomicAdd (full)"};
    for (int blocks : {256, 4096}) {
        int T = 256;
        auto gbs = [&](double ms) { return n * 4.0 / ms / 1e6; };
        double ms;
        ms = best_ms([&] { k<0><<<blocks, T>>>(in, out, n); }); std::printf("%5d blocks  %-24s %7.3f ms %6.0f GB/s\n", blocks, names[0], ms, gbs(ms));
        ms = best_ms([&] { k<1><<<blocks, T>>>(in, out, n); }); std::printf("%5d blocks  %-24s %7.3f ms %6.0f GB/s\n", blocks, names[1], ms, gbs(ms));
        ms = best_ms([&] { k<2><<<blocks, T>>>(in, out, n); }); std::printf("%5d blocks  %-24s %7.3f ms %6.0f GB/s\n", blocks, names[2], ms, gbs(ms));
        ms = best_ms([&] { k<3><<<blocks, T>>>(in, out, n); }); std::printf("%5d blocks  %-24s %7.3f ms %6.0f GB/s\n", blocks, names[3], ms, gbs(ms));
        ms = best_ms([&] { k_fixed<<<blocks, T>>>(in, out, n, blocks * T); }); std::printf("%5d blocks  %-24s %7.3f ms %6.0f GB/s\n", blocks, "loop, fixed stride", ms, gbs(ms));
    }
    return 0;
}
