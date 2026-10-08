// M1: runtime API surface most programs touch -- device query, warpSize inside
// a kernel, streams + async copies + events, pinned host memory, managed memory,
// float atomics, constant memory, device-side printf.
#include "common.h"
#include <vector>

__constant__ float coeff[4];

__global__ void probe(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = warpSize;
        printf("device printf ok, warpSize=%d\n", warpSize);
    }
}

__global__ void axpb(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = coeff[0] * x[i] + coeff[1];
}

__global__ void float_atomics(float *sum, double *dsum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, 1.0f);
        atomicAdd(dsum, 0.5);
    }
}

int main() {
    int rc = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s, %d multiprocessors, %.1f GiB, warpSize %d, shared/block %zu KiB\n",
                p.name, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, p.warpSize,
                p.sharedMemPerBlock / 1024);

    int *dw, w = 0;
    CK(cudaMalloc(&dw, sizeof(int)));
    probe<<<1, 64>>>(dw);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&w, dw, sizeof(int), cudaMemcpyDeviceToHost));
    rc |= finish("warpSize_is_32", w == 32 && p.warpSize == 32);

    // two streams, pinned memory, async copies, events
    const int n = 1 << 22;
    float *h0, *h1, *d0, *d1;
    CK(cudaMallocHost(&h0, n * sizeof(float)));
    CK(cudaMallocHost(&h1, n * sizeof(float)));
    CK(cudaMalloc(&d0, n * sizeof(float)));
    CK(cudaMalloc(&d1, n * sizeof(float)));
    for (int i = 0; i < n; i++) { h0[i] = float(i % 100); h1[i] = float(i % 50); }
    float c[4] = {2.0f, 1.0f, 0, 0};
    CK(cudaMemcpyToSymbol(coeff, c, sizeof c));
    cudaStream_t s0, s1;
    CK(cudaStreamCreate(&s0));
    CK(cudaStreamCreate(&s1));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s0));
    CK(cudaMemcpyAsync(d0, h0, n * sizeof(float), cudaMemcpyHostToDevice, s0));
    CK(cudaMemcpyAsync(d1, h1, n * sizeof(float), cudaMemcpyHostToDevice, s1));
    axpb<<<(n + 255) / 256, 256, 0, s0>>>(d0, n);
    axpb<<<(n + 255) / 256, 256, 0, s1>>>(d1, n);
    CK(cudaMemcpyAsync(h0, d0, n * sizeof(float), cudaMemcpyDeviceToHost, s0));
    CK(cudaMemcpyAsync(h1, d1, n * sizeof(float), cudaMemcpyDeviceToHost, s1));
    CK(cudaEventRecord(e1, s0));
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    bool ok = true;
    for (int i = 0; i < n && ok; i++)
        ok = h0[i] == 2.0f * (i % 100) + 1.0f && h1[i] == 2.0f * (i % 50) + 1.0f;
    char d[64];
    std::snprintf(d, sizeof d, "stream 0 took %.3f ms", ms);
    rc |= finish("streams_events_pinned_constant", ok, d);

    // managed memory touched by both sides
    float *m;
    CK(cudaMallocManaged(&m, 1024 * sizeof(float)));
    for (int i = 0; i < 1024; i++) m[i] = float(i);
    axpb<<<4, 256>>>(m, 1024);
    CK(cudaDeviceSynchronize());
    ok = true;
    for (int i = 0; i < 1024 && ok; i++) ok = m[i] == 2.0f * i + 1.0f;
    rc |= finish("managed_memory", ok);

    // float / double atomicAdd (native on Xe2)
    float *fs;
    double *ds;
    CK(cudaMalloc(&fs, sizeof(float)));
    CK(cudaMalloc(&ds, sizeof(double)));
    CK(cudaMemset(fs, 0, sizeof(float)));
    CK(cudaMemset(ds, 0, sizeof(double)));
    const int na = 1 << 20;
    float fms = gpu_ms([&] { float_atomics<<<na / 256, 256>>>(fs, ds, na); });
    float fv;
    double dv;
    CK(cudaMemcpy(&fv, fs, sizeof fv, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&dv, ds, sizeof dv, cudaMemcpyDeviceToHost));
    std::snprintf(d, sizeof d, "%.3f ms for 2x%d contended atomics", fms, na);
    rc |= finish("float_double_atomicAdd", fv == float(na) && dv == 0.5 * na, d);
    return rc;
}
