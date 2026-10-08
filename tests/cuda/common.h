// Shared helpers for the acceptance tests. Plain CUDA: every test also builds with nvcc.
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CK(call)                                                                   \
    do {                                                                           \
        cudaError_t e_ = (call);                                                   \
        if (e_ != cudaSuccess) {                                                   \
            std::printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #call,        \
                        cudaGetErrorString(e_));                                   \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

static inline int finish(const char *name, bool ok, const char *detail = "") {
    std::printf("%s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
    return ok ? 0 : 1;
}

// Milliseconds spent on the GPU by `fn`, measured with CUDA events.
template <class F>
static inline float gpu_ms(F fn, int reps = 1) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) fn();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return ms / reps;
}
