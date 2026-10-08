// M1: the basics -- cudaMalloc, cudaMemcpy, <<<>>> launch, grid-stride indexing.
#include "common.h"
#include <vector>

__global__ void add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 24;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = 1.0f - i; }
    float *da, *db, *dc;
    CK(cudaMalloc(&da, n * sizeof(float)));
    CK(cudaMalloc(&db, n * sizeof(float)));
    CK(cudaMalloc(&dc, n * sizeof(float)));
    CK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int threads = 256, blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(da, db, dc, n);
    CK(cudaGetLastError());
    float ms = gpu_ms([&] { add<<<blocks, threads>>>(da, db, dc, n); }, 20);
    CK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n && ok; i++) ok = c[i] == a[i] + b[i];
    char d[96];
    std::snprintf(d, sizeof d, "%.3f ms, %.0f GB/s", ms, 3.0 * n * sizeof(float) / ms / 1e6);
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc));
    return finish("vector_add", ok, d);
}
