// M1: tiled SGEMM with __shared__ memory and __syncthreads().
#include "common.h"
#include <vector>

#define TILE 16

__global__ void matmul(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < n / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * n + col] = acc;
}

int main() {
    const int n = 1024;
    std::vector<float> A(n * n), B(n * n), C(n * n);
    for (int i = 0; i < n * n; i++) { A[i] = (i % 13) * 0.1f - 0.5f; B[i] = (i % 7) * 0.2f - 0.6f; }
    float *dA, *dB, *dC;
    size_t bytes = size_t(n) * n * sizeof(float);
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
    CK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    dim3 block(TILE, TILE), grid(n / TILE, n / TILE);
    matmul<<<grid, block>>>(dA, dB, dC, n);
    CK(cudaGetLastError());
    float ms = gpu_ms([&] { matmul<<<grid, block>>>(dA, dB, dC, n); }, 10);
    CK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    // spot-check 512 entries against a double-precision CPU result
    double worst = 0;
    for (int s = 0; s < 512; s++) {
        int r = (s * 977) % n, c = (s * 389) % n;
        double ref = 0;
        for (int k = 0; k < n; k++) ref += double(A[r * n + k]) * B[k * n + c];
        worst = std::fmax(worst, std::fabs(ref - C[r * n + c]));
    }
    char d[96];
    std::snprintf(d, sizeof d, "%.3f ms, %.0f GFLOP/s, max err %.2e", ms, 2.0 * n * n * n / ms / 1e6, worst);
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
    return finish("matmul_shared", worst < 1e-2, d);
}
