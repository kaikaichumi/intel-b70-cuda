// M1: three classic kernels in one binary -- block scan (shared memory,
// Hillis-Steele), transpose (padded shared tile), histogram (shared + global
// integer atomics).
#include "common.h"
#include <vector>

__global__ void block_scan(const int *in, int *out, int n) {
    extern __shared__ int buf[];
    int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
    buf[t] = g < n ? in[g] : 0;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int v = t >= off ? buf[t - off] : 0;
        __syncthreads();
        buf[t] += v;
        __syncthreads();
    }
    if (g < n) out[g] = buf[t];
}

#define TD 32
__global__ void transpose(const float *in, float *out, int w, int h) {
    __shared__ float tile[TD][TD + 1];
    int x = blockIdx.x * TD + threadIdx.x, y = blockIdx.y * TD + threadIdx.y;
    for (int j = 0; j < TD; j += blockDim.y)
        if (x < w && y + j < h) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * w + x];
    __syncthreads();
    x = blockIdx.y * TD + threadIdx.x;
    y = blockIdx.x * TD + threadIdx.y;
    for (int j = 0; j < TD; j += blockDim.y)
        if (x < h && y + j < w) out[(y + j) * h + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void histogram(const unsigned char *in, unsigned *bins, int n) {
    __shared__ unsigned local[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local[in[i]], 1u);
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) atomicAdd(&bins[i], local[i]);
}

static int test_scan() {
    const int n = 1 << 20, bs = 512;
    std::vector<int> h(n), o(n);
    for (int i = 0; i < n; i++) h[i] = (i * 7) % 5 - 2;
    int *di, *dout;
    CK(cudaMalloc(&di, n * sizeof(int)));
    CK(cudaMalloc(&dout, n * sizeof(int)));
    CK(cudaMemcpy(di, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    block_scan<<<n / bs, bs, bs * sizeof(int)>>>(di, dout, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(o.data(), dout, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int b = 0; b < n / bs && ok; b++) {
        int s = 0;
        for (int t = 0; t < bs && ok; t++) { s += h[b * bs + t]; ok = o[b * bs + t] == s; }
    }
    CK(cudaFree(di)); CK(cudaFree(dout));
    return finish("scan_dynamic_shared", ok);
}

static int test_transpose() {
    const int w = 2048, h = 1536;
    std::vector<float> a(w * h), t(w * h);
    for (int i = 0; i < w * h; i++) a[i] = float(i);
    float *din, *dout;
    CK(cudaMalloc(&din, w * h * sizeof(float)));
    CK(cudaMalloc(&dout, w * h * sizeof(float)));
    CK(cudaMemcpy(din, a.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    dim3 grid(w / TD, h / TD), block(TD, 8);
    float ms = gpu_ms([&] { transpose<<<grid, block>>>(din, dout, w, h); }, 20);
    CK(cudaGetLastError());
    CK(cudaMemcpy(t.data(), dout, w * h * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int y = 0; y < h && ok; y++)
        for (int x = 0; x < w && ok; x++) ok = t[x * h + y] == a[y * w + x];
    char d[64];
    std::snprintf(d, sizeof d, "%.3f ms, %.0f GB/s", ms, 2.0 * w * h * sizeof(float) / ms / 1e6);
    CK(cudaFree(din)); CK(cudaFree(dout));
    return finish("transpose", ok, d);
}

static int test_histogram() {
    const int n = 1 << 24;
    std::vector<unsigned char> h(n);
    std::vector<unsigned> ref(256, 0), got(256);
    for (int i = 0; i < n; i++) { h[i] = (unsigned char)((i * 2654435761u) >> 24); ref[h[i]]++; }
    unsigned char *din;
    unsigned *dbins;
    CK(cudaMalloc(&din, n));
    CK(cudaMalloc(&dbins, 256 * sizeof(unsigned)));
    CK(cudaMemcpy(din, h.data(), n, cudaMemcpyHostToDevice));
    CK(cudaMemset(dbins, 0, 256 * sizeof(unsigned)));
    histogram<<<256, 256>>>(din, dbins, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(got.data(), dbins, 256 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool ok = got == ref;
    CK(cudaFree(din)); CK(cudaFree(dbins));
    return finish("histogram_atomics", ok);
}

int main() { return test_scan() | test_transpose() | test_histogram(); }
