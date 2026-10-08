// Which access shape fails at a constant offset into dynamic shared memory?
#include <cuda_runtime.h>
#include <cstdio>

// V0: lane0-of-each-warp writes p[warp], thread 0 reads p[0..3]        (BlockReduce's pattern)
// V1: all lanes write p[warp] (uniform address, uniform value), thread 0 reads
// V2: lane0 writes p[warp], every lane reads p[lane & 3] (vector read), lane 0 reports
// V3: every lane writes p[tid] (vector write), thread 0 reads p[0], p[32], p[64], p[96] (scalar reads)
// V4: like V0 but through a volatile pointer
// V5: like V0 but offset is runtime (control)
template <int OFF, int V>
__global__ void k(float *out, int rt_off) {
  extern __shared__ char smem_[];
  float *p = reinterpret_cast<float *>(smem_ + (V == 5 ? rt_off : OFF));
  volatile float *vp = p;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  if (V == 0 || V == 5) { if (lane == 0) p[warp] = warp + 1.0f; }
  if (V == 1) { p[warp] = warp + 1.0f; }
  if (V == 2) { if (lane == 0) p[warp] = warp + 1.0f; }
  if (V == 3) { p[threadIdx.x] = threadIdx.x / 32 + 1.0f; }
  if (V == 4) { if (lane == 0) vp[warp] = warp + 1.0f; }
  __syncthreads();
  float s = 0;
  if (V == 0 || V == 1 || V == 5) { if (threadIdx.x == 0) for (int i = 0; i < 4; ++i) s += p[i]; }
  if (V == 2) { float v = p[lane & 3]; for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); if (threadIdx.x == 0) s = v / 8.0f; }
  if (V == 3) { if (threadIdx.x == 0) for (int i = 0; i < 4; ++i) s += p[i * 32]; }
  if (V == 4) { if (threadIdx.x == 0) for (int i = 0; i < 4; ++i) s += vp[i]; }
  if (threadIdx.x == 0) out[blockIdx.x] = s;  // want 1+2+3+4 = 10
}

static float *d_out;
template <int OFF>
void run(size_t total) {
  const char *names[] = {"lane0-write/t0-read", "uniform-write/t0-read", "lane0-write/vector-read", "vector-write/t0-scalar-read", "volatile", "runtime-offset"};
  printf("offset %6d total %7zu:", OFF, total);
  auto one = [&](auto kern, int v) {
    cudaFuncSetAttribute((const void *)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)total);
    cudaMemset(d_out, 0, 8);
    kern<<<2, 128, total>>>(d_out, OFF);
    cudaError_t e = cudaDeviceSynchronize();
    float ho[2]; cudaMemcpy(ho, d_out, 8, cudaMemcpyDeviceToHost);
    printf("  %s=%s", names[v], e ? "ERR" : (ho[0] == 10.0f && ho[1] == 10.0f) ? "ok" : ho[0] == 0 ? "ZERO" : "bad");
  };
  one(k<OFF, 0>, 0); one(k<OFF, 1>, 1); one(k<OFF, 2>, 2); one(k<OFF, 3>, 3); one(k<OFF, 4>, 4); one(k<OFF, 5>, 5);
  printf("\n");
}

int main() {
  cudaMalloc(&d_out, 8);
  run<32768>(32768 + 4096); run<32832>(32832 + 4096); run<32832>(131072); run<33024>(33024 + 4096);
  run<34816>(34816 + 4096); run<36864>(36864 + 4096); run<40960>(40960 + 4096); run<50688>(50688 + 4096);
  run<50688>(131072); run<51200>(51200 + 4096); run<65472>(65472 + 4096); run<66560>(66560 + 4096);
  run<100352>(100352 + 4096);
  return 0;
}
