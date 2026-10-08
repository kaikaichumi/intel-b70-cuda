// M3 micro-benchmarks, SYCL baseline (icpx -fsycl -O3). Same tests, sizes and
// work-group size as bench/micro.cu, timed with SYCL event profiling.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

using namespace sycl;

template <class F> static double best_ms(queue &q, F f, int reps = 10) {
    f().wait();  // warm-up / JIT
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f().wait();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    queue q{gpu_selector_v, property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<info::device::name>().c_str());
    const size_t n = 64ull << 20, T = 256, R = (n + T - 1) / T * T;
    float *a = malloc_device<float>(n, q), *b = malloc_device<float>(n, q), *c = malloc_device<float>(n, q);
    auto nd = nd_range<1>(R, T);
    // same hash noise as micro.cu (defeats Xe2 memory compression)
    auto fill = [&](float *x, unsigned seed) {
        q.parallel_for(nd, [=](nd_item<1> it) {
            size_t i = it.get_global_id(0);
            if (i < n) {
                unsigned h = (unsigned)i * 2654435761u ^ seed;
                h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
                x[i] = (float)(h & 0xffffff) * 5.96e-8f + 0.5f;
            }
        });
    };
    fill(a, 1u); fill(b, 2u); fill(c, 3u); q.wait();
    double gb2 = 2.0 * n * 4 / 1e6, gb3 = 3.0 * n * 4 / 1e6;
    std::printf("stream_copy   %8.1f GB/s\n", gb2 / best_ms(q, [&] { return q.parallel_for(nd, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0); if (i < n) c[i] = a[i]; }); }));
    std::printf("stream_scale  %8.1f GB/s\n", gb2 / best_ms(q, [&] { return q.parallel_for(nd, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0); if (i < n) b[i] = 3.0f * c[i]; }); }));
    std::printf("stream_add    %8.1f GB/s\n", gb3 / best_ms(q, [&] { return q.parallel_for(nd, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0); if (i < n) c[i] = a[i] + b[i]; }); }));
    std::printf("stream_triad  %8.1f GB/s\n", gb3 / best_ms(q, [&] { return q.parallel_for(nd, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0); if (i < n) a[i] = b[i] + 3.0f * c[i]; }); }));

    const int L = 2000;
    double ms = best_ms(q, [&] {
        event e;
        for (int i = 0; i < L; i++) e = q.parallel_for(nd_range<1>(32, 32), [=](nd_item<1>) {});
        return e; }, 5);
    std::printf("launch        %8.2f us/kernel\n", ms * 1000.0 / L);

    float *slots = malloc_device<float>(1024, q);
    q.memset(slots, 0, 1024 * 4).wait();
    const size_t AB = 1024, AT = 256; const int IT = 64;
    ms = best_ms(q, [&] { return q.parallel_for(nd_range<1>(AB * AT, AT), [=](nd_item<1> it) {
        int i = (int)it.get_global_id(0);
        for (int k = 0; k < IT; k++)
            atomic_ref<float, memory_order::relaxed, memory_scope::device>(slots[(i + k) & 1023]) += 1.0f;
    }); });
    std::printf("atomic_f32    %8.2f G ops/s\n", (double)AB * AT * IT / ms / 1e6);
    return 0;
}
