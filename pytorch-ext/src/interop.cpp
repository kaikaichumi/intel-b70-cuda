// libb70torch: glue between PyTorch's XPU streams and the chipStar (CUDA) runtime
// on the same Intel GPU. Compiled with plain g++ against torch + SYCL headers;
// the few chipStar entry points it needs are declared by hand below so no HIP
// header has to be parsed by g++.
//
//  * b70torch_init(): hand chipStar the Level Zero driver/device/context behind
//    torch's current XPU stream, so cudaMalloc'd and torch-allocated memory
//    live in one context and every pointer is valid on both sides.
//  * b70torch_stream_acquire(dev): torch queue -> CUDA stream ordering, returns
//    the cudaStream_t an extension should launch on. Marks the stream pending.
//  * Pending streams are flushed (CUDA stream -> torch queue ordering) before
//    the next ATen operator runs (RecordFunction callback), or explicitly via
//    b70torch_flush(). So "torch op -> extension kernel -> torch op" is ordered
//    without the extension doing anything.
//
// Sync modes (B70TORCH_SYNC): "event" (default) uses GPU-side events in both
// directions; "host" waits on the host instead -- slower, kept for debugging.
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <ATen/record_function.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
// chipStar / HIP runtime (libCHIP.so)
typedef struct ihipStream_t *hipStream_t;
typedef struct ihipEvent_t *hipEvent_t;
int hipInitFromNativeHandles(const uintptr_t *handles, int n);
void *hipGetHipEventFromNativeEvent(void *native);
void *hipGetNativeEventFromHipEvent(void *event);
int hipStreamCreateWithFlags(hipStream_t *s, unsigned flags);
int hipStreamWaitEvent(hipStream_t s, hipEvent_t e, unsigned flags);
int hipStreamSynchronize(hipStream_t s);
int hipEventCreateWithFlags(hipEvent_t *e, unsigned flags);
int hipEventRecord(hipEvent_t e, hipStream_t s);
int hipEventSynchronize(hipEvent_t e);
int hipEventQuery(hipEvent_t e);
int hipDeviceSynchronize(void);
const char *hipGetErrorString(int);
// ours (defined below)
void b70torch_flush();
}

namespace {
constexpr unsigned hipStreamNonBlocking = 1, hipEventDisableTiming = 2;
constexpr int hipSuccess = 0;

bool g_event_mode = true;
std::once_flag g_init_once;
int g_init_rc = -1;

// B70TORCH_PROFILE=1: per-phase time of the two sync directions, printed at exit
struct Prof {
    const char *name;
    double us = 0;
    long n = 0;
};
Prof g_prof[] = {{"in: barrier"}, {"in: convert"}, {"in: hipStreamWaitEvent"}, {"in: ring"},
                 {"out: hipEventSynchronize"}, {"out: hipEventRecord"}, {"out: make_event"},
                 {"out: queue barrier"}};
bool g_profile = false;
struct Timer {
    Prof &p;
    std::chrono::steady_clock::time_point t0;
    explicit Timer(int i) : p(g_prof[i]), t0(std::chrono::steady_clock::now()) {}
    ~Timer() {
        if (g_profile) {
            p.us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            p.n++;
        }
    }
};
void print_profile() {
    for (auto &p : g_prof)
        if (p.n) std::fprintf(stderr, "[b70torch] %-26s %8ld calls  %8.1f us avg\n", p.name, p.n, p.us / p.n);
}

void check(int rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "[b70torch] %s failed: %s\n", what, hipGetErrorString(rc));
        std::abort();
    }
}

struct StreamState {
    hipStream_t hip = nullptr;
    sycl::queue *queue = nullptr;
    c10::DeviceIndex device = 0;
    bool pending = false;
    // sync-in: keep the SYCL events (and chipStar's wrappers) alive until the
    // CUDA stream has consumed them
    std::deque<std::pair<sycl::event, hipEvent_t>> in_flight;
    // sync-out: a small pool of CUDA events, reused once they completed
    std::vector<hipEvent_t> out_events;
    size_t out_next = 0;
};

std::mutex g_mtx;
std::unordered_map<sycl::queue *, StreamState> g_streams;  // keyed by torch's queue
std::vector<StreamState *> g_pending;

int do_init() {
    const char *m = std::getenv("B70TORCH_SYNC");
    if (m && std::strcmp(m, "host") == 0) g_event_mode = false;
    if (std::getenv("B70TORCH_PROFILE")) {
        g_profile = true;
        std::atexit(print_profile);
    }
    sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
    auto drv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device().get_platform());
    auto dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    uintptr_t handles[4] = {(uintptr_t)drv, (uintptr_t)dev, (uintptr_t)ctx, 0};
    int rc = hipInitFromNativeHandles(handles, 4);
    if (rc != hipSuccess) {
        std::fprintf(stderr, "[b70torch] hipInitFromNativeHandles: %s\n", hipGetErrorString(rc));
        return rc;
    }
    // flush pending CUDA work before any ATen operator consumes it
    at::addGlobalCallback(
        at::RecordFunctionCallback(
            [](const at::RecordFunction &) -> std::unique_ptr<at::ObserverContext> {
                b70torch_flush();
                return nullptr;
            },
            nullptr)
            .scopes({at::RecordScope::FUNCTION}));
    return 0;
}

StreamState &state_for(c10::DeviceIndex dev) {  // g_mtx held
    sycl::queue *q = &c10::xpu::getCurrentXPUStream(dev).queue();
    auto it = g_streams.find(q);
    if (it != g_streams.end()) return it->second;
    StreamState &s = g_streams[q];
    s.queue = q;
    s.device = dev;
    check(hipStreamCreateWithFlags(&s.hip, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    return s;
}

// torch queue -> CUDA stream
void sync_in(StreamState &s) {
    if (!g_event_mode) {
        s.queue->wait();
        return;
    }
    sycl::event e;
    { Timer t(0); e = s.queue->ext_oneapi_submit_barrier(); }
    hipEvent_t he;
    {
        Timer t(1);
        auto ze = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(e);
        he = static_cast<hipEvent_t>(hipGetHipEventFromNativeEvent(ze));
    }
    if (!he) {  // nothing was queued: no ordering needed
        return;
    }
    { Timer t(2); check(hipStreamWaitEvent(s.hip, he, 0), "hipStreamWaitEvent"); }
    Timer t(3);
    s.in_flight.emplace_back(std::move(e), he);
    while (s.in_flight.size() > 64) {  // old entries have long completed
        s.in_flight.front().first.wait();
        s.in_flight.pop_front();
    }
}

// CUDA stream -> torch queue
void sync_out(StreamState &s) {
    if (!g_event_mode) {
        check(hipStreamSynchronize(s.hip), "hipStreamSynchronize");
        return;
    }
    if (s.out_events.empty()) {
        s.out_events.resize(16, nullptr);
        for (auto &e : s.out_events)
            check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "hipEventCreateWithFlags");
    }
    hipEvent_t he = s.out_events[s.out_next];
    s.out_next = (s.out_next + 1) % s.out_events.size();
    // the event is re-recorded only after everything that waited on its previous
    // recording has been enqueued behind it; make sure that recording completed
    { Timer t(4); check(hipEventSynchronize(he), "hipEventSynchronize"); }
    { Timer t(5); check(hipEventRecord(he, s.hip), "hipEventRecord"); }
    void *ze = hipGetNativeEventFromHipEvent(he);
    if (!ze) return;
    sycl::event se;
    {
        Timer t(6);
        se = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
            {static_cast<ze_event_handle_t>(ze), sycl::ext::oneapi::level_zero::ownership::keep},
            s.queue->get_context());
    }
    Timer t(7);
    s.queue->ext_oneapi_submit_barrier({se});
}
}  // namespace

extern "C" {

int b70torch_init() {
    std::call_once(g_init_once, [] { g_init_rc = do_init(); });
    return g_init_rc;
}

void b70torch_flush() {
    std::lock_guard<std::mutex> lock(g_mtx);
    for (StreamState *s : g_pending) {
        sync_out(*s);
        s->pending = false;
    }
    g_pending.clear();
}

void *b70torch_stream_acquire(int device) {
    if (b70torch_init() != 0) std::abort();
    std::lock_guard<std::mutex> lock(g_mtx);
    c10::DeviceIndex dev = device < 0 ? c10::xpu::current_device() : (c10::DeviceIndex)device;
    StreamState &s = state_for(dev);
    sync_in(s);
    if (!s.pending) {
        s.pending = true;
        g_pending.push_back(&s);
    }
    return s.hip;
}

int b70torch_current_device() { return (int)c10::xpu::current_device(); }
int b70torch_device_count() { return (int)c10::xpu::device_count(); }

// torch.cuda.synchronize() equivalent that covers both runtimes
void b70torch_device_synchronize() {
    b70torch_flush();
    check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    c10::xpu::syncStreamsOnDevice(c10::xpu::current_device());
}
}
