// Where do chipStar's ~480 us per cudaEventRecord go on the B70? Time each Level
// Zero operation its recordEvent() performs, and the single-engine alternative
// (one barrier that signals a kernel-timestamp event on the compute list).
// Build: cc -O2 ze_record_probe.c -lze_loader -o ze_record_probe
#include <stdio.h>
#include <time.h>
#include <level_zero/ze_api.h>

#define CK(x) do { ze_result_t r_ = (x); if (r_) { printf("%s -> 0x%x\n", #x, r_); return 1; } } while (0)
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    CK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t n = 1; ze_driver_handle_t drv; CK(zeDriverGet(&n, &drv));
    n = 1; ze_device_handle_t dev; CK(zeDeviceGet(drv, &n, &dev));
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx;
    CK(zeContextCreate(drv, &cd, &ctx));

    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.ordinal = 0; ze_command_list_handle_t compute; CK(zeCommandListCreateImmediate(ctx, dev, &qd, &compute));
    qd.ordinal = 1; ze_command_list_handle_t copy;    CK(zeCommandListCreateImmediate(ctx, dev, &qd, &copy));

    ze_event_pool_desc_t pd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count = 16;
    ze_event_pool_handle_t pool; CK(zeEventPoolCreate(ctx, &pd, 0, NULL, &pool));
    pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    ze_event_pool_handle_t tspool; CK(zeEventPoolCreate(ctx, &pd, 0, NULL, &tspool));
    ze_event_handle_t ev[4], ts;
    for (int i = 0; i < 4; i++) {
        ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC}; ed.index = i;
        ed.signal = ZE_EVENT_SCOPE_FLAG_HOST; ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        CK(zeEventCreate(pool, &ed, &ev[i]));
    }
    ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC}; ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    CK(zeEventCreate(tspool, &ed, &ts));

    ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void *shared; CK(zeMemAllocShared(ctx, &dd, &hd, 64, 64, dev, &shared));
    uint64_t host_ts, dev_ts, dst;
    const int N = 200;
    double t;

    t = now_us();
    for (int i = 0; i < N; i++) CK(zeDeviceGetGlobalTimestamps(dev, &host_ts, &dev_ts));
    printf("zeDeviceGetGlobalTimestamps                 %8.2f us\n", (now_us() - t) / N);

    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendWriteGlobalTimestamp(copy, (uint64_t *)shared, ev[0], 0, NULL));
        CK(zeEventHostSynchronize(ev[0], UINT64_MAX)); CK(zeEventHostReset(ev[0]));
    }
    printf("timestamp write on copy engine + host wait  %8.2f us\n", (now_us() - t) / N);

    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendWriteGlobalTimestamp(copy, (uint64_t *)shared, ev[0], 0, NULL));
        CK(zeCommandListAppendMemoryCopy(copy, &dst, shared, 8, ev[1], 1, &ev[0]));
        CK(zeCommandListAppendBarrier(compute, ev[2], 1, &ev[1]));
        CK(zeCommandListAppendBarrier(compute, ev[3], 1, &ev[1]));
        CK(zeEventHostSynchronize(ev[3], UINT64_MAX));
        for (int k = 0; k < 4; k++) CK(zeEventHostReset(ev[k]));
    }
    printf("chipStar-style record (copy->compute) + wait %7.2f us\n", (now_us() - t) / N);

    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendWriteGlobalTimestamp(compute, (uint64_t *)shared, ev[0], 0, NULL));
        CK(zeCommandListAppendMemoryCopy(compute, &dst, shared, 8, ev[1], 1, &ev[0]));
        CK(zeCommandListAppendBarrier(compute, ev[2], 1, &ev[1]));
        CK(zeCommandListAppendBarrier(compute, ev[3], 1, &ev[1]));
        CK(zeEventHostSynchronize(ev[3], UINT64_MAX));
        for (int k = 0; k < 4; k++) CK(zeEventHostReset(ev[k]));
    }
    printf("same sequence, all on the compute engine     %7.2f us\n", (now_us() - t) / N);

    // hypothesis: the 8-byte copy into ordinary (non-USM) host memory is the slow part
    uint64_t *usm_host; CK(zeMemAllocHost(ctx, &hd, 64, 64, (void **)&usm_host));
    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendWriteGlobalTimestamp(copy, (uint64_t *)shared, ev[0], 0, NULL));
        CK(zeCommandListAppendMemoryCopy(copy, usm_host, shared, 8, ev[1], 1, &ev[0]));
        CK(zeCommandListAppendBarrier(compute, ev[2], 1, &ev[1]));
        CK(zeCommandListAppendBarrier(compute, ev[3], 1, &ev[1]));
        CK(zeEventHostSynchronize(ev[3], UINT64_MAX));
        for (int k = 0; k < 4; k++) CK(zeEventHostReset(ev[k]));
    }
    printf("chipStar-style, copy into USM host memory    %7.2f us\n", (now_us() - t) / N);

    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendWriteGlobalTimestamp(compute, usm_host, ev[0], 0, NULL));
        CK(zeEventHostSynchronize(ev[0], UINT64_MAX));
        CK(zeEventHostReset(ev[0]));
    }
    printf("timestamp straight into USM host, compute    %7.2f us\n", (now_us() - t) / N);

    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendBarrier(compute, ts, 0, NULL));
        CK(zeEventHostSynchronize(ts, UINT64_MAX));
        ze_kernel_timestamp_result_t r; CK(zeEventQueryKernelTimestamp(ts, &r));
        CK(zeEventHostReset(ts));
    }
    printf("single barrier + kernel-timestamp event      %7.2f us\n", (now_us() - t) / N);

    t = now_us();
    for (int i = 0; i < N; i++) {
        CK(zeCommandListAppendBarrier(compute, ts, 0, NULL));
        CK(zeEventHostReset(ts));
    }
    printf("  (append only, no wait)                     %7.2f us\n", (now_us() - t) / N);
    return 0;
}
