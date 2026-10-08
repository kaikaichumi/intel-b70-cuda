// ze_probe: dump the Level Zero properties of every GPU that matter for mapping
// CUDA onto it (compute layout, SLM, sub-groups, atomics, SPIR-V, caches, engines).
// Build: cc -O2 ze_probe.c -lze_loader -o ze_probe
#include <stdio.h>
#include <string.h>
#include <level_zero/ze_api.h>

#define CHECK(x) do { ze_result_t r_ = (x); if (r_ != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "%s failed: 0x%x\n", #x, r_); return 1; } } while (0)

static void flags(const char *name, unsigned v, const char *const *names, int n) {
    printf("  %-34s", name);
    for (int i = 0; i < n; i++)
        if (v & (1u << i)) printf(" %s", names[i]);
    printf("\n");
}

int main(void) {
    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t ndrv = 0;
    CHECK(zeDriverGet(&ndrv, NULL));
    ze_driver_handle_t drv[8];
    if (ndrv > 8) ndrv = 8;
    CHECK(zeDriverGet(&ndrv, drv));
    for (uint32_t d = 0; d < ndrv; d++) {
        ze_driver_properties_t dp = {ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
        CHECK(zeDriverGetProperties(drv[d], &dp));
        ze_api_version_t api;
        CHECK(zeDriverGetApiVersion(drv[d], &api));
        printf("driver %u: version 0x%x, API %u.%u\n", d, dp.driverVersion,
               ZE_MAJOR_VERSION(api), ZE_MINOR_VERSION(api));

        uint32_t ndev = 0;
        CHECK(zeDeviceGet(drv[d], &ndev, NULL));
        ze_device_handle_t dev[8];
        if (ndev > 8) ndev = 8;
        CHECK(zeDeviceGet(drv[d], &ndev, dev));
        for (uint32_t i = 0; i < ndev; i++) {
            ze_device_ip_version_ext_t ipv = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT};
            ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &ipv};
            CHECK(zeDeviceGetProperties(dev[i], &p));
            printf("\ndevice %u: %s  (vendor 0x%x, device 0x%x, IP version 0x%x)\n",
                   i, p.name, p.vendorId, p.deviceId, ipv.ipVersion);
            printf("  core clock                         %u MHz\n", p.coreClockRate);
            printf("  slices x subslices x EUs x threads %u x %u x %u x %u  (= %u EUs)\n",
                   p.numSlices, p.numSubslicesPerSlice, p.numEUsPerSubslice,
                   p.numThreadsPerEU,
                   p.numSlices * p.numSubslicesPerSlice * p.numEUsPerSubslice);
            printf("  physical EU SIMD width             %u\n", p.physicalEUSimdWidth);
            printf("  max mem alloc                      %.1f GiB\n", p.maxMemAllocSize / 1073741824.0);
            printf("  timer resolution                   %lu ns\n", (unsigned long)p.timerResolution);

            ze_device_compute_properties_t c = {ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
            CHECK(zeDeviceGetComputeProperties(dev[i], &c));
            printf("  max work-group size                %u  (x %u, y %u, z %u)\n",
                   c.maxTotalGroupSize, c.maxGroupSizeX, c.maxGroupSizeY, c.maxGroupSizeZ);
            printf("  max group count                    %u x %u x %u\n",
                   c.maxGroupCountX, c.maxGroupCountY, c.maxGroupCountZ);
            printf("  max shared local memory            %u KiB\n", c.maxSharedLocalMemory / 1024);
            printf("  sub-group sizes                   ");
            for (uint32_t s = 0; s < c.numSubGroupSizes; s++) printf(" %u", c.subGroupSizes[s]);
            printf("\n");

            ze_float_atomic_ext_properties_t fa = {ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES};
            ze_device_module_properties_t mp = {ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES, &fa};
            CHECK(zeDeviceGetModuleProperties(dev[i], &mp));
            printf("  SPIR-V version                     %u.%u\n",
                   ZE_MAJOR_VERSION(mp.spirvVersionSupported), ZE_MINOR_VERSION(mp.spirvVersionSupported));
            static const char *const mflags[] = {"fp16", "fp64", "int64-atomics", "dp4a", "dpas"};
            flags("module flags", mp.flags, mflags, 5);
            static const char *const fpf[] = {"denorm", "inf-nan", "round-nearest", "round-zero",
                                              "round-inf", "fma", "round-div-sqrt", "soft-float"};
            flags("fp32 caps", mp.fp32flags, fpf, 8);
            flags("fp16 caps", mp.fp16flags, fpf, 8);
            flags("fp64 caps", mp.fp64flags, fpf, 8);
            printf("  max kernel argument size           %u bytes\n", mp.maxArgumentsSize);
            printf("  printf buffer                      %u KiB\n", mp.printfBufferSize / 1024);
            static const char *const af[] = {"global-load-store", "global-add", "global-min-max",
                                             "", "", "", "", "", "", "", "", "", "", "", "", "",
                                             "local-load-store", "local-add", "local-min-max"};
            flags("fp16 atomics", fa.fp16Flags, af, 19);
            flags("fp32 atomics", fa.fp32Flags, af, 19);
            flags("fp64 atomics", fa.fp64Flags, af, 19);

            uint32_t nmem = 0;
            CHECK(zeDeviceGetMemoryProperties(dev[i], &nmem, NULL));
            ze_device_memory_properties_t mem[4];
            if (nmem > 4) nmem = 4;
            for (uint32_t k = 0; k < nmem; k++) {
                memset(&mem[k], 0, sizeof mem[k]);
                mem[k].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
            }
            CHECK(zeDeviceGetMemoryProperties(dev[i], &nmem, mem));
            for (uint32_t k = 0; k < nmem; k++)
                printf("  memory %-27s %.1f GiB, %u MHz, bus %u bit\n", mem[k].name,
                       mem[k].totalSize / 1073741824.0, mem[k].maxClockRate, mem[k].maxBusWidth);

            uint32_t ncache = 0;
            CHECK(zeDeviceGetCacheProperties(dev[i], &ncache, NULL));
            ze_device_cache_properties_t cache[4];
            if (ncache > 4) ncache = 4;
            for (uint32_t k = 0; k < ncache; k++) {
                memset(&cache[k], 0, sizeof cache[k]);
                cache[k].stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
            }
            CHECK(zeDeviceGetCacheProperties(dev[i], &ncache, cache));
            for (uint32_t k = 0; k < ncache; k++)
                printf("  cache %u                            %.1f MiB%s\n", k,
                       cache[k].cacheSize / 1048576.0,
                       (cache[k].flags & ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL) ? " (user control)" : "");

            uint32_t nq = 0;
            CHECK(zeDeviceGetCommandQueueGroupProperties(dev[i], &nq, NULL));
            ze_command_queue_group_properties_t q[8];
            if (nq > 8) nq = 8;
            for (uint32_t k = 0; k < nq; k++) {
                memset(&q[k], 0, sizeof q[k]);
                q[k].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
            }
            CHECK(zeDeviceGetCommandQueueGroupProperties(dev[i], &nq, q));
            for (uint32_t k = 0; k < nq; k++)
                printf("  queue group %u                      %s%s%s x %u engines\n", k,
                       (q[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) ? "compute " : "",
                       (q[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ? "copy " : "",
                       (q[k].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS) ? "cooperative " : "",
                       q[k].numQueues);

            ze_device_memory_access_properties_t ma = {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
            CHECK(zeDeviceGetMemoryAccessProperties(dev[i], &ma));
            static const char *const mac[] = {"rw", "atomic", "concurrent", "concurrent-atomic"};
            flags("host alloc access", ma.hostAllocCapabilities, mac, 4);
            flags("device alloc access", ma.deviceAllocCapabilities, mac, 4);
            flags("shared single-device access", ma.sharedSingleDeviceAllocCapabilities, mac, 4);
            flags("shared system access", ma.sharedSystemAllocCapabilities, mac, 4);
        }
    }
    return 0;
}
