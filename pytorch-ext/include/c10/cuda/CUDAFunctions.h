// b70torch shim for <c10/cuda/CUDAFunctions.h>
#pragma once
#include <cuda_runtime_api.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace c10 {
namespace cuda {

inline DeviceIndex device_count() noexcept { return (DeviceIndex)b70torch_device_count(); }
inline DeviceIndex device_count_ensure_non_zero() {
  auto n = device_count();
  TORCH_CHECK(n > 0, "No Intel GPU (b70) available");
  return n;
}
inline DeviceIndex current_device() { return (DeviceIndex)b70torch_current_device(); }
inline void set_device(DeviceIndex) {}
inline void device_synchronize() { b70torch_device_synchronize(); }
inline void warn_or_error_on_sync() {}
inline int32_t driver_version() { int v = 0; cudaDriverGetVersion(&v); return v; }

inline cudaError_t GetDeviceCount(int *count) { return cudaGetDeviceCount(count); }
inline cudaError_t GetDevice(DeviceIndex *d) { *d = current_device(); return cudaSuccess; }
inline cudaError_t SetDevice(DeviceIndex) { return cudaSuccess; }
inline cudaError_t MaybeSetDevice(DeviceIndex) { return cudaSuccess; }
inline DeviceIndex ExchangeDevice(DeviceIndex) { return current_device(); }
inline DeviceIndex MaybeExchangeDevice(DeviceIndex) { return current_device(); }
inline void SetTargetDevice() {}

inline bool hasPrimaryContext(DeviceIndex) { return true; }
inline std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() { return current_device(); }

}  // namespace cuda
}  // namespace c10
