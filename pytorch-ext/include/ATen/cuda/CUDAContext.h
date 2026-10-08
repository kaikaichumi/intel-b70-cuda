// b70torch shim for <ATen/cuda/CUDAContext.h>, the header almost every CUDA
// extension includes. Streams come from c10/cuda/CUDAStream.h (b70 shim),
// device properties from the chipStar runtime. No cuBLAS/cuSPARSE handles.
#pragma once
#include <cuda_runtime.h>
#include <ATen/Context.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <mutex>
#include <vector>

namespace at {
namespace cuda {

inline bool is_available() { return c10::cuda::device_count() > 0; }
inline DeviceIndex device_count() { return c10::cuda::device_count(); }
inline DeviceIndex current_device() { return c10::cuda::current_device(); }
inline void set_device(DeviceIndex d) { c10::cuda::set_device(d); }
inline void device_synchronize() { c10::cuda::device_synchronize(); }

inline cudaDeviceProp *getDeviceProperties(DeviceIndex device) {
  static std::mutex mtx;
  static std::vector<cudaDeviceProp *> props;
  std::lock_guard<std::mutex> lock(mtx);
  if ((size_t)device >= props.size()) props.resize(device + 1, nullptr);
  if (!props[device]) {
    auto *p = new cudaDeviceProp();
    AT_CUDA_CHECK(cudaGetDeviceProperties(p, device));
    props[device] = p;
  }
  return props[device];
}
inline cudaDeviceProp *getCurrentDeviceProperties() { return getDeviceProperties(current_device()); }
inline int warp_size() { return 32; }
inline bool canDeviceAccessPeer(DeviceIndex, DeviceIndex) { return false; }
inline Allocator *getCUDADeviceAllocator() { return at::getCPUAllocator(); }

}  // namespace cuda
}  // namespace at
