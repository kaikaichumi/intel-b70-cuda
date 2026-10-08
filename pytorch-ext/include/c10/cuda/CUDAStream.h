// b70torch shim for <c10/cuda/CUDAStream.h>: at::cuda::getCurrentCUDAStream()
// and friends, backed by libb70torch (see pytorch-ext/src/interop.cpp).
// Kept free of SYCL/XPU headers so it also compiles inside .cu files.
#pragma once
#include <cuda_runtime_api.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <cstdint>

extern "C" {
void *b70torch_stream_acquire(int device);
void b70torch_flush();
int b70torch_current_device();
int b70torch_device_count();
void b70torch_device_synchronize();
}

namespace c10 {
namespace cuda {

class CUDAStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit CUDAStream(DeviceIndex device = -1)
      : device_(device < 0 ? (DeviceIndex)b70torch_current_device() : device),
        stream_((cudaStream_t)b70torch_stream_acquire(device_)) {}
  CUDAStream(Unchecked, c10::Stream s)
      : device_(s.device_index()),
        stream_((cudaStream_t)b70torch_stream_acquire(device_)) {}
  explicit CUDAStream(c10::Stream s) : CUDAStream(UNCHECKED, s) {}

  bool operator==(const CUDAStream &o) const { return stream_ == o.stream_; }
  bool operator!=(const CUDAStream &o) const { return stream_ != o.stream_; }
  operator cudaStream_t() const { return stream_; }
  operator c10::Stream() const { return unwrap(); }
  cudaStream_t stream() const { return stream_; }
  DeviceIndex device_index() const { return device_; }
  DeviceType device_type() const { return DeviceType::XPU; }
  Device device() const { return Device(DeviceType::XPU, device_); }
  StreamId id() const { return 0; }
  int priority() const { return 0; }
  bool query() const { return cudaStreamQuery(stream_) == cudaSuccess; }
  void synchronize() const { cudaStreamSynchronize(stream_); }
  c10::Stream unwrap() const { return c10::Stream(c10::Stream::UNSAFE, device(), 0); }
  struct c10::StreamData3 pack3() const { return unwrap().pack3(); }
  static CUDAStream unpack3(StreamId, DeviceIndex device_index, DeviceType) {
    return CUDAStream(device_index);
  }
  static std::tuple<int, int> priority_range() { return {0, 0}; }

 private:
  DeviceIndex device_;
  cudaStream_t stream_;
};

inline CUDAStream getCurrentCUDAStream(DeviceIndex device = -1) { return CUDAStream(device); }
inline CUDAStream getDefaultCUDAStream(DeviceIndex device = -1) { return CUDAStream(device); }
inline CUDAStream getStreamFromPool(const bool = false, DeviceIndex device = -1) {
  return CUDAStream(device);
}
inline CUDAStream getStreamFromPool(const int, DeviceIndex device = -1) { return CUDAStream(device); }
inline CUDAStream getStreamFromExternal(cudaStream_t, DeviceIndex device) { return CUDAStream(device); }
inline void setCurrentCUDAStream(CUDAStream) {}
inline std::ostream &operator<<(std::ostream &os, const CUDAStream &s) {
  return os << "cuda stream " << (void *)s.stream() << " (b70, xpu:" << (int)s.device_index() << ")";
}

}  // namespace cuda
}  // namespace c10

namespace at {
namespace cuda {
using c10::cuda::CUDAStream;
using c10::cuda::getCurrentCUDAStream;
using c10::cuda::getDefaultCUDAStream;
using c10::cuda::getStreamFromExternal;
using c10::cuda::getStreamFromPool;
using c10::cuda::setCurrentCUDAStream;
}  // namespace cuda
}  // namespace at

namespace std {
template <>
struct hash<c10::cuda::CUDAStream> {
  size_t operator()(c10::cuda::CUDAStream s) const noexcept {
    return std::hash<void *>{}((void *)s.stream());
  }
};
}  // namespace std
