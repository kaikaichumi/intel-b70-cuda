// b70torch shim for <ATen/cuda/CUDAEvent.h>: a CUDA event on the b70 stream.
#pragma once
#include <cuda_runtime_api.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>
#include <utility>

namespace at {
namespace cuda {

struct CUDAEvent {
  CUDAEvent() noexcept = default;
  explicit CUDAEvent(unsigned int flags) noexcept : flags_(flags) {}
  ~CUDAEvent() { if (event_) cudaEventDestroy(event_); }
  CUDAEvent(const CUDAEvent &) = delete;
  CUDAEvent &operator=(const CUDAEvent &) = delete;
  CUDAEvent(CUDAEvent &&o) noexcept { std::swap(flags_, o.flags_); std::swap(event_, o.event_); std::swap(device_, o.device_); }
  CUDAEvent &operator=(CUDAEvent &&o) noexcept {
    if (this != &o) { std::swap(flags_, o.flags_); std::swap(event_, o.event_); std::swap(device_, o.device_); }
    return *this;
  }
  operator cudaEvent_t() const { return event(); }
  cudaEvent_t event() const { return event_; }
  bool isCreated() const { return event_ != nullptr; }
  DeviceIndex device_index() const { return device_; }
  bool query() const { return !event_ || cudaEventQuery(event_) == cudaSuccess; }
  void record() { record(getCurrentCUDAStream()); }
  void recordOnce(const CUDAStream &s) { if (!was_recorded_) record(s); }
  void record(const CUDAStream &s) {
    if (!event_) { device_ = s.device_index(); AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_)); }
    AT_CUDA_CHECK(cudaEventRecord(event_, s.stream()));
    was_recorded_ = true;
  }
  void block(const CUDAStream &s) { if (event_) AT_CUDA_CHECK(cudaStreamWaitEvent(s.stream(), event_, 0)); }
  float elapsed_time(const CUDAEvent &o) const {
    float ms = 0; AT_CUDA_CHECK(cudaEventElapsedTime(&ms, event_, o.event_)); return ms;
  }
  void synchronize() const { if (event_) AT_CUDA_CHECK(cudaEventSynchronize(event_)); }

 private:
  unsigned int flags_ = cudaEventDisableTiming;
  bool was_recorded_ = false;
  DeviceIndex device_ = -1;
  cudaEvent_t event_ = nullptr;
};

}  // namespace cuda
}  // namespace at
