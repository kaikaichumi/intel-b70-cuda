// b70torch shim for <c10/cuda/CUDAGuard.h>: device/stream guards over the XPU
// device guard implementation that torch registers at runtime.
#pragma once
#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <optional>

namespace c10 {
namespace cuda {

namespace detail {
inline Device as_xpu(Device d) {
  return Device(DeviceType::XPU, d.index());
}
}  // namespace detail

struct CUDAGuard {
  explicit CUDAGuard() = delete;
  explicit CUDAGuard(DeviceIndex index) : guard_(Device(DeviceType::XPU, index)) {}
  explicit CUDAGuard(Device device) : guard_(detail::as_xpu(device)) {}
  CUDAGuard(const CUDAGuard &) = delete;
  CUDAGuard &operator=(const CUDAGuard &) = delete;
  void set_device(Device device) { guard_.reset_device(detail::as_xpu(device)); }
  void reset_device(Device device) { guard_.reset_device(detail::as_xpu(device)); }
  void set_index(DeviceIndex index) { guard_.reset_device(Device(DeviceType::XPU, index)); }
  Device original_device() const { return guard_.original_device(); }
  Device current_device() const { return guard_.current_device(); }

 private:
  c10::DeviceGuard guard_;
};

struct OptionalCUDAGuard {
  explicit OptionalCUDAGuard() = default;
  explicit OptionalCUDAGuard(std::optional<Device> device) {
    if (device) guard_.emplace(detail::as_xpu(*device));
  }
  explicit OptionalCUDAGuard(std::optional<DeviceIndex> index) {
    if (index) guard_.emplace(Device(DeviceType::XPU, *index));
  }
  explicit OptionalCUDAGuard(Device device) : guard_(std::in_place, detail::as_xpu(device)) {}
  explicit OptionalCUDAGuard(DeviceIndex index) : guard_(std::in_place, Device(DeviceType::XPU, index)) {}
  OptionalCUDAGuard(const OptionalCUDAGuard &) = delete;
  OptionalCUDAGuard &operator=(const OptionalCUDAGuard &) = delete;
  void set_device(Device device) { reset_device(device); }
  void reset_device(Device device) {
    if (guard_) guard_->reset_device(detail::as_xpu(device));
    else guard_.emplace(detail::as_xpu(device));
  }
  void set_index(DeviceIndex index) { reset_device(Device(DeviceType::XPU, index)); }
  std::optional<Device> original_device() const {
    return guard_ ? std::optional<Device>(guard_->original_device()) : std::nullopt;
  }
  std::optional<Device> current_device() const {
    return guard_ ? std::optional<Device>(guard_->current_device()) : std::nullopt;
  }
  void reset() { guard_.reset(); }

 private:
  std::optional<c10::DeviceGuard> guard_;
};

// There is one CUDA stream per torch stream here, so a stream guard only has
// to switch the device.
struct CUDAStreamGuard {
  explicit CUDAStreamGuard() = delete;
  explicit CUDAStreamGuard(Stream stream) : guard_(stream.device()) {}
  CUDAStreamGuard(const CUDAStreamGuard &) = delete;
  CUDAStreamGuard &operator=(const CUDAStreamGuard &) = delete;
  void reset_stream(Stream stream) { guard_.reset_device(stream.device()); }
  CUDAStream original_stream() const { return CUDAStream(guard_.original_device().index()); }
  CUDAStream current_stream() const { return CUDAStream(guard_.current_device().index()); }
  Device current_device() const { return guard_.current_device(); }
  Device original_device() const { return guard_.original_device(); }

 private:
  c10::DeviceGuard guard_;
};

struct OptionalCUDAStreamGuard {
  explicit OptionalCUDAStreamGuard() = default;
  explicit OptionalCUDAStreamGuard(Stream stream) : guard_(std::in_place, stream.device()) {}
  explicit OptionalCUDAStreamGuard(std::optional<Stream> stream) {
    if (stream) guard_.emplace(stream->device());
  }
  void reset_stream(Stream stream) {
    if (guard_) guard_->reset_device(stream.device());
    else guard_.emplace(stream.device());
  }
  void reset() { guard_.reset(); }

 private:
  std::optional<c10::DeviceGuard> guard_;
};

struct CUDAMultiStreamGuard {
  explicit CUDAMultiStreamGuard(ArrayRef<CUDAStream>) {}
};

}  // namespace cuda
}  // namespace c10

namespace at {
namespace cuda {
using c10::cuda::CUDAGuard;
using c10::cuda::CUDAMultiStreamGuard;
using c10::cuda::CUDAStreamGuard;
using c10::cuda::OptionalCUDAGuard;
using c10::cuda::OptionalCUDAStreamGuard;
}  // namespace cuda
}  // namespace at
