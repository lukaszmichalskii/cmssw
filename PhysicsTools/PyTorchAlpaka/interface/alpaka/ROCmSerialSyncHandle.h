#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_ROCmSerialSyncHandle_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_ROCmSerialSyncHandle_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

using namespace cms::alpakatools;

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  // Define the base class to avoid typename declarations
  class ROCmSerialSyncHandleBase {
  public:
    virtual ~ROCmSerialSyncHandleBase() = default;
    virtual void copyToHost(Queue&) = 0;
    virtual void copyToDevice(Queue&) = 0;
    virtual void* ptr() = 0;
  };

  // Helper class to provide SerialSync backend fallback on ROCmAsync based modules
  template <typename T>
  class ROCmSerialSyncHandle : public ROCmSerialSyncHandleBase {
  public:
    explicit ROCmSerialSyncHandle(const void* device_ptr, const size_t size, const size_t stride)
        : device_ptr_(device_ptr), extent_(Vec1D{size * stride}), h_buf_(make_host_buffer<T[]>(size * stride)) {}

    // Synchronization responsibility move to the caller
    void copyToHost(Queue& queue) {
      auto d_view =
          alpaka::createView(alpaka::getDev(queue), const_cast<T*>(static_cast<const T*>(device_ptr_)), extent_);
      alpaka::memcpy(queue, h_buf_.value(), d_view);
    }

    // Synchronization responsibility move to the caller
    void copyToDevice(Queue& queue) {
      auto d_view =
          alpaka::createView(alpaka::getDev(queue), const_cast<T*>(static_cast<const T*>(device_ptr_)), extent_);
      alpaka::memcpy(queue, d_view, h_buf_.value());
    }

    void* ptr() { return alpaka::getPtrNative(h_buf_.value()); }

  private:
    const void* device_ptr_;
    const Vec1D extent_;
    std::optional<host_buffer<T[]>> h_buf_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_ROCmSerialSyncHandle_h