#ifndef PhysicsTools_PyTorchAlpaka_interface_Policy_h
#define PhysicsTools_PyTorchAlpaka_interface_Policy_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace cms::torch::alpakatools {

  // Default no-ops policy for fully supported backends:
  // - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // - ALPAKA_ACC_GPU_CUDA_ENABLED
  struct DefaultPolicy {
    explicit DefaultPolicy(const void*, size_t, size_t) {}
    template <typename TQueue> void copyToHost(TQueue&) const noexcept {}
    template <typename TQueue> void copyToDevice(TQueue&) const noexcept {}
  };

  // Generic fallback mechanism that provides a host-resident mirror of device memory blob.
  // Manages a host-side buffer that mirrors device memory, enabling CPU-based inference
  // when ROCm execution is not available.
  template <typename T>
  struct ROCmAsyncPolicy {
    ROCmAsyncPolicy(const void* d_ptr, size_t ncols, size_t nelems)
        : d_ptr_(d_ptr), extent_(alpaka_common::Vec1D{ncols * nelems}),
          h_buf_(cms::alpakatools::make_host_buffer<T[]>(ncols * nelems)) {}

    // Synchronization (if applicable) responsibility move to the caller
    template <typename TQueue>
    void copyToHost(TQueue& queue) {
      auto d_view = alpaka::createView(alpaka::getDev(queue), const_cast<T*>(static_cast<const T*>(d_ptr_)), extent_);
      alpaka::memcpy(queue, h_buf_, d_view);
    }

    // Synchronization (if applicable) responsibility move to the caller
    template <typename TQueue>
    void copyToDevice(TQueue& queue) {
      auto d_view =
          alpaka::createView(alpaka::getDev(queue), const_cast<T*>(static_cast<const T*>(d_ptr_)), extent_);
      alpaka::memcpy(queue, d_view, h_buf_);
    }

    const void* hostPtr() const noexcept { return alpaka::getPtrNative(h_buf_); }

  private:
    const void* d_ptr_;
    alpaka_common::Vec1D extent_;
    cms::alpakatools::host_buffer<T[]> h_buf_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_Policy_h