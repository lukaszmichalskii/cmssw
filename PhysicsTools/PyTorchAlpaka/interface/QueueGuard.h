#ifndef PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h 
#define PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h

#include <type_traits>

#include <alpaka/alpaka.hpp>
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/GetDevice.h"

namespace cms::torch::alpakatools {

  // Default no-op implementation for platforms where no special handling is needed.
  // - ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // - ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  // - ALPAKA_ACC_GPU_HIP_ENABLED (AMD ROCm/HIP backend not yet supported, see below)
  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  struct QueueGuardTraits {
    static void set(const TQueue&) noexcept { /* no-op default */ }
    static void reset() noexcept { /* no-op default */ }
  };

  template <typename TQueue>
    requires ::alpaka::isQueue<TQueue>
  class QueueGuard {
  public:
    explicit QueueGuard(const TQueue &queue) { QueueGuardTraits<TQueue>::set(queue); }
    ~QueueGuard() { QueueGuardTraits<TQueue>::reset(); }
  };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

  #include <c10/cuda/CUDAStream.h>

  template <>
  struct QueueGuardTraits<alpaka_cuda_async::Queue> {
    // setCurrentCUDAStream() is assumed to not throw exceptions on the later-than-first calls.
    // see: https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDAStream.cpp#L373
    // Internal torch implementation of CUDA stream handling is based on a `thread_local`
    // see: https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDAStream.cpp#L169
    // follows the semantics of "current device" of CUDA itself (but not of Alpaka)
    // 
    // TODO: `noexcept` is used to avoid exceptions in the destructor, which for 100% clarity
    // restore the previous state (but currently not required for correctness).
    static void set(const alpaka_cuda_async::Queue &queue) noexcept {
      auto dev = ALPAKA_ACCELERATOR_NAMESPACE::torch::getDevice(queue);
      auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), dev.index());
      c10::cuda::setCurrentCUDAStream(stream);
    }
    static void reset() noexcept { /* no-op, can consider caching previous state and restoring it */}
  };

#elif ALPAKA_ACC_GPU_HIP_ENABLED

  // AMD ROCm/HIP backend not yet supported (though Alpaka HIP backend is available), the CPU fallback is used.
  // When CMSSW provide `pytorch-hip` counterpart in addition to `pytorch-cuda`, this can be implemented analogously to CUDA above.
  // See: 
  // - https://docs.pytorch.org/docs/stable/notes/hip.html
  // - https://github.com/pytorch/pytorch/tree/v2.6.0/c10/cuda#readme c10::cuda -> c10::hip
  //
  //
  // #include <c10/hip/HIPStream.h>
  //
  // template <>
  // struct QueueGuardTraits<alpaka_rocm_async::Queue> {
  //   static void set(const alpaka_rocm_async::Queue &queue) noexcept {
  //     auto dev = ALPAKA_ACCELERATOR_NAMESPACE::torch::getDevice(queue);
  //     auto stream = c10::hip::getStreamFromExternal(queue.getNativeHandle(), dev.index());
  //     c10::hip::setCurrentCUDAStream(stream);
  //   }
  //   static void reset() noexcept { /* no-op */ }
  // };

#endif

}  // namespace cms::torch::alpakatools 

#endif  // PhysicsTools_PyTorchAlpaka_interface_QueueGuard_h