#ifndef PhysicsTools_PyTorch_interface_common_h
#define PhysicsTools_PyTorch_interface_common_h

#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <thread>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

//////////////////////////////////////////////////////////////
// TORCH COMMON
//////////////////////////////////////////////////////////////

namespace cms::torch_tools {

inline torch::jit::script::Module load_model(const std::string &model_path) {
  try {
    return torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    throw std::runtime_error("Error loading the model");
  }
}

}  // namespace torch_tools

//////////////////////////////////////////////////////////////
// TORCH ALPAKA COMMON
//////////////////////////////////////////////////////////////

namespace cms::torch_alpaka_common {

template <typename TQueue>
class QueueGuard {
 public:
  QueueGuard() {}
  ~QueueGuard() {}

  void set(const TQueue &queue) {}
  void reset() {}
};

template <typename TQueue>
class MultithreadingGuard {
 public:
  MultithreadingGuard() {}
  ~MultithreadingGuard() { reset();}

  void set(const TQueue &queue) { static DisableMultithreading disabler; }
  void reset() { static EnableMultithreading enabler; }

 private:
  class DisableMultithreading {
   friend MultithreadingGuard;
   private:
    DisableMultithreading() {
      at::set_num_threads(1);
      at::set_num_interop_threads(1);
    }
  };
  
  class EnableMultithreading {
   friend MultithreadingGuard;
   private:
    EnableMultithreading() {
      const auto num_threads = std::thread::hardware_concurrency();
      at::set_num_threads(num_threads);
      at::set_num_interop_threads(num_threads);
    }
  }; 
};


#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
constexpr c10::DeviceType kDeviceType = c10::DeviceType::CUDA;

template <>
class QueueGuard<alpaka_cuda_async::Queue> {
 public:
  QueueGuard() : native_stream_(c10::cuda::getCurrentCUDAStream()) {}
  ~QueueGuard() { reset(); }

  void set(const alpaka_cuda_async::Queue &queue) {
    auto dev = torch::Device(kDeviceType, alpaka::getDev(queue).getNativeHandle());
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), dev.index());
    c10::cuda::setCurrentCUDAStream(stream);
    set_ = true;
  }
  
  void reset() {
    if (!set_) 
      return;
    c10::cuda::setCurrentCUDAStream(native_stream_);
    set_ = false;
  }

 private:
  c10::cuda::CUDAStream native_stream_;
  bool set_ = true;
};
#elif ALPAKA_ACC_GPU_HIP_ENABLED
constexpr c10::DeviceType kDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
constexpr c10::DeviceType kDeviceType = c10::DeviceType::CPU;

template <>
class QueueGuard<alpaka_serial_sync::Queue>
    : public MultithreadingGuard<alpaka_serial_sync::Queue> {
  using MultithreadingGuard<alpaka_serial_sync::Queue>::MultithreadingGuard;
};

#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
constexpr c10::DeviceType kDeviceType = c10::DeviceType::CPU;

template <>
class QueueGuard<alpaka_tbb_async::Queue>
    : public MultithreadingGuard<alpaka_tbb_async::Queue> {
  using MultithreadingGuard<alpaka_tbb_async::Queue>::MultithreadingGuard;
};
#else
#error "Could not define the torch device type from alpaka backend."
#endif

}  // namespace cms::torch_alpaka_common

//////////////////////////////////////////////////////////////

namespace cms::torch_alpaka_tools {

inline torch::Device device(ALPAKA_ACCELERATOR_NAMESPACE::Device &dev) {
  return torch::Device(cms::torch_alpaka_common::kDeviceType, dev.getNativeHandle());
}

inline torch::Device device(ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
  const auto& dev = alpaka::getDev(queue);  
  return torch::Device(cms::torch_alpaka_common::kDeviceType, dev.getNativeHandle());
}

inline torch::Device device() {
  const auto devices = cms::alpakatools::devices<ALPAKA_ACCELERATOR_NAMESPACE::Platform>();
  assert(!devices.empty());
  const auto& dev = devices[0];
  return torch::Device(cms::torch_alpaka_common::kDeviceType, dev.getNativeHandle());
}

inline torch::jit::script::Module load_model(const std::string &model_path) {
  auto model = cms::torch_tools::load_model(model_path);
  model.to(device());
  return model;
}

}  // namespace cms::torch_alpaka_tools

#endif  // PhysicsTools_PyTorch_interface_common_h
