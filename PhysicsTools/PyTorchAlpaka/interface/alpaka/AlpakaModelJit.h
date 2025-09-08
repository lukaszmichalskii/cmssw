#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModelJit_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModelJit_h

#include "alpaka/alpaka.hpp"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/TorchModelJit.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Converter.h"
#include "PhysicsTools/PyTorchAlpaka/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/GetDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  class AlpakaModelJit : public cms::torch::TorchModelJit {
  public:
    using cms::torch::TorchModelJit::forward;
    using cms::torch::TorchModelJit::to;

    // Default model loads to CPU memory space, to be moved to accelerator memory space later in async fashion.
    explicit AlpakaModelJit(const std::string &model_path)
        : cms::torch::TorchModelJit(model_path) {}
    
    // Loads model to alpaka accelerator specified memory space.
    template <typename T>
      requires ::alpaka::isDevice<T> || ::alpaka::isQueue<T>
    explicit AlpakaModelJit(const std::string &model_path, const T &acc_mem)
        : cms::torch::TorchModelJit(model_path, getDevice(acc_mem)) {}

    // Forward pass (inference) of model with SoA metadata input/output.
    // Allows to run inference directly using SoA portable objects/collections without excessive copies and conversions.
    // Refer: PhysicsTools/PyTorch/interface/Converter.h for details about wrapping memory layouts.
    // TODO: add support for multi-output models (without temporary mem copy)
    template <typename InMemLayout, typename OutMemLayout>
    void forward(const cms::torch::alpakatools::ModelMetadata<InMemLayout, OutMemLayout> &metadata) {
      auto input_tensor = cms::torch::alpakatools::Converter::convert_input(metadata, device_);
      cms::torch::alpakatools::Converter::convert_output(metadata, device_) = model_.forward(input_tensor).toTensor();
    }

    // Move model to specified device memory space. Async load (in default stream if not overridden by the caller)
    template <typename T>
      requires ::alpaka::isDevice<T> || ::alpaka::isQueue<T>
    void to(const T &acc_mem) {
      if constexpr (std::is_same_v<::alpaka::Dev<T>, ::alpaka::DevCpu>) {
        this->TorchModelJit::to(getDevice(acc_mem));
        return;
      }
// #ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
      // ROCm/HIP not yet directly supported → fallback to CPU inference
      this->TorchModelJit::to(getDevice(acc_mem));
      return;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED 
      // this->TorchModelJit::to(getDevice(acc_mem), true);
    }
    
    // Overload for Queue to simplify the interface for the common case of async execution.
    void to(const Queue &queue) {
      this->AlpakaModelJit::to(::alpaka::getDev(queue));
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModelJit_h