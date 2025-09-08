#ifndef PhysicsTools_PyTorch_interface_TorchModelJit_h
#define PhysicsTools_PyTorch_interface_TorchModelJit_h

#include <string>
#include <vector>

#include "PhysicsTools/PyTorch/interface/Converter.h"
#include "PhysicsTools/PyTorch/interface/ScriptModuleLoad.h"
#include "PhysicsTools/PyTorch/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"

namespace cms::torch {

  // Wrapper of torch::jit::script::Module: https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
  class TorchModelJit {
  public:
    explicit TorchModelJit(const std::string &model_path)
        : model_(cms::torch::load(model_path)), device_(::torch::kCPU) {}

    explicit TorchModelJit(const std::string &model_path, ::torch::Device device)
        : model_(cms::torch::load(model_path, device)), device_(device) {}

    // Move model to specified device memory space. Async load (in default stream if not overridden by the caller)
    void to(::torch::Device device, bool non_blocking = false) {
      if (device == device_)
        return;
      model_.to(device, non_blocking);
      device_ = device;
    }

    // Forward pass (inference) of model, returns torch::IValue (multi output support). Match native torchlib interface.
    ::torch::IValue forward(std::vector<::torch::IValue> &inputs) {
      return model_.forward(inputs);
    }
     
    // Get model current device information.
    ::torch::Device device() const {
      return device_;
    }

  protected:
    ::torch::jit::script::Module model_;  // underlying JIT model
    ::torch::Device device_;              // device where the model is allocated (default CPU)
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_ModelJit_h