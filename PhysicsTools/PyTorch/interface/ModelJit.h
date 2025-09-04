#ifndef PhysicsTools_PyTorch_interface_ModelJit_h
#define PhysicsTools_PyTorch_interface_ModelJit_h

#include <string>
#include <vector>

#include "PhysicsTools/PyTorch/interface/JitLoad.h"
#include "PhysicsTools/PyTorch/interface/TorchLib.h"

namespace cms::torch {

  /**
   * @brief wrapper of torch::jit::script::Module
   * @see: https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
   */
  class ModelJit {
  public:
    explicit ModelJit(const std::string &model_path);
    explicit ModelJit(const std::string &model_path, ::torch::Device device);

    void to(::torch::Device device, bool non_blocking = false);
    ::torch::IValue forward(std::vector<::torch::IValue> &inputs);
    ::torch::Device device() const;

    template <typename InMemLayout, typename OutMemLayout>
    void forward(const ModelMetadata<InMemLayout, OutMemLayout> &metadata) {
      auto input_tensor = Converter::convert_input(metadata, device_);
      // TODO: add support for multi-output models (without temporary mem copy)
      Converter::convert_output(metadata, device_) = model_.forward(input_tensor).toTensor();
    }

  protected:
    ::torch::jit::script::Module model_;
    ::torch::Device device_;
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_ModelJit_h