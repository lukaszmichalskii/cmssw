#ifndef PhysicsTools_PyTorch_interface_model_h
#define PhysicsTools_PyTorch_interface_model_h

#include "PhysicsTools/PyTorch/interface/common.h"


namespace cms::torch_alpaka {

class Model {
 public:
  Model() = default;
  Model(const std::string &model_path);
  void to(const torch::Device &device) const;
  const torch::Device device() const;

  template <typename TSoALayoutIn, typename TSoALayoutOut>
  void forward(torch_alpaka_tools::ModelMetadata metadata,
               std::byte *inputs,
               std::byte *outputs) const;

 private:
  mutable torch::jit::script::Module model_;
  mutable torch::Device device_ = torch::Device(torch::kCPU);
};

//////////////////////////////////////////////////////////////

Model::Model(const std::string &model_path) {
  model_ = cms::torch_tools::load_model(model_path);
}

void Model::to(const torch::Device &device) const {
  device_ = device;
  model_.to(device_);
}

const torch::Device Model::device() const {
  return device_;
}

template <typename TSoALayoutIn, typename TSoALayoutOut>
void Model::forward(torch_alpaka_tools::ModelMetadata metadata,
                    std::byte *inputs,
                    std::byte *outputs) const {
  std::vector<torch::jit::IValue> input_tensor = 
      torch_alpaka_tools::Converter<TSoALayoutIn>::convert_input(metadata, device_, inputs);
  torch_alpaka_tools::Converter<TSoALayoutOut>::convert_output(metadata, device_, outputs) = 
      model_.forward(input_tensor).toTensor();
}

}  // namespace cms::torch_alpaka

#endif  // PhysicsTools_PyTorch_interface_model_h