#ifndef PhysicsTools_PyTorch_interface_model_h
#define PhysicsTools_PyTorch_interface_model_h

#include "PhysicsTools/PyTorch/interface/common.h"


namespace cms::torch_alpaka {

class Model {
 public:
  Model() = default;
  Model(const std::string &model_path);
  void to(const torch::Device &device);
  const torch::Device device() const;

 private:
  torch::jit::script::Module model_;
  torch::Device device_ = cms::torch_alpaka_tools::device();
};

//////////////////////////////////////////////////////////////

Model::Model(const std::string &model_path) {
  model_ = cms::torch_alpaka_tools::load_model(model_path);
}

void Model::to(const torch::Device &device) {
  model_.to(device);
  device_ = device;
}

const torch::Device Model::device() const {
  return device_;
}

}  // namespace cms::torch_alpaka

#endif  // PhysicsTools_PyTorch_interface_model_h