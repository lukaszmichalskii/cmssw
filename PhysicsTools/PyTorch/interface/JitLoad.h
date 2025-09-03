#ifndef PhysicsTools_PyTorch_interface_JitLoad_h
#define PhysicsTools_PyTorch_interface_JitLoad_h

// TODO: find a better way to resolve PyTorch and ROOT's ClassDef macro clash
#ifdef ClassDef
#undef ClassDef
#endif

#include <torch/script.h>
#include <torch/torch.h>

namespace cms::torch {

  ::torch::jit::script::Module load(const std::string &model_path, std::optional<::torch::Device> dev = std::nullopt);

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_JitLoad_h