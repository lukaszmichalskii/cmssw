#ifndef PhysicsTools_PyTorch_interface_ModelJitAlpaka_h
#define PhysicsTools_PyTorch_interface_ModelJitAlpaka_h

#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/ModelJit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  using namespace cms::torch;

  /**
   * @brief wrapper of cms::torch::ModelJit with Alpaka support
   * @see: PhysicsTools/PyTorch/interface/ModelJit.h
   */
  class ModelJitAlpaka : public ModelJit {
  public:
    explicit ModelJitAlpaka(const std::string &model_path);
    explicit ModelJitAlpaka(const std::string &model_path, const Device &dev);
    explicit ModelJitAlpaka(const std::string &model_path, const Queue &queue);

    void to(const Device &dev, bool non_blocking = false);
    void to(const Queue &queue, bool non_blocking = false);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorch_interface_ModelJitAlpaka_h