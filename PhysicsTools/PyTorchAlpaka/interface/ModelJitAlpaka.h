#ifndef PhysicsTools_PyTorch_interface_ModelJitAlpaka_h
#define PhysicsTools_PyTorch_interface_ModelJitAlpaka_h

#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/ModelJit.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/DeviceUtils.h"

namespace cms::torch::alpaka {

  /**
   * @brief wrapper of cms::torch::ModelJit with Alpaka support
   * @see: PhysicsTools/PyTorch/interface/ModelJit.h
   */
  class ModelJitAlpaka : public cms::torch::ModelJit {
  public:
    explicit ModelJitAlpaka(const std::string &model_path) : cms::torch::ModelJit(model_path) {}

    template <typename T>
    explicit ModelJitAlpaka(const std::string &model_path, const T &target_dev)
        : cms::torch::ModelJit(model_path, ALPAKA_ACCELERATOR_NAMESPACE::torch::alpakatools::device(target_dev)) {}

    template <typename T>
    void to(const T &target_dev) {
      if constexpr (std::is_same_v<::alpaka::Dev<T>, ::alpaka::DevCpu>) {
        ModelJit::to(ALPAKA_ACCELERATOR_NAMESPACE::torch::alpakatools::device(target_dev));
        return;
      }
      ModelJit::to(ALPAKA_ACCELERATOR_NAMESPACE::torch::alpakatools::device(target_dev), true);
    }
  };

}  // namespace cms::torch::alpaka

#endif  // PhysicsTools_PyTorch_interface_ModelJitAlpaka_h