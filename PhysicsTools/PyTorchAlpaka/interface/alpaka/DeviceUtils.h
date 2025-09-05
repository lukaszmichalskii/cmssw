// ROCm/HIP backend not yet supported, see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75
#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_DeviceUtils_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_DeviceUtils_h

#include "alpaka/alpaka.hpp"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/TorchLib.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch::alpakatools {

  inline ::torch::Device device(const Device &dev) {
    if (kDevice == kDevHost)
      return ::torch::Device(kDevHost);
    return ::torch::Device(kDevice, dev.getNativeHandle());
  }

  inline ::torch::Device device(const Queue &queue) { return device(::alpaka::getDev(queue)); }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_DeviceUtils_h