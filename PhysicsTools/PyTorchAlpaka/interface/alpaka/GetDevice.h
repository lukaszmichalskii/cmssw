#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_DeviceUtils_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_DeviceUtils_h

#include "alpaka/alpaka.hpp"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  inline ::torch::Device getDevice(const Device &device) {
    return (kDevice == kDevHost)
           ? ::torch::Device(kDevHost)
           : ::torch::Device(kDevice, device.getNativeHandle());
  }

  inline ::torch::Device getDevice(const Queue &queue) { return getDevice(::alpaka::getDev(queue)); }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_DeviceUtils_h