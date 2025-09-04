#ifndef PhysicsTools_PyTorch_interface_Config_h
#define PhysicsTools_PyTorch_interface_Config_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/TorchLib.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  constexpr auto kDevHost = c10::DeviceType::CPU;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  constexpr auto kDevice = c10::DeviceType::CUDA;
// #elif ALPAKA_ACC_GPU_HIP_ENABLED
//   constexpr auto kDevice = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  constexpr auto kDevice = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  constexpr auto kDevice = c10::DeviceType::CPU;
#else
#error "Could not define the torch device type."
#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorch_interface_Config_h