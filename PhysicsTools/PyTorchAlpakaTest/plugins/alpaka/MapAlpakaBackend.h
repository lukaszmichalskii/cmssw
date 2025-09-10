#ifndef PhysicsTools_PyTorch_plugins_alpaka_MapAlpakaBackend_h
#define PhysicsTools_PyTorch_plugins_alpaka_MapAlpakaBackend_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  constexpr auto kAlpakaBackend = "CudaAsync";
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  constexpr auto kAlpakaBackend = "ROCmAsync";
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  constexpr auto kAlpakaBackend = "SerialSync";
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  constexpr auto kAlpakaBackend = "TbbAsync";
#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

#endif  // PhysicsTools_PyTorch_plugins_alpaka_MapAlpakaBackend_h