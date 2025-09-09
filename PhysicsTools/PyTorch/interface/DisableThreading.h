#ifndef PhysicsTools_PyTorch_interface_SetThreading_h
#define PhysicsTools_PyTorch_interface_SetThreading_h

#include <mutex>

#include "PhysicsTools/PyTorch/interface/TorchCompat.h"

namespace cms::torch {

  // Disable PyTorch's threading model: https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  // PyTorch can internally spawn multiple threads including intra-op threading within ops (element-wise, GEMM, convolutions) 
  // which are not controlled by framework. All Torch CPU based operations will run single-threaded in CMSSW workflows.
  inline void disableThreading() {
    // Global call, has to be called once before any PyTorch code is executed (PhysicsTools/PyTorch/plugins/PyTorchService.cc)
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
  }

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_SetThreading_h