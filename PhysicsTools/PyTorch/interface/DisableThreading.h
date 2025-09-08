#ifndef PhysicsTools_PyTorch_interface_SetThreading_h
#define PhysicsTools_PyTorch_interface_SetThreading_h

#include <mutex>

#include "PhysicsTools/PyTorch/interface/TorchCompat.h"

namespace cms::torch {

  // Disable PyTorch's threading model: https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  inline void disableThreading() {
    // Global call, has to be called once per job before any PyTorch code is executed (PhysicsTools/PyTorch/plugins/PyTorchService.cc)
    static std::once_flag threading_guard_flag;
    std::call_once(threading_guard_flag, [] {
      at::set_num_threads(1);
      at::set_num_interop_threads(1);
    });
  }

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_SetThreading_h