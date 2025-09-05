#ifndef PhysicsTools_PyTorch_interface_FwkGuards_h
#define PhysicsTools_PyTorch_interface_FwkGuards_h

#include <mutex>

#include "PhysicsTools/PyTorch/interface/TorchLib.h"

namespace cms::torch {

  /**
   * @brief Sets the guard to disable multi-threading and control PyTorch's threading model.
   * @note Global call.
   */
  inline void set_threading_guard() {
    static std::once_flag threading_guard_flag;
    std::call_once(threading_guard_flag, [] {
      at::set_num_threads(1);
      at::set_num_interop_threads(1);
    });
  }

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_FwkGuards_h