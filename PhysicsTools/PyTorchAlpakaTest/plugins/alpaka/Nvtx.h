#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_Nvtx_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_Nvtx_h

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class Nvtx {
  public:
    explicit Nvtx(const char* msg) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      id_ = nvtxRangeStartA(msg);
#endif
    }

    void end() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      if (active_) {
        active_ = false;
        nvtxRangeEnd(id_);
      }
#endif
    }

    ~Nvtx() { end(); }

  private:
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    nvtxRangeId_t id_;
    bool active_ = true;
#endif
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_Nvtx_h