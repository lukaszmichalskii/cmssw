#ifndef PhysicsTools_PyTorch_test_Nvtx_h
#define PhysicsTools_PyTorch_test_Nvtx_h

#if defined(__CUDACC__) || defined(USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace torchtest {

  class Nvtx {
  public:
    explicit Nvtx(const char* msg) {
#if defined(__CUDACC__) || defined(USE_NVTX)
      id_ = nvtxRangeStartA(msg);
#endif
    }

    void end() {
#if defined(__CUDACC__) || defined(USE_NVTX)
      if (active_) {
        active_ = false;
        nvtxRangeEnd(id_);
      }
#endif
    }

    ~Nvtx() { end(); }

  private:
#if defined(__CUDACC__) || defined(USE_NVTX)
    nvtxRangeId_t id_;
#endif
    bool active_ = true;
  };

}  // namespace torchtest

#endif  // PhysicsTools_PyTorch_test_Nvtx_h
