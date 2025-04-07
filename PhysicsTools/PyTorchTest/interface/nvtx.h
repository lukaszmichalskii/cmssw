#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#include <nvtx3/nvToolsExt.h>
#endif

namespace torch_alpaka {

class NVTXScopedRange {
  public:
   NVTXScopedRange(const char* msg) {
 #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
     defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
     id_ = nvtxRangeStartA(msg);
 #endif
   }
 
   void end() {
 #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
     defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
     if (active_) {
       active_ = false;
       nvtxRangeEnd(id_);
     }
 #endif
   }
 
   ~NVTXScopedRange() { end(); }
 
  private:
 #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
     defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
   nvtxRangeId_t id_;
   bool active_ = true;
 #endif
 };

}  // namespace torch_alpaka