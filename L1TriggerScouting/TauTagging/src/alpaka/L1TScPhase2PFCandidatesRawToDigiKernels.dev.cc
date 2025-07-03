// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "L1TriggerScouting/TauTagging/interface/alpaka/L1TScPhase2PFCandidatesRawToDigiKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  class RawToDigiKernel {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, PFCandidateCollection::ConstView pf_candidates) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          int32_t head = 10;
          int32_t span = (pf_candidates.metadata().size() > head) ? head : pf_candidates.metadata().size();
          for (int32_t idx = 0; idx < span; ++idx) {
            printf("[%d] | %3.2f | %3.2f | %3.2f |\n", idx, 
                pf_candidates.pt()[idx], 
                pf_candidates.eta()[idx], 
                pf_candidates.phi()[idx]);
          }
          printf("PFCandidateCollection size: %d\n", pf_candidates.metadata().size());
        }
      }
  };

  void L1TScPhase2PFCandidatesRawToDigiKernels::RawToDigi(
      Queue& queue, PFCandidateCollection& pf_candidates) const {
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(
        pf_candidates.const_view().metadata().size(), threads_per_block);      
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, RawToDigiKernel{}, pf_candidates.const_view());    
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc
