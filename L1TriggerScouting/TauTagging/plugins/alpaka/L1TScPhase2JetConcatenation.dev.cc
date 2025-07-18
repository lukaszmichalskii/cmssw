#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2JetConcatenation.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  class ConcatenateKernel {
  public:
    ALPAKA_FN_ACC void operator()(const Acc1D& acc, PFCandidateCollection::ConstView pf_candidates, CLUEsteringCollection::ConstView clue_collection) const {
      if (once_per_grid(acc)) {
        printf("ConcatenateKernel OK\n");
      }
    }
  };

  void concatenate(Queue& queue, const PFCandidateCollection& pf_candidates, const CLUEsteringCollection& clue_collection) {
    uint32_t threads_per_block = 64;
    uint32_t blocks_per_grid = divide_up_by(pf_candidates.view().metadata().size(), threads_per_block);      
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, ConcatenateKernel{}, pf_candidates.const_view(), clue_collection.const_view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels