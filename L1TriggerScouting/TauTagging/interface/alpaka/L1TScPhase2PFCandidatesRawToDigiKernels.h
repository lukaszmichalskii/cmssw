#ifndef L1_TRIGGER_SCOUTING__TAU_TAGGING__INTERFACE__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_KERNELS_H
#define L1_TRIGGER_SCOUTING__TAU_TAGGING__INTERFACE__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_KERNELS_H

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  class L1TScPhase2PFCandidatesRawToDigiKernels {
    public:
      void RawToDigi(Queue& queue, PFCandidateCollection& pf_candidates) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

#endif  // L1_TRIGGER_SCOUTING__TAU_TAGGING__INTERFACE__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_KERNELS_H