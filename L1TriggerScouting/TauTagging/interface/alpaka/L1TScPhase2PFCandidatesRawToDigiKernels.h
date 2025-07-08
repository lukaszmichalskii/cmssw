#ifndef L1_TRIGGER_SCOUTING__TAU_TAGGING__INTERFACE__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_KERNELS_H
#define L1_TRIGGER_SCOUTING__TAU_TAGGING__INTERFACE__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_KERNELS_H

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  void PrintPFCandidateCollection(Queue& queue, PFCandidateCollection& pf_candidates);
  void RawToDigi(Queue& queue, data_t *pf_data, PFCandidateCollection& pf_candidates);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1_TRIGGER_SCOUTING__TAU_TAGGING__INTERFACE__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_KERNELS_H