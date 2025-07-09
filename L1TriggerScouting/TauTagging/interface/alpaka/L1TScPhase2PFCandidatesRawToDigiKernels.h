#ifndef L1TriggerScouting_TauTagging_interface_alpaka_L1TScPhase2PFCandidatesRawToDigiKernels_h
#define L1TriggerScouting_TauTagging_interface_alpaka_L1TScPhase2PFCandidatesRawToDigiKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/OrbitEventIndexMapCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  void PrintPFCandidateCollection(Queue& queue, PFCandidateCollection& pf_candidates);
  void RawToDigi(Queue& queue, data_t *pf_data, PFCandidateCollection& pf_candidates);
  void AssociateOrbitEventIndex(
      Queue& queue, data_t *h_data, OrbitEventIndexMapCollection& orbit_association_map);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_TauTagging_interface_alpaka_L1TScPhase2PFCandidatesRawToDigiKernels_h