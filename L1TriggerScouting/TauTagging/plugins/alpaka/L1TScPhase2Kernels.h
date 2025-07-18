#ifndef L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2Kernels_h
#define L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2Kernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/CLUEsteringCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace ::l1sc;

  int max(Queue &queue, const int* data, const size_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2Kernels_h