#ifndef L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2CLUEstering_h
#define L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2CLUEstering_h

#include <alpaka/alpaka.hpp>

#include "CLUEstering/CLUEstering.hpp"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/CLUEsteringCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  using namespace ::l1sc;

  class L1TScPhase2CLUEstering {
  public:
    explicit L1TScPhase2CLUEstering(float dc, float rhoc, float dm);

    void run(Queue& queue, PFCandidateCollection& pf_candidates, CLUEsteringCollection& clue_collection);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

#endif  // L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2CLUEstering_h