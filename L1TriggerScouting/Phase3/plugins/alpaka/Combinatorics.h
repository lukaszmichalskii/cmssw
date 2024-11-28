#ifndef L1TriggerScouting_Phase3_plugins_alpaka_Combinatorics_h
#define L1TriggerScouting_Phase3_plugins_alpaka_Combinatorics_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Combinatorics {
public:
  PuppiCollection Combinatorial(Queue& queue, PuppiCollection const& data) const;
};


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_Combinatorics_h
