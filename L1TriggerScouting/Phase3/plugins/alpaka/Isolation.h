#ifndef L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h
#define L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Isolation {
public:
  void Isolate(Queue& queue, PuppiCollection& data) const;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h