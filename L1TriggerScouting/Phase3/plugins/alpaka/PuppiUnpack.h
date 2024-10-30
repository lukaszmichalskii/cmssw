#ifndef L1TriggerScouting_Phase3_plugins_alpaka_PuppiUnpack_h
#define L1TriggerScouting_Phase3_plugins_alpaka_PuppiUnpack_h

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class PuppiUnpack {

  public:
    void Fill(Queue& queue, PuppiCollection& collection, int value = 1) const;
    void Assert(Queue& queue, PuppiCollection const& collection, int value = 1) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_PuppiUnpack_h