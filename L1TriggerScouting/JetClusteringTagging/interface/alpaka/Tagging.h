#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Tagging {
public:
  void Tag(Queue& queue, PuppiCollection& data);
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h