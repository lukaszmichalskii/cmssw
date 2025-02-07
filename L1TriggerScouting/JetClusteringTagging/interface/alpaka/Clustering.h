#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Clustering {
public:
  uint32_t Cluster(Queue& queue, PuppiCollection const& data) const;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h