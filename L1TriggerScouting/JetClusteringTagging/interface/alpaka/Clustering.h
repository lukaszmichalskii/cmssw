#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class SeededConeClustering {
public:
  void Cluster(Queue& queue, PuppiCollection const& data, uint32_t clusters_num);
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h