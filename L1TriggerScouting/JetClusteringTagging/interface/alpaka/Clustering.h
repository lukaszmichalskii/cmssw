#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h

// libs
#include <alpaka/alpaka.hpp>
// typedefs
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// heterogeneous
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class SeededConeClustering {
public:
  void Cluster(Queue& queue, PuppiCollection const& data, ClustersCollection& clusters, uint32_t clusters_num);
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Clustering_h