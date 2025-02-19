#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_ClusteringNode_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_ClusteringNode_h

// libs
#include <alpaka/alpaka.hpp>
// fw core
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// heterogeneous
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// typedefs
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// clustering
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Clustering.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {


class ClusteringNode : public stream::EDProducer<> {

public:
  ClusteringNode(const edm::ParameterSet& params);
  ~ClusteringNode() override = default;

  void produce(device::Event& event, const device::EventSetup& event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  ClustersCollection Cluster(Queue &queue, PuppiCollection const& data);

  const device::EDGetToken<PuppiCollection> device_in_token_;
  device::EDPutToken<ClustersCollection> device_out_token_;

  SeededConeClustering clustering_;
  uint32_t clusters_num_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_ClusteringNode_h
