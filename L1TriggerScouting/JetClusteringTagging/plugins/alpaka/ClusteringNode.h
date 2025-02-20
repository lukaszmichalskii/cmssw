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

/**
 * Consume device side product and invoke clustering algorithm to form jets.
 * Implicit device memory allocation and transfers with reduced copy operations.
 * The product stays on device memory and can be automatically transferred to host if needed.
 *
 * @brief Jet clustering for puppi dataset node.
 */
class ClusteringNode : public stream::EDProducer<> {

public:
  ClusteringNode(const edm::ParameterSet& params);
  ~ClusteringNode() override = default;

  /**
   * @brief cmssw callback for node
   */
  void produce(device::Event& event, const device::EventSetup& event_setup) override;

  /**
   * @brief Declare parameters for node
   */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /**
   * @brief Cluster puppi data to form jets primitives @see <<PuppiCollection>>
   */
  ClustersCollection Cluster(Queue &queue, PuppiCollection const& data);

  // utils
  std::unique_ptr<SeededConeClustering> clustering_ = nullptr;  /**< algorithm used for clustering */
  // tokens
  const device::EDGetToken<PuppiCollection> device_in_token_;  /**< device read data */
  device::EDPutToken<ClustersCollection> device_out_token_;  /**< device write data */
  // params
  uint32_t clusters_num_;  /**< number of clusters to init algorithm */
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_ClusteringNode_h
