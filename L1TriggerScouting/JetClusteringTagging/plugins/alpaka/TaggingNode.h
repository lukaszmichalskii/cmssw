#ifndef L1TriggerScouting_JetClusteringTagging_alpaka_TaggingNode_h
#define L1TriggerScouting_JetClusteringTagging_alpaka_TaggingNode_h

// libs
#include <alpaka/alpaka.hpp>
#include <chrono>
// fw core
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// heterogeneous
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// typedefs
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/JetsCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// inference 
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Tagging.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

/**
 * Consume device side product and invoke jet tagging and pt regression DNN based algorithm.
 * Implicit device memory allocation and transfers with reduced copy operations.
 * The product stays on device memory and can be automatically transferred to host if needed.
 *
 * @brief Jet tagging and pt regression with ONNX runtime inference.
 */
class TaggingNode : public stream::EDProducer<> {

public:
  TaggingNode(const edm::ParameterSet& params);
  ~TaggingNode() override = default;

  /**
   * @brief cmssw callback for node
   */
  void produce(device::Event& event, const device::EventSetup& event_setup) override;

  /**
   * @brief Declare parameters for node
   */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /**
   * @brief collect stats at the beginning of processing stream
   */
  void beginStream(edm::StreamID stream) override;

  /**
   * @brief collect stats at the end of processing stream
   */
  void endStream() override;

private:
  /**
   * @brief Tag multiclass jets and run pt regression. 
   */
  JetsCollection Tag(Queue &queue, PuppiCollection const& data, ClustersCollection const& clusters);

  // utils
  std::unique_ptr<Tagging> tagging_ = nullptr;  /**< inference runtime algorithm */
  // tokens
  const device::EDGetToken<PuppiCollection> device_data_token_;  /**< read device data ptr */
  const device::EDGetToken<ClustersCollection> device_clusters_token_;  /**< read device clusters ptr */
  device::EDPutToken<JetsCollection> device_out_token_;  /**< device write data */
  // params
  std::string model_;
  std::string backend_;
  // stats
  std::chrono::high_resolution_clock::time_point start_stamp_, end_stamp_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_TaggingNode_h
