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
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// inference 
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class TaggingNode : public stream::EDProducer<> {

public:
  TaggingNode(const edm::ParameterSet& params);
  ~TaggingNode() override = default;

  void produce(device::Event& event, const device::EventSetup& event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStream(edm::StreamID stream) override;
  void endStream() override;

private:
  const device::EDGetToken<PuppiCollection> device_data_token_;
  const device::EDGetToken<ClustersCollection> device_clusters_token_;

  std::string model_;
  std::string backend_;
  std::unique_ptr<Ort::Env> env_ = nullptr;
  std::unique_ptr<Ort::Session> session_ = nullptr;
  std::unique_ptr<Ort::RunOptions> options_ = nullptr;

  std::vector<std::string> input_node_strings_;
  std::vector<const char*> input_node_names_;
  std::map<std::string, std::vector<int64_t>> input_node_dims_;

  std::vector<std::string> output_node_strings_;
  std::vector<const char*> output_node_names_;
  std::map<std::string, std::vector<int64_t>> output_node_dims_;

  std::chrono::high_resolution_clock::time_point start_stamp_, end_stamp_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_TaggingNode_h
