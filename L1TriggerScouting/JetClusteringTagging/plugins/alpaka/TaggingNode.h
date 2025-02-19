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

class TaggingNode : public stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {

public:
  TaggingNode(const edm::ParameterSet& params, const cms::Ort::ONNXRuntime *onnx_runtime);
  ~TaggingNode() override = default;

  void produce(device::Event& event, const device::EventSetup& event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStream(edm::StreamID stream) override;
  void endStream() override;

  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &params);
  static void globalEndJob(const cms::Ort::ONNXRuntime *onnx_runtime);

private:
  const device::EDGetToken<PuppiCollection> device_data_token_;
  const device::EDGetToken<ClustersCollection> device_clusters_token_;
  std::vector<std::string> input_names_ = {"inputs"};
  std::vector<std::vector<int64_t>> input_shapes_;
  cms::Ort::FloatArrays model_data_;

  std::chrono::high_resolution_clock::time_point start_stamp_, end_stamp_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_TaggingNode_h
