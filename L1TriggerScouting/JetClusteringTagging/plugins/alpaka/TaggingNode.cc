#include "L1TriggerScouting/JetClusteringTagging/plugins/alpaka/TaggingNode.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

TaggingNode::TaggingNode(edm::ParameterSet const& params, const cms::Ort::ONNXRuntime *onnx_runtime)
  : device_data_token_(consumes(params.getParameter<edm::InputTag>("data"))),
    device_clusters_token_(consumes(params.getParameter<edm::InputTag>("clusters"))),
    input_shapes_() {
  model_data_.emplace_back(10, 0);
}

std::unique_ptr<cms::Ort::ONNXRuntime> TaggingNode::initializeGlobalCache(const edm::ParameterSet &params) {
  auto sess_opts = cms::Ort::ONNXRuntime::defaultSessionOptions(cms::Ort::Backend::cpu);
  return std::make_unique<cms::Ort::ONNXRuntime>(
    params.getParameter<edm::FileInPath>("model").fullPath(), &sess_opts);
}

void TaggingNode::globalEndJob(const cms::Ort::ONNXRuntime *cache) {}

void TaggingNode::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto t1 = std::chrono::high_resolution_clock::now();

  auto const& data = event.get(device_data_token_);
  auto const& clusters = event.get(device_clusters_token_);

  std::vector<float> &group_data = model_data_[0];
  for (size_t i = 0; i < 10; i++){
      group_data[i] = float(i);
  }

  // run prediction and get outputs
  std::vector<float> outputs = globalCache()->run(input_names_, model_data_, input_shapes_)[0];

  // print the input and output data
  std::cout << "input data -> ";
  for (auto &i: group_data) { std::cout << i << " "; }
  std::cout << std::endl << "output data -> ";
  for (auto &i: outputs) { std::cout << i << " "; }
  std::cout << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Tagging: OK [" << duration.count() << " us]" << std::endl;
  std::cout << "-------------------------------------" << std::endl;
}  

void TaggingNode::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data");
  desc.add<edm::InputTag>("clusters");
  desc.add<edm::FileInPath>("model");
  descriptions.addWithDefaultLabel(desc);
}

void TaggingNode::beginStream(edm::StreamID) {
  std::cout << "=====================================" << std::endl;
  start_stamp_ = std::chrono::high_resolution_clock::now();
}

void TaggingNode::endStream() {
  end_stamp_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_stamp_ - start_stamp_);
  std::cout << "-------------------------------------" << std::endl;
  std::cout << "JetClusteringTagging (" << duration.count() << " ms)" << std::endl;
  std::cout << "=====================================" << std::endl;
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TaggingNode);
