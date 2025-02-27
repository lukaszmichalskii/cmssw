#include "L1TriggerScouting/JetClusteringTagging/plugins/alpaka/TaggingNode.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

TaggingNode::TaggingNode(edm::ParameterSet const& params)
  : device_data_token_(consumes(params.getParameter<edm::InputTag>("data"))),
    device_clusters_token_(consumes(params.getParameter<edm::InputTag>("clusters"))),
    device_out_token_{produces()},
    model_(params.getParameter<edm::FileInPath>("model").fullPath()),
    backend_(params.getParameter<std::string>("backend")) {
  tagging_ = std::make_unique<Tagging>(model_, backend_);
}

void TaggingNode::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto t1 = std::chrono::high_resolution_clock::now();

  auto const& data = event.get(device_data_token_);
  auto const& clusters = event.get(device_clusters_token_);
  auto jets = Tag(event.queue(), data, clusters);
  event.emplace(device_out_token_, std::move(jets));

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Tagging (" << backend_ << "): OK [" << duration.count() << " us]" << std::endl;
  std::cout << "-------------------------------------" << std::endl;
}  

void TaggingNode::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data");
  desc.add<edm::InputTag>("clusters");
  desc.add<edm::FileInPath>("model");
  desc.add<std::string>("backend");
  descriptions.addWithDefaultLabel(desc);
}

void TaggingNode::beginStream(edm::StreamID) {
  std::cout << "=====================================" << std::endl;
  start_stamp_ = std::chrono::high_resolution_clock::now();
}

void TaggingNode::endStream() {
  end_stamp_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_stamp_ - start_stamp_);
  std::cout << "JetClusteringTagging (" << duration.count() << " ms)" << std::endl;
  std::cout << "=====================================" << std::endl;
}

JetsCollection TaggingNode::Tag(Queue &queue, PuppiCollection const& data, ClustersCollection const& clusters) {
  // auto jets = JetsCollection(data.const_view().metadata().size(), queue);
  auto jets = JetsCollection(1486848*2, queue);
  tagging_->Tag(queue, data, clusters, jets);
  return jets;
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TaggingNode);
