#include "JetClusteringTagging.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

JetClusteringTagging::JetClusteringTagging(edm::ParameterSet const& params)
  : raw_token_{consumes<SDSRawDataCollection>(params.getParameter<edm::InputTag>("src"))} {}

void JetClusteringTagging::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto raw_data_collection = event.getHandle(raw_token_);
}  

void JetClusteringTagging::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fedIDs");
  descriptions.addWithDefaultLabel(desc);
}

void JetClusteringTagging::beginStream(edm::StreamID) {
  start_stamp_ = std::chrono::high_resolution_clock::now();
}

void JetClusteringTagging::endStream() {
  end_stamp_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_stamp_ - start_stamp_);
  std::cout << "OK: JetClusteringTagging " << duration.count() << " ms" << std::endl;
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(JetClusteringTagging);
