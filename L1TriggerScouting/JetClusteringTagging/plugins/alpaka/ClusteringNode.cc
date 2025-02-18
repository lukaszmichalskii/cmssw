#include "ClusteringNode.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {


ClusteringNode::ClusteringNode(edm::ParameterSet const& params)
  : device_in_token_(consumes(params.getParameter<edm::InputTag>("data"))),
    device_out_token_{produces()},
    clusters_num_(params.getParameter<uint32_t>("clustersNum")) {}


void ClusteringNode::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto t1 = std::chrono::high_resolution_clock::now();

  auto const& data = event.get(device_in_token_);
  clustering_.Cluster(event.queue(), data, clusters_num_);
  auto collection = PuppiCollection(1, event.queue());
  event.emplace(device_out_token_, std::move(collection));

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Clustering: OK [" << duration.count() << " us]" << std::endl;
  std::cout << "-------------------------------------" << std::endl;
}


void ClusteringNode::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data");
  desc.add<uint32_t>("clustersNum");
  descriptions.addWithDefaultLabel(desc);
}


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(ClusteringNode);
