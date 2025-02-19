#include "L1TriggerScouting/JetClusteringTagging/plugins/alpaka/ClusteringNode.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {


ClusteringNode::ClusteringNode(edm::ParameterSet const& params)
  : device_in_token_(consumes(params.getParameter<edm::InputTag>("data"))),
    device_out_token_{produces()},
    clusters_num_(params.getParameter<uint32_t>("clustersNum")) {}


ClustersCollection ClusteringNode::Cluster(Queue& queue, PuppiCollection const& data) {
  auto t1 = std::chrono::high_resolution_clock::now();

  auto clusters_associations = ClustersCollection(data.const_view().metadata().size(), queue);
  clustering_.Cluster(queue, data, clusters_associations, clusters_num_);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Clustering: OK [" << duration.count() << " us]" << std::endl;
  std::cout << "-------------------------------------" << std::endl;

  return clusters_associations;
}    


void ClusteringNode::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto const& data = event.get(device_in_token_);
  auto clusters = Cluster(event.queue(), data);
  event.emplace(device_out_token_, std::move(clusters));
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
