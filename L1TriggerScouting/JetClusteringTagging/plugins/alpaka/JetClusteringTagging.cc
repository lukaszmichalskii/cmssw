#include "JetClusteringTagging.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

JetClusteringTagging::JetClusteringTagging(edm::ParameterSet const& params)
  : raw_token_{consumes<SDSRawDataCollection>(params.getParameter<edm::InputTag>("src"))},
    fed_ids_(params.getParameter<std::vector<uint32_t>>("fedIDs")),
    clusters_num_(params.getParameter<uint32_t>("clustersNum")) {}

void JetClusteringTagging::unpacking(Queue &queue, const SDSRawDataCollection &raw_data) {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "-------------------------------------" << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////////

  std::vector<uint64_t> headers_buffer, buffer;

  for (auto &fed_id : fed_ids_) {
    const auto &src = raw_data.FEDData(fed_id);
    const auto chunk_begin = reinterpret_cast<const uint64_t*>(src.data());
    const auto chunk_end = reinterpret_cast<const uint64_t*>(src.data() + src.size());

    for (auto ptr = chunk_begin; ptr != chunk_end;) {
      if (*ptr == 0) 
        continue;

      headers_buffer.insert(headers_buffer.end(), ptr, ptr + 1);
      auto chunk_size = (*ptr) & 0xFFF;
      ptr++;
      buffer.insert(buffer.end(), ptr, ptr + chunk_size);
      ptr += chunk_size; 
    }
  }

  data_ = PuppiCollection(buffer.size(), queue);
  unpacking_.Unpack(queue, headers_buffer, buffer, data_);

  //////////////////////////////////////////////////////////////////////////////////////////

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Unpack: OK [" << duration.count() << " us]" << std::endl;
} 

void JetClusteringTagging::clustering(Queue &queue) {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "-------------------------------------" << std::endl;
  
  //////////////////////////////////////////////////////////////////////////////////////////

  clustering_.Cluster(queue, data_, clusters_num_);

  //////////////////////////////////////////////////////////////////////////////////////////

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Clustering: OK [" << duration.count() << " us]" << std::endl;
} 

void JetClusteringTagging::tagging(Queue &queue) {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "-------------------------------------" << std::endl;
  
  //////////////////////////////////////////////////////////////////////////////////////////

  tagging_.Tag(queue, data_);

  //////////////////////////////////////////////////////////////////////////////////////////

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Tagging: OK [" << duration.count() << " us]" << std::endl;
} 

void JetClusteringTagging::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto raw_data_collection = event.getHandle(raw_token_);
  unpacking(event.queue(), *raw_data_collection);
  clustering(event.queue());
  tagging(event.queue());
}  

void JetClusteringTagging::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<uint32_t>>("fedIDs");
  desc.add<uint32_t>("clustersNum");
  descriptions.addWithDefaultLabel(desc);
}

void JetClusteringTagging::beginStream(edm::StreamID) {
  std::cout << "=====================================" << std::endl;
  start_stamp_ = std::chrono::high_resolution_clock::now();
}

void JetClusteringTagging::endStream() {
  end_stamp_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_stamp_ - start_stamp_);
  std::cout << "-------------------------------------" << std::endl;
  std::cout << "JetClusteringTagging (" << duration.count() << " ms)" << std::endl;
  std::cout << "=====================================" << std::endl;
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(JetClusteringTagging);
