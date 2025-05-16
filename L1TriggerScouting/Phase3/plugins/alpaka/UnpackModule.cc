#include <chrono>
#include "UnpackModule.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

UnpackModule::UnpackModule(edm::ParameterSet const& config)
  : raw_token_{consumes<SDSRawDataCollection>(config.getParameter<edm::InputTag>("src"))},
    token_{produces()},
    fed_ids_(config.getParameter<std::vector<unsigned int>>("fedIDs")) {}

template<typename T>
std::tuple<std::vector<T>, std::vector<T>> UnpackModule::MemoryScan(const SDSRawDataCollection &raw_data) {
  std::vector<T> headers_buffer;
  std::vector<T> buffer; 

  for (auto &fed_id : fed_ids_) {
    const auto &src = raw_data.FEDData(fed_id);
    const auto chunk_begin = reinterpret_cast<const T*>(src.data());
    const auto chunk_end = reinterpret_cast<const T*>(src.data() + src.size());

    for (auto ptr = chunk_begin; ptr != chunk_end;) {
      if (*ptr == 0) {
        ptr++;
        continue;
      }

      headers_buffer.insert(headers_buffer.end(), ptr, ptr + 1);
      // Header readout
      auto chunk_size = (*ptr) & 0xFFF; // number of trailing words
      ptr++; // shift ptr from header bits
      buffer.insert(buffer.end(), ptr, ptr + chunk_size);
      ptr += chunk_size; // shift ptr to process next header
    }
  }
  return {std::move(buffer), std::move(headers_buffer)};
}

PuppiCollection UnpackModule::UnpackCollection(Queue &queue, const SDSRawDataCollection &raw_data) {  
  auto [buffer, headers] = MemoryScan<uint64_t>(raw_data);
  PuppiCollection collection(buffer.size(), queue);
  utils_.Unpacking(queue, headers, buffer, collection);
  return collection;
}

void UnpackModule::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto s = std::chrono::high_resolution_clock::now();

  auto raw_data_collection = event.getHandle(raw_token_);
  auto product = UnpackCollection(event.queue(), *raw_data_collection);
  event.emplace(token_, std::move(product));

  auto e = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
  std::cout << "-------------------------------------" << std::endl;
  std::cout << "I/O (H2D): OK [" << duration.count() << " us]" << std::endl;
}

void UnpackModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fedIDs");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(UnpackModule);
