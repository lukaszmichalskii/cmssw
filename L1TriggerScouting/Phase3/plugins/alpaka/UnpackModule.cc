#include <chrono>
#include "UnpackModule.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

UnpackModule::UnpackModule(edm::ParameterSet const& config)
  : raw_token_{consumes<SDSRawDataCollection>(config.getParameter<edm::InputTag>("src"))},
    token_{produces()},
    fed_ids_(config.getParameter<std::vector<unsigned int>>("fedIDs")) {}

template<typename T>
std::tuple<std::vector<T>, std::vector<T>> UnpackModule::MemoryScan(const SDSRawDataCollection &raw_data) {
  std::vector<T> headers_buffer(3564);
  std::vector<T> buffer;
  buffer.reserve(120000); // up to 120k stubs

  size_t header_idx = 0;

  for (auto &fed_id : fed_ids_) {
    const auto &src = raw_data.FEDData(fed_id);
    const auto chunk_begin = reinterpret_cast<const T*>(src.data());
    const auto chunk_end = reinterpret_cast<const T*>(src.data() + src.size());

    for (auto ptr = chunk_begin; ptr < chunk_end;) {
      if (*ptr == 0) {
        ++ptr;
        continue;
      }

      if (header_idx < 3564) [[likely]] {
        headers_buffer[header_idx++] = *ptr;
      }

      auto chunk_size = (*ptr) & 0xFFF;
      ++ptr;

      const size_t remaining = chunk_end - ptr;
      const size_t copy_count = std::min<size_t>(chunk_size, remaining);

      buffer.insert(buffer.end(), ptr, ptr + copy_count);
      ptr += copy_count;
    }
  }

  return {std::move(buffer), std::move(headers_buffer)};
}

PuppiCollection UnpackModule::UnpackCollection(Queue &queue, const SDSRawDataCollection &raw_data) {  
  // auto s1 = std::chrono::high_resolution_clock::now();
  auto [buffer, headers] = MemoryScan<uint64_t>(raw_data);
  // std::cout << buffer.size() << ", " << headers.size() << std::endl;
  // auto e1 = std::chrono::high_resolution_clock::now();
  // auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1);
  // std::cout << "I/O (memscan): OK [" << duration1.count() << " us]" << std::endl;

  // auto s2 = std::chrono::high_resolution_clock::now();
  PuppiCollection collection(buffer.size(), queue);
  // alpaka::wait(queue);
  // auto e2 = std::chrono::high_resolution_clock::now();
  // auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2);
  // std::cout << "I/O (malloc): OK [" << duration2.count() << " us]" << std::endl;

  // auto s3 = std::chrono::high_resolution_clock::now();
  utils_.Unpacking(queue, headers, buffer, collection);
  // alpaka::wait(queue);
  // auto e3 = std::chrono::high_resolution_clock::now();
  // auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(e3 - s3);
  // std::cout << "I/O (kernels): OK [" << duration3.count() << " us]" << std::endl;
  return collection;
}

void UnpackModule::produce(device::Event& event, device::EventSetup const& event_setup) {
  std::cout << "-------------------------------------" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();

  auto raw_data_collection = event.getHandle(raw_token_);
  auto product = UnpackCollection(event.queue(), *raw_data_collection);
  event.emplace(token_, std::move(product));
  alpaka::wait(event.queue());

  auto e = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
  // std::cout << "-------------------------------------" << std::endl;
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
