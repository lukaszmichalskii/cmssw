#include <chrono>

#include "PuppiRawToDigiProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

PuppiRawToDigiProducer::PuppiRawToDigiProducer(edm::ParameterSet const& config)
  : raw_token_{consumes<SDSRawDataCollection>(config.getParameter<edm::InputTag>("src"))},
    token_{produces()},
    fed_ids_(config.getParameter<std::vector<unsigned int>>("fed_ids")) {}

void PuppiRawToDigiProducer::Summary(const long &duration) {
  std::cout << "Parameters: " << std::endl;
  std::cout << "--fed_ids = (" << fed_ids_.size() << ") ";
  if (!fed_ids_.empty()) {
    for (const auto& fed_id : fed_ids_)
      std::cout << fed_id << "; ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Processing took: " << duration << " ns" << std::endl;
}

std::chrono::high_resolution_clock::time_point PuppiRawToDigiProducer::Tick() {
  return std::chrono::high_resolution_clock::now();
}

void PuppiRawToDigiProducer::LogSeparator() {
  std::cout << std::endl;
  std::cout << "===============================================================" << std::endl;
  std::cout << std::endl;
}

template<typename T>
std::tuple<std::vector<T>, std::vector<T>> PuppiRawToDigiProducer::MemoryScan(const SDSRawDataCollection &raw_data) {
  std::vector<T> headers_buffer;
  std::vector<T> buffer; 

  for (auto &fed_id : fed_ids_) {
    const auto &src = raw_data.FEDData(fed_id);
    const auto chunk_begin = reinterpret_cast<const T*>(src.data());
    const auto chunk_end = reinterpret_cast<const T*>(src.data() + src.size());

    for (auto ptr = chunk_begin; ptr != chunk_end;) {
      if (*ptr == 0) 
        continue;

      headers_buffer.insert(headers_buffer.end(), ptr, ptr + 1);
      // Header readout
      auto chunk_size = (*ptr) & 0xFFF; // number of trailing words
      ptr++; // shift ptr from header bits
      buffer.insert(buffer.end(), ptr, ptr + chunk_size);
      ptr += chunk_size; // shift ptr to process next header
    }
  }
  return std::make_tuple(std::move(buffer), std::move(headers_buffer));
}

std::unique_ptr<PuppiCollection> PuppiRawToDigiProducer::UnpackCollection(Queue &queue, const SDSRawDataCollection &raw_data) {  
  // Scan raw memory and extract mem blocks
  auto [buffer, headers] = MemoryScan<uint64_t>(raw_data);

  std::cout << "\tHeaders: " << headers.size() << std::endl;
  for (size_t idx = 0; idx < headers.size(); idx++) {
    std::bitset<64> bit64h(headers[idx]);
    std::cout << "\t" << bit64h << std::endl;
    if (idx == 5)
      break;
  }
  std::cout << std::endl;

  std::cout << "\tParticles: " << buffer.size() << std::endl;
  for (size_t idx = 0; idx < buffer.size(); idx++) {
    std::bitset<64> bit64p(buffer[idx]);
    std::cout << "\t" << bit64p << std::endl;
    if (idx == 5)
      break;
  }
  std::cout << std::endl;

  // Instantiate on device
  PuppiCollection collection(buffer.size(), queue);
  unpacker_.ProcessHeaders(queue, headers, collection);
  unpacker_.ProcessData(queue, buffer, collection);

  PuppiHostCollection host_collection(collection.view().metadata().size(), queue);
  alpaka::memcpy(queue, host_collection.buffer(), collection.const_buffer());
  alpaka::wait(queue);

  for (int32_t idx = 0; idx < host_collection.view().metadata().size(); idx++) {
    std::cout << "\t" << host_collection.view().bx()[idx] << "; ";
    std::cout << host_collection.view().offsets()[idx] << "; ";
    std::cout << host_collection.view()[idx].pt() << "; ";
    std::cout << host_collection.view()[idx].eta() << "; ";
    std::cout << host_collection.view()[idx].phi() << "; ";
    std::cout << host_collection.view()[idx].z0() << "; ";
    std::cout << host_collection.view()[idx].dxy() << "; ";
    std::cout << host_collection.view()[idx].puppiw() << "; ";
    std::cout << host_collection.view()[idx].pdgId() << "; ";
    std::cout << static_cast<int>(host_collection.view()[idx].quality()) << "; ";
    std::cout << std::endl;

    if (idx == 3570)
      break;
  }

  return std::make_unique<PuppiCollection>(std::move(collection));
}

void PuppiRawToDigiProducer::produce(device::Event& event, device::EventSetup const& event_setup) {
  LogSeparator();
  auto start = Tick();

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CODE BLOCK /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto raw_data_collection = event.getHandle(raw_token_);
  event.put(token_, UnpackCollection(event.queue(), *raw_data_collection));

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// END CODE BLOCK /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto end = Tick();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  
  Summary(duration.count());
  LogSeparator();
}

void PuppiRawToDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fed_ids");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PuppiRawToDigiProducer);
