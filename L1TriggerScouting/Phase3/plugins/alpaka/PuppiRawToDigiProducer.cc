#include <chrono>

#include "PuppiRawToDigiProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

PuppiRawToDigiProducer::PuppiRawToDigiProducer(edm::ParameterSet const& config)
  : raw_token_{consumes<SDSRawDataCollection>(config.getParameter<edm::InputTag>("src"))},
    token_{produces()},
    fed_ids_(config.getParameter<std::vector<unsigned int>>("fedIDs")) {}

void PuppiRawToDigiProducer::Summary(const long &duration) {
  std::cout << "Decoding raw to digi took: " << duration << " ms"  << std::endl;
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

PuppiCollection PuppiRawToDigiProducer::UnpackCollection(Queue &queue, const SDSRawDataCollection &raw_data) {  
  // Scan raw memory and extract mem blocks
  auto [buffer, headers] = MemoryScan<uint64_t>(raw_data);

  // Instantiate on device
  PuppiCollection collection(buffer.size(), queue);
  
  // Launch kernels wrapper
  unpacker_.ProcessHeaders(queue, headers, collection);
  unpacker_.ProcessData(queue, buffer, collection);


  // Collection stays on device afterwards
  return collection;
}

void PuppiRawToDigiProducer::produce(device::Event& event, device::EventSetup const& event_setup) {
  LogSeparator();
  auto start = Tick();

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CODE BLOCK /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  // Get data from Event
  auto raw_data_collection = event.getHandle(raw_token_);

  // Unpack collection on device
  auto product = UnpackCollection(event.queue(), *raw_data_collection);
  std::cout << "Size of PuppiCollection after decoding: " << product.view().metadata().size() << std::endl;
  PuppiHostCollection h_collection(product.view().metadata().size(), event.queue());
  alpaka::memcpy(event.queue(), h_collection.buffer(), product.const_buffer());
  alpaka::wait(event.queue());

  // Puppi collection on device
  std::cout << "\tPuppi collection on device:\n\t";
  for (uint32_t i = 5; i < h_collection.view().bx().size(); ++i) {
    std::cout << h_collection.view().bx()[i] << "; ";
    std::cout << h_collection.view().offsets()[h_collection.view().bx()[i]] << "; ";
    std::cout << h_collection.view().offsets()[h_collection.view().bx()[i+1]] << "; ";
    std::cout << h_collection.view().offsets()[h_collection.view().bx()[i+1]] - h_collection.view().offsets()[h_collection.view().bx()[i]] << "; ";
    std::cout << h_collection.view().metadata().size() << std::endl;
    break; // one line only for debugging
  }

  std::cout << std::endl;

  // Put device product into event (transferred to host automatically if needed)
  event.emplace(token_, std::move(product));

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// END CODE BLOCK /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto end = Tick();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  Summary(duration.count());
  LogSeparator();
}

void PuppiRawToDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fedIDs");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PuppiRawToDigiProducer);
