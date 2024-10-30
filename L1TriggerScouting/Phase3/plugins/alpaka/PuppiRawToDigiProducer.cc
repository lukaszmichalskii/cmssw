#include <chrono>

#include "PuppiRawToDigiProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

PuppiRawToDigiProducer::PuppiRawToDigiProducer(edm::ParameterSet const& config)
  : raw_token_{consumes<SDSRawDataCollection>(config.getParameter<edm::InputTag>("src"))},
    fed_ids_(config.getParameter<std::vector<unsigned int>>("fed_ids")),
    token_{produces()}, 
    size_{config.getParameter<int32_t>("size")} {}

void PuppiRawToDigiProducer::Summary(const long &duration) {
  std::cout << "Parameters: " << std::endl;
  std::cout << "--size = " << size_ << std::endl;
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

void PuppiRawToDigiProducer::produce(device::Event& event, device::EventSetup const& event_setup) {
  LogSeparator();
  auto start = Tick();

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CODE BLOCK /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto raw_data_collection = event.getHandle(raw_token_);
  auto d_collection = std::make_unique<PuppiCollection>(size_, event.queue());
  // run the algorithm, potentially asynchronously
  unpacker_.Fill(event.queue(), *d_collection, 1);
  unpacker_.Assert(event.queue(), *d_collection, 1);
  event.put(token_, std::move(d_collection));

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
  desc.add<int32_t>("size", 1);
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fed_ids");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PuppiRawToDigiProducer);
