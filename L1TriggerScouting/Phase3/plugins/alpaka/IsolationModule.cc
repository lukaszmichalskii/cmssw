#include <chrono>
#include "IsolationModule.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

IsolationModule::IsolationModule(edm::ParameterSet const& params)
  : raw_token_(consumes(params.getParameter<edm::InputTag>("src"))),
    token_{produces()} {}

void IsolationModule::produce(device::Event& event, device::EventSetup const& event_setup) {

  auto& raw_data_collection = const_cast<PuppiCollection&>(event.get(raw_token_));
  w3pi_candidates += raw_data_collection.view().bx().size();
  // std::cout << "Size of pariticles: " << raw_data_collection.view().metadata().size() << std::endl;

  ///////////////////////////////////////////////////////////////////////////////////////////// 
  // Analysis
  /////////////////////////////////////////////////////////////////////////////////////////////
  auto s = std::chrono::high_resolution_clock::now();

  utils_.Isolate(event.queue(), raw_data_collection);
  alpaka::wait(event.queue());

  auto e = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
  std::cout << "Analysis: OK [" << duration.count() << " us]" << std::endl;

  ///////////////////////////////////////////////////////////////////////////////////////////// 
  // I/O -> D2H
  /////////////////////////////////////////////////////////////////////////////////////////////
  auto s2 = std::chrono::high_resolution_clock::now();

  PuppiHostCollection host_data(raw_data_collection.view().metadata().size(), event.queue());
  alpaka::memcpy(event.queue(), host_data.buffer(), raw_data_collection.buffer());
  event.emplace(token_, std::move(raw_data_collection));
  alpaka::wait(event.queue());

  auto e2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2);
  std::cout << "I/O (D2H): OK [" << duration2.count() << " us]" << std::endl; 

  for (size_t idx = 0; idx < host_data.view().bx().size(); ++idx) {
    auto span_begin = host_data.view().offsets()[idx];
    auto span_end = host_data.view().offsets()[idx + 1];
    if (span_end - span_begin == 0)
      continue;
    int ct = 0;  
    for (uint32_t i = span_begin; i < span_end; ++i) {
      ct += host_data.view().selection()[i];
    }
    if (ct > 0) {
      w3pi_ct_ = w3pi_ct_ + static_cast<uint32_t>(ct / 3);
    }
  }
}

void IsolationModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  descriptions.addWithDefaultLabel(desc);
}

void IsolationModule::beginStream(edm::StreamID) {
  start_ = std::chrono::high_resolution_clock::now();
  std::cout << "=====================================" << std::endl;
}

void IsolationModule::endStream() {
  end_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
  std::cout << "-------------------------------------" << std::endl;
  std::cout << "W3Pi: " << w3pi_candidates << " -> " << w3pi_ct_ << "; (" << duration.count() << " ms)" << std::endl;
  std::cout << "=====================================" << std::endl;
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(IsolationModule);
