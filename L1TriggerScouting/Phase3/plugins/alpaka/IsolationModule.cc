#include <chrono>

#include "IsolationModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

IsolationModule::IsolationModule(edm::ParameterSet const& params)
  : raw_token_(consumes(params.getParameter<edm::InputTag>("src"))),
    token_{produces()} {}

void IsolationModule::Summary(const long &duration) {
  std::cout << "W3PI OK -> " << product_ << " (" << duration << " us)"  << std::endl;
}

std::chrono::high_resolution_clock::time_point IsolationModule::Tick() {
  return std::chrono::high_resolution_clock::now();
}

void IsolationModule::LogSeparator() {
  std::cout << std::endl;
  std::cout << "===============================================================" << std::endl;
  std::cout << std::endl;
}

size_t IsolationModule::Isolate(Queue &queue, PuppiCollection const& raw_collection) {  
  return isolation_.Isolate(queue, raw_collection);
}

void IsolationModule::produce(device::Event& event, device::EventSetup const& event_setup) {
  // LogSeparator();
  auto start = Tick();

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CODE BLOCK /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto& raw_data_collection = event.get(raw_token_);
  product_ = 0;
  product_ = Isolate(event.queue(), raw_data_collection);
  w3pi_num_ += raw_data_collection.view().bx().size();
  w3pi_results_ += product_;
  event.emplace(token_, std::move(PuppiCollection(1, event.queue())));

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// END CODE BLOCK /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto end = Tick();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  Summary(duration.count());
  LogSeparator();
}

void IsolationModule::beginStream(edm::StreamID) {
  w3pi_results_ = 0;
  w3pi_num_ = 0;
  start_ = std::chrono::high_resolution_clock::now();
}

void IsolationModule::endStream() {
  end_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
  std::cout << "W3PI: " << w3pi_num_ << " -> " << w3pi_results_ << " (" << duration.count() << " ms)" << std::endl;
}


void IsolationModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(IsolationModule);
