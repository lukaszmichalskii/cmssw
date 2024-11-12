#include <chrono>

#include "IsolationModule.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

IsolationModule::IsolationModule(edm::ParameterSet const& params)
  : raw_token_(consumes(params.getParameter<edm::InputTag>("src"))),
    token_{produces()} {}

void IsolationModule::Summary(const long &duration) {
  std::cout << "Isolation module took: " << duration << " ms"  << std::endl;
}

std::chrono::high_resolution_clock::time_point IsolationModule::Tick() {
  return std::chrono::high_resolution_clock::now();
}

void IsolationModule::LogSeparator() {
  std::cout << std::endl;
  std::cout << "===============================================================" << std::endl;
  std::cout << std::endl;
}

PuppiCollection IsolationModule::Isolate(Queue &queue, PuppiCollection const& raw_collection) {  
  return isolation_.Isolate(queue, raw_collection);
}

void IsolationModule::produce(device::Event& event, device::EventSetup const& event_setup) {
  LogSeparator();
  auto start = Tick();

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CODE BLOCK /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto& raw_data_collection = event.get(raw_token_);
  auto product = Isolate(event.queue(), raw_data_collection);
  std::cout << "Size of PuppiCollection after isolation: " << product.view().metadata().size() << std::endl;
  event.emplace(token_, std::move(product));

  //////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// END CODE BLOCK /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////

  auto end = Tick();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  Summary(duration.count());
  LogSeparator();
}

void IsolationModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(IsolationModule);
