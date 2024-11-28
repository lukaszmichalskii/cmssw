#include <chrono>
#include "CombinatoricsModule.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

CombinatoricsModule::CombinatoricsModule(edm::ParameterSet const& params)
  : raw_token_(consumes(params.getParameter<edm::InputTag>("src"))),
    token_{produces()} {}

void CombinatoricsModule::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto s = std::chrono::high_resolution_clock::now();

  auto& raw_data_collection = event.get(raw_token_);
  auto product = utils_.Combinatorial(event.queue(), raw_data_collection);
  event.emplace(token_, std::move(product));

  auto e = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
  std::cout << "Combinatorics: OK [" << duration.count() << " us]" << std::endl;
}

void CombinatoricsModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(CombinatoricsModule);