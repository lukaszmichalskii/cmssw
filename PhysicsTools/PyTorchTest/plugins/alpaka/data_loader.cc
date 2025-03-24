#include <alpaka/alpaka.hpp>
#include <cstdint>

#include "DataFormats/PyTorchTest/interface/alpaka/torch_alpaka.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class DataLoader : public global::EDProducer<> {
 public:
  DataLoader(const edm::ParameterSet &params);

  void produce(edm::StreamID sid, device::Event &event, const device::EventSetup &event_setup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDPutToken<SimpleCollection> simple_collection_put_token_;
};

//////////////////////////////////////////////////////////////

DataLoader::DataLoader(edm::ParameterSet const& params)
  : EDProducer<>(params),
    simple_collection_put_token_{produces()} {}

void DataLoader::produce(edm::StreamID sid, device::Event &event, const device::EventSetup &event_setup) const {
  SimpleCollection collection(10, event.queue());
  collection.zeroInitialise(event.queue());
  event.emplace(simple_collection_put_token_, std::move(collection));
}

void DataLoader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(DataLoader);