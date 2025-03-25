#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

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
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchTest/plugins/alpaka/kernels.h"
#include "PhysicsTools/PyTorch/interface/common.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Inference2 : public stream::EDProducer<> {
 public:
 Inference2(const edm::ParameterSet &params);

  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<SimpleCollection> simple_collection_get_token_;
  const device::EDPutToken<SimpleCollection> simple_collection_put_token_;
  std::shared_ptr<Kernels> kernels_ = nullptr;
};

//////////////////////////////////////////////////////////////

Inference2::Inference2(edm::ParameterSet const& params)
  : stream::EDProducer<>(params),
    simple_collection_get_token_{consumes(params.getParameter<edm::InputTag>("input"))},
    simple_collection_put_token_{produces()},
    kernels_(std::make_shared<Kernels>()) {}

void Inference2::produce(device::Event &event, const device::EventSetup &event_setup) { 
  auto tid = cms::torch_alpaka_tools::queue_id(event.queue());
  std::cout << "(Inference2) hash=" << tid << std::endl;

  std::cout << "(Inference2) backend=" << cms::torch_alpaka_tools::device(event.queue()) << std::endl;    
  const auto &input_collection = event.get(simple_collection_get_token_);
  const size_t size = input_collection.const_view().metadata().size();
  SimpleCollection collection(size, event.queue());
  kernels_->FillSimpleCollection(event.queue(), collection, 0.87f);
  std::cout << "(Inference2) kernel OK" << cms::torch_alpaka_tools::device(event.queue()) << std::endl;    
  
  /////////////////////////////////////////////////////////////////////////////////
  // DEBUG  
  /////////////////////////////////////////////////////////////////////////////////

//   SimpleCollectionHost input_collection_host(size, cms::alpakatools::host());
//   alpaka::memcpy(event.queue(), input_collection_host.buffer(), input_collection.buffer());
//   alpaka::wait(event.queue());
//   std::cout << "(Inference2) Inputs:" << std::endl;
//   std::cout << "|  x  |" << std::endl;
//   for (size_t idx = 0; idx < size; idx++) {
//     printf("| %1.1f |\n", 
//       input_collection_host.view().x()[idx]);
//   }
  // std::cout << "|  x  |  y  |  z  |" << std::endl;
  // for (size_t idx = 0; idx < size; idx++) {
  //   printf("| %1.1f | %1.1f | %1.1f |\n", 
  //     input_collection_host.view().x()[idx], input_collection_host.view().y()[idx], input_collection_host.view().z()[idx]);
  // }
    
//   SimpleCollectionHost collection_host(size, cms::alpakatools::host());
//   alpaka::memcpy(event.queue(), collection_host.buffer(), collection.buffer());
//   alpaka::wait(event.queue());
//   std::cout << "(Inference2) Outputs:" << std::endl;
//   std::cout << "|  x  |" << std::endl;
//   for (size_t idx = 0; idx < size; idx++) {
//     printf("| %1.1f |\n", 
//       collection_host.view().x()[idx]);
//   }
//   std::cout << "|  x  |  y  |  z  |" << std::endl;
  // for (size_t idx = 0; idx < size; idx++) {
  //   printf("| %1.1f | %1.1f | %1.1f |\n", 
  //     collection_host.view().x()[idx], collection_host.view().y()[idx], collection_host.view().z()[idx]);
  // }
  
  /////////////////////////////////////////////////////////////////////////////////
  // END DEBUG 
  /////////////////////////////////////////////////////////////////////////////////

  event.emplace(simple_collection_put_token_, std::move(collection));
  std::cout << "(Inference2) OK (" << cms::torch_alpaka_tools::device(event.queue()) << ")" << std::endl;  
}

void Inference2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Inference2);