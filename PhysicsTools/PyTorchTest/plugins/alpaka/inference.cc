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
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/common.h"
#include "PhysicsTools/PyTorch/interface/model.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Inference : public global::EDProducer<> {
 public:
 Inference(const edm::ParameterSet &params);

  void produce(edm::StreamID sid, device::Event &event, const device::EventSetup &event_setup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<SimpleCollection> simple_collection_get_token_;
  const device::EDPutToken<SimpleCollection> simple_collection_put_token_;
  const std::string model_path_;
  cms::torch_alpaka::Model model_;
};

//////////////////////////////////////////////////////////////

Inference::Inference(edm::ParameterSet const& params)
  : EDProducer<>(params),
    simple_collection_get_token_{consumes(params.getParameter<edm::InputTag>("input"))},
    simple_collection_put_token_{produces()},
    model_path_(params.getParameter<edm::FileInPath>("modelPath").fullPath()),
    model_(cms::torch_alpaka::Model(params.getParameter<edm::FileInPath>("modelPath").fullPath())) {}

void Inference::produce(edm::StreamID sid, device::Event &event, const device::EventSetup &event_setup) const {  
  // auto quard = cms::torch_alpaka_common::QueueGuard<Queue>(); 
  // quard.set(event.queue());
  const auto dev = cms::torch_alpaka_tools::device(event.queue());
  const auto dev_model = model_.device();
  assert(dev == dev_model);

  const auto &tmp = event.get(simple_collection_get_token_);
  const size_t batch_size = tmp.const_view().metadata().size();

  SimpleCollection input_collection(batch_size, event.queue());
  SimpleCollection collection(batch_size, event.queue());

  cms::torch_alpaka_tools::InputMetadata input_mask(cms::torch_alpaka_common::Float, 3);
  cms::torch_alpaka_tools::OutputMetadata output_mask(cms::torch_alpaka_common::Float, 3);
  cms::torch_alpaka_tools::ModelMetadata model_metadata(batch_size, input_mask, output_mask);

  model_.forward<SimpleSoA, SimpleSoA>(
    model_metadata, input_collection.buffer().data(), collection.buffer().data());
    
  SimpleCollectionHost collection_host(batch_size, cms::alpakatools::host());
  alpaka::memcpy(event.queue(), collection_host.buffer(), collection.buffer());
  alpaka::wait(event.queue());

  std::cout << "|  x  |  y  |  z  |" << std::endl;
  for (size_t idx = 0; idx < batch_size; idx++) {
    printf("| %1.1f | %1.1f | %1.1f |\n", 
      collection_host.view().x()[idx], collection_host.view().y()[idx], collection_host.view().z()[idx]);
  }
  
  event.emplace(simple_collection_put_token_, std::move(collection));
  // quard.reset();
}

void Inference::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input");
  desc.add<edm::FileInPath>("modelPath");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Inference);