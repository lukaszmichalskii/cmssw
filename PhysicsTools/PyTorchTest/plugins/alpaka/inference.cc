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
#include "PhysicsTools/PyTorch/interface/common.h"
#include "PhysicsTools/PyTorch/interface/model.h"
#include "PhysicsTools/PyTorchTest/plugins/alpaka/kernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Inference : public stream::EDProducer<edm::GlobalCache<cms::torch_alpaka::Model>> {
 public:
  Inference(const edm::ParameterSet &params, const cms::torch_alpaka::Model *cache);

  static std::unique_ptr<cms::torch_alpaka::Model> initializeGlobalCache(const edm::ParameterSet &params);
  static void globalEndJob(const cms::torch_alpaka::Model *cache);
  void produce(device::Event &event, const device::EventSetup &event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:  
  const device::EDGetToken<SimpleCollection> simple_collection_get_token_;
  const device::EDPutToken<SimpleCollection> simple_collection_put_token_;
  std::shared_ptr<Kernels> kernels_ = nullptr;
  bool bind_stream_ = true;
  int64_t queue_id_ = -1;
  std::unique_ptr<cms::torch_alpaka_common::QueueGuard<Queue>> guard_ = nullptr;
};

//////////////////////////////////////////////////////////////

Inference::Inference(edm::ParameterSet const& params, const cms::torch_alpaka::Model *cache)
  : stream::EDProducer<edm::GlobalCache<cms::torch_alpaka::Model>>(params),
    simple_collection_get_token_{consumes(params.getParameter<edm::InputTag>("input"))},
    simple_collection_put_token_{produces()},
    kernels_(std::make_shared<Kernels>()),
    guard_(std::make_unique<cms::torch_alpaka_common::QueueGuard<Queue>>()) {}

std::unique_ptr<cms::torch_alpaka::Model> Inference::initializeGlobalCache(const edm::ParameterSet &param) {
  auto model_path = param.getParameter<edm::FileInPath>("modelPath").fullPath();
  return std::make_unique<cms::torch_alpaka::Model>(model_path);
}

void Inference::globalEndJob(const cms::torch_alpaka::Model *cache) {
}

void Inference::produce(device::Event &event, const device::EventSetup &event_setup) { 
  if (bind_stream_) {
    guard_->set(event.queue());
    bind_stream_ = false;
    queue_id_ = cms::torch_alpaka_tools::queue_id(event.queue());
  }
  auto curr_queue_id = cms::torch_alpaka_tools::queue_id(event.queue());
  std::cout << "(Inference) hash=" << curr_queue_id << "; cache=" << queue_id_ << std::endl;
  // assert(curr_queue_id == queue_id_);

  const auto &tmp = event.get(simple_collection_get_token_);
  const size_t batch_size = tmp.const_view().metadata().size();

  SimpleCollection input_collection(batch_size, event.queue());
  SimpleCollection collection(batch_size, event.queue());

  kernels_->FillSimpleCollection(event.queue(), input_collection, 1.0f);
  std::cout << "(Inference) kernel OK" << globalCache()->device() << std::endl;    

  std::cout << "(Inference) tensors=" << cms::torch_alpaka_tools::device(event.queue()) << std::endl;
  cms::torch_alpaka_tools::InputMetadata input_mask(cms::torch_alpaka_common::Float, 1);
  cms::torch_alpaka_tools::OutputMetadata output_mask(cms::torch_alpaka_common::Float, 1);
  cms::torch_alpaka_tools::ModelMetadata model_metadata(batch_size, input_mask, output_mask);

  std::cout << "(Inference) model=" << globalCache()->device() << std::endl;
  if (cms::torch_alpaka_tools::device(event.queue()) != globalCache()->device()) 
    globalCache()->to(cms::torch_alpaka_tools::device(event.queue()));
  std::cout << "(Inference) model=" << globalCache()->device() << std::endl;  
  assert(cms::torch_alpaka_tools::device(event.queue()) == globalCache()->device());  
  std::cout << "(Inference) inference (" << cms::torch_alpaka_tools::device(event.queue()) << ")" << globalCache()->device() << std::endl;  
  globalCache()->forward<SimpleSoA, SimpleSoA>(
    model_metadata, input_collection.buffer().data(), collection.buffer().data());

  /////////////////////////////////////////////////////////////////////////////////
  // DEBUG  
  /////////////////////////////////////////////////////////////////////////////////

  // SimpleCollectionHost input_collection_host(batch_size, cms::alpakatools::host());
  // alpaka::memcpy(event.queue(), input_collection_host.buffer(), input_collection.buffer());
  // alpaka::wait(event.queue());
  // std::cout << "(Inference) Inputs:" << std::endl;
  // std::cout << "|  x  |" << std::endl;
  // for (size_t idx = 0; idx < batch_size; idx++) {
  //   printf("| %1.1f |\n", 
  //     input_collection_host.view().x()[idx]);
  // }
  // std::cout << "|  x  |  y  |  z  |" << std::endl;
  // for (size_t idx = 0; idx < batch_size; idx++) {
  //   printf("| %1.1f | %1.1f | %1.1f |\n", 
  //     input_collection_host.view().x()[idx], input_collection_host.view().y()[idx], input_collection_host.view().z()[idx]);
  // }
    
  // SimpleCollectionHost collection_host(batch_size, cms::alpakatools::host());
  // alpaka::memcpy(event.queue(), collection_host.buffer(), collection.buffer());
  // alpaka::wait(event.queue());
  // std::cout << "(Inference) Outputs:" << std::endl;
  // std::cout << "|  x  |" << std::endl;
  // for (size_t idx = 0; idx < batch_size; idx++) {
  //   printf("| %1.1f |\n", 
  //     collection_host.view().x()[idx]);
  // }
  // std::cout << "|  x  |  y  |  z  |" << std::endl;
  // for (size_t idx = 0; idx < batch_size; idx++) {
  //   printf("| %1.1f | %1.1f | %1.1f |\n", 
  //     collection_host.view().x()[idx], collection_host.view().y()[idx], collection_host.view().z()[idx]);
  // }
  
  /////////////////////////////////////////////////////////////////////////////////
  // END DEBUG 
  /////////////////////////////////////////////////////////////////////////////////

  event.emplace(simple_collection_put_token_, std::move(collection));
  std::cout << "(Inference) OK (" <<  globalCache()->device() << ")" << std::endl;  
}

void Inference::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input");
  desc.add<edm::FileInPath>("modelPath");
  descriptions.addWithDefaultLabel(desc);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Inference);