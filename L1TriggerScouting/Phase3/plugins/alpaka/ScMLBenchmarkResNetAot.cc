#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorch/interface/Nvtx.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using AotModel = cms::torch::alpaka::Model<cms::torch::alpaka::CompilationType::kAheadOfTime>;

  class ScMLBenchmarkResNetAot : public stream::EDProducer<edm::GlobalCache<AotModel>> {
  public:
    ScMLBenchmarkResNetAot(const edm::ParameterSet &params, const AotModel *cache);

    static std::unique_ptr<AotModel> initializeGlobalCache(const edm::ParameterSet &params);
    static void globalEndJob(const AotModel *cache);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const uint32_t batch_size_;
  };

  ScMLBenchmarkResNetAot::ScMLBenchmarkResNetAot(edm::ParameterSet const &params, const AotModel *cache)
      : EDProducer<edm::GlobalCache<AotModel>>(params),
        batch_size_(params.getParameter<uint32_t>("batchSize")) {}

  std::unique_ptr<AotModel> ScMLBenchmarkResNetAot::initializeGlobalCache(const edm::ParameterSet &param) {
    auto model_path = param.getParameter<edm::FileInPath>("modelPath").fullPath();
    return std::make_unique<AotModel>(model_path);
  }

  void ScMLBenchmarkResNetAot::globalEndJob(const AotModel *cache) {}

  void ScMLBenchmarkResNetAot::produce(device::Event &event, const device::EventSetup &event_setup) {
    cms::torch::alpaka::Guard<Queue> guard(event.queue());
    // sanity check
    assert(cms::torch::alpaka::queue_hash(event.queue()) == cms::torch::alpaka::current_stream_hash(event.queue()));
    if (cms::torch::alpaka::device(event.queue()) != globalCache()->device())
      globalCache()->to(event.queue());
    assert(cms::torch::alpaka::device(event.queue()) == globalCache()->device());

    auto inputs = torchportable::ResNetInputCollection(batch_size_, event.queue());
    auto outputs = torchportable::ResNetOutputCollection(batch_size_, event.queue());
    alpaka::wait(event.queue());
    
    // metadata for automatic tensor conversion
    auto input_records = inputs.view().records();
    auto output_records = outputs.view().records();

    cms::torch::alpaka::SoAMetadata<torchportable::ResNetInputSoA> inputs_metadata(batch_size_);
    inputs_metadata.append_block("features", input_records.r(), input_records.g(), input_records.b());

    cms::torch::alpaka::SoAMetadata<torchportable::ResNetOutputSoA> outputs_metadata(batch_size_);
    outputs_metadata.append_block("probs", output_records.probs());

    cms::torch::alpaka::ModelMetadata<torchportable::ResNetInputSoA, torchportable::ResNetOutputSoA> metadata(
        inputs_metadata, outputs_metadata);
   
    // inference
    alpaka::wait(event.queue());
    auto t1 = std::chrono::high_resolution_clock::now();
    globalCache()->forward(metadata);
    alpaka::wait(event.queue());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "(AotResNetInference) OK - " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              << " us" << std::endl;
  }

  void ScMLBenchmarkResNetAot::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("batchSize");
    desc.add<edm::FileInPath>("modelPath");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(ScMLBenchmarkResNetAot);