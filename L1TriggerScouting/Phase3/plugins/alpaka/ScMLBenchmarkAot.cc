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

  class ScMLBenchmarkAot : public stream::EDProducer<edm::GlobalCache<AotModel>> {
  public:
    ScMLBenchmarkAot(const edm::ParameterSet &params, const AotModel *cache);

    static std::unique_ptr<AotModel> initializeGlobalCache(const edm::ParameterSet &params);
    static void globalEndJob(const AotModel *cache);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const uint32_t batch_size_;
  };

  ScMLBenchmarkAot::ScMLBenchmarkAot(edm::ParameterSet const &params, const AotModel *cache)
      : EDProducer<edm::GlobalCache<AotModel>>(params),
        batch_size_(params.getParameter<uint32_t>("batchSize")) {}

  std::unique_ptr<AotModel> ScMLBenchmarkAot::initializeGlobalCache(const edm::ParameterSet &param) {
    auto model_path = param.getParameter<edm::FileInPath>("modelPath").fullPath();
    return std::make_unique<AotModel>(model_path);
  }

  void ScMLBenchmarkAot::globalEndJob(const AotModel *cache) {}

  void ScMLBenchmarkAot::produce(device::Event &event, const device::EventSetup &event_setup) {
    cms::torch::alpaka::Guard<Queue> guard(event.queue());
    // sanity check
    assert(cms::torch::alpaka::queue_hash(event.queue()) == cms::torch::alpaka::current_stream_hash(event.queue()));

    auto inputs = torchportable::InputCollection(batch_size_, event.queue());
    auto outputs = torchportable::OutputCollection(batch_size_, event.queue());

    // metadata for automatic tensor conversion
    auto input_records = inputs.view().records();
    auto output_records = outputs.view().records();

    cms::torch::alpaka::SoAMetadata<torchportable::InputSoA> inputs_metadata(batch_size_);
    inputs_metadata.append_block("features", 
        input_records.f1(), 
        input_records.f2(), 
        input_records.f3(),
        input_records.f4(),
        input_records.f5(),
        input_records.f6(),
        input_records.f7(),
        input_records.f8(),
        input_records.f9(),
        input_records.f10(),
        input_records.f11(),
        input_records.f12(),
        input_records.f13(),
        input_records.f14(),
        input_records.f15(),
        input_records.f16());

    cms::torch::alpaka::SoAMetadata<torchportable::OutputSoA> outputs_metadata(batch_size_);
    outputs_metadata.append_block("preds", 
        output_records.c1(), 
        output_records.c2(),
        output_records.c3(),
        output_records.c4(),
        output_records.c5(),
        output_records.c6(),
        output_records.c7(),
        output_records.c8(),
        output_records.c9(),
        output_records.c10());

    cms::torch::alpaka::ModelMetadata<torchportable::InputSoA, torchportable::OutputSoA> metadata(
        inputs_metadata, outputs_metadata);

    if (cms::torch::alpaka::device(event.queue()) != globalCache()->device())
      globalCache()->to(event.queue());
    assert(cms::torch::alpaka::device(event.queue()) == globalCache()->device());
    alpaka::wait(event.queue());

    // inference
    auto t1 = std::chrono::high_resolution_clock::now();
    globalCache()->forward(metadata);
    alpaka::wait(event.queue());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "(AotInference) OK - " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              << " us" << std::endl;
  }

  void ScMLBenchmarkAot::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("batchSize");
    desc.add<edm::FileInPath>("modelPath");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(ScMLBenchmarkAot);