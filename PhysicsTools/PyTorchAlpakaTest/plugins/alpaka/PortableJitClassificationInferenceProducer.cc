#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/FwkGuards.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/ModelJitAlpaka.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/EventTimer.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/Nvtx.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;
  using namespace cms::torch;

  class PortableJitClassificationInferenceProducer : public stream::EDProducer<> {
  public:
    PortableJitClassificationInferenceProducer(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<ClassificationDeviceCollection> classification_token_;
    std::unique_ptr<torch::ModelJitAlpaka> model_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  PortableJitClassificationInferenceProducer::PortableJitClassificationInferenceProducer(edm::ParameterSet const &params)
      : EDProducer<>(params),
        particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
        classification_token_{produces()},
        model_(std::make_unique<torch::ModelJitAlpaka>(params.getParameter<edm::FileInPath>("model").fullPath())) {}

  void PortableJitClassificationInferenceProducer::produce(device::Event &event,
                                                           const device::EventSetup &event_setup) {
    Nvtx produce_range(
        fmt::format("PortableJitClassificationInferenceProducer::produce({})", event.id().event()).c_str());
    auto timer =
        EventTimer(fmt::format("PortableJitClassificationInferenceProducer({})", model_->device().str()), event);

    // in/out collections
    auto &particle_collection = const_cast<ParticleDeviceCollection &>(event.get(particles_token_));
    const auto batch_size = particle_collection.const_view().metadata().size();
    auto classification_collection = ClassificationDeviceCollection(batch_size, event.queue());
    classification_collection.zeroInitialise(event.queue());

    // records
    auto input_records = particle_collection.view().records();
    auto output_records = classification_collection.view().records();

    // input tensor definition
    SoAMetadata<ParticleSoA> inputs_metadata(batch_size);
    inputs_metadata.append_block("features", input_records.pt(), input_records.eta(), input_records.phi());

    // output tensor definition
    SoAMetadata<ClassificationSoA> outputs_metadata(batch_size);
    outputs_metadata.append_block("classes", output_records.c1(), output_records.c2());

    // metadata for automatic tensor conversion
    ModelMetadata<ParticleSoA, ClassificationSoA> metadata(inputs_metadata, outputs_metadata);

    // inference
    {
      Nvtx torchlib_range(
          fmt::format("PortableJitClassificationInferenceProducer::torchlib({})", event.id().event()).c_str());
      torch::Guard guard(event.queue());
      model_->to(event.queue(), true /**< async */);
      model_->forward(metadata);
    }

    event.emplace(classification_token_, std::move(classification_collection));
  }

  void PortableJitClassificationInferenceProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::FileInPath>("model");
    desc.add<edm::InputTag>("particles");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::PortableJitClassificationInferenceProducer);