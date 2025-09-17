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
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/EventTimer.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/Nvtx.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/MapAlpakaBackend.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;
  using namespace cms::torch::alpakatools;

  class ParticleClassificationProducer : public stream::EDProducer<> {
  public:
    ParticleClassificationProducer(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          classification_token_{produces()},
          model_(std::make_unique<torch::AlpakaModel>(params.getParameter<edm::FileInPath>("model").fullPath())),
          verbose_{params.getUntrackedParameter<bool>("verbose")} {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      QueueGuard<Queue> guard(event.queue());
      Nvtx produce_range(
        fmt::format("Classification::produce(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      auto timer =
          EventTimer(fmt::format("ParticleClassificationProducer({})", kAlpakaBackend).c_str(), event, verbose_);

      // in/out collections
      Nvtx alloc_range(
          fmt::format("Classification::malloc(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      // TODO: hide const_cast from end-user code
      auto &particle_collection = const_cast<ParticleDeviceCollection &>(event.get(particles_token_));
      const auto batch_size = particle_collection.const_view().metadata().size();
      auto classification_collection = ClassificationDeviceCollection(batch_size, event.queue());
      classification_collection.zeroInitialise(event.queue());
      alloc_range.end();

      // records
      auto input_records = particle_collection.view().records();
      auto output_records = classification_collection.view().records();

      // input tensor definition
      Nvtx metadata_range(
          fmt::format("Classification::metadata(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      SoAMetadata<ParticleSoA> inputs_metadata(batch_size);
      inputs_metadata.append_block("features", input_records.pt(), input_records.eta(), input_records.phi());

      // output tensor definition
      SoAMetadata<ClassificationSoA> outputs_metadata(batch_size);
      outputs_metadata.append_block("classes", output_records.c1(), output_records.c2());

      // metadata for automatic tensor conversion
      ModelMetadata<ParticleSoA, ClassificationSoA> metadata(inputs_metadata, outputs_metadata);
      metadata_range.end();

      // inference
      Nvtx torchlib_range(
          fmt::format("Classification::torchlib(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      // santity check 
      // assert(QueueHash<Queue>::alpakaQueue(event.queue()) == QueueHash<Queue>::pytorchQueue(event.queue()));
      model_->to(event.queue());
      model_->forward(event.queue(), metadata);
      
      event.emplace(classification_token_, std::move(classification_collection));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<edm::InputTag>("particles");
      desc.addUntracked<bool>("verbose", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<ClassificationDeviceCollection> classification_token_;
    std::unique_ptr<torch::AlpakaModel> model_;
    const bool verbose_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::ParticleClassificationProducer);