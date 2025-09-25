#include "PhysicsTools/PyTorchAlpakaTest/interface/ResNetSoA.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/ResNetDeviceCollection.h"
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

  class ResNetInferenceProducer : public stream::EDProducer<> {
  public:
    ResNetInferenceProducer(const edm::ParameterSet &params)
        : EDProducer<>(params),
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          verbose_{params.getUntrackedParameter<bool>("verbose")},
          batch_size_(params.getParameter<uint32_t>("batchSize")) {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      QueueGuard<Queue> guard(event.queue());
      Nvtx produce_range(
        fmt::format("ResNet::produce(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      auto timer =
          EventTimer(fmt::format("ResNetInferenceProducer({})", kAlpakaBackend).c_str(), event, verbose_);

      // in/out collections
      Nvtx alloc_range(
          fmt::format("ResNet::malloc(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());

      auto images = ImageDeviceCollection(batch_size_, event.queue());
      auto logits = LogitsDeviceCollection(batch_size_, event.queue());
      images.zeroInitialise(event.queue());
      logits.zeroInitialise(event.queue());
      alloc_range.end();

      // records
      auto image_records = images.view().records();
      auto logits_records = logits.view().records();

      // input tensor definition
      Nvtx metadata_range(
          fmt::format("ResNet::metadata(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      SoAMetadata<Image> inputs_metadata(batch_size_);
      inputs_metadata.append_block("image", image_records.r(), image_records.g(), image_records.b());

      // output tensor definition
      SoAMetadata<Logits> outputs_metadata(batch_size_);
      outputs_metadata.append_block("logits", logits_records.logits());

      // metadata for automatic tensor conversion
      ModelMetadata<Image, Logits> metadata(inputs_metadata, outputs_metadata);
      metadata_range.end();

      // inference
      Nvtx torchlib_range(
          fmt::format("ResNet::torchlib(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      // santity check 
      // assert(QueueHash<Queue>::alpakaQueue(event.queue()) == QueueHash<Queue>::pytorchQueue(event.queue()));
      model_.to(event.queue());
      model_.forward(event.queue(), metadata);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("batchSize");
      desc.addUntracked<bool>("verbose", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    torch::AlpakaModel model_;
    const bool verbose_;
    const uint32_t batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::ResNetInferenceProducer);