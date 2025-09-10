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

  class ParticleRegressionProducer : public stream::EDProducer<> {
  public:
    ParticleRegressionProducer(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          regression_token_{produces()},
          model_(std::make_unique<torch::AlpakaModel>(params.getParameter<edm::FileInPath>("model").fullPath())),
          verbose_{params.getUntrackedParameter<bool>("verbose")} {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      Nvtx produce_range(
        fmt::format("Regression::produce(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      auto timer =
          EventTimer(fmt::format("ParticleRegressionProducer({})", kAlpakaBackend).c_str(), event, verbose_);

      // in/out collections
      Nvtx alloc_range(
          fmt::format("Regression::malloc(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      // TODO: hide const_cast from end-user code
      auto &particle_collection = const_cast<ParticleDeviceCollection &>(event.get(particles_token_));
      const auto batch_size = particle_collection.const_view().metadata().size();
      auto regression_collection = RegressionDeviceCollection(batch_size, event.queue());
      regression_collection.zeroInitialise(event.queue());
      alloc_range.end();

      // records
      auto input_records = particle_collection.view().records();
      auto output_records = regression_collection.view().records();

      // input tensor definition
      Nvtx metadata_range(
          fmt::format("Regression::metadata(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      SoAMetadata<ParticleSoA> inputs_metadata(batch_size);
      inputs_metadata.append_block("features", input_records.pt(), input_records.eta(), input_records.phi());

      // output tensor definition
      SoAMetadata<RegressionSoA> outputs_metadata(batch_size);
      outputs_metadata.append_block("preds", output_records.reco_pt());

      // metadata for automatic tensor conversion
      ModelMetadata<ParticleSoA, RegressionSoA> metadata(inputs_metadata, outputs_metadata);
      metadata_range.end();

      // inference
      {
        Nvtx torchlib_range(
            fmt::format("Regression::torchlib(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
        QueueGuard<Queue> guard(event.queue());
        // santity check 
        assert(QueueHash<Queue>::alpakaQueue(event.queue()) == QueueHash<Queue>::pytorchQueue(event.queue()));
        model_->to(event.queue());
        model_->forward(metadata);
      }

      event.emplace(regression_token_, std::move(regression_collection));
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
    const device::EDPutToken<RegressionDeviceCollection> regression_token_;
    std::unique_ptr<torch::AlpakaModel> model_;
    const bool verbose_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::ParticleRegressionProducer);