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
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/EventTimer.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/Nvtx.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/RandomCollectionFillingKernel.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/MapAlpakaBackend.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;

  class PortableCollectionProducer : public stream::EDProducer<> {
  public:
    PortableCollectionProducer(const edm::ParameterSet &params)
        : EDProducer<>(params), 
          particles_token_{produces()}, 
          batch_size_(params.getParameter<uint32_t>("batchSize")),
          verbose_{params.getUntrackedParameter<bool>("verbose")} {}


    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      Nvtx produce_range(fmt::format("PortableCollectionProducer::produce(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      auto timer = EventTimer(fmt::format("PortableCollectionProducer ({})", kAlpakaBackend).c_str(), event, verbose_);
      auto collection = ParticleDeviceCollection(batch_size_, event.queue());
      randomFillParticleCollection(event.queue(), collection);
      event.emplace(particles_token_, std::move(collection));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<uint32_t>("batchSize");
      desc.addUntracked<bool>("verbose", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<ParticleDeviceCollection> particles_token_;
    const uint32_t batch_size_;
    const bool verbose_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::PortableCollectionProducer);