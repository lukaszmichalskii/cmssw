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
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/EventTimer.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/Nvtx.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/RandomCollectionFillingKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;

  class PortableCollectionProducer : public stream::EDProducer<> {
  public:
    PortableCollectionProducer(const edm::ParameterSet &params)
        : EDProducer<>(params), 
          particles_token_{produces()}, 
          batch_size_(params.getParameter<uint32_t>("batchSize")) {}


    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      Nvtx produce_range(fmt::format("PortableCollectionProducer::produce({})", event.id().event()).c_str());
      auto timer = EventTimer(fmt::format("PortableCollectionProducer ({})", alpaka::getName(alpaka::getDev(event.queue()))).c_str(), event);
      auto collection = ParticleDeviceCollection(batch_size_, event.queue());
      randomFillParticleCollection(event.queue(), collection);
      event.emplace(particles_token_, std::move(collection));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<uint32_t>("batchSize");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<ParticleDeviceCollection> particles_token_;
    const uint32_t batch_size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::PortableCollectionProducer);