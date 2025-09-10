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
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/RecoMergeKernel.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/MapAlpakaBackend.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;

  class ReconstructionMergeProducer : public stream::EDProducer<> {
  public:
    ReconstructionMergeProducer(const edm::ParameterSet &params)
        : EDProducer<>(params), 
          classification_token_(consumes(params.getParameter<edm::InputTag>("classification"))),
          regression_token_(consumes(params.getParameter<edm::InputTag>("regression"))),
          reco_token_{produces()},
          verbose_{params.getUntrackedParameter<bool>("verbose")} {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      Nvtx produce_range(fmt::format("Reco::produce(event: {}, stream: {}, device: {}, queue: {})", event.id().event(), static_cast<int>(event.streamID().value()), formatDevice(event.device()), QueueHash<Queue>::alpakaQueue(event.queue())).c_str());
      auto timer = EventTimer(fmt::format("ReconstructionMergeProducer ({})", kAlpakaBackend).c_str(), event, verbose_);

      auto &regression_collection = event.get(regression_token_);
      auto &classification_collection = event.get(classification_token_);
      auto collection = ReconstructionDeviceCollection(regression_collection.const_view().metadata().size(), event.queue());
      collection.zeroInitialise(event.queue());

      merge(event.queue(), collection, classification_collection, regression_collection);

      event.emplace(reco_token_, std::move(collection));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("classification");
      desc.add<edm::InputTag>("regression");
      desc.addUntracked<bool>("verbose", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<ClassificationDeviceCollection> classification_token_;
    const device::EDGetToken<RegressionDeviceCollection> regression_token_;
    const device::EDPutToken<ReconstructionDeviceCollection> reco_token_;
    const bool verbose_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::ReconstructionMergeProducer);