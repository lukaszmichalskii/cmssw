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

  class PortableJitRegressionInferenceProducer : public stream::EDProducer<> {
  public:
    PortableJitRegressionInferenceProducer(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<ParticleDeviceCollection> particles_token_;
    const device::EDPutToken<RegressionDeviceCollection> regression_token_;
    std::unique_ptr<torch::ModelJitAlpaka> model_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  PortableJitRegressionInferenceProducer::PortableJitRegressionInferenceProducer(edm::ParameterSet const &params)
      : EDProducer<>(params),
        particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
        regression_token_{produces()},
        model_(std::make_unique<torch::ModelJitAlpaka>(params.getParameter<edm::FileInPath>("model").fullPath())) {}

  void PortableJitRegressionInferenceProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
    Nvtx produce_range(fmt::format("PortableJitRegressionInferenceProducer::produce({})", event.id().event()).c_str());
    auto timer = EventTimer(fmt::format("PortableJitRegressionInferenceProducer({})", model_->device().str()), event);

    // in/out collections
    auto &particle_collection = const_cast<ParticleDeviceCollection &>(event.get(particles_token_));
    const auto batch_size = particle_collection.const_view().metadata().size();
    auto regression_collection = RegressionDeviceCollection(batch_size, event.queue());
    regression_collection.zeroInitialise(event.queue());

    // records
    auto input_records = particle_collection.view().records();
    auto output_records = regression_collection.view().records();

    // input tensor definition
    SoAMetadata<ParticleSoA> inputs_metadata(batch_size);
    inputs_metadata.append_block("features", input_records.pt(), input_records.eta(), input_records.phi());

    // output tensor definition
    SoAMetadata<RegressionSoA> outputs_metadata(batch_size);
    outputs_metadata.append_block("preds", output_records.reco_pt());

    // metadata for automatic tensor conversion
    // ModelMetadata<ParticleSoA, RegressionSoA> metadata(inputs_metadata, outputs_metadata);

    // inference
    {
      Nvtx torchlib_range(
          fmt::format("PortableJitRegressionInferenceProducer::torchlib({})", event.id().event()).c_str());
      torch::Guard guard(event.queue());
      model_->to(event.queue(), true /**< async */);
    }

    event.emplace(regression_token_, std::move(regression_collection));
  }

  void PortableJitRegressionInferenceProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::FileInPath>("model");
    desc.add<edm::InputTag>("particles");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::PortableJitRegressionInferenceProducer);