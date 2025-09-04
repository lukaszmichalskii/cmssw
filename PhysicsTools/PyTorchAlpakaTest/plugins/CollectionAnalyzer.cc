#include <cassert>

#include <fmt/format.h>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Common.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/GetBackendTag.h"

namespace torchtest {

  using namespace torchportabletest;

  class CollectionAnalyzer : public edm::stream::EDAnalyzer<> {
  public:
    CollectionAnalyzer(const edm::ParameterSet& params)
        : particles_token_{consumes(params.getParameter<edm::InputTag>("particles"))},
          particles_backend_{consumes(torchtest::getBackendTag(params.getParameter<edm::InputTag>("particles")))},
          classification_token_{consumes(params.getParameter<edm::InputTag>("classification"))},
          classification_backend_{consumes(torchtest::getBackendTag(params.getParameter<edm::InputTag>("classification")))},
          regression_token_{consumes(params.getParameter<edm::InputTag>("regression"))},
          regression_backend_{consumes(torchtest::getBackendTag(params.getParameter<edm::InputTag>("regression")))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("particles");
      desc.add<edm::InputTag>("classification");
      desc.add<edm::InputTag>("regression");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::Event const& event, edm::EventSetup const&) override {
      auto const& particle_collection = event.get(particles_token_);
      auto const particle_collection_backend = static_cast<cms::alpakatools::Backend>(event.get(particles_backend_));

      auto const& classififcation_collection = event.get(classification_token_);
      auto const classifiacation_collection_backend = static_cast<cms::alpakatools::Backend>(event.get(classification_backend_));

      auto const& regression_collection = event.get(regression_token_);
      auto const regression_collection_backend = static_cast<cms::alpakatools::Backend>(event.get(regression_backend_));

      torchtest::printParticleCollection(particle_collection, particle_collection_backend, event);
      torchtest::printClassificationCollection(classififcation_collection, classifiacation_collection_backend, event);
      torchtest::printRegressionCollection(regression_collection, regression_collection_backend, event);
    }

  private:
    const edm::EDGetTokenT<ParticleHostCollection> particles_token_;
    const edm::EDGetTokenT<unsigned short> particles_backend_;

    const edm::EDGetTokenT<ClassificationHostCollection> classification_token_;
    const edm::EDGetTokenT<unsigned short> classification_backend_;

    const edm::EDGetTokenT<RegressionHostCollection> regression_token_;
    const edm::EDGetTokenT<unsigned short> regression_backend_;
  };

}  // namespace torchtest

DEFINE_FWK_MODULE(torchtest::CollectionAnalyzer);