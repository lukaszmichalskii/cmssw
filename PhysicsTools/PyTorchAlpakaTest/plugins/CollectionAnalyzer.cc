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
          particles_backend_{consumes(torchtest::getBackendTag(params.getParameter<edm::InputTag>("particles")))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("particles");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::Event const& event, edm::EventSetup const&) override {
      auto const& particle_collection = event.get(particles_token_);
      auto const particle_collection_backend = static_cast<cms::alpakatools::Backend>(event.get(particles_backend_));
      
      torchtest::printParticleCollection(particle_collection, particle_collection_backend, event);
    }

  private:
    const edm::EDGetTokenT<ParticleHostCollection> particles_token_;
    const edm::EDGetTokenT<unsigned short> particles_backend_;
  };

}  // namespace torchtest

DEFINE_FWK_MODULE(torchtest::CollectionAnalyzer);