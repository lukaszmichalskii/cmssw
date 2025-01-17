#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitFlatTable.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/L1ScoutingPuppi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "L1TriggerScouting/Utilities/interface/BxOffsetsFiller.h"
#include "fastjet/ClusterSequence.hh"

#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include <algorithm>
#include <array>
#include <iostream>

class ScPhase2PuppiAKJetsDemo : public edm::stream::EDProducer<> {
public:
  explicit ScPhase2PuppiAKJetsDemo(const edm::ParameterSet &);
  ~ScPhase2PuppiAKJetsDemo() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;

  edm::EDGetTokenT<OrbitCollection<l1t::PFCandidate>> candidateToken_;
  edm::EDGetTokenT<OrbitCollection<l1Scouting::Puppi>> structToken_;

};

ScPhase2PuppiAKJetsDemo::ScPhase2PuppiAKJetsDemo(const edm::ParameterSet &iConfig) {
  // consume l1t::PFCandidate
  // produce flat table of jets
  candidateToken_ = consumes<OrbitCollection<l1t::PFCandidate>>(iConfig.getParameter<edm::InputTag>("src"));
  produces<l1ScoutingRun3::OrbitFlatTable>("AK4Jets");
}

ScPhase2PuppiAKJetsDemo::~ScPhase2PuppiAKJetsDemo(){};

void ScPhase2PuppiAKJetsDemo::beginStream(edm::StreamID) {
}

void ScPhase2PuppiAKJetsDemo::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {

  using namespace fastjet;

  l1ScoutingRun3::BxOffsetsFillter bxOffsetsFiller;
  bxOffsetsFiller.start();

  edm::Handle<OrbitCollection<l1t::PFCandidate>> src;
  iEvent.getByToken(candidateToken_, src);

  // containers for output products
  // jet 3 momenta
  std::vector<float> pt;
  std::vector<float> eta;
  std::vector<float> phi;

  for (unsigned int bx = 1; bx <= OrbitCollection<l1t::PFCandidate>::NBX; ++bx) {
    auto span = (*src).bxIterator(bx);
    //const l1t::PFCandidate *cands = &span.front();
    //auto size = span.size();

    //std::vector<l1t::PFCandidate> particles;
    std::vector<PseudoJet> particles;
    for (unsigned i = 0; i < span.size(); i++) {
      particles.push_back(PseudoJet(span[i].px(), span[i].py(), span[i].pz(), span[i].energy()));
    }

    // choose a jet definition
    double R = 0.4;
    JetDefinition jet_def(antikt_algorithm, R);
    // run the clustering, extract the jets
    ClusterSequence cs(particles, jet_def);
    vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets());    

    for(auto it = jets.begin(); it != jets.end(); it++){
      pt.push_back((*it).pt());
      eta.push_back((*it).eta());
      phi.push_back((*it).phi());
    }
    
    // add the number of jets to the BX offset
    bxOffsetsFiller.addBx(bx, jets.size());
  } // loop over BX

  auto bxOffsets = bxOffsetsFiller.done();
  auto tab = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(bxOffsets, "AK4Jets", true);
  tab->addColumn<float>("pt", pt, "Jet pt");
  tab->addColumn<float>("eta", eta, "Jet eta");
  tab->addColumn<float>("phi", phi, "Jet phi");
  iEvent.put(std::move(tab), "AK4Jets");
}

void ScPhase2PuppiAKJetsDemo::endStream() {
}

void ScPhase2PuppiAKJetsDemo::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2PuppiAKJetsDemo);
