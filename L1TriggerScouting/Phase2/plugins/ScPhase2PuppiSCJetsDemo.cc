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


#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include <algorithm>
#include <array>
#include <iostream>

class ScPhase2PuppiSCJetsDemo : public edm::stream::EDProducer<> {
public:
  explicit ScPhase2PuppiSCJetsDemo(const edm::ParameterSet &);
  ~ScPhase2PuppiSCJetsDemo() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;

  edm::EDGetTokenT<OrbitCollection<l1t::PFCandidate>> candidateToken_;
  edm::EDGetTokenT<OrbitCollection<l1Scouting::Puppi>> structToken_;

};

ScPhase2PuppiSCJetsDemo::ScPhase2PuppiSCJetsDemo(const edm::ParameterSet &iConfig) {
  // consume l1t::PFCandidate
  // TODO what to produce?
  //    - flat table of jets?
  candidateToken_ = consumes<OrbitCollection<l1t::PFCandidate>>(iConfig.getParameter<edm::InputTag>("src"));
  produces<l1ScoutingRun3::OrbitFlatTable>("SC4Jets");
}

ScPhase2PuppiSCJetsDemo::~ScPhase2PuppiSCJetsDemo(){};

void ScPhase2PuppiSCJetsDemo::beginStream(edm::StreamID) {
}

l1t::PFJet makeJet_SW(const std::vector<l1t::PFCandidate>& parts) {
  l1t::PFCandidate seed = parts.at(0);

  auto sumpt = [](float a, const l1t::PFCandidate b) { return a + b.pt(); };

  // Sum the pt
  float pt = std::accumulate(parts.begin(), parts.end(), 0., sumpt);

  // pt weighted d eta
  std::vector<float> pt_deta;
  pt_deta.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pt_deta.begin(), [&seed, &pt](const l1t::PFCandidate part) {
    return (part.pt() / pt) * (part.eta() - seed.eta());
  });
  // Accumulate the pt weighted etas. Init to the seed eta, start accumulating at begin()+1 to skip seed
  float eta = std::accumulate(pt_deta.begin() + 1, pt_deta.end(), seed.eta());

  // pt weighted d phi
  std::vector<float> pt_dphi;
  pt_dphi.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pt_dphi.begin(), [&seed, &pt](const l1t::PFCandidate part) {
    return (part.pt() / pt) * reco::deltaPhi(part.phi(), seed.phi());
  });
  // Accumulate the pt weighted phis. Init to the seed phi, start accumulating at begin()+1 to skip seed
  float phi = std::accumulate(pt_dphi.begin() + 1, pt_dphi.end(), seed.phi());

  l1t::PFJet jet(pt, eta, phi);
  /*for (auto it = parts.begin(); it != parts.end(); it++) {
    jet.addConstituent(*it);
  }*/

  return jet;
}

std::vector<l1t::PFJet> L1SeedCone_processEvent_SW(std::vector<l1t::PFCandidate>& work,
                                     unsigned nJets, float coneSize) {
  // The floating point algorithm simulation
  std::sort(work.begin(), work.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) {
    return (i.pt() > j.pt());
  });
  std::vector<l1t::PFJet> jets;
  jets.reserve(nJets);
  while (!work.empty() && jets.size() < nJets) {
    // Take the first (highest pt) candidate as a seed
    l1t::PFCandidate seed = work.at(0);
    // Get the particles within a coneSize of the seed
    std::vector<l1t::PFCandidate> particlesInCone;
    std::copy_if(
        work.begin(), work.end(), std::back_inserter(particlesInCone), [&](const l1t::PFCandidate part) {
          return reco::deltaR<l1t::PFCandidate, l1t::PFCandidate>(seed, part) <= coneSize;
        });
    jets.push_back(makeJet_SW(particlesInCone));
    // remove the clustered particles
    work.erase(std::remove_if(work.begin(),
                              work.end(),
                              [&](const l1t::PFCandidate part) {
                                return reco::deltaR<l1t::PFCandidate, l1t::PFCandidate>(seed, part) <= coneSize;
                              }),
              work.end());
  }
  return jets;
}

void ScPhase2PuppiSCJetsDemo::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {

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

    std::vector<l1t::PFCandidate> particles;
    for (unsigned i = 0; i < span.size(); i++) {
      particles.push_back(span[i]);
    }
    // SC Jets interfaces
    // std::vector<l1t::PFJet> processEvent_SW(std::vector<edm::Ptr<l1t::PFCandidate>>& parts) const;
    // std::vector<l1t::PFJet> processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& parts) const;

    std::vector<l1t::PFJet> jets = L1SeedCone_processEvent_SW(particles, 12, 0.4);
    for(auto it = jets.begin(); it != jets.end(); it++){
      pt.push_back((*it).pt());
      eta.push_back((*it).eta());
      phi.push_back((*it).phi());
    }
    
    bxOffsetsFiller.addBx(bx, jets.size()); // TODO guessing that this should contain number of objects per BX
  } // loop over BX

  //std::cout << "pt size: " << pt.size() << std::endl;
  //std::cout << "bxOffsetsFiller: " << std::endl;

  auto bxOffsets = bxOffsetsFiller.done();
  auto tab = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(bxOffsets, "SC4Jets", true);
  tab->addColumn<float>("pt", pt, "Jet pt");
  tab->addColumn<float>("eta", eta, "Jet eta");
  tab->addColumn<float>("phi", phi, "Jet phi");
  iEvent.put(std::move(tab), "SC4Jets");
}

void ScPhase2PuppiSCJetsDemo::endStream() {
}

void ScPhase2PuppiSCJetsDemo::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2PuppiSCJetsDemo);
