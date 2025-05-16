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
#include "DataFormats/L1TParticleFlow/interface/L1ScoutingPuppi.h"
#include "L1TriggerScouting/Utilities/interface/BxOffsetsFiller.h"

#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include <algorithm>
#include <array>
#include <iostream>

class ScPhase2PuppiW3PiDemo : public edm::stream::EDProducer<> {
public:
  explicit ScPhase2PuppiW3PiDemo(const edm::ParameterSet &);
  ~ScPhase2PuppiW3PiDemo() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;
  template <typename T>
  void runObj(const OrbitCollection<T> &src,
              edm::Event &out,
              unsigned long &nTry,
              unsigned long &nPass,
              const std::string &bxLabel);
  void runSOA(const l1Scouting::PuppiSOA &src, edm::Event &out);

  bool doCandidate_, doStruct_, doSOA_;
  edm::EDGetTokenT<OrbitCollection<l1t::PFCandidate>> candidateToken_;
  edm::EDGetTokenT<OrbitCollection<l1Scouting::Puppi>> structToken_;
  edm::EDGetTokenT<l1Scouting::PuppiSOA> soaToken_;

  std::chrono::high_resolution_clock::time_point start_, end_;

  struct Cuts {
    float minpt1 = 7;   // 9
    float minpt2 = 12;  // 15
    float minpt3 = 15;  // 20
    float mindeltar2 = 0.5 * 0.5;
    float minmass = 60;   // 60
    float maxmass = 100;  // 100
    float mindr2 = 0.01 * 0.01;
    float maxdr2 = 0.25 * 0.25;
    float maxiso = 2.0;  //0.4
  } cuts;

  template <typename T>
  bool isolation(unsigned int pidex, const T *cands, unsigned int size) const;

  template <typename T>
  bool isolation(unsigned int pidex, const T *cands, unsigned int size, unsigned int &cache) const {
    if (cache == 0)
      cache = isolation(pidex, cands, size) ? 1 : 2;
    return (cache == 1);
  }

  bool isolation(unsigned int pidex,
                 unsigned int npx,
                 const float *eta,
                 const float *phi,
                 const float *pt,
                 unsigned int &cache) const {
    if (cache == 0)
      cache = isolation(pidex, npx, eta, phi, pt) ? 1 : 2;
    return (cache == 1);
  }
  bool isolation(unsigned int pidex, unsigned int npx, const float *eta, const float *phi, const float *pt) const;
  bool deltar(float eta1, float eta2, float phi1, float phi2) const;
  static float tripletmass(const std::array<unsigned int, 3> &t, const float *pts, const float *etas, const float *phis);

  unsigned long countCandidate_, countStruct_, countSOA_;
  unsigned long passCandidate_, passStruct_, passSOA_;
};

ScPhase2PuppiW3PiDemo::ScPhase2PuppiW3PiDemo(const edm::ParameterSet &iConfig)
    : doCandidate_(iConfig.getParameter<bool>("runCandidate")),
      doStruct_(iConfig.getParameter<bool>("runStruct")),
      doSOA_(iConfig.getParameter<bool>("runSOA")) {
  if (doCandidate_) {
    candidateToken_ = consumes<OrbitCollection<l1t::PFCandidate>>(iConfig.getParameter<edm::InputTag>("src"));
    produces<std::vector<unsigned>>("selectedBxCandidate");
    produces<l1ScoutingRun3::OrbitFlatTable>("w3piCandidate");
  }
  if (doStruct_) {
    structToken_ = consumes<OrbitCollection<l1Scouting::Puppi>>(iConfig.getParameter<edm::InputTag>("src"));
    produces<std::vector<unsigned>>("selectedBx");
    produces<l1ScoutingRun3::OrbitFlatTable>("w3pi");
  }
  if (doSOA_) {
    soaToken_ = consumes<l1Scouting::PuppiSOA>(iConfig.getParameter<edm::InputTag>("src"));
    produces<l1Scouting::PuppiSOA>();
  }
}

ScPhase2PuppiW3PiDemo::~ScPhase2PuppiW3PiDemo(){};

void ScPhase2PuppiW3PiDemo::beginStream(edm::StreamID) {
  countCandidate_ = 0;
  countStruct_ = 0;
  countSOA_ = 0;
  passCandidate_ = 0;
  passStruct_ = 0;
  passSOA_ = 0;
  // start_ = std::chrono::high_resolution_clock::now();
  // std::cout << "=====================================" << std::endl;
}

void ScPhase2PuppiW3PiDemo::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (doCandidate_) {
    edm::Handle<OrbitCollection<l1t::PFCandidate>> src;
    iEvent.getByToken(candidateToken_, src);
    runObj(*src, iEvent, countCandidate_, passCandidate_, "Candidate");
  }
  if (doStruct_) {
    // auto t = std::chrono::high_resolution_clock::now();
    edm::Handle<OrbitCollection<l1Scouting::Puppi>> src;
    iEvent.getByToken(structToken_, src);
    runObj(*src, iEvent, countStruct_, passStruct_, "");
    // std::cout << "Analysis: OK [" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t).count() << " us]" << std::endl;
  }
  if (doSOA_) {
    // auto t = std::chrono::high_resolution_clock::now();
    edm::Handle<l1Scouting::PuppiSOA> src;
    iEvent.getByToken(soaToken_, src);
    runSOA(*src, iEvent);
    // std::cout << "Analysis: OK [" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t).count() << " us]" << std::endl;
  }
}

void ScPhase2PuppiW3PiDemo::endStream() {
  end_ = std::chrono::high_resolution_clock::now();
  if (doCandidate_)
    edm::LogImportant("ScPhase2AnalysisSummary")
        << "W3Pi Candidate analysis: " << countCandidate_ << " -> " << passCandidate_;
  if (doStruct_) {
    auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    std::cout << "-------------------------------------" << std::endl;
    edm::LogImportant("ScPhase2AnalysisSummary") << "W3Pi Struct analysis: " << countStruct_ << " -> " << passStruct_;
    std::cout << "=====================================" << std::endl;
  }
  if (doSOA_) {
    auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    std::cout << "-------------------------------------" << std::endl;
    edm::LogImportant("ScPhase2AnalysisSummary") << "W3Pi: " << countSOA_ << " -> " << passSOA_ << " (" << t.count() << " ms)";
    std::cout << "=====================================" << std::endl;
  }
}

template <typename T>
void ScPhase2PuppiW3PiDemo::runObj(const OrbitCollection<T> &src,
                                   edm::Event &iEvent,
                                   unsigned long &nTry,
                                   unsigned long &nPass,
                                   const std::string &label) {
  auto s = std::chrono::high_resolution_clock::now();
  l1ScoutingRun3::BxOffsetsFillter bxOffsetsFiller;
  bxOffsetsFiller.start();
  auto ret = std::make_unique<std::vector<unsigned>>();
  std::vector<float> masses;
  std::vector<uint8_t> i0s, i1s, i2s;
  ROOT::RVec<unsigned int> ix;   // pions
  ROOT::RVec<unsigned int> iso;  //stores whether a particle passes isolation test so we don't calculate reliso twice
  std::array<unsigned int, 3> bestTriplet;  // best triplet
  float bestTripletScore, bestTripletMass;
  for (unsigned int bx = 1; bx <= OrbitCollection<T>::NBX; ++bx) {
    nTry++;
    auto range = src.bxIterator(bx);
    const T *cands = &range.front();
    auto size = range.size();
    ix.clear();
    int intermediatecut = 0;
    int highcut = 0;
    for (unsigned int i = 0; i < size; ++i) {  //make list of all hadrons
      if ((std::abs(cands[i].pdgId()) == 211 or std::abs(cands[i].pdgId()) == 11)) {
        if (cands[i].pt() >= cuts.minpt1) {
          ix.push_back(i);
          if (cands[i].pt() >= cuts.minpt2)
            intermediatecut++;
          if (cands[i].pt() >= cuts.minpt3)
            highcut++;
        }
      }
    }
    unsigned int npions = ix.size();
    if (highcut < 1 || intermediatecut < 2 || npions < 3)
      continue;
    iso.resize(npions);
    std::fill(iso.begin(), iso.end(), 0);
    bestTripletScore = 0;

    for (unsigned int i1 = 0; i1 < npions; ++i1) {
      if (cands[ix[i1]].pt() < cuts.minpt3)
        continue;  //high pt cut
      if (isolation(ix[i1], cands, size, iso[i1]) == 0)
        continue;  //check iso of high pt pion
      for (unsigned int i2 = 0; i2 < npions; ++i2) {
        if (i2 == i1 || cands[ix[i2]].pt() < cuts.minpt2)
          continue;
        if (cands[ix[i2]].pt() > cands[ix[i1]].pt() || (cands[ix[i2]].pt() == cands[ix[i1]].pt() and i2 < i1))
          continue;  //intermediate pt cut
        if (!deltar(cands[ix[i1]].eta(), cands[ix[i2]].eta(), cands[ix[i1]].phi(), cands[ix[i2]].phi()))
          continue;  //angular sep of top 2 pions
        for (unsigned int i3 = 0; i3 < npions; ++i3) {
          if (i3 == i1 or i3 == i2)
            continue;
          if (cands[ix[i2]].pt() < cuts.minpt1)
            continue;  //low pt cut
          if (cands[ix[i3]].pt() > cands[ix[i1]].pt() || (cands[ix[i3]].pt() == cands[ix[i1]].pt() and i3 < i1))
            continue;
          if (cands[ix[i3]].pt() > cands[ix[i2]].pt() || (cands[ix[i3]].pt() == cands[ix[i2]].pt() and i3 < i2))
            continue;
          std::array<unsigned int, 3> tr{{ix[i1], ix[i2], ix[i3]}};  //triplet of indeces

          if (std::abs(cands[ix[i1]].charge() + cands[ix[i2]].charge() + cands[ix[i3]].charge()) == 1) {
            //make Lorentz vectors for each triplet
            auto mass = (cands[ix[i1]].p4() + cands[ix[i2]].p4() + cands[ix[i3]].p4()).mass();
            if (mass >= cuts.minmass and mass <= cuts.maxmass) {  //MASS test
              if (deltar(cands[ix[i1]].eta(), cands[ix[i3]].eta(), cands[ix[i1]].phi(), cands[ix[i3]].phi()) and
                  deltar(cands[ix[i2]].eta(), cands[ix[i3]].eta(), cands[ix[i2]].phi(), cands[ix[i3]].phi())) {
                //ISOLATION test for lower 4 pions
                bool isop = isolation(ix[i2], cands, size, iso[i2]) && isolation(ix[i3], cands, size, iso[i3]);
                if (isop == true) {
                  float ptsum = cands[ix[i1]].pt() + cands[ix[i2]].pt() + cands[ix[i3]].pt();
                  nPass++;
                  if (ptsum > bestTripletScore) {
                    std::copy_n(tr.begin(), 3, bestTriplet.begin());
                    bestTripletScore = ptsum;
                    bestTripletMass = mass;
                  }
                }  // iso
              }    // delta R
            }      // mass
          }        //charge
        }          //low pt cut
      }            //intermediate pt cut
    }              //high pt cut

    if (bestTripletScore > 0) {
      ret->emplace_back(bx);
      // nPass++;
      masses.push_back(bestTripletMass);
      i0s.push_back(bestTriplet[0]);
      i1s.push_back(bestTriplet[1]);
      i2s.push_back(bestTriplet[2]);
      bxOffsetsFiller.addBx(bx, 1);
    }
  }  // loop on BXs
  auto e = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
  std::cout << "Analysis: OK [" << duration.count() << " us]" << std::endl;

  auto s2 = std::chrono::high_resolution_clock::now();
  iEvent.put(std::move(ret), "selectedBx" + label);
  // now we make the table
  auto bxOffsets = bxOffsetsFiller.done();
  auto tab = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(bxOffsets, "W3Pi" + label, true);
  tab->addColumn<float>("mass", masses, "3-pion invariant mass");
  tab->addColumn<uint8_t>("i0", i0s, "leading pion");
  tab->addColumn<uint8_t>("i1", i1s, "subleading pion");
  tab->addColumn<uint8_t>("i2", i2s, "trailing pion");
  iEvent.put(std::move(tab), "w3pi" + label);
  auto e2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2);
  std::cout << "I/O (D2H): OK [" << duration2.count() << " us]" << std::endl;
}

void ScPhase2PuppiW3PiDemo::runSOA(const l1Scouting::PuppiSOA &src, edm::Event &iEvent) {
  auto s = std::chrono::high_resolution_clock::now();
  auto ret = std::make_unique<l1Scouting::PuppiSOA>();
  std::vector<uint32_t> &offsets = ret->offsets;
  offsets.push_back(0);
  ROOT::RVec<unsigned int> ix;   // pions
  ROOT::RVec<unsigned int> iso;  //stores whether a particle passes isolation test so we don't calculate reliso twice
  ROOT::RVec<int> charge;        //stores whether a particle passes isolation test so we don't calculate reliso twice
  std::array<unsigned int, 3> bestTriplet;  // best triplet
  float bestTripletScore;
  int global_int_cut = 0;
  int global_high_cut = 0;
  int global_l1_pass = 0;

  size_t ctr = 0;
  size_t passed = 0;
  auto full = 0;
  int ct = 0;
  std::vector<int> times;
  for (unsigned int ibx = 0, nbx = src.bx.size(); ibx < nbx; ++ibx) {
    // if (ibx != 3526)
    //   continue;
    countSOA_++;
    auto se = std::chrono::high_resolution_clock::now();
    unsigned int offs = src.offsets[ibx];
    unsigned int size = src.offsets[ibx + 1] - offs;
    const float *pts = &src.pt[offs];
    const float *etas = &src.eta[offs];
    const float *phis = &src.phi[offs];
    const int16_t *pdgIds = &src.pdgId[offs];
    const auto *z0 = &src.z0[offs];
    const auto *dxy = &src.dxy[offs];
    const auto *puppiw = &src.puppiw[offs];
    const auto *quality = &src.quality[offs];
    ix.clear();
    charge.clear();
    int intermediatecut = 0;
    int highcut = 0;
    // auto t = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < size; ++i) {  //make list of all hadrons
      if ((std::abs(pdgIds[i]) == 211 or std::abs(pdgIds[i]) == 11)) {
        if (pts[i] >= cuts.minpt1) {
          ix.push_back(i);
          charge.push_back(abs(pdgIds[i]) == 11 ? (pdgIds[i] > 0 ? -1 : +1) : (pdgIds[i] > 0 ? +1 : -1));
          if (pts[i] >= cuts.minpt2)
            intermediatecut++;
          if (pts[i] >= cuts.minpt3)
            highcut++;
        }
      }
    }
    // ct += ix.size();
    // times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t).count());
    // printf("s: %zu, i: %d, h: %d (%d, %d) -> %d\n", ix.size(), intermediatecut, highcut, src.offsets[ibx], src.offsets[ibx + 1], src.offsets[ibx + 1] - src.offsets[ibx]);
    auto es = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(es - se);
    full += duration.count();

    global_int_cut += intermediatecut;
    global_high_cut += highcut;
    global_l1_pass += ix.size();
    unsigned int npions = ix.size();
    if (highcut < 1 || intermediatecut < 2 || npions < 3)
      continue;
    ctr++;
    iso.resize(npions);
    std::fill(iso.begin(), iso.end(), 0);
    bestTripletScore = 0;

    for (unsigned int i1 = 0; i1 < npions; ++i1) {
      // if (ix[i1] != 0) break;
      // printf("id: %d; ", ix[i1]);
      // printf("pt: %f; ",pts[ix[i1]]);
      // printf("eta: %f; ", etas[ix[i1]]);
      // printf("phi: %f; ", phis[ix[i1]]);
      // printf("z0: %f; ", z0[ix[i1]]);
      // printf("dxy: %f; ", dxy[ix[i1]]);
      // printf("puppiw: %f; ", puppiw[ix[i1]]);
      // printf("pdgId: %d; ", pdgIds[ix[i1]]);
      // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i1]]));
      // printf("\n");
      if (pts[ix[i1]] < cuts.minpt3)
        continue;  //high pt cut
      if (isolation(ix[i1], size, etas, phis, pts, iso[i1]) == 0)
        continue;  //check iso of high pt pion
      // printf("0");
      // printf("PASSED: %d", ix[i1]);
      for (unsigned int i2 = 0; i2 < npions; ++i2) {
        // if (ix[i2] != 8) continue;
        // printf("\nid: %d; ", ix[i1]);
        // printf("pt: %f; ",pts[ix[i1]]);
        // printf("eta: %f; ", etas[ix[i1]]);
        // printf("phi: %f; ", phis[ix[i1]]);
        // printf("z0: %f; ", z0[ix[i1]]);
        // printf("dxy: %f; ", dxy[ix[i1]]);
        // printf("puppiw: %f; ", puppiw[ix[i1]]);
        // printf("pdgId: %d; ", pdgIds[ix[i1]]);
        // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i1]]));
        // printf("\n");
        // printf("id: %d; ", ix[i2]);
        // printf("pt: %f; ",pts[ix[i2]]);
        // printf("eta: %f; ", etas[ix[i2]]);
        // printf("phi: %f; ", phis[ix[i2]]);
        // printf("z0: %f; ", z0[ix[i2]]);
        // printf("dxy: %f; ", dxy[ix[i2]]);
        // printf("puppiw: %f; ", puppiw[ix[i2]]);
        // printf("pdgId: %d; ", pdgIds[ix[i2]]);
        // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i2]]));
        // printf("\n");
        // printf("id: %d; ", ix[i3]);
        // printf("pt: %f; ",pts[ix[i3]]);
        // printf("eta: %f; ", etas[ix[i3]]);
        // printf("phi: %f; ", phis[ix[i3]]);
        // printf("z0: %f; ", z0[ix[i3]]);
        // printf("dxy: %f; ", dxy[ix[i3]]);
        // printf("puppiw: %f; ", puppiw[ix[i3]]);
        // printf("pdgId: %d; ", pdgIds[ix[i3]]);
        // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i3]]));
        // printf("\n");

        if (i2 == i1 || pts[ix[i2]] < cuts.minpt2)
          continue;
        if (pts[ix[i2]] > pts[ix[i1]] || (pts[ix[i2]] == pts[ix[i1]] and i2 < i1)) {
            // printf("M0 ");
            continue;
          }
        if (!deltar(etas[ix[i1]], etas[ix[i2]], phis[ix[i1]], phis[ix[i2]]))
         {
            continue;
          }
        // printf("2");

        for (unsigned int i3 = 0; i3 < npions; ++i3) {
          // if (ix[i3] != 11) continue;
          if (i3 == i1 or i3 == i2)
            continue;
          if (pts[ix[i2]] < cuts.minpt1) {
            // printf("low_pt_cut ");
            continue;
          }

          if (pts[ix[i3]] > pts[ix[i1]] || (pts[ix[i3]] == pts[ix[i1]] and i3 < i1)) {
            // printf("M1 ");
            continue;
          }
          if (pts[ix[i3]] > pts[ix[i2]] || (pts[ix[i3]] == pts[ix[i2]] and i3 < i2)) {
            // printf("M2 ");
            continue;
          }
          std::array<unsigned int, 3> tr{{ix[i1], ix[i2], ix[i3]}};  //triplet of indeces
          // printf("3\n");
          // printf("%d, %d, %d -> (%d, %d) ? %d\n", ix[i1], ix[i2], ix[i3], src.offsets[ibx], src.offsets[ibx + 1], std::abs(charge[i1] + charge[i2] + charge[i3]) == 1);

          if (std::abs(charge[i1] + charge[i2] + charge[i3]) == 1) {

            //make Lorentz vectors for each triplet
            // if (pdgIds[ix[i1]] != 211 && pdgIds[ix[i2]] != 11 && pdgIds[ix[i3]] != -11) continue;
            auto mass = tripletmass(tr, pts, etas, phis);
            // printf("%d, %d, %d -> (%d, %d) ? %f\n", ix[i1], ix[i2], ix[i3], src.offsets[ibx], src.offsets[ibx + 1], mass);
            // if (mass < 160 && mass > 140) {
            //   printf("mass -> %f", mass);
            //   printf("\nid: %d; ", ix[i1]);
            //   printf("pt: %f; ",pts[ix[i1]]);
            //   printf("eta: %f; ", etas[ix[i1]]);
            //   printf("phi: %f; ", phis[ix[i1]]);
            //   printf("z0: %f; ", z0[ix[i1]]);
            //   printf("dxy: %f; ", dxy[ix[i1]]);
            //   printf("puppiw: %f; ", puppiw[ix[i1]]);
            //   printf("pdgId: %d; ", pdgIds[ix[i1]]);
            //   printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i1]]));
            //   printf("\n");
            //   printf("id: %d; ", ix[i2]);
            //   printf("pt: %f; ",pts[ix[i2]]);
            //   printf("eta: %f; ", etas[ix[i2]]);
            //   printf("phi: %f; ", phis[ix[i2]]);
            //   printf("z0: %f; ", z0[ix[i2]]);
            //   printf("dxy: %f; ", dxy[ix[i2]]);
            //   printf("puppiw: %f; ", puppiw[ix[i2]]);
            //   printf("pdgId: %d; ", pdgIds[ix[i2]]);
            //   printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i2]]));
            //   printf("\n");
            //   printf("id: %d; ", ix[i3]);
            //   printf("pt: %f; ",pts[ix[i3]]);
            //   printf("eta: %f; ", etas[ix[i3]]);
            //   printf("phi: %f; ", phis[ix[i3]]);
            //   printf("z0: %f; ", z0[ix[i3]]);
            //   printf("dxy: %f; ", dxy[ix[i3]]);
            //   printf("puppiw: %f; ", puppiw[ix[i3]]);
            //   printf("pdgId: %d; ", pdgIds[ix[i3]]);
            //   printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i3]]));
            //   printf("\n\n");
            // }
            if (mass >= cuts.minmass and mass <= cuts.maxmass) {  //MASS test

              // printf("4");

              // printf("\nid: %d; ", ix[i1]);
              // printf("pt: %f; ",pts[ix[i1]]);
              // printf("eta: %f; ", etas[ix[i1]]);
              // printf("phi: %f; ", phis[ix[i1]]);
              // printf("z0: %f; ", z0[ix[i1]]);
              // printf("dxy: %f; ", dxy[ix[i1]]);
              // printf("puppiw: %f; ", puppiw[ix[i1]]);
              // printf("pdgId: %d; ", pdgIds[ix[i1]]);
              // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i1]]));
              // printf("\n");
              // printf("\nid: %d; ", ix[i2]);
              // printf("pt: %f; ",pts[ix[i2]]);
              // printf("eta: %f; ", etas[ix[i2]]);
              // printf("phi: %f; ", phis[ix[i2]]);
              // printf("z0: %f; ", z0[ix[i2]]);
              // printf("dxy: %f; ", dxy[ix[i2]]);
              // printf("puppiw: %f; ", puppiw[ix[i2]]);
              // printf("pdgId: %d; ", pdgIds[ix[i2]]);
              // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i2]]));
              // printf("\n");
              // printf("\nid: %d; ", ix[i3]);
              // printf("pt: %f; ",pts[ix[i3]]);
              // printf("eta: %f; ", etas[ix[i3]]);
              // printf("phi: %f; ", phis[ix[i3]]);
              // printf("z0: %f; ", z0[ix[i3]]);
              // printf("dxy: %f; ", dxy[ix[i3]]);
              // printf("puppiw: %f; ", puppiw[ix[i3]]);
              // printf("pdgId: %d; ", pdgIds[ix[i3]]);
              // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i3]]));
              // printf("\n");
              // printf("%d %d %d\n", ix[i1], ix[i2], ix[i3]);
              if (deltar(etas[ix[i1]], etas[ix[i3]], phis[ix[i1]], phis[ix[i3]]) and
                  deltar(etas[ix[i2]], etas[ix[i3]], phis[ix[i2]], phis[ix[i3]])) {
                // printf("5");
                //ISOLATION test for lower 4 pions
                bool isop = isolation(ix[i2], size, etas, phis, pts, iso[i2]) &&
                            isolation(ix[i3], size, etas, phis, pts, iso[i3]);
                if (isop == true) {
                  // printf("6");
                  float ptsum = pts[ix[i1]] + pts[ix[i2]] + pts[ix[i3]];
                  // printf("%d: %f; %f; %f; %f;\n", ix[i1], pts[ix[i1]], pts[ix[i2]], pts[ix[i3]], ptsum); 
                  passSOA_++;
                  if (ptsum > bestTripletScore) {
                    // printf("7");
                    std::copy_n(tr.begin(), 3, bestTriplet.begin());
                    bestTripletScore = ptsum;
                    // printf("\nid: %d; ", ix[i1]);
                    // printf("pt: %f; ",pts[ix[i1]]);
                    // printf("eta: %f; ", etas[ix[i1]]);
                    // printf("phi: %f; ", phis[ix[i1]]);
                    // printf("z0: %f; ", z0[ix[i1]]);
                    // printf("dxy: %f; ", dxy[ix[i1]]);
                    // printf("puppiw: %f; ", puppiw[ix[i1]]);
                    // printf("pdgId: %d; ", pdgIds[ix[i1]]);
                    // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i1]]));
                    // printf("\n");
                    // printf("id: %d; ", ix[i2]);
                    // printf("pt: %f; ",pts[ix[i2]]);
                    // printf("eta: %f; ", etas[ix[i2]]);
                    // printf("phi: %f; ", phis[ix[i2]]);
                    // printf("z0: %f; ", z0[ix[i2]]);
                    // printf("dxy: %f; ", dxy[ix[i2]]);
                    // printf("puppiw: %f; ", puppiw[ix[i2]]);
                    // printf("pdgId: %d; ", pdgIds[ix[i2]]);
                    // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i2]]));
                    // printf("\n");
                    // printf("id: %d; ", ix[i3]);
                    // printf("pt: %f; ",pts[ix[i3]]);
                    // printf("eta: %f; ", etas[ix[i3]]);
                    // printf("phi: %f; ", phis[ix[i3]]);
                    // printf("z0: %f; ", z0[ix[i3]]);
                    // printf("dxy: %f; ", dxy[ix[i3]]);
                    // printf("puppiw: %f; ", puppiw[ix[i3]]);
                    // printf("pdgId: %d; ", pdgIds[ix[i3]]);
                    // printf("quality: %d; ", static_cast<unsigned short>(quality[ix[i3]]));
                    // printf("\n");
                  }
                }  // iso
              } else {
                // printf("WARNING ======== ANG_SEP -> ");
              }    // delta R
            } else {
              // printf("WARNING ======== MASS -> ");
            }     // mass
          } else {
            // printf("WARNING ======== CHARGE -> ");
          }       //charge
        } //low pt cut
      } //intermediate pt cut
    }            //high pt cut

    if (bestTripletScore > 0) {
      offsets.push_back(offsets.back() + size);
      ret->bx.push_back(src.bx[ibx]);
      ret->pt.insert(ret->pt.end(), pts, pts + size);
      ret->eta.insert(ret->eta.end(), etas, etas + size);
      ret->phi.insert(ret->phi.end(), phis, phis + size);
      ret->pdgId.insert(ret->pdgId.end(), pdgIds, pdgIds + size);
      ret->z0.insert(ret->z0.end(), &src.z0[offs], &src.z0[offs + size]);
      ret->dxy.insert(ret->dxy.end(), &src.dxy[offs], &src.dxy[offs + size]);
      ret->puppiw.insert(ret->puppiw.end(), &src.puppiw[offs], &src.puppiw[offs + size]);
      ret->quality.insert(ret->quality.end(), &src.quality[offs], &src.quality[offs + size]);
      // passSOA_++;
      // printf("%d: (%d, %d) -> Score: %.2f\n", ibx, src.offsets[ibx], src.offsets[ibx + 1], bestTripletScore); 

      // printf("Increment");
    }

    // std::cout << "Idx: " << ibx << "; [" << src.offsets[ibx] << ", " << src.offsets[ibx+1] << "]; "<< "Best Score: " << bestTripletScore << std::endl << std::endl;
    // std::cout << "Puppi collection on device:\n";
    // for (uint32_t i = 0; i < size; ++i) {
    //   std::cout << "id: " << i << "; ";
    //   std::cout << "pt: " << pts[i] << "; ";
    //   std::cout << "eta: " << etas[i] << "; ";
    //   std::cout << "phi: " << phis[i] << "; ";
    //   std::cout << "z0: " << z0[i] << "; ";
    //   std::cout << "dxy: " << dxy[i] << "; ";
    //   std::cout << "puppiw: " << puppiw[i] << "; ";
    //   std::cout << "pdgId: " << pdgIds[i] << "; ";
    //   std::cout << "quality: " << static_cast<unsigned short>(quality[i]) << "; " << std::endl;
    // }
    // break;
  
  }  // loop on BXs

  // std::cout << "Particles Num L1 Filter: " << global_l1_pass << std::endl;
  // std::cout << "Paritcles Num L1 IntCut: " << global_int_cut << std::endl;
  // std::cout << "Paritcles Num L1  HiCut: "  << global_high_cut << std::endl;
  // std::cout << "Candidates Num L1: " << ctr << std::endl;
  // std::cout << "W3Pi Num: " << passed << std::endl;
  // std::cout << std::endl;

  auto e = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);
  std::cout << "Analysis: OK [" << duration.count() << " us]" << std::endl;

  auto s2 = std::chrono::high_resolution_clock::now();
  iEvent.put(std::move(ret));
  auto e2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2);
  std::cout << "I/O (D2H): OK [" << duration2.count() << " us]" << std::endl;
  // auto mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
  // std::cout << "Kernel: OK [" << mean << " ns]" << std::endl;
  // std::cout << "Filtered Size: " << ct << std::endl; 
  // std::cout << "Isolation Module: OK [" << full << " us]" << std::endl;

}

//TEST functions
template <typename T>
bool ScPhase2PuppiW3PiDemo::isolation(unsigned int pidex, const T *cands, unsigned int size) const {
  bool passed = false;
  float psum = 0;
  float eta = cands[pidex].eta();
  float phi = cands[pidex].phi();
  for (unsigned int j = 0u; j < size; ++j) {  //loop over other particles
    if (pidex == j)
      continue;
    float deta = eta - cands[j].eta(), dphi = ROOT::VecOps::DeltaPhi<float>(phi, cands[j].phi());
    float dr2 = deta * deta + dphi * dphi;
    if (dr2 >= cuts.mindr2 && dr2 <= cuts.maxdr2)
      psum += cands[j].pt();
  }
  if (psum <= cuts.maxiso * cands[pidex].pt())
    passed = true;
  // printf("accu: %f\n", psum);
  return passed;
}

bool ScPhase2PuppiW3PiDemo::isolation(
    unsigned int pidex, unsigned int npx, const float *eta, const float *phi, const float *pt) const {
  bool passed = false;
  float psum = 0;
  for (unsigned int j = 0u, n = npx; j < n; ++j) {  //loop over other particles
    if (pidex == j)
      continue;
    // printf("idx: %d; etat: %f; eta: %f; phit: %f; phi: %f;\n", j, eta[pidex], eta[j], phi[pidex], phi[j]);  
    float deta = eta[pidex] - eta[j], dphi = ROOT::VecOps::DeltaPhi<float>(phi[pidex], phi[j]);
    // printf("deta: %f; dphi: %f; ", deta, dphi);
    float dr2 = deta * deta + dphi * dphi;
    // printf("dr: %f;\n", dr2);
    if (dr2 >= cuts.mindr2 && dr2 <= cuts.maxdr2)
      psum += pt[j];
  }
  if (psum <= cuts.maxiso * pt[pidex])
    passed = true;
  // printf("accu: %f\n", psum);
  return passed;
}

bool ScPhase2PuppiW3PiDemo::deltar(float eta1, float eta2, float phi1, float phi2) const {
  bool passed = true;
  float deta = eta1 - eta2;
  float dphi = ROOT::VecOps::DeltaPhi<float>(phi1, phi2);
  float dr2 = deta * deta + dphi * dphi;
  // printf("ang: %f; deta: %f; dphi: %f\n", dr2, deta, dphi);
  if (dr2 < cuts.mindeltar2) {
    passed = false;
    return passed;
  }
  return passed;
}

float ScPhase2PuppiW3PiDemo::tripletmass(const std::array<unsigned int, 3> &t,
                                         const float *pts,
                                         const float *etas,
                                         const float *phis) {
  ROOT::Math::PtEtaPhiMVector p1(pts[t[0]], etas[t[0]], phis[t[0]], 0.1396);
  ROOT::Math::PtEtaPhiMVector p2(pts[t[1]], etas[t[1]], phis[t[1]], 0.1396);
  ROOT::Math::PtEtaPhiMVector p3(pts[t[2]], etas[t[2]], phis[t[2]], 0.1396);
  float mass = (p1 + p2 + p3).M();
  return mass;
}

void ScPhase2PuppiW3PiDemo::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.add<bool>("runStruct", true);
  desc.add<bool>("runCandidate", false);
  desc.add<bool>("runSOA", false);
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2PuppiW3PiDemo);
