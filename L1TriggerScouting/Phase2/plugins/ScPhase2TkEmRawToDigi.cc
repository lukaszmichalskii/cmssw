#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1TParticleFlow/interface/L1ScoutingTkEm.h"
#include "L1TriggerScouting/Phase2/interface/l1tkemUnpack.h"

class ScPhase2TkEmRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScPhase2TkEmRawToDigi(const edm::ParameterSet &);
  ~ScPhase2TkEmRawToDigi() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  edm::EDGetTokenT<SDSRawDataCollection> rawToken_;
  std::vector<unsigned int> fedIDs_;

  // temporary storage
  std::vector<std::vector<l1Scouting::TkEm>> structBufferEm_;
  std::vector<std::vector<l1Scouting::TkEle>> structBufferEle_;

  void unpackFromRaw(uint64_t datalow, uint32_t datahigh, std::vector<l1Scouting::TkEm> &outBuffer);
  void unpackFromRaw(uint64_t datalow, uint32_t datahigh, std::vector<l1Scouting::TkEle> &outBuffer);
};

ScPhase2TkEmRawToDigi::ScPhase2TkEmRawToDigi(const edm::ParameterSet &iConfig)
    : rawToken_(consumes<SDSRawDataCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      fedIDs_(iConfig.getParameter<std::vector<unsigned int>>("fedIDs")) {
  structBufferEm_.resize(OrbitCollection<l1Scouting::TkEm>::NBX + 1);
  structBufferEle_.resize(OrbitCollection<l1Scouting::TkEle>::NBX + 1);
  produces<OrbitCollection<l1Scouting::TkEm>>();
  produces<OrbitCollection<l1Scouting::TkEle>>();
  produces<unsigned int>("nbx");
}

ScPhase2TkEmRawToDigi::~ScPhase2TkEmRawToDigi(){};

void ScPhase2TkEmRawToDigi::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<SDSRawDataCollection> feds;
  iEvent.getByToken(rawToken_, feds);

  unsigned int ntotEm = 0, ntotEle = 0, nbx = 0, reforbit = iEvent.id().event();
  for (auto &fedId : fedIDs_) {
    const FEDRawData &src = feds->FEDData(fedId);
    const uint64_t *begin = reinterpret_cast<const uint64_t *>(src.data());
    const uint64_t *end = reinterpret_cast<const uint64_t *>(src.data() + src.size());
    for (auto p = begin; p != end;) {
      if ((*p) == 0) {
        ++p;
        continue;
      }
      unsigned int bx = ((*p) >> 12) & 0xFFF;
      unsigned int nwords = (*p) & 0xFFF;
      unsigned int orbit = ((*p) >> 24) & 0xFFFFFFFFFlu;
      if (reforbit != orbit) {
        throw cms::Exception("CorruptData") << "Data for orbit " << reforbit << ", fedId " << fedId
                                            << " has header with mismatching orbit number " << orbit << std::endl;
      }
      nbx++;
      unsigned int negamma = (nwords * 2) / 3;
      assert(negamma >= 12);
      unsigned int ntkem = 12;  // always 12, then followed by some number of tkEle
      unsigned int ntkele = negamma - ntkem;
      ++p;
      assert(bx < OrbitCollection<l1Scouting::TkEm>::NBX);
      std::vector<l1Scouting::TkEm> &outputBufferEm = structBufferEm_[bx + 1];
      std::vector<l1Scouting::TkEle> &outputBufferEle = structBufferEle_[bx + 1];
      outputBufferEm.reserve(ntkem);
      outputBufferEle.reserve(ntkele);
      const uint32_t *ptr32 = reinterpret_cast<const uint32_t *>(p);
      for (unsigned int i = 0; i < negamma; ++i, ptr32 += 3) {
        uint64_t datalow;
        uint32_t datahigh;
        if ((i & 1) == 0) {
          datalow = *reinterpret_cast<const uint64_t *>(ptr32);
          datahigh = *(ptr32 + 2);
        } else {
          datalow = *reinterpret_cast<const uint64_t *>(ptr32 + 1);
          datahigh = *ptr32;
        }
        if (i < ntkem) {
          unpackFromRaw(datalow, datahigh, outputBufferEm);
          ntotEm++;
        } else {
          unpackFromRaw(datalow, datahigh, outputBufferEle);
          ntotEle++;
        }
      }
      p += nwords;
    }
  }

  auto outEm = std::make_unique<OrbitCollection<l1Scouting::TkEm>>(structBufferEm_, ntotEm);
  auto outEle = std::make_unique<OrbitCollection<l1Scouting::TkEle>>(structBufferEle_, ntotEle);
  iEvent.put(std::move(outEm));
  iEvent.put(std::move(outEle));
  iEvent.put(std::make_unique<unsigned int>(nbx), "nbx");
}

void ScPhase2TkEmRawToDigi::unpackFromRaw(uint64_t datalow,
                                          uint32_t datahigh,
                                          std::vector<l1Scouting::TkEm> &outBuffer) {
  float pt, eta, phi, isolation;
  uint8_t quality;
  l1tkemUnpack::readshared(datalow, datahigh, pt, eta, phi, quality, isolation);
  outBuffer.emplace_back(pt, eta, phi, quality, isolation);
}

void ScPhase2TkEmRawToDigi::unpackFromRaw(uint64_t datalow,
                                          uint32_t datahigh,
                                          std::vector<l1Scouting::TkEle> &outBuffer) {
  float pt, eta, phi, isolation, z0;
  uint8_t quality;
  int8_t charge;
  l1tkemUnpack::readshared(datalow, datahigh, pt, eta, phi, quality, isolation);
  l1tkemUnpack::readele(datalow, datahigh, charge, z0);
  outBuffer.emplace_back(pt, eta, phi, quality, isolation, charge, z0);
}

void ScPhase2TkEmRawToDigi::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fedIDs");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2TkEmRawToDigi);
