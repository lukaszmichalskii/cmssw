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
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/L1ScoutingPuppi.h"
#include "L1TriggerScouting/Phase2/interface/unpack.h"
#include <chrono>
#include <numeric>

float *pt_;
float *eta_;
float *phi_;
float *mass_;
float *z0_;
float *dxy_;
float *puppiw_;
uint16_t *hwPt_;
uint16_t *hwPuppiW_;
int16_t *pdgId_;
int16_t *hwEta_;
int16_t *hwPhi_;
int16_t *hwZ0_;
int8_t *hwDxy_;
uint8_t *pid_;
uint8_t *hwQuality_;
l1t::PFCandidate::ParticleType *type_;
int8_t *charge_;

class ScPhase2PuppiRawToDigiCPU : public edm::stream::EDProducer<> {
public:
  explicit ScPhase2PuppiRawToDigiCPU(const edm::ParameterSet &params);
  ~ScPhase2PuppiRawToDigiCPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void GetParams(const edm::ParameterSet &params);
  void Init();
  void produce(edm::Event &event, const edm::EventSetup &event_setup) override;

  template <typename T> std::unique_ptr<OrbitCollection<T>> Unpack(
    unsigned int orbit, const SDSRawDataCollection &feds, std::vector<std::vector<T>> &buffer);
  // helpers
  void UnpackRawBatch(const uint64_t *data, size_t size, std::vector<l1t::PFCandidate> &out_buffer);
  void UnpackRawBatch(const uint64_t *data, size_t size, std::vector<l1Scouting::Puppi> &out_buffer);
  void UnpackRawSIMD(const uint64_t *data, size_t size, std::vector<l1t::PFCandidate> &out_buffer);
  void UnpackRawSIMD(const uint64_t *data, size_t size, std::vector<l1Scouting::Puppi> &out_buffer);
  void UnpackRaw(uint64_t data, std::vector<l1t::PFCandidate> &out_buffer);
  void UnpackRaw(uint64_t data, std::vector<l1Scouting::Puppi> &out_buffer);
  void UnpackRawInline(uint64_t data, std::vector<l1t::PFCandidate> &out_buffer);
  void UnpackRawInline(uint64_t data, std::vector<l1Scouting::Puppi> &out_buffer);

  edm::EDGetTokenT<SDSRawDataCollection> raw_token_;
  std::vector<unsigned int> fed_ids_;
  bool is_candidate_unpack_enabled_, is_struct_unpack_enabled_;
  bool inline_enabled_;
  bool batch_processing_enabled_;
  bool simd_enabled_;

  template<typename T> using TempBuffer = std::vector<std::vector<T>>; /**< temporary buffer repr */
  TempBuffer<l1t::PFCandidate> pf_candidates_buffer;
  TempBuffer<l1Scouting::Puppi> struct_buffer_;
  unsigned int nbx_;
};

/**
 * Constructor
 */
ScPhase2PuppiRawToDigiCPU::ScPhase2PuppiRawToDigiCPU(const edm::ParameterSet &params) {
  GetParams(params);
  Init();
}

/**
 * Default destructor
 */
ScPhase2PuppiRawToDigiCPU::~ScPhase2PuppiRawToDigiCPU(){};

/**
 * Parse parameters
 */
void ScPhase2PuppiRawToDigiCPU::GetParams(const edm::ParameterSet &params) {
  raw_token_ = consumes<SDSRawDataCollection>(params.getParameter<edm::InputTag>("src"));
  fed_ids_ = params.getParameter<std::vector<unsigned int>>("fedIDs");
  is_candidate_unpack_enabled_ = params.getParameter<bool>("runCandidateUnpacker");
  is_struct_unpack_enabled_ = params.getParameter<bool>("runStructUnpacker");
  inline_enabled_ = params.getParameter<bool>("runInlineVersion");
  batch_processing_enabled_ = params.getParameter<bool>("runBatchProcessing");
  simd_enabled_ = params.getParameter<bool>("runSIMDVersion");
}

/**
 * Initialize state
 */
void ScPhase2PuppiRawToDigiCPU::Init() {
  if (is_candidate_unpack_enabled_) {
    pf_candidates_buffer.resize(OrbitCollection<l1t::PFCandidate>::NBX + 1);
    produces<OrbitCollection<l1t::PFCandidate>>();
  }
  if (is_struct_unpack_enabled_) {
    struct_buffer_.resize(OrbitCollection<l1Scouting::Puppi>::NBX + 1);
    produces<OrbitCollection<l1Scouting::Puppi>>();
  }
  if (is_candidate_unpack_enabled_ || is_struct_unpack_enabled_) {
    produces<unsigned int>("nbx");
  }
}



void ScPhase2PuppiRawToDigiCPU::produce(edm::Event &event, const edm::EventSetup &event_setup) {
  edm::Handle<SDSRawDataCollection> scoutingRawDataCollection;
  event.getByToken(raw_token_, scoutingRawDataCollection);
  if (is_candidate_unpack_enabled_) {
    event.put(Unpack(event.id().event(), *scoutingRawDataCollection, pf_candidates_buffer));
  }
  if (is_struct_unpack_enabled_) {
    event.put(Unpack(event.id().event(), *scoutingRawDataCollection, struct_buffer_));
  }
  if (is_candidate_unpack_enabled_ || is_struct_unpack_enabled_) {
    event.put(std::make_unique<unsigned int>(nbx_), "nbx");
  }
}

template <typename T>
std::unique_ptr<OrbitCollection<T>> ScPhase2PuppiRawToDigiCPU::Unpack(
    unsigned int orbit, const SDSRawDataCollection &feds, std::vector<std::vector<T>> &buffer) {
  unsigned int ntot = 0;
  unsigned int mx_size = 0;
  nbx_ = 0;

  for (auto &fedId : fed_ids_) {
    const FEDRawData &src = feds.FEDData(fedId);
    const uint64_t *begin = reinterpret_cast<const uint64_t *>(src.data());
    const uint64_t *end = reinterpret_cast<const uint64_t *>(src.data() + src.size());
    for (auto p = begin; p != end;) {
      if ((*p) == 0) {
        ++p;
        continue;
      }
      unsigned int nwords = (*p) & 0xFFF;
      ++p;
      if (nwords > mx_size)
        mx_size = nwords;
      // mx_size = std::max(mx_size, nwords);
    }
  }
  std::cout << "mx_size = " << mx_size << std::endl;
  

  if (batch_processing_enabled_) {
    pt_ = new float[mx_size];
    eta_ = new float[mx_size];
    phi_ = new float[mx_size];
    mass_ = new float[mx_size];
    z0_ = new float[mx_size];
    dxy_ = new float[mx_size];
    puppiw_ = new float[mx_size];
    hwPt_ = new uint16_t[mx_size];
    hwPuppiW_ = new uint16_t[mx_size];
    pdgId_ = new int16_t[mx_size];
    hwEta_ = new int16_t[mx_size];
    hwPhi_ = new int16_t[mx_size];
    hwZ0_ = new int16_t[mx_size];
    hwDxy_ = new int8_t[mx_size];
    pid_ = new uint8_t[mx_size];
    hwQuality_ = new uint8_t[mx_size];
    type_ = new l1t::PFCandidate::ParticleType[mx_size];
    charge_ = new int8_t[mx_size];
  }
  int keeper = 0;
  for (auto &fedId : fed_ids_) {
    const FEDRawData &src = feds.FEDData(fedId);
    const uint64_t *begin = reinterpret_cast<const uint64_t *>(src.data());
    const uint64_t *end = reinterpret_cast<const uint64_t *>(src.data() + src.size());
    keeper++;
    for (auto p = begin; p != end;) {
      if ((*p) == 0) {
        ++p;
        continue;
      }
      unsigned int bx = ((*p) >> 12) & 0xFFF;
      unsigned int orbitno = ((*p) >> 24) & 0xFFFFFFFFFlu;
      unsigned int nwords = (*p) & 0xFFF;
      if (orbitno != orbit) {
        throw cms::Exception("CorruptData") 
          << "Data for orbit " << orbit 
          << ", fedId " << fedId                      
          << " has header with mismatching orbit number " << orbitno << std::endl;
      }
      nbx_++;
      ++p;
      assert(bx < OrbitCollection<T>::NBX);
      std::vector<T> &output_buffer = buffer[bx + 1];
      output_buffer.reserve(nwords);
      if (batch_processing_enabled_) {
        if (simd_enabled_) {
          UnpackRawSIMD(p, nwords, output_buffer);
        } else {
          UnpackRawBatch(p, nwords, output_buffer);
        }
        ntot = ntot + nwords;
        p = p + nwords;
      } else {
        for (unsigned int i = 0; i < nwords; ++i, ++p) {
          uint64_t data = *p;
          if (inline_enabled_) {
            UnpackRawInline(data, output_buffer);
          } else {
            UnpackRaw(data, output_buffer);
          }
          ntot++;
        }
      }
    }
  }
  std::cout << "Keeper = " << keeper << std::endl;
  
  if (batch_processing_enabled_) {
    delete[] pt_;
    delete[] eta_;
    delete[] phi_;
    delete[] mass_;
    delete[] z0_;
    delete[] dxy_;
    delete[] puppiw_;
    delete[] hwPt_;
    delete[] hwPuppiW_;
    delete[] pdgId_;
    delete[] hwEta_;
    delete[] hwPhi_;
    delete[] hwZ0_;
    delete[] hwDxy_;
    delete[] pid_;
    delete[] hwQuality_;
    delete[] type_;
    delete[] charge_;
  }
  return std::make_unique<OrbitCollection<T>>(buffer, ntot);
}

void ScPhase2PuppiRawToDigiCPU::UnpackRawSIMD(const uint64_t *data, size_t size, std::vector<l1t::PFCandidate> &out_buffer) {
  unpack::boosted::cpu::ReadSharedBatch(data, size, pt_, eta_, phi_);
  unpack::boosted::cpu::ReadSharedBatch(data, size, hwPt_, hwEta_, hwPhi_);
  unpack::boosted::cpu::DecodePIDsBatch(data, pid_, size);
  
  unpack::boosted::cpu::MapParticleBatch(pid_, pdgId_, size);
  unpack::boosted::cpu::MapTypeBatch(pid_, type_, size);
  unpack::boosted::cpu::MapMassBatch(pid_, mass_, size);
  unpack::boosted::cpu::MapChargeBatch(pid_, charge_, size); 

  for (size_t i = 0; i < size; ++i) {
    reco::Particle::PolarLorentzVector p4(pt_[i], eta_[i], phi_[i], mass_[i]);
    if (pid_[i] > 1) {
      unpack::readcharged(data[i], z0_[i], dxy_[i], hwQuality_[i]);
      unpack::readcharged(data[i], hwZ0_[i], hwDxy_[i], hwQuality_[i]);
    } else {
      unpack::readneutral(data[i], puppiw_[i], hwQuality_[i]);
      unpack::readneutral(data[i], hwPuppiW_[i], hwQuality_[i]);
    }
    out_buffer.emplace_back(type_[i], charge_[i], p4, puppiw_[i], hwPt_[i], hwEta_[i], hwPhi_[i]);
    if (pid_[i] > 1) {
      out_buffer.back().setZ0(z0_[i]);
      out_buffer.back().setDxy(dxy_[i]);
      out_buffer.back().setHwZ0(hwZ0_[i]);
      out_buffer.back().setHwDxy(hwDxy_[i]);
      out_buffer.back().setHwTkQuality(hwQuality_[i]);
    } else {
      out_buffer.back().setHwPuppiWeight(hwPuppiW_[i]);
      out_buffer.back().setHwEmID(hwQuality_[i]);
    }
    out_buffer.back().setEncodedPuppi64(data[i]);
  }
}

void ScPhase2PuppiRawToDigiCPU::UnpackRawSIMD(const uint64_t *data, size_t size, std::vector<l1Scouting::Puppi> &out_buffer) {
}

void ScPhase2PuppiRawToDigiCPU::UnpackRawBatch(const uint64_t *data, size_t size, std::vector<l1t::PFCandidate> &out_buffer) {
  unpack::batch::ReadSharedBatch(data, size, pt_, eta_, phi_);
  unpack::batch::ReadSharedBatch(data, size, hwPt_, hwEta_, hwPhi_);
  unpack::batch::DecodePIDsBatch(data, pid_, size);
  
  unpack::batch::MapParticleBatch(pid_, pdgId_, size);
  unpack::batch::MapTypeBatch(pid_, type_, size);
  unpack::batch::MapMassBatch(pid_, mass_, size);
  unpack::batch::MapChargeBatch(pid_, charge_, size); 

  for (size_t i = 0; i < size; ++i) {
    reco::Particle::PolarLorentzVector p4(pt_[i], eta_[i], phi_[i], mass_[i]);
    if (pid_[i] > 1) {
      unpack::readcharged(data[i], z0_[i], dxy_[i], hwQuality_[i]);
      unpack::readcharged(data[i], hwZ0_[i], hwDxy_[i], hwQuality_[i]);
    } else {
      unpack::readneutral(data[i], puppiw_[i], hwQuality_[i]);
      unpack::readneutral(data[i], hwPuppiW_[i], hwQuality_[i]);
    }
    out_buffer.emplace_back(type_[i], charge_[i], p4, puppiw_[i], hwPt_[i], hwEta_[i], hwPhi_[i]);
    if (pid_[i] > 1) {
      out_buffer.back().setZ0(z0_[i]);
      out_buffer.back().setDxy(dxy_[i]);
      out_buffer.back().setHwZ0(hwZ0_[i]);
      out_buffer.back().setHwDxy(hwDxy_[i]);
      out_buffer.back().setHwTkQuality(hwQuality_[i]);
    } else {
      out_buffer.back().setHwPuppiWeight(hwPuppiW_[i]);
      out_buffer.back().setHwEmID(hwQuality_[i]);
    }
    out_buffer.back().setEncodedPuppi64(data[i]);
  }
}

void ScPhase2PuppiRawToDigiCPU::UnpackRawBatch(const uint64_t *data, size_t size, std::vector<l1Scouting::Puppi> &out_buffer) {
}


void ScPhase2PuppiRawToDigiCPU::UnpackRaw(uint64_t data, std::vector<l1t::PFCandidate> &out_buffer) {
  float pt, eta, phi, mass, z0 = 0, dxy = 0, puppiw = 1;
  uint16_t hwPt, hwPuppiW = 1 << 8;
  int16_t pdgId, hwEta, hwPhi, hwZ0 = 0;
  int8_t hwDxy = 0;
  uint8_t pid, hwQuality;
  l1t::PFCandidate::ParticleType type;
  int charge;
  unpack::readshared(data, pt, eta, phi);
  unpack::readshared(data, hwPt, hwEta, hwPhi);
  pid = (data >> 37) & 0x7;
  unpack::assignpdgid(pid, pdgId);
  unpack::assignCMSSWPFCandidateId(pid, type);
  unpack::assignmass(pid, mass);
  unpack::assigncharge(pid, charge);
  reco::Particle::PolarLorentzVector p4(pt, eta, phi, mass);
  if (pid > 1) {
    unpack::readcharged(data, z0, dxy, hwQuality);
    unpack::readcharged(data, hwZ0, hwDxy, hwQuality);
  } else {
    unpack::readneutral(data, puppiw, hwQuality);
    unpack::readneutral(data, hwPuppiW, hwQuality);
  }
  out_buffer.emplace_back(type, charge, p4, puppiw, hwPt, hwEta, hwPhi);
  if (pid > 1) {
    out_buffer.back().setZ0(z0);
    out_buffer.back().setDxy(dxy);
    out_buffer.back().setHwZ0(hwZ0);
    out_buffer.back().setHwDxy(hwDxy);
    out_buffer.back().setHwTkQuality(hwQuality);
  } else {
    out_buffer.back().setHwPuppiWeight(hwPuppiW);
    out_buffer.back().setHwEmID(hwQuality);
  }
  out_buffer.back().setEncodedPuppi64(data);
}

void ScPhase2PuppiRawToDigiCPU::UnpackRawInline(uint64_t data, std::vector<l1t::PFCandidate> &out_buffer) {
  float pt, eta, phi, mass, z0 = 0, dxy = 0, puppiw = 1;
  uint16_t hwPt, hwPuppiW = 1 << 8;
  int16_t pdgId, hwEta, hwPhi, hwZ0 = 0;
  int8_t hwDxy = 0;
  uint8_t pid, hwQuality;
  l1t::PFCandidate::ParticleType type;
  int charge;
  unpack::inlined::readshared(data, pt, eta, phi);
  unpack::inlined::readshared(data, hwPt, hwEta, hwPhi);
  pid = (data >> 37) & 0x7;
  unpack::inlined::assignpdgid(pid, pdgId);
  unpack::inlined::assignCMSSWPFCandidateId(pid, type);
  unpack::inlined::assignmass(pid, mass);
  unpack::inlined::assigncharge(pid, charge);
  reco::Particle::PolarLorentzVector p4(pt, eta, phi, mass);
  if (pid > 1) {
    unpack::inlined::readcharged(data, z0, dxy, hwQuality);
    unpack::inlined::readcharged(data, hwZ0, hwDxy, hwQuality);
  } else {
    unpack::inlined::readneutral(data, puppiw, hwQuality);
    unpack::inlined::readneutral(data, hwPuppiW, hwQuality);
  }
  out_buffer.emplace_back(type, charge, p4, puppiw, hwPt, hwEta, hwPhi);
  if (pid > 1) {
    out_buffer.back().setZ0(z0);
    out_buffer.back().setDxy(dxy);
    out_buffer.back().setHwZ0(hwZ0);
    out_buffer.back().setHwDxy(hwDxy);
    out_buffer.back().setHwTkQuality(hwQuality);
  } else {
    out_buffer.back().setHwPuppiWeight(hwPuppiW);
    out_buffer.back().setHwEmID(hwQuality);
  }
  out_buffer.back().setEncodedPuppi64(data);
}

void ScPhase2PuppiRawToDigiCPU::UnpackRaw(uint64_t data, std::vector<l1Scouting::Puppi> &out_buffer) {
  float pt, eta, phi, z0 = 0, dxy = 0, puppiw = 1;
  uint8_t quality;
  unpack::readshared(data, pt, eta, phi);
  uint8_t pid = (data >> 37) & 0x7;
  if (pid > 1) {
    unpack::readcharged(data, z0, dxy, quality);
  } else {
    unpack::readneutral(data, puppiw, quality);
  }
  out_buffer.emplace_back(pt, eta, phi, pid, z0, dxy, puppiw, quality);
}

void ScPhase2PuppiRawToDigiCPU::UnpackRawInline(uint64_t data, std::vector<l1Scouting::Puppi> &out_buffer) {
  float pt, eta, phi, z0 = 0, dxy = 0, puppiw = 1;
  uint8_t quality;
  unpack::inlined::readshared(data, pt, eta, phi);
  uint8_t pid = (data >> 37) & 0x7;
  if (pid > 1) {
    unpack::inlined::readcharged(data, z0, dxy, quality);
  } else {
    unpack::inlined::readneutral(data, puppiw, quality);
  }
  out_buffer.emplace_back(pt, eta, phi, pid, z0, dxy, puppiw, quality);
}

/**
 * @brief Populate the descriptions with the allowed parameters for this module
 */
void ScPhase2PuppiRawToDigiCPU::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int>>("fedIDs");
  desc.add<bool>("runCandidateUnpacker", false);
  desc.add<bool>("runStructUnpacker", true);
  desc.add<bool>("runInlineVersion", false);
  desc.add<bool>("runBatchProcessing", false);
  desc.add<bool>("runSIMDVersion", false);
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScPhase2PuppiRawToDigiCPU);  // register module
