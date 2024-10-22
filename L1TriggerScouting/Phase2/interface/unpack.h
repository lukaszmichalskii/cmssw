#ifndef L1TriggerScouting_Phase2_unpack_h
#define L1TriggerScouting_Phase2_unpack_h

#include <cstdint>
#include <cmath>

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

/** @brief Mapping lookup table for particle pdgid */
alignas(16) static constexpr int16_t PARTICLE_DGROUP_MAP[8] = 
  {130, 22, -211, 211, 11, -11, 13, -13};

/** @brief Mapping lookup table for particle masses */
alignas(16) static constexpr float MASS_DMAP[8] = 
  {0.5, 0.0, 0.13, 0.13, 0.0005, 0.0005, 0.105, 0.105};

/** @brief Mapping lookup table for particle types */
alignas(16) static constexpr l1t::PFCandidate::ParticleType PARTICLE_DTYPE_MAP[8] = {
  l1t::PFCandidate::NeutralHadron,
  l1t::PFCandidate::Photon,
  l1t::PFCandidate::ChargedHadron,
  l1t::PFCandidate::ChargedHadron,
  l1t::PFCandidate::Electron,
  l1t::PFCandidate::Electron,
  l1t::PFCandidate::Muon,
  l1t::PFCandidate::Muon
};

/** @brief Precomputed 3.14/720 constant */
static constexpr auto PRECOMPUTED_PI_720 = static_cast<float>(M_PI / 720.0);

/** 
 * @brief unpacking utilities base implementations 
 */
namespace unpack {

// mapping
void assignpdgid(uint8_t pid, short int &pdgid);
void assignCMSSWPFCandidateId(uint8_t pid, l1t::PFCandidate::ParticleType &id);
void assignmass(uint8_t pid, float &mass);
void assigncharge(uint8_t pid, int &charge);
// decoding
void readshared(const uint64_t data, uint16_t &pt, int16_t &eta, int16_t &phi); // int
void readshared(const uint64_t data, float &pt, float &eta, float &phi); // float
void readcharged(const uint64_t data, int16_t &z0, int8_t &dxy, uint8_t &quality); // int
void readcharged(const uint64_t data, float &z0, float &dxy, uint8_t &quality); // float
void readcharged(const uint64_t data, uint8_t pid, float &z0, float &dxy); // float
void readneutral(const uint64_t data, uint16_t &wpuppi, uint8_t &id); // int
void readneutral(const uint64_t data, float &wpuppi, uint8_t &id); // float
void readneutral(const uint64_t data, uint8_t pid, float &wpuppi); // float

}  // namespace unpack

namespace unpack::inlined {

inline void assignpdgid(uint8_t pid, short int &pdgid) {
  pdgid = PARTICLE_DGROUP_MAP[pid];
};

inline void assignCMSSWPFCandidateId(uint8_t pid, l1t::PFCandidate::ParticleType &id) {
  id = PARTICLE_DTYPE_MAP[pid];
};

inline void assignmass(uint8_t pid, float &mass) {
  mass = MASS_DMAP[pid];
};

inline void assigncharge(uint8_t pid, int &charge) {
  charge = (pid > 1 ? (pid & 1 ? +1 : -1) : 0);
};

inline void readshared(const uint64_t data, uint16_t &pt, int16_t &eta, int16_t &phi) {
  pt = data & 0x3FFF;
  eta = ((data >> 25) & 1) ? ((data >> 14) | (-0x800)) : ((data >> 14) & (0xFFF));
  phi = ((data >> 36) & 1) ? ((data >> 26) | (-0x400)) : ((data >> 26) & (0x7FF));
}; // int

inline void readshared(const uint64_t data, float &pt, float &eta, float &phi) {
  uint16_t ptint = data & 0x3FFF;
  pt = ptint * 0.25f;
  int etaint = ((data >> 25) & 1) ? ((data >> 14) | (-0x800)) : ((data >> 14) & (0xFFF));
  eta = etaint * PRECOMPUTED_PI_720;
  int phiint = ((data >> 36) & 1) ? ((data >> 26) | (-0x400)) : ((data >> 26) & (0x7FF));
  phi = phiint * PRECOMPUTED_PI_720;
}; // float

inline void readcharged(const uint64_t data, int16_t &z0, int8_t &dxy, uint8_t &quality) {
  z0 = ((data >> 49) & 1) ? ((data >> 40) | (-0x200)) : ((data >> 40) & 0x3FF);
  dxy = ((data >> 57) & 1) ? ((data >> 50) | (-0x100)) : ((data >> 50) & 0xFF);
  quality = (data >> 58) & 0x7;  //3 bits
}; // int

inline void readcharged(const uint64_t data, float &z0, float &dxy, uint8_t &quality) {
  int z0int = ((data >> 49) & 1) ? ((data >> 40) | (-0x200)) : ((data >> 40) & 0x3FF);
  z0 = z0int * .05f;  //conver to centimeters

  int dxyint = ((data >> 57) & 1) ? ((data >> 50) | (-0x100)) : ((data >> 50) & 0xFF);
  dxy = dxyint * 0.05f;          // PLACEHOLDER
  quality = (data >> 58) & 0x7;  //3 bits
}; // float

inline void readcharged(const uint64_t data, uint8_t pid, float &z0, float &dxy) {
  int z0int = ((data >> 49) & 1) ? ((data >> 40) | (-0x200)) : ((data >> 40) & 0x3FF);
  z0 = (pid > 1) * z0int * .05f;  //conver to centimeters
  int dxyint = ((data >> 57) & 1) ? ((data >> 50) | (-0x100)) : ((data >> 50) & 0xFF);
  dxy = (pid > 1) * dxyint * 0.05f;  // PLACEHOLDER
}; // float

inline void readneutral(const uint64_t data, uint16_t &wpuppi, uint8_t &id) {
  wpuppi = (data >> 40) & 0x3FF;
  id = (data >> 50) & 0x3F;
};

inline void readneutral(const uint64_t data, float &wpuppi, uint8_t &id) {
  int wpuppiint = (data >> 40) & 0x3FF;
  wpuppi = wpuppiint * (1 / 256.f);
  id = (data >> 50) & 0x3F;
};

inline void readneutral(const uint64_t data, uint8_t pid, float &wpuppi) {
  int wpuppiint = (data >> 40) & 0x3FF;
  wpuppi = pid > 1 ? wpuppiint * float(1 / 256.f) : 1.0f;
};
}

namespace unpack::batch {

void ReadShared(const std::uint64_t data, std::uint16_t &pt, std::int16_t &eta, std::int16_t &phi);
void ReadShared(const uint64_t data, float &pt, float &eta, float &phi);
void ReadSharedBatch(const uint64_t* data, const std::size_t size, std::uint16_t* pts, std::int16_t* etas, std::int16_t* phis);
void ReadSharedBatch(const uint64_t* data, const std::size_t size, float* pts, float* etas, float* phis);

void DecodePID(const uint64_t &data, std::uint8_t &decoded);
void DecodePIDsBatch(const std::uint64_t *data, std::uint8_t *decoded, const std::size_t size);
void MapParticle(std::uint8_t pid, std::int16_t &mapped_pid);
void MapParticleBatch(const std::uint8_t* pids, std::int16_t* mapped_pids, const std::size_t size);
void MapType(const std::uint8_t pid, l1t::PFCandidate::ParticleType &type);
void MapTypeBatch(const std::uint8_t* pids, l1t::PFCandidate::ParticleType* types, const std::size_t size);
void MapMass(const std::uint8_t pid, float &mass);
void MapMassBatch(const std::uint8_t* pids, float* mapped_masses, const std::size_t size);
void MapCharge(const std::uint8_t pid, std::int8_t &charge);
void MapChargeBatch(const std::uint8_t* pids, std::int8_t* charges, const std::size_t size);

}  // namespace unpack::batch


/**
 * Collection of utility functions for boosted unpacking on CPU arch.
 * Includes: AVX2, AVX512, multi-threaded.
 * Singnatures: ? (no suffix)
 * <?> : base refactored implementation on single core
 * <V> : vectorized / SIMD based
 * <T> : multi-threaded @note by default all hardware threads used @param th_count to specify number of threads.)
 * 
 * @brief unpacking utilities boosted for CPU execution.
 */
namespace unpack::boosted::cpu {

void ReadShared(const std::uint64_t data, std::uint16_t &pt, std::int16_t &eta, std::int16_t &phi);
void ReadShared(const uint64_t data, float &pt, float &eta, float &phi);
void ReadSharedBatch(const uint64_t* data, const std::size_t size, std::uint16_t* pts, std::int16_t* etas, std::int16_t* phis);
void ReadSharedBatch(const uint64_t* data, const std::size_t size, float* pts, float* etas, float* phis);

void DecodePID(const uint64_t &data, std::uint8_t &decoded);
void DecodePIDsBatch(const std::uint64_t *data, std::uint8_t *decoded, const std::size_t size);
void MapParticle(std::uint8_t pid, std::int16_t &mapped_pid);
void MapParticleBatch(const std::uint8_t* pids, std::int16_t* mapped_pids, const std::size_t size);
void MapType(const std::uint8_t pid, l1t::PFCandidate::ParticleType &type);
void MapTypeBatch(const std::uint8_t* pids, l1t::PFCandidate::ParticleType* types, const std::size_t size);
void MapMass(const std::uint8_t pid, float &mass);
void MapMassBatch(const std::uint8_t* pids, float* mapped_masses, const std::size_t size);
void MapCharge(const std::uint8_t pid, std::int8_t &charge);
void MapChargeBatch(const std::uint8_t* pids, std::int8_t* charges, const std::size_t size);

}  // namespace unpack::boosted::cpu


namespace unpack::boosted::gpu {

}  // namespace unpack::boosted::gpu


#endif  // L1TriggerScouting_Phase2_interface_unpack_h