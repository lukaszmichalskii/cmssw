#include "L1TriggerScouting/Phase2/interface/puppi_unpack.h"

namespace puppi_unpack::cpu {

void ReadCharge(const uint64_t data, int16_t &z0, int8_t &dxy, uint8_t &quality) {
  z0 = data >> 40 & 0x3FF | -(data >> 49 & 1) << 9; 
  dxy = data >> 50 & 0xFF | -(data >> 57 & 1) << 7; 
  quality = data >> 58 & 0x7;
}

void ReadCharge(const uint64_t data, float &z0, float &dxy, uint8_t &quality) {
  z0 = (data >> 40 & 0x3FF | -(data >> 49 & 1) << 9) * 0.05f;
  dxy = (data >> 50 & 0xFF | -(data >> 57 & 1) << 7) * 0.05f;
  quality = (data >> 58) & 0x7;
}

void ReadNeutral(const uint64_t data, uint16_t &wpuppi, uint8_t &id) {
  wpuppi = static_cast<uint16_t>((data >> 40) & 0x3FF);  
  id = static_cast<uint8_t>((data >> 50) & 0x3F); 
}

void ReadNeutral(const uint64_t data, float &wpuppi, uint8_t &id) {
  wpuppi = static_cast<float>(((data >> 40) & 0x3FF)) * (1.0f / 256.0f); 
  id = static_cast<uint8_t>((data >> 50) & 0x3F);
}

void ReadShared(const std::uint64_t data, std::uint16_t &pt, std::int16_t &eta, std::int16_t &phi) {
  pt = data & 0x3FFF;
  eta = data >> 14 & 0xFFF | -(data >> 25 & 1) << 11;
  phi = data >> 26 & 0x7FF | -(data >> 36 & 1) << 10;
}

void ReadShared(const uint64_t data, float &pt, float &eta, float &phi) {
  pt = static_cast<uint16_t>(data & 0x3FFF) * 0.25f; 
  eta = (static_cast<int>(data >> 14 & 0xFFF) | -static_cast<int>(data >> 25 & 1) << 11) * PRECOMPUTED_PI_720;  
  phi = (static_cast<int>(data >> 26 & 0x7FF) | -static_cast<int>(data >> 36 & 1) << 10) * PRECOMPUTED_PI_720; 
}

void ReadSharedBatch(const uint64_t* data, const std::size_t size, std::uint16_t* pts, std::int16_t* etas, std::int16_t* phis) {
  for (std::size_t i = 0; i < size; i++) {
    ReadShared(data[i], pts[i], etas[i], phis[i]);
  }
}

void ReadSharedBatch(const uint64_t* data, const std::size_t size, float* pts, float* etas, float* phis) {
  for (std::size_t i = 0; i < size; i++) {
    ReadShared(data[i], pts[i], etas[i], phis[i]);
  }
}

void DecodePID(const uint64_t &data, std::uint8_t &decoded) {
  decoded = data >> 37 & 0x7;
}

void DecodePIDsBatch(const std::uint64_t *data, std::uint8_t *decoded, const std::size_t size) {
  for (std::size_t i = 0; i < size; i++) {
    DecodePID(data[i], decoded[i]);
  }
}

void MapParticle(std::uint8_t pid, std::int16_t &mapped_pid) {
  mapped_pid = PARTICLE_DGROUP_MAP[pid];
}

void MapParticleBatch(const std::uint8_t* pids, std::int16_t* mapped_pids, const std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    MapParticle(pids[i], mapped_pids[i]);
  }
}

void MapType(const std::uint8_t pid, l1t::PFCandidate::ParticleType &type) {
  type = PARTICLE_DTYPE_MAP[pid];
}

void MapTypeBatch(const std::uint8_t* pids, l1t::PFCandidate::ParticleType* types, const std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    MapType(pids[i], types[i]);
  }
}

void MapMass(const std::uint8_t pid, float &mass) {
  mass = MASS_DMAP[pid];
}

void MapMassBatch(const std::uint8_t* pids, float* mapped_masses, const std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    MapMass(pids[i], mapped_masses[i]);
  }
}

void MapCharge(const std::uint8_t pid, std::int8_t &charge) {
  charge = static_cast<std::int8_t>((pid > 1) * (2 * (pid & 1) - 1));
}

void MapChargeBatch(const std::uint8_t* pids, std::int8_t* charges, const std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    MapCharge(pids[i], charges[i]);
  }
}

}  // namespace puppi_unpack

