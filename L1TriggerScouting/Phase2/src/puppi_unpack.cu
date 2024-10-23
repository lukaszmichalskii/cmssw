#include "L1TriggerScouting/Phase2/interface/puppi_unpack.h"

namespace puppi_unpack::gpu {

__device__ void ReadShared(const uint64_t data, uint16_t& pt, int16_t& eta, int16_t& phi) {
  pt = data & 0x3FFF;
  eta = (data >> 14 & 0xFFF) | (-(data >> 25 & 1) << 11);
  phi = (data >> 26 & 0x7FF) | (-(data >> 36 & 1) << 10);
}

__device__ void ReadShared(const uint64_t data, float& pt, float& eta, float& phi) {
  pt = static_cast<uint16_t>(data & 0x3FFF) * 0.25f;
  eta = (static_cast<int>(data >> 14 & 0xFFF) | (-static_cast<int>(data >> 25 & 1) << 11)) * PRECOMPUTED_PI_720;
  phi = (static_cast<int>(data >> 26 & 0x7FF) | (-static_cast<int>(data >> 36 & 1) << 10)) * PRECOMPUTED_PI_720;
}

__global__ void ReadSharedBatch(const uint64_t* data, size_t size, uint16_t* pts, int16_t* etas, int16_t* phis) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    ReadShared(data[i], pts[i], etas[i], phis[i]);
  }
}

__global__ void ReadSharedBatch(const uint64_t* data, size_t size, float* pts, float* etas, float* phis) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    ReadShared(data[i], pts[i], etas[i], phis[i]);
  }
}

__device__ void DecodePID(const uint64_t& data, uint8_t& decoded) { decoded = (data >> 37) & 0x7; }

__global__ void DecodePIDsBatch(const uint64_t* data, uint8_t* decoded, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    DecodePID(data[i], decoded[i]);
  }
}

__device__ void MapParticle(uint8_t pid, int16_t& mapped_pid) { mapped_pid = PARTICLE_DGROUP_MAP[pid]; }

__global__ void MapParticleBatch(const uint8_t* pids, int16_t* mapped_pids, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    MapParticle(pids[i], mapped_pids[i]);
  }
}

__device__ void MapType(const uint8_t pid, l1t::PFCandidate::ParticleType& type) { 
  type = PARTICLE_DTYPE_MAP[pid]; 
}

__global__ void MapTypeBatch(const uint8_t* pids, l1t::PFCandidate::ParticleType* types, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    MapType(pids[i], types[i]);
  }
}

__device__ void MapMass(const uint8_t pid, float& mass) { mass = MASS_DMAP[pid]; }

__global__ void MapMassBatch(const uint8_t* pids, float* mapped_masses, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    MapMass(pids[i], mapped_masses[i]);
  }
}

__device__ void MapCharge(const uint8_t pid, int8_t& charge) {
  charge = static_cast<int8_t>((pid > 1) * (2 * (pid & 1) - 1));
}

__global__ void MapChargeBatch(const uint8_t* pids, int8_t* charges, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    MapCharge(pids[i], charges[i]);
  }
}

}  // namespace puppi_unpack::gpu