#ifndef L1TriggerScouting_Phase2_puppi_unpack_h
#define L1TriggerScouting_Phase2_puppi_unpack_h

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h> 

/**
 * Collection of utility functions for unpacking on CPU arch.
 * @brief unpacking utilities for CPU execution.
 */
namespace puppi_unpack::cpu {

    constexpr auto PRECOMPUTED_PI_720 = static_cast<float>(M_PI / 720.0);  
    constexpr int16_t PARTICLE_DGROUP_MAP[8] = {130, 22, -211, 211, 11, -11, 13, -13};
    constexpr float MASS_DMAP[8] =  {0.5, 0.0, 0.13, 0.13, 0.0005, 0.0005, 0.105, 0.105};
    constexpr l1t::PFCandidate::ParticleType PARTICLE_DTYPE_MAP[8] = {
    l1t::PFCandidate::NeutralHadron,
    l1t::PFCandidate::Photon,
    l1t::PFCandidate::ChargedHadron,
    l1t::PFCandidate::ChargedHadron,
    l1t::PFCandidate::Electron,
    l1t::PFCandidate::Electron,
    l1t::PFCandidate::Muon,
    l1t::PFCandidate::Muon
    };    

    void ReadCharge(const uint64_t data, int16_t &z0, int8_t &dxy, uint8_t &quality);
    void ReadCharge(const uint64_t data, float &z0, float &dxy, uint8_t &quality);
    void ReadNeutral(const uint64_t data, uint16_t &wpuppi, uint8_t &id);
    void ReadNeutral(const uint64_t data, float &wpuppi, uint8_t &id);
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
    
}  // namespace puppi_unpack

/**
 * Collection of utility functions for unpacking on GPU arch.
 * @brief unpacking utilities for GPU execution.
 */
namespace puppi_unpack::gpu {

    constexpr __device__ auto PRECOMPUTED_PI_720 = static_cast<float>(M_PI / 720.0);   
    constexpr __device__ int16_t PARTICLE_DGROUP_MAP[8] = {130, 22, -211, 211, 11, -11, 13, -13};
    constexpr __device__ float MASS_DMAP[8] =  {0.5, 0.0, 0.13, 0.13, 0.0005, 0.0005, 0.105, 0.105};
    constexpr __device__ l1t::PFCandidate::ParticleType PARTICLE_DTYPE_MAP[8] = {
    l1t::PFCandidate::NeutralHadron,
    l1t::PFCandidate::Photon,
    l1t::PFCandidate::ChargedHadron,
    l1t::PFCandidate::ChargedHadron,
    l1t::PFCandidate::Electron,
    l1t::PFCandidate::Electron,
    l1t::PFCandidate::Muon,
    l1t::PFCandidate::Muon
    };

    __device__ void ReadShared(const uint64_t data, uint16_t& pt, int16_t& eta, int16_t& phi);
    __device__ void ReadShared(const uint64_t data, float& pt, float& eta, float& phi);
    __global__ void ReadSharedBatch(const uint64_t* data, size_t size, uint16_t* pts, int16_t* etas, int16_t* phis);
    __global__ void ReadSharedBatch(const uint64_t* data, size_t size, float* pts, float* etas, float* phis);
    __device__ void DecodePID(const uint64_t& data, uint8_t& decoded);
    __global__ void DecodePIDsBatch(const uint64_t* data, uint8_t* decoded, size_t size);
    __device__ void MapParticle(uint8_t pid, int16_t& mapped_pid);
    __global__ void MapParticleBatch(const uint8_t* pids, int16_t* mapped_pids, size_t size);
    __device__ void MapType(const uint8_t pid, l1t::PFCandidate::ParticleType& type);
    __global__ void MapTypeBatch(const uint8_t* pids, l1t::PFCandidate::ParticleType* types, size_t size);
    __device__ void MapMass(const uint8_t pid, float& mass);
    __global__ void MapMassBatch(const uint8_t* pids, float* mapped_masses, size_t size);
    __device__ void MapCharge(const uint8_t pid, int8_t& charge);
    __global__ void MapChargeBatch(const uint8_t* pids, int8_t* charges, size_t size);
    
}  // namespace puppi_unpack


#endif  // L1TriggerScouting_Phase2_puppi_unpack_h