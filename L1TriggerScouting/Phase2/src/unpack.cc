#include "L1TriggerScouting/Phase2/interface/unpack.h"
#include <omp.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// UNPACK BASE /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace unpack {

void assignpdgid(uint8_t pid, short int &pdgid) {
  pdgid = PARTICLE_DGROUP_MAP[pid];
};

void assignCMSSWPFCandidateId(uint8_t pid, l1t::PFCandidate::ParticleType &id) {
  id = PARTICLE_DTYPE_MAP[pid];
};

void assignmass(uint8_t pid, float &mass) {
  mass = MASS_DMAP[pid];
};

void assigncharge(uint8_t pid, int &charge) {
  charge = (pid > 1 ? (pid & 1 ? +1 : -1) : 0);
};

void readshared(const uint64_t data, uint16_t &pt, int16_t &eta, int16_t &phi) {
  pt = data & 0x3FFF;
  eta = ((data >> 25) & 1) ? ((data >> 14) | (-0x800)) : ((data >> 14) & (0xFFF));
  phi = ((data >> 36) & 1) ? ((data >> 26) | (-0x400)) : ((data >> 26) & (0x7FF));
}; // int

void readshared(const uint64_t data, float &pt, float &eta, float &phi) {
  uint16_t ptint = data & 0x3FFF;
  pt = ptint * 0.25f;
  int etaint = ((data >> 25) & 1) ? ((data >> 14) | (-0x800)) : ((data >> 14) & (0xFFF));
  eta = etaint * PRECOMPUTED_PI_720;
  int phiint = ((data >> 36) & 1) ? ((data >> 26) | (-0x400)) : ((data >> 26) & (0x7FF));
  phi = phiint * PRECOMPUTED_PI_720;
}; // float

void readcharged(const uint64_t data, int16_t &z0, int8_t &dxy, uint8_t &quality) {
  z0 = ((data >> 49) & 1) ? ((data >> 40) | (-0x200)) : ((data >> 40) & 0x3FF);
  dxy = ((data >> 57) & 1) ? ((data >> 50) | (-0x100)) : ((data >> 50) & 0xFF);
  quality = (data >> 58) & 0x7;  //3 bits
}; // int

void readcharged(const uint64_t data, float &z0, float &dxy, uint8_t &quality) {
  int z0int = ((data >> 49) & 1) ? ((data >> 40) | (-0x200)) : ((data >> 40) & 0x3FF);
  z0 = z0int * .05f;  //conver to centimeters

  int dxyint = ((data >> 57) & 1) ? ((data >> 50) | (-0x100)) : ((data >> 50) & 0xFF);
  dxy = dxyint * 0.05f;          // PLACEHOLDER
  quality = (data >> 58) & 0x7;  //3 bits
}; // float

void readcharged(const uint64_t data, uint8_t pid, float &z0, float &dxy) {
  int z0int = ((data >> 49) & 1) ? ((data >> 40) | (-0x200)) : ((data >> 40) & 0x3FF);
  z0 = (pid > 1) * z0int * .05f;  //conver to centimeters
  int dxyint = ((data >> 57) & 1) ? ((data >> 50) | (-0x100)) : ((data >> 50) & 0xFF);
  dxy = (pid > 1) * dxyint * 0.05f;  // PLACEHOLDER
}; // float

void readneutral(const uint64_t data, uint16_t &wpuppi, uint8_t &id) {
  wpuppi = (data >> 40) & 0x3FF;
  id = (data >> 50) & 0x3F;
};

void readneutral(const uint64_t data, float &wpuppi, uint8_t &id) {
  int wpuppiint = (data >> 40) & 0x3FF;
  wpuppi = wpuppiint * (1 / 256.f);
  id = (data >> 50) & 0x3F;
};

void readneutral(const uint64_t data, uint8_t pid, float &wpuppi) {
  int wpuppiint = (data >> 40) & 0x3FF;
  wpuppi = pid > 1 ? wpuppiint * float(1 / 256.f) : 1.0f;
};

}  // namespace unpack

namespace unpack::batch {  

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
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; i++) {
    ReadShared(data[i], pts[i], etas[i], phis[i]);
  }
}

void ReadSharedBatch(const uint64_t* data, const std::size_t size, float* pts, float* etas, float* phis) {
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; i++) {
    ReadShared(data[i], pts[i], etas[i], phis[i]);
  }
}

void DecodePID(const uint64_t &data, std::uint8_t &decoded) {
  decoded = data >> 37 & 0x7;
}

void DecodePIDsBatch(const std::uint64_t *data, std::uint8_t *decoded, const std::size_t size) {
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; i++) {
    DecodePID(data[i], decoded[i]);
  }
}

void MapParticle(std::uint8_t pid, std::int16_t &mapped_pid) {
  mapped_pid = PARTICLE_DGROUP_MAP[pid];
}

void MapParticleBatch(const std::uint8_t* pids, std::int16_t* mapped_pids, const std::size_t size) {
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; ++i) {
    MapParticle(pids[i], mapped_pids[i]);
  }
}

void MapType(const std::uint8_t pid, l1t::PFCandidate::ParticleType &type) {
  type = PARTICLE_DTYPE_MAP[pid];
}

void MapTypeBatch(const std::uint8_t* pids, l1t::PFCandidate::ParticleType* types, const std::size_t size) {
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; ++i) {
    MapType(pids[i], types[i]);
  }
}

void MapMass(const std::uint8_t pid, float &mass) {
  mass = MASS_DMAP[pid];
}

void MapMassBatch(const std::uint8_t* pids, float* mapped_masses, const std::size_t size) {
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; ++i) {
    MapMass(pids[i], mapped_masses[i]);
  }
}

void MapCharge(const std::uint8_t pid, std::int8_t &charge) {
  charge = static_cast<std::int8_t>((pid > 1) * (2 * (pid & 1) - 1));
}


void MapChargeBatch(const std::uint8_t* pids, std::int8_t* charges, const std::size_t size) {
  #pragma omp parallel for
  for (std::size_t i = 0; i < size; ++i) {
    MapCharge(pids[i], charges[i]);
  }
}

}  // namespace unpack::batch

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// BOOSTED CPU ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace unpack::boosted::cpu {

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

#if defined(__AVX2__) || defined(__AVX__)
void DecodePIDsBatch(const std::uint64_t *data, std::uint8_t *decoded, const std::size_t size) {
  if (size <= 0 || decoded == nullptr || data == nullptr)
    throw std::invalid_argument("mem corrupted for in buffers");
  std::size_t it;
  constexpr std::size_t parallel_items = 4;  // 4x 64-bit
  const __m256i mask_val = _mm256_set1_epi64x(0x7);  // lower 3 bits

  for (it = 0; it + parallel_items <= size; it += parallel_items) {
    const __m256i r_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data[it]));
    const __m256i shifted_data = _mm256_srli_epi64(r_data, 37);
    const __m256i masked_data = _mm256_and_si256(shifted_data, mask_val);
    std::uint64_t tmp[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), masked_data);
    decoded[it + 0] = static_cast<std::uint8_t>(tmp[0]);
    decoded[it + 1] = static_cast<std::uint8_t>(tmp[1]);
    decoded[it + 2] = static_cast<std::uint8_t>(tmp[2]);
    decoded[it + 3] = static_cast<std::uint8_t>(tmp[3]);
  }

  for (; it < size; it++) {
    DecodePID(data[it], decoded[it]);
  }
}
#endif

void MapParticle(std::uint8_t pid, std::int16_t &mapped_pid) {
  mapped_pid = PARTICLE_DGROUP_MAP[pid];
}

#if defined(__AVX512F__) && defined(__AVX512CD__)
void MapParticleBatch(const std::uint8_t *pids, std::int16_t *mapped_pids, const std::size_t size) {
  std::size_t it;
  constexpr std::uint8_t mem_register_size = 32; // 32 * 8-bit integers in a 512-bit register (AVX512)

  for (it = 0; it + mem_register_size < size; it += mem_register_size) {
    const __m512i r_data = _mm512_loadu_si512(&pids[it]);

    // TODO: this should be done in safer manner. Refactor unnecessary moves between registers.
    uint8_t bytes[64];
    _mm512_storeu_si512(&bytes[0], r_data);

    const __m512i out = _mm512_set_epi16(
        PARTICLE_DGROUP_MAP[bytes[31]],
        PARTICLE_DGROUP_MAP[bytes[30]],
        PARTICLE_DGROUP_MAP[bytes[29]],
        PARTICLE_DGROUP_MAP[bytes[28]],
        PARTICLE_DGROUP_MAP[bytes[27]],
        PARTICLE_DGROUP_MAP[bytes[26]],
        PARTICLE_DGROUP_MAP[bytes[25]],
        PARTICLE_DGROUP_MAP[bytes[24]],
        PARTICLE_DGROUP_MAP[bytes[23]],
        PARTICLE_DGROUP_MAP[bytes[22]],
        PARTICLE_DGROUP_MAP[bytes[21]],
        PARTICLE_DGROUP_MAP[bytes[20]],
        PARTICLE_DGROUP_MAP[bytes[19]],
        PARTICLE_DGROUP_MAP[bytes[18]],
        PARTICLE_DGROUP_MAP[bytes[17]],
        PARTICLE_DGROUP_MAP[bytes[16]],
        PARTICLE_DGROUP_MAP[bytes[15]],
        PARTICLE_DGROUP_MAP[bytes[14]],
        PARTICLE_DGROUP_MAP[bytes[13]],
        PARTICLE_DGROUP_MAP[bytes[12]],
        PARTICLE_DGROUP_MAP[bytes[11]],
        PARTICLE_DGROUP_MAP[bytes[10]],
        PARTICLE_DGROUP_MAP[bytes[9]],
        PARTICLE_DGROUP_MAP[bytes[8]],
        PARTICLE_DGROUP_MAP[bytes[7]],
        PARTICLE_DGROUP_MAP[bytes[6]],
        PARTICLE_DGROUP_MAP[bytes[5]],
        PARTICLE_DGROUP_MAP[bytes[4]],
        PARTICLE_DGROUP_MAP[bytes[3]],
        PARTICLE_DGROUP_MAP[bytes[2]],
        PARTICLE_DGROUP_MAP[bytes[1]],
        PARTICLE_DGROUP_MAP[bytes[0]]
    );

    _mm512_storeu_si512(&mapped_pids[it], out);
  }

  for (; it < size; it++) {
    mapped_pids[it] = PARTICLE_DGROUP_MAP[pids[it]];
  }
}
#elif defined(__AVX2__) || defined(__AVX__)
void MapParticleBatch(const std::uint8_t *pids, std::int16_t *mapped_pids, const std::size_t size) {
  std::size_t it;
  constexpr std::uint8_t parallel_items = 16;  // Process 16 items per iteration (128 bits)

  // Broadcast base address of PARTICLE_DGROUP_MAP for gather operation
  const int *base_ptr = PARTICLE_DGROUP_MAP;  // Assuming PARTICLE_DGROUP_MAP is int32_t for compatibility with gather

  for (it = 0; it + parallel_items <= size; it += parallel_items) {
    // Load 16 uint8_t values from pids into a 256-bit register
    __m256i pids_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pids[it]));

    // Convert the 8-bit integers (uint8_t) in pids_vec to 32-bit integers for the gather operation
    __m256i pids_int32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(pids_vec));

    // Gather the corresponding elements from PARTICLE_DGROUP_MAP using the gathered pids
    __m256i mapped_pids_vec = _mm256_i32gather_epi32(base_ptr, pids_int32, 4);  // 4 is the byte offset (int32_t)

    // Store the results back into the mapped_pids array
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&mapped_pids[it]), mapped_pids_vec);
  }

  for (; it < size; it++) {
    mapped_pids[it] = PARTICLE_DGROUP_MAP[pids[it]];
  }
}
// void MapParticleBatch(const std::uint8_t *pids, std::int16_t *mapped_pids, const std::size_t size) {
//   std::size_t it;
//   constexpr std::uint8_t parallel_items = 16;

//   for (it = 0; it + parallel_items <= size; it += parallel_items) {
//     const __m256i r_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pids[it]));
//     const __m256i out = _mm256_set_epi16(
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 15)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 14)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 13)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 12)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 11)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 10)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 9)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 8)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 7)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 6)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 5)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 4)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 3)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 2)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 1)],
//       PARTICLE_DGROUP_MAP[_mm256_extract_epi8(r_data, 0)]
//     );
//     _mm256_storeu_si256(reinterpret_cast<__m256i *>(&mapped_pids[it]), out);
//   }

//   for (; it < size; it++) {
//     mapped_pids[it] = PARTICLE_DGROUP_MAP[pids[it]];
//   }
// }
#endif

void MapType(const std::uint8_t pid, l1t::PFCandidate::ParticleType &type) {
  type = PARTICLE_DTYPE_MAP[pid];
}

#if defined(__AVX2__) || defined(__AVX__)
void MapTypeBatch(const std::uint8_t *pids, l1t::PFCandidate::ParticleType *mapped_pids, const std::size_t size) {
  std::size_t it;
  constexpr std::uint8_t parallel_items = 32;

  for (it = 0; it + parallel_items <= size; it += parallel_items) {
    const __m256i r_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pids[it]));
    const __m256i out = _mm256_set_epi8(
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 31)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 30)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 29)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 28)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 27)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 26)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 25)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 24)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 23)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 22)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 21)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 20)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 19)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 18)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 17)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 16)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 15)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 14)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 13)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 12)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 11)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 10)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 9)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 8)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 7)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 6)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 5)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 4)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 3)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 2)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 1)],
      PARTICLE_DTYPE_MAP[_mm256_extract_epi8(r_data, 0)]
    );
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&mapped_pids[it]), out);
  }

  for (; it < size; it++) {
    mapped_pids[it] = static_cast<l1t::PFCandidate::ParticleType>(PARTICLE_DTYPE_MAP[pids[it]]);
  }
}
#endif

void MapMass(const std::uint8_t pid, float &mass) {
  mass = MASS_DMAP[pid];
}

#if defined(__AVX2__) || defined(__AVX__)
void MapMassBatch(const std::uint8_t *pids, float *masses, const std::size_t size) {
  std::size_t it;
  constexpr std::uint8_t parallel_items = 8;  // 8 floats fits in 256 bits registers

  for (it = 0; it + parallel_items <= size; it += parallel_items) {
    const __m256i r_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pids[it]));
    const __m256 out = _mm256_set_ps(
      MASS_DMAP[_mm256_extract_epi8(r_data, 7)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 6)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 5)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 4)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 3)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 2)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 1)],
      MASS_DMAP[_mm256_extract_epi8(r_data, 0)]
    );
    _mm256_storeu_ps(&masses[it], out);
  }

  for (; it < size; it++) {
    masses[it] = MASS_DMAP[pids[it]];
  }
}
#endif

void MapCharge(const std::uint8_t pid, std::int8_t &charge) {
  charge = static_cast<std::int8_t>((pid > 1) * (2 * (pid & 1) - 1));
}

#if defined(__AVX512F__) && defined(__AVX512CD__)
void MapChargeBatch(const std::uint8_t* pids, std::int8_t* charges, const size_t size) {
  const __m512i v_pos = _mm512_set1_epi8(1);
  const __m512i v_neg = _mm512_set1_epi8(-1);

  std::size_t it;
  constexpr std::uint8_t parallel_items = 64;

  for (it = 0; it + parallel_items <= size; it += parallel_items) {
    const __m512i pids_simd = _mm512_loadu_si512(&pids[it]);
    const __mmask64 pid_gt_1 = _mm512_cmpgt_epu8_mask(pids_simd, _mm512_set1_epi8(1));
    const __m512i pid_and_1 = _mm512_and_si512(pids_simd, v_pos);
    const __mmask64 is_positive = _mm512_cmpeq_epu8_mask(pid_and_1, v_pos);
    auto result = _mm512_mask_blend_epi8(is_positive, v_neg, v_pos);
    result = _mm512_maskz_mov_epi8(pid_gt_1, result);
    _mm512_storeu_si512(&charges[it], result);
  }

  for (; it < size; it++) {
    MapCharge(pids[it], charges[it]);
  }
}
#elif defined(__AVX2__) || defined(__AVX__)
void MapChargeBatch(const std::uint8_t* pids, std::int8_t* charges, const size_t size) {
  const __m256i v_pos = _mm256_set1_epi8(1);
  const __m256i v_neg = _mm256_set1_epi8(-1);

  std::size_t it;
  constexpr std::uint8_t parallel_items = 32;
  _mm_prefetch(reinterpret_cast<const char*>(pids), _MM_HINT_T0);

  for (it = 0; it + parallel_items <= size; it += parallel_items) {
    _mm_prefetch(reinterpret_cast<const char*>(&pids[it + parallel_items]), _MM_HINT_T0);

    const __m256i pids_simd = _mm256_load_si256(reinterpret_cast<const __m256i *>(&pids[it]));
    const __m256i pid_gt_1 = _mm256_cmpgt_epi8(pids_simd, _mm256_set1_epi8(1));
    const __m256i pid_and_1 = _mm256_and_si256(pids_simd, v_pos);
    const __m256i is_positive = _mm256_cmpeq_epi8(pid_and_1, v_pos);
    __m256i result = _mm256_blendv_epi8(v_neg, v_pos, is_positive);
    result = _mm256_and_si256(result, pid_gt_1);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&charges[it]), result);
  }

  for (; it < size; it++) {
    MapCharge(pids[it], charges[it]);
  }
}
#endif

}  // namespace unpack::bosted::cpu
