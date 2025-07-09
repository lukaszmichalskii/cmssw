#include "L1TriggerScouting/TauTagging/interface/alpaka/L1TScPhase2PFCandidatesRawToDigiKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // CUDA always has a warp size of 32
  inline constexpr int warpSize = 32;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  // HIP/ROCm defines warpSize as a constant expression in device code, with value 32 or 64 depending on the target device
  inline constexpr int warpSize = ::warpSize;
#else
  // CPU back-ends always have a warp size of 1
  inline constexpr int warpSize = 1;
#endif

  template <typename T>
  ALPAKA_FN_ACC T decodeBits(uint64_t word, unsigned int start, unsigned int width) {
    static_assert(std::is_integral<T>::value, "extract_unsigned_bits expects integral types");
    uint64_t mask = (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
    return static_cast<T>((word >> start) & mask);
  }

  template <typename T>
  ALPAKA_FN_ACC T decodeBitsSigned(uint64_t word, unsigned int start, unsigned int width) {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "extract_signed_bits expects signed integral types");
    uint64_t raw = (word >> start) & ((1ULL << width) - 1);
    if (raw & (1ULL << (width - 1))) 
      raw |= (~0ULL << width);  // manual sign extension
    return static_cast<T>(raw);
  }

  class RawToDigiKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, data_t* pf_data, PFCandidateCollection::View pf_candidates) const {
        constexpr int16_t PARTICLE_DGROUP_MAP[8] = {
            130, 22, -211, 211, 11, -11, 13, -13};
        constexpr float PI_720 = alpaka::math::constants::pi / 720.0f;

        for (int32_t idx : cms::alpakatools::uniform_elements(acc, pf_candidates.metadata().size())) {
          uint64_t data = pf_data[idx];

          auto hwPt = decodeBits<uint16_t>(data, 0, 14);
          auto hwEta = decodeBitsSigned<int16_t>(data, 14, 12);
          auto hwPhi = decodeBitsSigned<int16_t>(data, 26, 11);
          auto pid = decodeBits<uint8_t>(data, 37, 3);

          pf_candidates.pt()[idx] = hwPt * 0.25f;
          pf_candidates.eta()[idx] = hwEta * PI_720;
          pf_candidates.phi()[idx] = hwPhi * PI_720;
          pf_candidates.pdgid()[idx] = PARTICLE_DGROUP_MAP[pid];

          if (pid > 1) {
            auto hwZ0 = decodeBitsSigned<int16_t>(data, 40, 10);
            auto hwDxy = decodeBitsSigned<int16_t>(data, 50, 8);
            // auto hwQual = decodeBits<uint8_t>(data, 58, 3);

            pf_candidates.z0()[idx] = hwZ0 * 0.05f;
            pf_candidates.dxy()[idx] = hwDxy * 0.00390625f;
            // pf_candidates.puppiw()[idx] = 1.0f;
            // pf_candidates.quality()[idx] = hwQual;
          } else {
            // auto hwPuppiw = decodeBits<uint16_t>(data, 40, 10);
            // auto hwQual = decodeBits<uint8_t>(data, 50, 6);

            pf_candidates.z0()[idx] = 0.0f;
            pf_candidates.dxy()[idx] = 0.0f;
            // pf_candidates.puppiw()[idx] = hwPuppiw * (1.0f / 256.0f);
            // pf_candidates.quality()[idx] = hwQual;
          }

          // pf_candidates.selection()[idx] = 0;
        }
      }
  };

  void RawToDigi(
      Queue& queue, data_t* pf_data, PFCandidateCollection& pf_candidates) {
    auto extent = Vec1D(pf_candidates.const_view().metadata().size());    
    auto pf_data_device = alpaka::allocAsyncBuf<data_t, Idx>(queue, extent);
    alpaka::memcpy(queue, pf_data_device, createView(cms::alpakatools::host(), pf_data, extent));  

    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(
        pf_candidates.const_view().metadata().size(), threads_per_block);      
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, RawToDigiKernel{}, pf_data_device.data(), pf_candidates.view());    
  }

  void AssociateOrbitEventIndex(
      Queue& queue, data_t *h_data, OrbitEventIndexMapCollection& orbit_association_map) {
    auto extent = Vec1D(orbit_association_map.const_view().metadata().size());    
    auto h_data_device = alpaka::allocAsyncBuf<data_t, Idx>(queue, extent);
    alpaka::memcpy(queue, h_data_device, createView(cms::alpakatools::host(), h_data, extent));  

    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(
        orbit_association_map.const_view().metadata().size(), threads_per_block);      
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(
        queue, 
        grid, 
        [] ALPAKA_FN_ACC(Acc1D const &acc, data_t* data, OrbitEventIndexMapCollection::View orbit_association_map) {
          if (cms::alpakatools::once_per_grid(acc))
            orbit_association_map.offsets()[0] = 0;
          
          for (int32_t idx : cms::alpakatools::uniform_elements(
              acc, orbit_association_map.metadata().size() - 1)) {
            auto range = decodeBits<uint32_t>(data[idx], 0, 12);
            orbit_association_map.offsets()[idx + 1] = range;
          }
        },
        h_data_device.data(), 
        orbit_association_map.view());

    auto pc = alpaka::allocAsyncBuf<int32_t, Idx>(queue, Vec1D{1});
    alpaka::memset(queue, pc, 0x0);
    alpaka::exec<Acc1D>(
        queue, 
        grid, 
        cms::alpakatools::multiBlockPrefixScan<uint32_t>{}, 
        orbit_association_map.view().offsets() + 1, 
        orbit_association_map.view().offsets() + 1,
        orbit_association_map.view().metadata().size(),
        blocks_per_grid,
        pc.data(),
        warpSize);

    alpaka::exec<Acc1D>(
        queue, 
        grid, 
        [] ALPAKA_FN_ACC(Acc1D const &acc, OrbitEventIndexMapCollection::ConstView orbit_association_map) {
          if (cms::alpakatools::once_per_grid(acc)) {
            for (int32_t idx : cms::alpakatools::uniform_elements(acc, orbit_association_map.metadata().size() - 1)) {
              printf("%d -> [%d, %d]\n", idx, 
                  orbit_association_map.offsets()[idx], 
                  orbit_association_map.offsets()[idx + 1]);
            }
          }
        },
        orbit_association_map.const_view());
  }

  void PrintPFCandidateCollection(
      Queue& queue, PFCandidateCollection& pf_candidates) {
    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(
        pf_candidates.const_view().metadata().size(), threads_per_block);      
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(
        queue, 
        grid, 
        [] ALPAKA_FN_ACC(Acc1D const &acc, PFCandidateCollection::ConstView pf_candidates) {
          if (cms::alpakatools::once_per_grid(acc)) {
            int32_t head = 10;
            int32_t span = (pf_candidates.metadata().size() > head) ? head : pf_candidates.metadata().size();
            for (int32_t idx = 0; idx < span; ++idx) {
              printf("[%3d] | %6.2f | %7.2f | %7.2f |\n", idx, 
                  pf_candidates.pt()[idx], 
                  pf_candidates.eta()[idx], 
                  pf_candidates.phi()[idx]);
            }
            printf("PFCandidateCollection size: %d\n", pf_candidates.metadata().size());
          }
        },
        pf_candidates.const_view());    
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels
