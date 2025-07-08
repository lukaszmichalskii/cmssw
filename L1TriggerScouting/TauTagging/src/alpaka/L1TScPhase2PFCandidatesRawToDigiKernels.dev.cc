#include "L1TriggerScouting/TauTagging/interface/alpaka/L1TScPhase2PFCandidatesRawToDigiKernels.h"

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

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

    // for (int32_t idx : uniform_elements(acc, out.metadata().size())) {
    //   uint64_t b = data[idx];

    //   uint16_t ptint = b & 0x3FFF;
    //   int etaint = ((b >> 25) & 1) ? ((b >> 14) | (-0x800)) : ((b >> 14) & 0xFFF);
    //   int phiint = ((b >> 36) & 1) ? ((b >> 26) | (-0x400)) : ((b >> 26) & 0x7FF);
    //   int16_t pid = (b >> 37) & 0x7;

    //   out.pt()[idx] = ptint * 0.25f;
    //   out.eta()[idx] = etaint * PI_C;
    //   out.phi()[idx] = phiint * PI_C;
    //   out.pdgId()[idx] = PARTICLE_DGROUP_MAP[pid];

    //   bool isCharged = pid > 1;
    //   int z0int = ((b >> 49) & 1) ? ((b >> 40) | (-0x200)) : ((b >> 40) & 0x3FF);
    //   int dxyint = ((b >> 57) & 1) ? ((b >> 50) | (-0x100)) : ((b >> 50) & 0xFF);
    //   int wpuppiint = (b >> 40) & 0x3FF;

    //   out.z0()[idx] = isCharged ? z0int * 0.05f : 0.0f;
    //   out.dxy()[idx] = isCharged ? dxyint * 0.05f : 0.0f;
    //   out.puppiw()[idx] = isCharged ? 1.0f : wpuppiint * (1 / 256.f);
    //   out.quality()[idx] = isCharged ? ((b >> 58) & 0x7) : ((b >> 50) & 0x3F);
    //   out.selection()[idx] = 0;
    // }

  class RawToDigiKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, data_t* pf_data, PFCandidateCollection::View pf_candidates) const {
        constexpr int16_t PARTICLE_DGROUP_MAP[8] = {
            130, 22, -211, 211, 11, -11, 13, -13};
        constexpr float PI_720 = alpaka::math::constants::pi / 720.0f;
        
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("RawToDigiKernel OK\n");
        }

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
              printf("[%d] | %3.2f | %3.2f | %3.2f |\n", idx, 
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
