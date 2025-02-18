// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Decoder.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

class DecodeHeadersKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, PuppiCollection::View out, size_t size) const {
    if (once_per_grid(acc)) { // prefix sum sequential workaround
      out.clusters_density() = 0;
      out.offsets()[0] = 0;
      for (uint32_t idx = 1; idx < out.offsets().size(); idx++) {
        if (idx <= size)
          out.offsets()[idx] = out.offsets()[idx-1] + static_cast<uint32_t>(data[idx-1] & 0xFFF);  
        else 
          out.offsets()[idx] = 0xFFFFFFFF;
      }
    }
    
    for (int32_t idx : uniform_elements(acc, size)) {
      out.bx()[idx] = static_cast<uint16_t>((data[idx] >> 12) & 0xFFF);
    }
  }
};

void Decoder::DecodeHeaders(Queue& queue, WorkDiv<Dim1D>& grid, std::vector<uint64_t>& data, PuppiCollection& collection) const {
  size_t size = data.size();
  auto device_buffer = CopyToDevice<uint64_t>(queue, data);
  std::vector<uint64_t>().swap(data);
  alpaka::exec<Acc1D>(queue, grid, DecodeHeadersKernel{}, device_buffer.data(), collection.view(), size);
}

class DecodeDataKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, PuppiCollection::View out) const {
    static constexpr int16_t PARTICLE_DGROUP_MAP[8] = {130, 22, -211, 211, 11, -11, 13, -13};
    static constexpr float PI_C = 3.14159265358979323846 / 720.0f;
    for (int32_t idx : uniform_elements(acc, out.metadata().size())) {
      uint64_t bits64 = data[idx];
      // out[idx].pt() = 0.25f * (bits64 & 0x3FFF);

      // readshared
      out.cluster_association()[idx] = 0;
      uint16_t ptint = bits64 & 0x3FFF;
      out.pt()[idx] = ptint * 0.25f;
      int etaint = ((bits64 >> 25) & 1) ? ((bits64 >> 14) | (-0x800)) : ((bits64 >> 14) & (0xFFF));
      out.eta()[idx] = etaint * float(M_PI / 720.);
      int phiint = ((bits64 >> 36) & 1) ? ((bits64 >> 26) | (-0x400)) : ((bits64 >> 26) & (0x7FF));
      out.phi()[idx] = phiint * float(M_PI / 720.);

      // out.eta()[idx] = PI_C * (((bits64 >> 25) & 1) ? ((bits64 >> 14) | (-0x800)) : ((bits64 >> 14) & (0xFFF)));
      // out.phi()[idx] = PI_C * (((bits64 >> 36) & 1) ? ((bits64 >> 26) | (-0x400)) : ((bits64 >> 26) & (0x7FF)));
      int16_t pid = (bits64 >> 37) & 0x7;
      out.pdgId()[idx] = PARTICLE_DGROUP_MAP[pid];

      if (pid > 1) { // Charged particle
        int z0int = ((bits64 >> 49) & 1) ? ((bits64 >> 40) | (-0x200)) : ((bits64 >> 40) & 0x3FF);
        out.z0()[idx] = z0int * .05f;
        int dxyint = ((bits64 >> 57) & 1) ? ((bits64 >> 50) | (-0x100)) : ((bits64 >> 50) & 0xFF);
        out.dxy()[idx] = dxyint * 0.05f; 
        out.quality()[idx] = (bits64 >> 58) & 0x7;
        out.puppiw()[idx] = 1.0f;
      } else {  // Neutral particle
        out.z0()[idx] = 0.0f;
        out.dxy()[idx] = 0.0f;
        int wpuppiint = (bits64 >> 40) & 0x3FF;
        out.puppiw()[idx] = wpuppiint * (1 / 256.f);
        out.quality()[idx] = (bits64 >> 50) & 0x3F;
      }
    }
  }
};

void Decoder::DecodeData(Queue& queue, WorkDiv<Dim1D>& grid, std::vector<uint64_t>& data, PuppiCollection& collection) const {
  auto device_buffer = CopyToDevice<uint64_t>(queue, data);
  std::vector<uint64_t>().swap(data);
  alpaka::exec<Acc1D>(queue, grid, DecodeDataKernel{}, device_buffer.data(), collection.view());
}

void Decoder::Decode(Queue& queue, std::vector<uint64_t>& headers, std::vector<uint64_t>& data, PuppiCollection& collection) {
  uint32_t threads_per_block = 1024;
  uint32_t blocks_per_grid = divide_up_by(data.size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

  DecodeHeaders(queue, grid, headers, collection);
  DecodeData(queue, grid, data, collection);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
