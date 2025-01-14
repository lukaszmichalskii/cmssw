// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "PuppiUnpack.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  template<typename T>
  auto CopyToDevice(Queue &queue, std::vector<T>& data) {
    PlatformHost platform;
    DevHost host = alpaka::getDevByIdx(platform, 0);
    // Copy data to device
    Vec<alpaka::DimInt<1>> extent(data.size());
    auto device_buffer = alpaka::allocAsyncBuf<T, Idx>(queue, extent);
    auto host_buffer = createView(host, data, extent); // alpaka::View can be used instead of alpaka::Buf
    alpaka::memcpy(queue, device_buffer, host_buffer);  
    alpaka::wait(queue);
    return device_buffer;
  }

  class ProcessHeadersKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, PuppiCollection::View out, size_t size) const {
      if (once_per_grid(acc)) { // prefix sum sequential workaround
        out.offsets()[0] = 0;
        for (uint32_t idx = 1; idx < out.offsets().size(); idx++) {
          if (idx <= size)
            out.offsets()[idx] = out.offsets()[idx-1] + static_cast<uint32_t>(data[idx-1] & 0xFFF);  
          else 
            out.offsets()[idx] = 0xFFFFFFFF;
          // printf("%d -> %d (%d)\n", idx, out.offsets()[idx-1] + static_cast<uint32_t>(data[idx-1] & 0xFFF), static_cast<uint32_t>(data[idx-1] & 0xFFF));
        }
      }
      for (int32_t idx : uniform_elements(acc, size)) {
        out.bx()[idx] = static_cast<uint16_t>((data[idx] >> 12) & 0xFFF);
      }
    }
  };

  void PuppiUnpack::ProcessHeaders(
      Queue& queue, std::vector<uint64_t>& data, PuppiCollection& collection) const {
    size_t size = data.size();
    auto device_buffer = CopyToDevice<uint64_t>(queue, data);
    std::vector<uint64_t>().swap(data);

    uint32_t threads_per_block = 64;
    uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, ProcessHeadersKernel{}, device_buffer.data(), collection.view(), size);
  }

  class ProcessDataKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, PuppiCollection::View out) const {
      static constexpr int16_t PARTICLE_DGROUP_MAP[8] = {130, 22, -211, 211, 11, -11, 13, -13};
      static constexpr float PI_C = 3.14159265358979323846 / 720.0f;
      for (int32_t idx : uniform_elements(acc, out.metadata().size())) {
        uint64_t bits64 = data[idx];
        // out[idx].pt() = 0.25f * (bits64 & 0x3FFF);

        // readshared
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

  void PuppiUnpack::ProcessData(
      Queue& queue, std::vector<uint64_t>& data, PuppiCollection& collection) const {
    size_t size = data.size();
    auto device_buffer = CopyToDevice<uint64_t>(queue, data);
    std::vector<uint64_t>().swap(data);

    uint32_t threads_per_block = 64;
    uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, ProcessDataKernel{}, device_buffer.data(), collection.view());
  }

  class FillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::View view, int value) const {
      for (int32_t idx : uniform_elements(acc, view.metadata().size())) {
        view[idx].pt() = static_cast<float>(value);
        view[idx].eta() = static_cast<float>(value);
        view[idx].phi() = static_cast<float>(value);
        view[idx].z0() = static_cast<float>(value);
        view[idx].dxy() = static_cast<float>(value);
        view[idx].puppiw() = static_cast<float>(value);
        view[idx].pdgId() = static_cast<int16_t>(value);
        view[idx].quality() = static_cast<uint8_t>(value);
      }
    }
  };

  void PuppiUnpack::Fill(Queue& queue, PuppiCollection& collection, int value) const {
    uint32_t items = 64;
    uint32_t groups = divide_up_by(collection->metadata().size(), items);
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view(), value);
  }

  class AssertKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView view, int value) const {
      for (int32_t idx : uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view.bx().size() == 3564);
        ALPAKA_ASSERT_ACC(view.offsets().size() == 3564+1);
        ALPAKA_ASSERT_ACC(view[idx].pt() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].eta() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].phi() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].z0() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].dxy() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].puppiw() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].pdgId() == static_cast<int16_t>(value));
        ALPAKA_ASSERT_ACC(view[idx].quality() == static_cast<uint8_t>(value));
      }
    }
  };

  void PuppiUnpack::Assert(Queue& queue, PuppiCollection const& collection, int value) const {
    auto workDiv = make_workdiv<Acc1D>(1, 32);
    alpaka::exec<Acc1D>(queue, workDiv, AssertKernel{}, collection.const_view(), value);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
