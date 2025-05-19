// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Unpack.h"


namespace cms::alpakatools {

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

}  // namespace cms::alpakatools

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

class UnpackHeadersKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, PuppiCollection::View out, size_t size) const {
    if (once_per_grid(acc)) {
      out.offsets()[0] = 0;
    }
    
    for (int32_t idx : uniform_elements(acc, size)) {
      if (idx < static_cast<int>(size)) {
        out.bx()[idx] = static_cast<uint16_t>((data[idx] >> 12) & 0xFFF);
        // aggregate -> to utilize parallel prefix sum
        uint32_t len = static_cast<uint32_t>(data[idx] & 0xFFF);
        out.offsets()[idx + 1] = len;
      }
    }
  }
};

void Unpack::UnpackHeaders(
    Queue& queue, std::vector<uint64_t>& data, PuppiCollection& collection) const {
  size_t size = data.size();
  auto device_buffer = CopyToDevice<uint64_t>(queue, data);
  std::vector<uint64_t>().swap(data);

  uint32_t threads_per_block = 1024;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

  // unpack
  alpaka::exec<Acc1D>(queue, grid, UnpackHeadersKernel{}, device_buffer.data(), collection.view(), size);

  // prefix sum -> one to many associator for batching
  auto pc = alpaka::allocAsyncBuf<int32_t, Idx>(queue, Vec<alpaka::DimInt<1>>{1});
  alpaka::memset(queue, pc, 0x0);
  alpaka::exec<Acc1D>(
      queue, 
      grid, 
      multiBlockPrefixScan<uint32_t>{}, 
      collection.view().offsets().data() + 1, 
      collection.view().offsets().data() + 1,
      size,
      blocks_per_grid,
      pc.data(),
      cms::alpakatools::warpSize);
}

class UnpackDataKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, PuppiCollection::View out) const {
    constexpr int16_t PARTICLE_DGROUP_MAP[8] = {130, 22, -211, 211, 11, -11, 13, -13};
    constexpr float PI_C = 3.14159265358979323846f / 720.0f;

    for (int32_t idx : uniform_elements(acc, out.metadata().size())) {
      uint64_t b = data[idx];

      uint16_t ptint = b & 0x3FFF;
      int etaint = ((b >> 25) & 1) ? ((b >> 14) | (-0x800)) : ((b >> 14) & 0xFFF);
      int phiint = ((b >> 36) & 1) ? ((b >> 26) | (-0x400)) : ((b >> 26) & 0x7FF);
      int16_t pid = (b >> 37) & 0x7;

      out.pt()[idx] = ptint * 0.25f;
      out.eta()[idx] = etaint * PI_C;
      out.phi()[idx] = phiint * PI_C;
      out.pdgId()[idx] = PARTICLE_DGROUP_MAP[pid];

      bool isCharged = pid > 1;
      int z0int = ((b >> 49) & 1) ? ((b >> 40) | (-0x200)) : ((b >> 40) & 0x3FF);
      int dxyint = ((b >> 57) & 1) ? ((b >> 50) | (-0x100)) : ((b >> 50) & 0xFF);
      int wpuppiint = (b >> 40) & 0x3FF;

      out.z0()[idx] = isCharged ? z0int * 0.05f : 0.0f;
      out.dxy()[idx] = isCharged ? dxyint * 0.05f : 0.0f;
      out.puppiw()[idx] = isCharged ? 1.0f : wpuppiint * (1 / 256.f);
      out.quality()[idx] = isCharged ? ((b >> 58) & 0x7) : ((b >> 50) & 0x3F);
      out.selection()[idx] = 0;
    }
  }
};

void Unpack::UnpackData(Queue& queue, std::vector<uint64_t>& data, PuppiCollection& collection) const {
  size_t size = data.size();
  auto device_buffer = CopyToDevice<uint64_t>(queue, data);
  std::vector<uint64_t>().swap(data);

  uint32_t threads_per_block = 1024;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, UnpackDataKernel{}, device_buffer.data(), collection.view());
}

void Unpack::Unpacking(Queue& queue, std::vector<uint64_t>& headers, std::vector<uint64_t>& data, PuppiCollection& collection) {
  // auto s1 = std::chrono::high_resolution_clock::now();
  UnpackHeaders(queue, headers, collection);
  // alpaka::wait(queue);
  // auto e1 = std::chrono::high_resolution_clock::now();
  // auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1);
  // std::cout << "I/O (headers): OK [" << duration1.count() << " us]" << std::endl;

  // auto s2 = std::chrono::high_resolution_clock::now();
  UnpackData(queue, data, collection);
  // alpaka::wait(queue);
  // auto e2 = std::chrono::high_resolution_clock::now();
  // auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2);
  // std::cout << "I/O (data): OK [" << duration2.count() << " us]" << std::endl;
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
