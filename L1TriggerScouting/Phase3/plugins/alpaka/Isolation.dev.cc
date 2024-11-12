// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Isolation.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

PlatformHost platform;
DevHost host = alpaka::getDevByIdx(platform, 0);

using namespace cms::alpakatools;


PuppiCollection Isolation::Isolate(Queue& queue, PuppiCollection const& raw_data) const {
  const size_t size = raw_data.view().metadata().size();
  // Prepare mask for filtering and isolation
  Vec<alpaka::DimInt<1>> extent(size);
  Vec<alpaka::DimInt<1>> var_extent(1); 

  // Allocate device memory
  auto device_mask = alpaka::allocAsyncBuf<uint8_t, Idx>(queue, extent);
  auto device_charge = alpaka::allocAsyncBuf<int8_t, Idx>(queue, extent);
  auto device_estimated_size = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_int_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_high_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto& device_offsets = raw_data.view().offsets();

  // Initialize device memory
  alpaka::memset(queue, device_mask, 0);
  alpaka::memset(queue, device_charge, 0);
  alpaka::memset(queue, device_estimated_size, 0);
  alpaka::memset(queue, device_int_cut_ct, 0);
  alpaka::memset(queue, device_high_cut_ct, 0);

  // Destination memory for data to be copied to debug and size estimation
  uint32_t* host_estimated_size = new uint32_t[1];
  uint32_t* host_int_cut_ct = new uint32_t[1];
  uint32_t* host_high_cut_ct = new uint32_t[1];
  std::vector<int8_t> host_charge(size);
  std::vector<uint8_t> host_mask(size);

  std::array<uint32_t, 3564+1> host_offsets{};
  alpaka::memcpy(queue, createView(host, host_offsets, Vec<alpaka::DimInt<1>>(3564+1)), createView(alpaka::getDev(queue), device_offsets, Vec<alpaka::DimInt<1>>(3564+1)));
  alpaka::wait(queue);

  // Combinatorics
  for (size_t bx_idx = 0; bx_idx < raw_data.const_view().bx().size(); bx_idx++) {
    auto begin = host_offsets[bx_idx];
    auto end = host_offsets[bx_idx+1];
    Filter(queue, raw_data.const_view(), begin, end, device_mask.data(), device_charge.data(), device_int_cut_ct.data(), device_high_cut_ct.data());
  }

  // Get number of particles that pass criteria
  EstimateSize(queue, device_mask.data(), size, device_estimated_size.data());
  
  alpaka::memcpy(queue, createView(host, host_estimated_size, Vec<alpaka::DimInt<1>>(1)), device_estimated_size);
  alpaka::memcpy(queue, createView(host, host_int_cut_ct, Vec<alpaka::DimInt<1>>(1)), device_int_cut_ct);
  alpaka::memcpy(queue, createView(host, host_high_cut_ct, Vec<alpaka::DimInt<1>>(1)), device_high_cut_ct);
  alpaka::memcpy(queue, createView(host, host_charge, extent), device_charge);
  alpaka::memcpy(queue, createView(host, host_mask, extent), device_mask);

  // Debug
  std::cout << "Particles Num L1 Filter: " << host_estimated_size[0] << std::endl;
  std::cout << "Paritcles Num L1 IntCut: " << host_int_cut_ct[0] << std::endl;
  std::cout << "Paritcles Num L1  HiCut: "  << host_high_cut_ct[0] << std::endl;
  std::cout << std::endl;

  // Return reduced particles set for further analysis
  PuppiCollection collection(host_estimated_size[0], queue);
  return collection;
}

template<typename TAcc>
ALPAKA_FN_ACC bool ConeIsolation(TAcc const& acc, int64_t thread_idx, PuppiCollection::View data) {
  // Cut thresholds
  const float min_threshold = 0.01; // mindr2
  const float max_threshold = 0.25; // maxdr2
  const float max_isolation_threshold = 2.0; // maxiso

  float accumulated = 0;
  for (int64_t idx = 0; idx < data.metadata().size(); idx++) {
    if (thread_idx == idx) 
      continue;
    auto delta_eta = data[thread_idx].eta() - data[idx].eta();
    auto sin_value = alpaka::math::sin(acc, data[thread_idx].phi() - data[idx].phi());
    auto cos_value = alpaka::math::cos(acc, data[thread_idx].phi() - data[idx].phi());
    auto delta_phi = alpaka::math::atan2(acc, sin_value, cos_value);
  
    float th_value = delta_eta * delta_eta + delta_phi * delta_phi;
    if (th_value >= min_threshold * min_threshold && th_value <= max_threshold * max_threshold) {
      accumulated += data[idx].pt();
    }
  }
  return accumulated <= max_isolation_threshold * data[thread_idx].pt();
}

class FilterKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename U, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, uint32_t begin, uint32_t end, T* __restrict__ mask, U* __restrict__ charge, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct) const {
    const uint8_t min_threshold = 7;  // minpt1
    const uint8_t int_threshold = 12;  // minpt2
    const uint8_t high_threshold = 15;  // minpt3
      
    for (uint32_t thread_idx : uniform_elements(acc, begin, end)) {
      if (abs(data.pdgId()[thread_idx]) == 211 || abs(data.pdgId()[thread_idx]) == 11) {
        if (data.pt()[thread_idx] >= min_threshold) {
          mask[thread_idx] = static_cast<uint8_t>(1);
          charge[thread_idx] = static_cast<int8_t>(abs(data.pdgId()[thread_idx]) == 11 ? (data.pdgId()[thread_idx] > 0 ? -1 : +1) : (data.pdgId()[thread_idx] > 0 ? +1 : -1));
          if (data.pt()[thread_idx] >= int_threshold)
            alpaka::atomicAdd(acc, &int_cut_ct[0], static_cast<uint32_t>(1));
          if (data.pt()[thread_idx] >= high_threshold)
            alpaka::atomicAdd(acc, &high_cut_ct[0], static_cast<uint32_t>(1));
        }
      }
    }
  }
};

template<typename T, typename U, typename Tc>
void Isolation::Filter(Queue& queue, PuppiCollection::ConstView const_view, uint32_t begin, uint32_t end, T* __restrict__ mask, U* __restrict__ charge, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct) const {
  auto size = end - begin;
  if (size == 0) 
    return;
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, FilterKernel{}, const_view, begin, end, mask, charge, int_cut_ct, high_cut_ct);
}

class EstimateSizeKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T* mask,  uint32_t const size, Tc* accumulator) const {
    // Naive slow summation for simplicity
    // TODO: replace with reduction in the future
    for (auto idx : uniform_elements(acc, 1, size)) {
      if (mask[idx] == static_cast<uint32_t>(1)) {
        alpaka::atomicAdd(acc, &accumulator[0], static_cast<uint32_t>(1));
      }
    }

    // // Reduction
    // auto& shared_mem = alpaka::declareSharedVar<uint32_t[1024], __COUNTER__>(acc);
    // // reverse than in CUDA x,y,z => z,y,x (first threads_per_block)
    // auto index = alpaka::Dim<TAcc>::value - 1u;  
    // // CUDA equivalents:
    // auto thread_idx = (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[index]);  // threadIdx.x
    // auto block_idx = (alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[index]);  // blockIdx.x
    // auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[index]; // blockIdx.x * TBlocksize + threadIdx.x
    // shared_mem[thread_idx] = mask[idx];
    // alpaka::syncBlockThreads(acc);

    // for (uint32_t s = 1; s < 1850; s *= 2) {
    //   if (thread_idx % (2 * s) == 0) {
    //     shared_mem[thread_idx] += shared_mem[thread_idx + s];
    //   }
    //   alpaka::syncBlockThreads(acc);
    // }

    // if (thread_idx == 0) {
    //   mask[block_idx] = shared_mem[0];
    // }
  }
};

template<typename T, typename Tc>
void Isolation::EstimateSize(Queue& queue, T* mask, uint32_t const size, Tc* accumulator) const {
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, EstimateSizeKernel{}, mask, size, accumulator);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
