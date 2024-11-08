// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Isolation.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

PuppiCollection Isolation::Isolate(Queue& queue, PuppiCollection const& raw_data) const {
  const size_t size = raw_data.view().metadata().size();
  // Prepare mask for filtering
  Vec<alpaka::DimInt<1>> extent(size);
  auto device_mask = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, extent);
  alpaka::memset(queue, device_mask, static_cast<uint32_t>(0));

  // Destination memory for data to be copied to debug and size estimation
  PlatformHost platform;
  DevHost host = alpaka::getDevByIdx(platform, 0);
  Vec<alpaka::DimInt<1>> mask_extent(size);
  std::vector<uint32_t> host_mask(size, 0);

  // Filter particles by types / cuts and applying cone isolation
  Filter(queue, raw_data.const_view(), device_mask.data());

  // Get number of particles that pass criteria
  EstimateSize(queue, device_mask.data(), size);

  alpaka::memcpy(queue, createView(host, host_mask, mask_extent), device_mask);
  auto post_size = host_mask[0];
  // Return reduced particles set for further analysis
  PuppiCollection collection(post_size, queue);

  return collection;
}

template<typename TAcc>
ALPAKA_FN_ACC bool ConeIsolation(TAcc const& acc, int64_t thread_idx, PuppiCollection::ConstView data) {
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
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, T* __restrict__ mask) const {
    const int min_threshold = 7;  // minpt1
    for (int64_t idx : uniform_elements(acc, data.metadata().size())) {
      auto pdgid_abs = abs(data[idx].pdgId()); 
      bool filter_pass = (data[idx].pt() >= min_threshold) && (pdgid_abs == 211 || pdgid_abs == 11);
      if (filter_pass) {
        if (ConeIsolation(acc, idx, data)) { 
          mask[idx] = 1;
        }
      }
    }
  }
};

template<typename T>
void Isolation::Filter(Queue& queue, PuppiCollection::ConstView const_view, T* __restrict__ mask) const {
  auto size = const_view.metadata().size();
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, FilterKernel{}, const_view, mask);
}

class EstimateSizeKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T* __restrict__ mask, uint32_t const size) const {
    // Naive slow summation for simplicity
    // TODO: replace with reduction in the future
    for (auto idx : uniform_elements(acc, 1, size)) {
      if (mask[idx] == static_cast<uint32_t>(1))
        alpaka::atomicAdd(acc, &mask[0], static_cast<uint32_t>(1));
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

template<typename T>
void Isolation::EstimateSize(Queue& queue, T* __restrict__ mask, uint32_t const size) const {
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, EstimateSizeKernel{}, mask, size);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
