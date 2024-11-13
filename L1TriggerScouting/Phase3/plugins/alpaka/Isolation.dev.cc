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

template<typename TAcc>
ALPAKA_FN_ACC float DeltaPhi(TAcc const& acc, float phi1, float phi2) {
  const float M_PI_CONST = 3.14159265358979323846;
  const float M_2_PI_CONST = 2.0 * M_PI_CONST;
  auto r = alpaka::math::fmod(acc, phi2 - phi1, M_2_PI_CONST);
  if (r < -M_PI_CONST)
    return r + M_2_PI_CONST;
  if (r > M_PI_CONST)
    return r - M_2_PI_CONST;
  return r;
}

template<typename TAcc>
ALPAKA_FN_ACC bool AngularSeparation(TAcc const& acc, PuppiCollection::ConstView data, uint32_t pidx, uint32_t idx) {
  float delta_eta = data.eta()[pidx] - data.eta()[idx];
  float delta_phi = DeltaPhi(acc, data.phi()[pidx], data.phi()[idx]);
  float ang_sep = delta_eta * delta_eta + delta_phi * delta_phi;
  if (ang_sep < 0.5 * 0.5)
    return false;
  return true;
}

template<typename TAcc>
ALPAKA_FN_ACC bool ConeIsolation(TAcc const& acc, PuppiCollection::ConstView data, int32_t thread_idx) {
  // Cut thresholds
  const float min_threshold = 0.01 * 0.01; // mindr2
  const float max_threshold = 0.25 * 0.25; // maxdr2
  const float max_isolation_threshold = 2.0; // maxiso

  float accumulated = 0;
  for (int32_t idx = 0; idx < data.metadata().size(); idx++) {
    if (thread_idx == idx) 
      continue;
    auto delta_eta = data.eta()[thread_idx] - data.eta()[idx];
    auto delta_phi = DeltaPhi(acc, data.phi()[thread_idx], data.phi()[idx]);
  
    float th_value = delta_eta * delta_eta + delta_phi * delta_phi;
    if (th_value >= min_threshold && th_value <= max_threshold) {
      accumulated += data.pt()[thread_idx];
    }
  }
  return accumulated <= max_isolation_threshold * data.pt()[thread_idx];
}

template<typename TAcc>
ALPAKA_FN_ACC float TripletMass(TAcc const& acc, PuppiCollection::ConstView data, uint32_t i, uint32_t j, uint32_t k) {
  // ROOT::Math::PtEtaPhiMVector p1(data.pt()[i], data.eta()[i], data.phi()[i], 0.1396);
  // ROOT::Math::PtEtaPhiMVector p2(data.pt()[j], data.eta()[j], data.phi()[j], 0.1396);
  // ROOT::Math::PtEtaPhiMVector p3(data.pt()[k], data.eta()[k], data.phi()[k], 0.1396);
  // return (p1 + p2 + p3).M();
  return 60.0f;
}

size_t Isolation::Isolate(Queue& queue, PuppiCollection const& raw_data) const {
  const size_t size = raw_data.view().metadata().size();
  size_t w3pi = 0;
  // Prepare mask for filtering and isolation
  Vec<alpaka::DimInt<1>> extent(size);
  Vec<alpaka::DimInt<1>> var_extent(1); 

  // Allocate device memory
  auto device_mask = alpaka::allocAsyncBuf<uint8_t, Idx>(queue, extent);
  auto device_charge = alpaka::allocAsyncBuf<int8_t, Idx>(queue, extent);
  auto device_isolation = alpaka::allocAsyncBuf<uint8_t, Idx>(queue, extent);
  auto device_estimated_size = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_int_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_high_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto& device_offsets = raw_data.view().offsets();

  auto device_partial_size = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_partial_int_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_partial_high_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_best_score = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);

  // Initialize device memory
  alpaka::memset(queue, device_mask, 0);
  alpaka::memset(queue, device_charge, 0);
  alpaka::memset(queue, device_isolation, 0);
  alpaka::memset(queue, device_estimated_size, 0);
  alpaka::memset(queue, device_int_cut_ct, 0);
  alpaka::memset(queue, device_high_cut_ct, 0);

  // Destination memory for data to be copied to debug and size estimation
  uint32_t* host_estimated_size = new uint32_t[1];
  uint32_t* host_int_cut_ct = new uint32_t[1];
  uint32_t* host_high_cut_ct = new uint32_t[1];
  std::vector<int8_t> host_charge(size, 0);
  std::vector<uint8_t> host_mask(size, 0);
  std::vector<uint8_t> host_isolation(size, 0);

  uint32_t* host_partial_size = new uint32_t[1];
  uint32_t* host_partial_int_cut_ct = new uint32_t[1];
  uint32_t* host_partial_high_cut_ct = new uint32_t[1];
  uint32_t* host_best_score = new uint32_t[1];

  host_estimated_size[0] = 0;
  host_int_cut_ct[0] = 0;
  host_high_cut_ct[0] = 0;

  std::array<uint32_t, 3564+1> host_offsets{};
  alpaka::memcpy(queue, createView(host, host_offsets, Vec<alpaka::DimInt<1>>(3564+1)), createView(alpaka::getDev(queue), device_offsets, Vec<alpaka::DimInt<1>>(3564+1)));
  alpaka::wait(queue);

  // Combinatorics
  size_t pass = 0;
  for (size_t bx_idx = 0; bx_idx < raw_data.const_view().bx().size(); bx_idx++) {
    auto begin = host_offsets[bx_idx];
    auto end = host_offsets[bx_idx+1];
    if (end - begin == 0) 
      continue;

    alpaka::memset(queue, device_partial_size, 0);
    alpaka::memset(queue, device_partial_int_cut_ct, 0);
    alpaka::memset(queue, device_partial_high_cut_ct, 0);
    alpaka::memset(queue, device_best_score, 0);

    Filter(queue, raw_data.const_view(), begin, end, device_mask.data(), device_charge.data(), device_partial_int_cut_ct.data(), device_partial_high_cut_ct.data());
    EstimateSize(queue, device_mask.data(), begin, end, device_partial_size.data());

    alpaka::memcpy(queue, createView(host, host_partial_size, Vec<alpaka::DimInt<1>>(1)), device_partial_size);
    alpaka::memcpy(queue, createView(host, host_partial_int_cut_ct, Vec<alpaka::DimInt<1>>(1)), device_partial_int_cut_ct);
    alpaka::memcpy(queue, createView(host, host_partial_high_cut_ct, Vec<alpaka::DimInt<1>>(1)), device_partial_high_cut_ct);

    host_estimated_size[0] += host_partial_size[0];
    host_int_cut_ct[0] += host_partial_int_cut_ct[0];
    host_high_cut_ct[0] += host_partial_high_cut_ct[0];

    if (host_partial_size[0] < 3 || host_partial_int_cut_ct[0] < 2 || host_partial_high_cut_ct[0] < 1) 
      continue;
    pass++;

    Combinatorics(queue, raw_data.const_view(), begin, end, device_mask.data(), device_charge.data(), device_isolation.data(), device_partial_size.data(), device_partial_int_cut_ct.data(), device_partial_high_cut_ct.data(), device_best_score.data());
    alpaka::memcpy(queue, createView(host, host_best_score, Vec<alpaka::DimInt<1>>(1)), device_best_score);

    std::cout << "Idx: " << bx_idx << "; [" << begin << ", " << end << "]; " << "Best Score: " << host_best_score[0] << std::endl;

    if (host_best_score[0] > 0)
      w3pi++;
  }

  // Debug
  std::cout << "Particles Num L1 Filter: " << host_estimated_size[0] << std::endl;
  std::cout << "Paritcles Num L1 IntCut: " << host_int_cut_ct[0] << std::endl;
  std::cout << "Paritcles Num L1  HiCut: "  << host_high_cut_ct[0] << std::endl;
  std::cout << "Candidates Num L1: " << pass << std::endl;
  std::cout << std::endl;

  return w3pi;
}
// struct Cuts {
//     float minpt1 = 7;   // 9
//     float minpt2 = 12;  // 15
//     float minpt3 = 15;  // 20
//     float mindeltar2 = 0.5 * 0.5;
//     float minmass = 40;   // 60
//     float maxmass = 150;  // 100
//     float mindr2 = 0.01 * 0.01;
//     float maxdr2 = 0.25 * 0.25;
//     float maxiso = 2.0;  //0.4
//   } cuts;

class CombinatoricsKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename U, typename Tc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, PuppiCollection::ConstView data, 
      uint32_t begin, uint32_t end, 
      T* __restrict__ mask, U* __restrict__ charge, T* __restrict__ isolation, 
      Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tc* __restrict__ best_score) const {
    const uint8_t min_threshold = 7;  // minpt1
    const uint8_t int_threshold = 12;  // minpt2
    const uint8_t high_threshold = 15;  // minpt3 

    for (auto thread_idx : uniform_elements(acc, begin, end)) {
      if (mask[thread_idx] != static_cast<uint8_t>(1))
        continue;  
      printf("o");    
      if (data.pt()[thread_idx] < high_threshold) //intermediate pt cut
        continue;
      printf("0");      
      if (ConeIsolation(acc, data, thread_idx) == 0)
        continue;
      for (uint32_t i = begin; i < end; i++) {
        printf("1");
        if (i == thread_idx || data.pt()[i] < int_threshold) // minpt2
          continue;
        if (data.pt()[i] > data.pt()[thread_idx] || (data.pt()[i] == data.pt()[thread_idx] && i < thread_idx)) //intermediate pt cut
          continue;
        if (!AngularSeparation(acc, data, thread_idx, i))  //angular sep of top 2 pions  
          continue; 
        for (uint32_t j = begin; j < end; j++) {
          printf("2");
          if (j == thread_idx || j == i)
            continue;
          if (data.pt()[i] < min_threshold) //low pt cut
            continue;
          if (data.pt()[j] > data.pt()[thread_idx] || (data.pt()[j] == data.pt()[thread_idx] && j < thread_idx))
            continue;
          if (data.pt()[j] > data.pt()[i] || (data.pt()[j] == data.pt()[i] && j < i))
            continue;

          if (abs(charge[thread_idx] + charge[i] + charge[j]) != 1)
            continue;
          printf("3");
          auto mass = TripletMass(acc, data, thread_idx, i, j);
          if (mass < 40 || mass > 150) // minmass maxmass
            continue;
          printf("4");
          if (AngularSeparation(acc, data, thread_idx, j) && AngularSeparation(acc, data, i, j)) {
            printf("5");
            if (ConeIsolation(acc, data, i) && ConeIsolation(acc, data, j)) {
              printf("6");
              auto pt_sum = data.pt()[thread_idx] + data.pt()[i] + data.pt()[j];  
              if (pt_sum > best_score[0]) {
                printf("7");
                best_score[0] = pt_sum;
              }
              // alpaka::atomicMax(acc, &best_score, pt_sum);
            }
          }          
        }
      }
    }
  }
};

template<typename T, typename U, typename Tc>
void Isolation::Combinatorics(
    Queue& queue, PuppiCollection::ConstView const_view,
    uint32_t begin, uint32_t end, 
    T* __restrict__ mask, U* __restrict__ charge, T* __restrict__ isolation, 
    Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tc* __restrict__ best_score) const {

  auto size = end - begin;
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, CombinatoricsKernel{}, const_view, begin, end, mask, charge, isolation, pions_num, int_cut_ct, high_cut_ct, best_score);
}

class FilterKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename U, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, uint32_t begin, uint32_t end, T* __restrict__ mask, U* __restrict__ charge, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct) const {
    const uint8_t min_threshold = 7;  // minpt1
    const uint8_t int_threshold = 12;  // minpt2
    const uint8_t high_threshold = 15;  // minpt3

    for (auto thread_idx : uniform_elements(acc, begin, end)) {
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
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, FilterKernel{}, const_view, begin, end, mask, charge, int_cut_ct, high_cut_ct);
}

class EstimateSizeKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T* mask, uint32_t begin, uint32_t end, Tc* accumulator) const {
    // Naive slow summation for simplicity
    // TODO: replace with reduction in the future
    for (auto idx : uniform_elements(acc, begin, end)) {
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
    // alpaka::syncBlockThreads(acc); // Shared mem init has to be sync

    // for (uint32_t s = 1; s < alpaka::getExtentalpaka::Block; s *= 2) {
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
void Isolation::EstimateSize(Queue& queue, T* __restrict__ mask, uint32_t begin, uint32_t end, Tc* __restrict__ accumulator) const {
  auto size = end - begin;
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, EstimateSizeKernel{}, mask, begin, end, accumulator);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
