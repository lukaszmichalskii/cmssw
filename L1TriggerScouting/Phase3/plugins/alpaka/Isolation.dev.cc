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
ALPAKA_FN_ACC float Energy(TAcc const& acc, float pt, float eta, float mass) {
  float pz = pt * alpaka::math::sinh(acc, eta);
  float p = alpaka::math::sqrt(acc, pt * pt + pz * pz);
  return alpaka::math::sqrt(acc, p * p + mass * mass);
}

template<typename TAcc>
ALPAKA_FN_ACC float MassInvariant(TAcc const& acc, PuppiCollection::ConstView data, uint32_t i, uint32_t j, uint32_t k) {
  auto px1 = data.pt()[i] * alpaka::math::cos(acc, data.phi()[i]);
  auto py1 = data.pt()[i] * alpaka::math::sin(acc, data.phi()[i]);
  auto pz1 = data.pt()[i] * alpaka::math::sinh(acc, data.eta()[i]);
  auto e1 = Energy(acc, data.pt()[i], data.eta()[i], 0.1396);

  auto px2 = data.pt()[j] * alpaka::math::cos(acc, data.phi()[j]);
  auto py2 = data.pt()[j] * alpaka::math::sin(acc, data.phi()[j]);
  auto pz2 = data.pt()[j] * alpaka::math::sinh(acc, data.eta()[j]);
  auto e2 = Energy(acc, data.pt()[j], data.eta()[j], 0.1396);

  auto px3 = data.pt()[k] * alpaka::math::cos(acc, data.phi()[k]);
  auto py3 = data.pt()[k] * alpaka::math::sin(acc, data.phi()[k]);
  auto pz3 = data.pt()[k] * alpaka::math::sinh(acc, data.eta()[k]);
  auto e3 = Energy(acc, data.pt()[k], data.eta()[k], 0.1396);

  auto t_energy = e1 + e2 + e3;
  auto t_px = px1 + px2 + px3;
  auto t_py = py1 + py2 + py3;
  auto t_pz = pz1 + pz2 + pz3;

  auto t_momentum = t_px * t_px + t_py * t_py + t_pz * t_pz;
  auto invariant_mass = t_energy * t_energy - t_momentum;

  return invariant_mass > 0 ? alpaka::math::sqrt(acc, invariant_mass) : 0.0f;
}

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
  static constexpr float ang_sep_lower_bound = 0.5 * 0.5;
  float delta_eta = data.eta()[pidx] - data.eta()[idx];
  float delta_phi = DeltaPhi(acc, data.phi()[pidx], data.phi()[idx]);
  float ang_sep = delta_eta * delta_eta + delta_phi * delta_phi;
  if (ang_sep < ang_sep_lower_bound)
    return false;
  return true;
}

template<typename TAcc>
ALPAKA_FN_ACC bool ConeIsolation(TAcc const& acc, PuppiCollection::ConstView data, uint32_t thread_idx, uint32_t span_begin, uint32_t span_end) {
  const float min_threshold = 0.01 * 0.01;
  const float max_threshold = 0.25 * 0.25; 
  const float max_isolation_threshold = 2.0;

  float accumulated = 0.0f;
  for (auto idx = span_begin; idx < span_end; idx++) {
    if (thread_idx == idx) 
      continue;
    auto delta_eta = data.eta()[thread_idx] - data.eta()[idx];
    auto delta_phi = DeltaPhi(acc, data.phi()[thread_idx], data.phi()[idx]);
    
    float th_value = delta_eta * delta_eta + delta_phi * delta_phi;
    if (th_value >= min_threshold && th_value <= max_threshold) {
      accumulated += data.pt()[idx];
    }
  }
  return accumulated <= max_isolation_threshold * data.pt()[thread_idx];
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
  auto device_estimated_size = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto& device_offsets = raw_data.view().offsets();

  auto device_partial_size = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_partial_int_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_partial_high_cut_ct = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, var_extent);
  auto device_best_score = alpaka::allocAsyncBuf<float, Idx>(queue, var_extent);

  // Initialize device memory
  alpaka::memset(queue, device_mask, 0);
  alpaka::memset(queue, device_charge, 0);
  alpaka::memset(queue, device_estimated_size, 0);

  // Destination memory for data to be copied to debug and size estimation
  uint32_t* host_estimated_size = new uint32_t[1];
  uint32_t* host_int_cut_ct = new uint32_t[1];
  uint32_t* host_high_cut_ct = new uint32_t[1];
  std::vector<int8_t> host_charge(size, 0);
  std::vector<uint8_t> host_mask(size, 0);

  uint32_t* host_partial_size = new uint32_t[1];
  uint32_t* host_partial_int_cut_ct = new uint32_t[1];
  uint32_t* host_partial_high_cut_ct = new uint32_t[1];
  float* host_best_score = new float[1];

  host_estimated_size[0] = 0;
  host_int_cut_ct[0] = 0;
  host_high_cut_ct[0] = 0;

  auto osize = raw_data.const_view().offsets().size();
  std::vector<uint32_t> host_offsets(osize);
  alpaka::memcpy(queue, createView(host, host_offsets, Vec<alpaka::DimInt<1>>(osize)), createView(alpaka::getDev(queue), device_offsets, Vec<alpaka::DimInt<1>>(osize)));
  alpaka::wait(queue);

  // Combinatorics
  size_t pass = 0;
  for (size_t bx_idx = 0; bx_idx < raw_data.const_view().bx().size(); bx_idx++) {
    auto begin = host_offsets[bx_idx];
    auto end = host_offsets[bx_idx+1];
    // printf("begin: %d, end: %d (%zu)\n", begin, end, bx_idx);
    if (end == 0xFFFFFFFF) // stream termination signal
      break;
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


    Combinatorics(queue, raw_data.const_view(), begin, end, device_mask.data(), device_charge.data(), device_partial_size.data(), device_partial_int_cut_ct.data(), device_partial_high_cut_ct.data(), device_best_score.data());
    alpaka::memcpy(queue, createView(host, host_best_score, Vec<alpaka::DimInt<1>>(1)), device_best_score);
    if (host_best_score[0] > 0)
      w3pi++;
  }

  // Debug
  // std::cout << "==========================================" << std::endl;
  // std::cout << "Particles Num L1 Filter: " << host_estimated_size[0] << std::endl;
  // std::cout << "Paritcles Num L1 IntCut: " << host_int_cut_ct[0] << std::endl;
  // std::cout << "Paritcles Num L1  HiCut: "  << host_high_cut_ct[0] << std::endl;
  // std::cout << "Candidates Num L1: " << pass << std::endl;
  // std::cout << "W3Pi Num: " << w3pi << std::endl;
  // std::cout << "Detected Particles: " << w3pi << std::endl;
  // std::cout << "==========================================" << std::endl;

  // std::cout << std::endl;

  return w3pi;
}

class CombinatoricsKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename U, typename Tc, typename Tf>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, PuppiCollection::ConstView data, 
      uint32_t begin, uint32_t end, 
      T* __restrict__ mask, U* __restrict__ charge, 
      Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tf* __restrict__ best_score) const {
    const uint8_t min_threshold = 7;  
    const uint8_t int_threshold = 12; 
    const uint8_t high_threshold = 15; 
    const float invariant_mass_upper_bound = 150.0;
    const float invariant_mass_lower_bound = 40.0;

    for (uint32_t thread_idx : uniform_elements(acc, begin, end)) {
      if (mask[thread_idx] == static_cast<uint8_t>(1)) {
        if (data.pt()[thread_idx] < high_threshold)
          continue;
        if (!ConeIsolation(acc, data, thread_idx, begin, end))
          continue;
        for (uint32_t i = begin; i < end; i++) {
          if (mask[i] == static_cast<uint8_t>(0))
            continue;
          if (i == thread_idx || data.pt()[i] < int_threshold)
            continue;
          if (data.pt()[i] > data.pt()[thread_idx] || (data.pt()[i] == data.pt()[thread_idx] && i < thread_idx))
            continue;
          if (!AngularSeparation(acc, data, thread_idx, i))
            continue;
          for (uint32_t j = begin; j < end; j++) {
            if (mask[j] == static_cast<uint8_t>(0))
              continue;
            if (j == thread_idx || j == i)
              continue;
            if (data.pt()[i] < min_threshold)
              continue;
            if (data.pt()[j] > data.pt()[thread_idx] || (data.pt()[j] == data.pt()[thread_idx] && j < thread_idx))
              continue;
            if (data.pt()[j] > data.pt()[i] || (data.pt()[j] == data.pt()[i] && j < i))
              continue;
            if (abs(charge[thread_idx] + charge[i] + charge[j]) != 1)
              continue;
            if (data.pdgId()[thread_idx] != 211 && data.pdgId()[i] != 11 && data.pdgId()[j] != -11) {
              // printf("Indices: [%d, %d, %d] -> FAILED\n", thread_idx - begin, i - begin, j - begin);
              continue;
            }
            auto mass = MassInvariant(acc, data, thread_idx, i, j);
            // printf("Indices: [%d, %d, %d], Mass: %.3f\n", thread_idx - begin, i - begin, j - begin, mass);
            if (mass < invariant_mass_lower_bound || mass > invariant_mass_upper_bound) 
              continue;
            if (AngularSeparation(acc, data, thread_idx, j) && AngularSeparation(acc, data, i, j)) {
              if (ConeIsolation(acc, data, i, begin, end) && ConeIsolation(acc, data, j, begin, end)) {
                float pt_sum = data.pt()[thread_idx] + data.pt()[i] + data.pt()[j]; 
                if (pt_sum > best_score[0]) {
                  // printf("Indices: [%d, %d, %d], Mass: %.0f, Range: (%d, %d)\n", thread_idx - begin, i - begin, j - begin, mass, begin, end);
                  // if (thread_idx - begin == 3 && i - begin == 8 && j - begin == 15) {
                  //   printf("\nid: %d; ", thread_idx - begin);
                  //   printf("pt: %f; ",data.pt()[thread_idx]);
                  //   printf("eta: %f; ", data.eta()[thread_idx]);
                  //   printf("phi: %f; ", data.phi()[thread_idx]);
                  //   printf("z0: %f; ", data.z0()[thread_idx]);
                  //   printf("dxy: %f; ", data.dxy()[thread_idx]);
                  //   printf("puppiw: %f; ", data.puppiw()[thread_idx]);
                  //   printf("pdgId: %d; ", data.pdgId()[thread_idx]);
                  //   printf("quality: %d; ", static_cast<unsigned short>(data.quality()[thread_idx]));
                  //   printf("\n");
                  //   printf("id: %d; ", i - begin);
                  //   printf("pt: %f; ",data.pt()[i]);
                  //   printf("eta: %f; ", data.eta()[i]);
                  //   printf("phi: %f; ", data.phi()[i]);
                  //   printf("z0: %f; ", data.z0()[i]);
                  //   printf("dxy: %f; ", data.dxy()[i]);
                  //   printf("puppiw: %f; ", data.puppiw()[i]);
                  //   printf("pdgId: %d; ", data.pdgId()[i]);
                  //   printf("quality: %d; ", static_cast<unsigned short>(data.quality()[i]));
                  //   printf("\n");
                  //   printf("id: %d; ", j - begin);
                  //   printf("pt: %f; ",data.pt()[j]);
                  //   printf("eta: %f; ", data.eta()[j]);
                  //   printf("phi: %f; ", data.phi()[j]);
                  //   printf("z0: %f; ", data.z0()[j]);
                  //   printf("dxy: %f; ", data.dxy()[j]);
                  //   printf("puppiw: %f; ", data.puppiw()[j]);
                  //   printf("pdgId: %d; ", data.pdgId()[j]);
                  //   printf("quality: %d; ", static_cast<unsigned short>(data.quality()[j]));
                  //   printf("\n");
                  // }
                  best_score[0] = pt_sum;
                }
              }
            }          
          }
        }
      }
    }
  }
};

template<typename T, typename U, typename Tc, typename Tf>
void Isolation::Combinatorics(
    Queue& queue, PuppiCollection::ConstView const_view,
    uint32_t begin, uint32_t end, 
    T* __restrict__ mask, U* __restrict__ charge, 
    Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tf* __restrict__ best_score) const {

  auto size = end - begin;
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, CombinatoricsKernel{}, const_view, begin, end, mask, charge, pions_num, int_cut_ct, high_cut_ct, best_score);
}

class FilterKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename U, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, uint32_t begin, uint32_t end, T* __restrict__ mask, U* __restrict__ charge, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct) const {
    const uint8_t min_threshold = 7; 
    const uint8_t int_threshold = 12;  
    const uint8_t high_threshold = 15;

    // for (auto thread_idx : uniform_elements(acc, begin, end)) {
    //   // printf("thread_idx: %d\n", thread_idx);
    //   if (abs(data.pdgId()[thread_idx]) == 211 || abs(data.pdgId()[thread_idx]) == 11) {
    //     // printf("pdgId: %d\n", data.pdgId()[thread_idx]);
    //     if (data.pt()[thread_idx] >= min_threshold) {
    //       // printf("pt: %f\n", data.pt()[thread_idx]);
    //       mask[thread_idx] = static_cast<uint8_t>(1);
    //       charge[thread_idx] = static_cast<int8_t>(abs(data.pdgId()[thread_idx]) == 11 ? (data.pdgId()[thread_idx] > 0 ? -1 : +1) : (data.pdgId()[thread_idx] > 0 ? +1 : -1));
    //       // printf("masking\n");
    //       if (data.pt()[thread_idx] >= int_threshold)
    //         alpaka::atomicAdd(acc, &int_cut_ct[0], static_cast<uint32_t>(1));
    //       if (data.pt()[thread_idx] >= high_threshold)
    //         alpaka::atomicAdd(acc, &high_cut_ct[0], static_cast<uint32_t>(1));
    //     }
    //   }
    //   // printf("next loop jump\n");
    // }

    for (auto thread_idx : uniform_elements(acc, begin, end)) {
      auto cls = alpaka::math::abs(acc, static_cast<int>(data.pdgId()[thread_idx]));
      if (cls == 211 || cls == 11) {
        auto pt = data.pt()[thread_idx];
        if (pt >= min_threshold) {
          mask[thread_idx] = static_cast<uint8_t>(1);
          charge[thread_idx] = static_cast<int8_t>(cls == 11 ? (cls > 0 ? -1 : +1) : (cls > 0 ? +1 : -1));
          if (pt >= int_threshold)
            alpaka::atomicAdd(acc, &int_cut_ct[0], static_cast<uint32_t>(1));
          if (pt >= high_threshold)
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
