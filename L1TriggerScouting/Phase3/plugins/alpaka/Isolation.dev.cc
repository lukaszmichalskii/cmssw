// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "Isolation.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

PlatformHost platform;
DevHost DEVICE_HOST = alpaka::getDevByIdx(platform, 0);

template<typename TAcc>
ALPAKA_FN_ACC float Charge(TAcc const& acc, PuppiCollection::ConstView data, uint32_t i, uint32_t j, uint32_t k) {
  auto cls1 = alpaka::math::abs(acc, static_cast<int>(data.pdgId()[i]));
  auto cls2 = alpaka::math::abs(acc, static_cast<int>(data.pdgId()[j]));
  auto cls3 = alpaka::math::abs(acc, static_cast<int>(data.pdgId()[k]));
  auto c1 = cls1 == 11 ? (cls1 > 0 ? -1 : +1) : (cls1 > 0 ? +1 : -1);
  auto c2 = cls2 == 11 ? (cls2 > 0 ? -1 : +1) : (cls2 > 0 ? +1 : -1);
  auto c3 = cls3 == 11 ? (cls3 > 0 ? -1 : +1) : (cls3 > 0 ? +1 : -1);
  return alpaka::math::abs(acc, c1 + c2 + c3) == 1;
}


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

class FilterKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data) const {
    const uint8_t SHARED_MEM_BLOCK = 128;
    const uint8_t min_threshold = 7; 
    const uint8_t int_threshold = 12;
    const uint8_t high_threshold = 15;
    const float invariant_mass_upper_bound = 150.0;
    const float invariant_mass_lower_bound = 40.0;

    // auto& high_cut_ct = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    // auto& int_cut_ct = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    auto& size = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    auto& mask = alpaka::declareSharedVar<int[SHARED_MEM_BLOCK], __COUNTER__>(acc);
    auto& best_score = alpaka::declareSharedVar<float, __COUNTER__>(acc);

    if (once_per_block(acc)) {
      size = 0;
      best_score = 0.0f;
      for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++)
        mask[idx] = 0;
    }

    uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
    for (uint32_t block_idx: independent_groups(acc, grid_dim)) {
      uint32_t begin = data.offsets()[block_idx];
      uint32_t end = data.offsets()[block_idx + 1];
      if (end == 0xFFFFFFFF)
        continue;
      if (end - begin == 0) 
        continue;
      uint32_t block_dim = end - begin;
      for (uint32_t tid: independent_group_elements(acc, block_dim)) {
        auto thread_idx = tid + begin; // global index
        auto cls = alpaka::math::abs(acc, static_cast<int>(data.pdgId()[thread_idx]));
        if (cls != 211 && cls != 11)
          continue;
        auto pt = data.pt()[thread_idx];
        if (pt < min_threshold)
          continue;
        alpaka::atomicAdd(acc, &mask[tid], 1);
      }

      alpaka::syncBlockThreads(acc);

      if (once_per_block(acc)) {
        for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
          if (mask[idx] == 1)
            alpaka::atomicAdd(acc, &size, 1);
        }
        if (size < 3)
          continue;
      }

      for (uint32_t tid: independent_group_elements(acc, block_dim)) {
        auto thread_idx = tid + begin; // global index
        if (mask[tid] == 0)
          continue;
        if (data.pt()[thread_idx] < high_threshold)
          continue;
        if (!ConeIsolation(acc, data, thread_idx, begin, end))
          continue;
        for (uint32_t i = begin; i < end; i++) {
          if (mask[i] == 0)
            continue;
          if (i == thread_idx || data.pt()[i] < int_threshold)
            continue;
          if (data.pt()[i] > data.pt()[thread_idx] || (data.pt()[i] == data.pt()[thread_idx] && i < thread_idx))
            continue;
          if (!AngularSeparation(acc, data, thread_idx, i))
            continue;
          for (uint32_t j = begin; j < end; j++) {
            if (mask[j] == 0)
              continue;
            if (j == thread_idx || j == i || data.pt()[i] < min_threshold)
              continue;
            if (data.pt()[j] > data.pt()[thread_idx] || (data.pt()[j] == data.pt()[thread_idx] && j < thread_idx))
              continue;
            if (data.pt()[j] > data.pt()[i] || (data.pt()[j] == data.pt()[i] && j < i))
              continue;
            if (data.pdgId()[thread_idx] != 211 && data.pdgId()[i] != 11 && data.pdgId()[j] != -11)
              continue;
            if (!Charge(acc, data, thread_idx, i, j))
              continue;
            auto mass = MassInvariant(acc, data, thread_idx, i, j);
            if (mass < invariant_mass_lower_bound || mass > invariant_mass_upper_bound) 
              continue;
            if (AngularSeparation(acc, data, thread_idx, j) && AngularSeparation(acc, data, i, j)) {
              if (ConeIsolation(acc, data, i, begin, end) && ConeIsolation(acc, data, j, begin, end)) {
                printf("%d, %d, %d", thread_idx, i, j);
                // float accumulated_pt = data.pt()[thread_idx] + data.pt()[i] + data.pt()[j]; 
                printf("%d, %d, %d", thread_idx, i, j);
                // if (accumulated_pt > best_score) {
                //   alpaka::atomicAdd(acc, &best_score, accumulated_pt);
                // }
              }
            }          
          }
        }
      }

      alpaka::syncBlockThreads(acc);

      if (once_per_block(acc)) {
        if (best_score > 0) {
          printf("(%d, %d) -> OK", begin, end);
        }
      }
    }
  }
};

PuppiCollection Isolation::Isolate(Queue& queue, PuppiCollection const& raw_data) const {
  // Accelerator setup
  uint32_t threads_per_block = 128; // conservative constraint of particles per single processing block on hardware.
  uint32_t blocks_per_grid = raw_data.view().bx().size();
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

  // Enqueue kernel
  auto t = std::chrono::high_resolution_clock::now();
  alpaka::exec<Acc1D>(queue, grid, FilterKernel{}, raw_data.const_view());
  alpaka::wait(queue);
  std::cout << "Kernel: OK [" << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t).count() << " ns]" << std::endl;

  return PuppiCollection(1, queue);
}

// class CombinatoricsKernel {
// public:
//   template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename U, typename Tc, typename Tf>
//   ALPAKA_FN_ACC void operator()(
//       TAcc const& acc, PuppiCollection::ConstView data, 
//       uint32_t begin, uint32_t end, 
//       T* __restrict__ mask, U* __restrict__ charge, 
//       Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tf* __restrict__ best_score) const {
//     const uint8_t min_threshold = 7;  
//     const uint8_t int_threshold = 12; 
//     const uint8_t high_threshold = 15; 
//     const float invariant_mass_upper_bound = 150.0;
//     const float invariant_mass_lower_bound = 40.0;

//     if (pions_num[0] < 3 || int_cut_ct[0] < 2 || high_cut_ct[0] < 1) 
//       return;

    // for (uint32_t thread_idx : uniform_elements(acc, begin, end)) {
    //   if (mask[thread_idx] == static_cast<uint8_t>(1)) {
    //     if (data.pt()[thread_idx] < high_threshold)
    //       continue;
    //     if (!utils::ConeIsolation(acc, data, thread_idx, begin, end))
    //       continue;
    //     for (uint32_t i = begin; i < end; i++) {
    //       if (mask[i] == static_cast<uint8_t>(0))
    //         continue;
    //       if (i == thread_idx || data.pt()[i] < int_threshold)
    //         continue;
    //       if (data.pt()[i] > data.pt()[thread_idx] || (data.pt()[i] == data.pt()[thread_idx] && i < thread_idx))
    //         continue;
    //       if (!utils::AngularSeparation(acc, data, thread_idx, i))
    //         continue;
    //       for (uint32_t j = begin; j < end; j++) {
    //         if (mask[j] == static_cast<uint8_t>(0))
    //           continue;
    //         if (j == thread_idx || j == i)
    //           continue;
    //         if (data.pt()[i] < min_threshold)
    //           continue;
    //         if (data.pt()[j] > data.pt()[thread_idx] || (data.pt()[j] == data.pt()[thread_idx] && j < thread_idx))
    //           continue;
    //         if (data.pt()[j] > data.pt()[i] || (data.pt()[j] == data.pt()[i] && j < i))
    //           continue;
    //         if (abs(charge[thread_idx] + charge[i] + charge[j]) != 1)
    //           continue;
    //         auto mass = utils::MassInvariant(acc, data, thread_idx, i, j);
    //         if (mass < invariant_mass_lower_bound || mass > invariant_mass_upper_bound) 
    //           continue;
    //         if (utils::AngularSeparation(acc, data, thread_idx, j) && utils::AngularSeparation(acc, data, i, j)) {
    //           if (utils::ConeIsolation(acc, data, i, begin, end) && utils::ConeIsolation(acc, data, j, begin, end)) {
    //             float pt_sum = data.pt()[thread_idx] + data.pt()[i] + data.pt()[j]; 
    //             if (pt_sum > best_score[0]) {
    //               best_score[0] = pt_sum;
    //             }
    //           }
    //         }          
    //       }
    //     }
    //   }
    // }
//   }
// };

// template<typename T, typename U, typename Tc, typename Tf>
// void Isolation::Combinatorics(
//     Queue& queue, PuppiCollection::ConstView const_view,
//     uint32_t begin, uint32_t end, 
//     T* __restrict__ mask, U* __restrict__ charge, 
//     Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tf* __restrict__ best_score) const {

//   auto size = end - begin;
//   uint32_t threads_per_block = 64;
//   uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);
//   auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
//   alpaka::exec<Acc1D>(queue, grid, CombinatoricsKernel{}, const_view, begin, end, mask, charge, pions_num, int_cut_ct, high_cut_ct, best_score);
// }


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
