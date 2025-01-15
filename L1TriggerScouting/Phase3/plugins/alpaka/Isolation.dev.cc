// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "Isolation.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

PlatformHost platform;
DevHost DEVICE_HOST = alpaka::getDevByIdx(platform, 0);


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
    const uint8_t min_threshold = 7; 
    const uint8_t int_threshold = 12;
    const uint8_t high_threshold = 15;

    // auto& high_cut_ct = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    // auto& int_cut_ct = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    auto& size = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    auto& mask = alpaka::declareSharedVar<int[128], __COUNTER__>(acc);

    if (once_per_block(acc)) {
      size = 0;
      for (auto idx = 0; idx < 128; idx++)
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
        for (auto idx = 0; idx < 128; idx++) {
          if (mask[idx] == 1)
            alpaka::atomicAdd(acc, &size, 1);
        }
        if (size < 3)
          continue;
        // printf("(%d, %d) -> %d\n", begin, end, size);
      }
        
      // for (uint32_t tid: independent_group_elements(acc, block_dim)) {
      //   auto thread_id = tid + begin; // global index
      //   if (mask[thread_idx] == 0)
      //     continue;
      // }
    }
    // printf("Size: %d (%d)\n", size[0], size[0] - prev_size);
    // prev_size = size[0];
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

//     for (uint32_t thread_idx : uniform_elements(acc, begin, end)) {
//       if (mask[thread_idx] == static_cast<uint8_t>(1)) {
//         if (data.pt()[thread_idx] < high_threshold)
//           continue;
//         if (!utils::ConeIsolation(acc, data, thread_idx, begin, end))
//           continue;
//         for (uint32_t i = begin; i < end; i++) {
//           if (mask[i] == static_cast<uint8_t>(0))
//             continue;
//           if (i == thread_idx || data.pt()[i] < int_threshold)
//             continue;
//           if (data.pt()[i] > data.pt()[thread_idx] || (data.pt()[i] == data.pt()[thread_idx] && i < thread_idx))
//             continue;
//           if (!utils::AngularSeparation(acc, data, thread_idx, i))
//             continue;
//           for (uint32_t j = begin; j < end; j++) {
//             if (mask[j] == static_cast<uint8_t>(0))
//               continue;
//             if (j == thread_idx || j == i)
//               continue;
//             if (data.pt()[i] < min_threshold)
//               continue;
//             if (data.pt()[j] > data.pt()[thread_idx] || (data.pt()[j] == data.pt()[thread_idx] && j < thread_idx))
//               continue;
//             if (data.pt()[j] > data.pt()[i] || (data.pt()[j] == data.pt()[i] && j < i))
//               continue;
//             if (abs(charge[thread_idx] + charge[i] + charge[j]) != 1)
//               continue;
//             auto mass = utils::MassInvariant(acc, data, thread_idx, i, j);
//             if (mass < invariant_mass_lower_bound || mass > invariant_mass_upper_bound) 
//               continue;
//             if (utils::AngularSeparation(acc, data, thread_idx, j) && utils::AngularSeparation(acc, data, i, j)) {
//               if (utils::ConeIsolation(acc, data, i, begin, end) && utils::ConeIsolation(acc, data, j, begin, end)) {
//                 float pt_sum = data.pt()[thread_idx] + data.pt()[i] + data.pt()[j]; 
//                 if (pt_sum > best_score[0]) {
//                   best_score[0] = pt_sum;
//                 }
//               }
//             }          
//           }
//         }
//       }
//     }
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
