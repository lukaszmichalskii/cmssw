// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Isolation.h"
#include "Utils.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

PlatformHost platform;
DevHost DEVICE_HOST = alpaka::getDevByIdx(platform, 0);


using namespace cms::alpakatools;

class FilterKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, T* mask) const {
    const uint8_t min_threshold = 7; 
    // const uint8_t int_threshold = 12;  
    // const uint8_t high_threshold = 15;

    // auto& shared_int_cut = alpaka::declareSharedVar<uint32_t[1], __COUNTER__>(acc);
    // auto& shared_high_cut = alpaka::declareSharedVar<uint32_t[1], __COUNTER__>(acc);

    // if (once_per_block(acc)) {
    //   shared_int_cut[0] = static_cast<uint32_t>(0);
    //   shared_high_cut[0] = static_cast<uint32_t>(0);
    // }

    for (auto thread_idx : uniform_elements(acc, data.metadata().size())) {
      auto pdgid_abs = abs(data.pdgId()[thread_idx]);
      auto pt = data.pt()[thread_idx];

      // if (thread_idx % 2 == 0) {
      //   mask[thread_idx] = static_cast<uint8_t>(1);
      // }

      if (pdgid_abs == 211 || pdgid_abs == 11) {
        if (data.pt()[thread_idx] >= min_threshold) {
          mask[thread_idx] = static_cast<uint8_t>(1);
        }
      }
    }

    // if (once_per_block(acc)) {
    //   alpaka::atomicAdd(acc, &int_cut_ct[0], shared_int_cut[0]);
    //   alpaka::atomicAdd(acc, &high_cut_ct[0], shared_high_cut[0]);
    // }
  }
};

class EstimateSizeKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T* mask, uint32_t const size, Tc* accumulator) const {
    // Naive slow summation for simplicity
    // TODO: replace with reduction in the future
    for (auto idx : uniform_elements(acc, size)) {
      if (mask[idx] == static_cast<uint32_t>(1)) {
        alpaka::atomicAdd(acc, &accumulator[0], static_cast<uint32_t>(1));
      }
    }
  }

    // Reduction
    // reverse than in CUDA x,y,z => z,y,x (first threads_per_block)
    // auto index = alpaka::Dim<TAcc>::value - 1u;  
    // // CUDA equivalents:
    // auto thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[index];  // threadIdx.x
    // auto block_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[index];  // blockIdx.x
    // auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[index]; // blockIdx.x * TBlocksize + threadIdx.x
    // auto block_dim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[index]; // blockDim.x

    // auto& shared_mem = alpaka::declareSharedVar<uint32_t[1024], __COUNTER__>(acc);
    // if (thread_idx < size) {
    //   shared_mem[thread_idx] = mask[idx];
    //   alpaka::syncBlockThreads(acc); 

    //   for (uint32_t s = 1; s < block_dim; s *= 2) {
    //     if (thread_idx % (2 * s) == 0) {
    //       shared_mem[thread_idx] += shared_mem[thread_idx + s];
    //     }
    //     alpaka::syncBlockThreads(acc);
    //   }

    //   if (thread_idx == 0) {
    //     alpaka::atomicAdd(acc, &accumulator[0], shared_mem[0]);
    //     // mask[block_idx] = shared_mem[0];
    //   }
    // }
  // }
};

class ExtractKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>, typename T, typename Tc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, PuppiCollection::View filtered, T* mask, Tc* ptr) const {
    for (auto idx : uniform_elements(acc, data.metadata().size())) {
      if (mask[idx] == static_cast<uint32_t>(1)) {
        auto i = alpaka::atomicAdd(acc, &ptr[0], static_cast<uint32_t>(1));
        filtered.pt()[i] = data.pt()[idx];
      }
    }
  }
};

PuppiCollection Isolation::Isolate(Queue& queue, PuppiCollection const& raw_data) const {
  const size_t size = raw_data.view().metadata().size();

  // Prepare mask for filtering and isolation
  Vec<alpaka::DimInt<1>> extent(size);
  Vec<alpaka::DimInt<1>> variable(1);

  // Allocate device memory
  auto mask = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, extent);
  auto estimated_size = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, variable);
  auto copy_ptr = alpaka::allocAsyncBuf<uint32_t, Idx>(queue, variable);

  // Initialize device memory
  alpaka::memset(queue, mask, 0);
  alpaka::memset(queue, estimated_size, 0);
  alpaka::memset(queue, copy_ptr, 0);

  // Accelerator setup
  uint32_t threads_per_block = 64;
  uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

  // Launch kernels
  alpaka::exec<Acc1D>(queue, grid, FilterKernel{}, raw_data.const_view(), mask.data());
  alpaka::exec<Acc1D>(queue, grid, EstimateSizeKernel{}, mask.data(), size, estimated_size.data());
  uint32_t h_estimated_size = 0;
  alpaka::memcpy(queue, createView(DEVICE_HOST, &h_estimated_size, variable), estimated_size);  
  PuppiCollection filtered_data(h_estimated_size, queue);
  alpaka::exec<Acc1D>(queue, grid, ExtractKernel{}, raw_data.const_view(), filtered_data.view(), mask.data(), copy_ptr.data());
  alpaka::wait(queue);
  std::cout << "Expected: " << size << std::endl;
  std::cout << "Estimated size: " << h_estimated_size << std::endl;
  std::cout << "Actual size: " << filtered_data.view().metadata().size() << std::endl;

  return filtered_data;
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
