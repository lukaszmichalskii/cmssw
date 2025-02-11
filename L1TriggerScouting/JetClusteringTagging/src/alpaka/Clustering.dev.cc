// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Clustering.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

class ClusteringKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::View data) const {
    const uint8_t SHARED_MEM_BLOCK = 128;
    auto& sorted_indices = alpaka::declareSharedVar<int[SHARED_MEM_BLOCK], __COUNTER__>(acc);
    auto& shared_pt = alpaka::declareSharedVar<float[SHARED_MEM_BLOCK], __COUNTER__>(acc); 
    auto& mask = alpaka::declareSharedVar<int[SHARED_MEM_BLOCK], __COUNTER__>(acc);
    auto& cluster = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    auto& valid_cluster = alpaka::declareSharedVar<bool, __COUNTER__>(acc);

    // TODO: can be skipped later, just for debugging purposes.
    // fill shared mem (EOF flags)
    if (once_per_block(acc)) {
      for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
        sorted_indices[idx] = -1;
        shared_pt[idx] = -1.0f;
        mask[idx] = 0;
      }
    }

    // define grid dimensions
    uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
    for (uint32_t block_idx: independent_groups(acc, grid_dim)) {
      // associate range to block
      uint32_t begin = data.offsets()[block_idx];
      uint32_t end = data.offsets()[block_idx + 1];
      if (end == 0xFFFFFFFF)
        continue;
      if (end - begin == 0)
        continue;

      // define block dimensions
      uint32_t block_dim = end - begin;
      // fill shared mem
      for (uint32_t tid : independent_group_elements(acc, block_dim)) {
        uint32_t thread_idx = tid + begin; // global index
        sorted_indices[tid] = tid;
        shared_pt[tid] = data.pt()[thread_idx];
      }
      // synchronize threads within block
      alpaka::syncBlockThreads(acc);

      // odd-even sort algorithm
      for (uint32_t i = 0; i < block_dim; i++) {
        for (uint32_t tid : independent_group_elements(acc, block_dim - 1)) {
          if (tid + 1 < block_dim) {
            if ((i % 2 == 0 && tid % 2 == 0) || (i % 2 == 1 && tid % 2 == 1)) {
              if (shared_pt[tid] < shared_pt[tid + 1]) {
                swap(acc, shared_pt[tid], shared_pt[tid + 1]);
                swap(acc, sorted_indices[tid], sorted_indices[tid + 1]);
              }
            }
          }
          // sync tree
          alpaka::syncBlockThreads(acc);
        }
      }

      // seeded cone jet clustering
      if (once_per_block(acc)) {
        cluster = 1;
        valid_cluster = false;
        for (uint32_t idx = 0; idx < block_dim; idx++) {
          if (cluster > 12)
            break;
          uint32_t seed_idx = sorted_indices[idx]; // highest pt seed
          if (mask[seed_idx] != 0) // already clustered
            continue;
          mask[seed_idx] = cluster;
          
          // seeded cone
          for (uint32_t it = 0; it < block_dim; it++) {
            uint32_t pidx = sorted_indices[it];
            if (mask[pidx] != 0) // already clustered
              continue;
            auto cone_constraint = DeltaR2(acc, data, seed_idx+begin, pidx+begin);
            if (cone_constraint > 0.4)
              continue;
            mask[pidx] = cluster;
            valid_cluster = true;
          }
          if (valid_cluster)
            alpaka::atomicAdd(acc, &cluster, 1);
          valid_cluster = false;
        }
      }

      // debug logs
      // if (once_per_block(acc)) {
      //   if (begin != 573)
      //     return;
      //   bool flag = true;
      //   for (auto idx = 0; idx < SHARED_MEM_BLOCK - 1; idx++) {
      //     auto curr = shared_pt[idx];
      //     auto next = shared_pt[idx+1];
      //     if (next <= curr) 
      //       continue;
      //     flag = false;
      //     break;
      //   }
      //   printf("\n");
      //   printf("\nAssert Sorted: %d -> (%d, %d) = %d\n", flag, begin, end, end-begin);
      //   for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
      //     if (sorted_indices[idx] == -1)
      //       break;
      //     printf("|%6d", sorted_indices[idx]);
      //   }
      //   printf("\n");
      //   for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
      //     if (shared_pt[idx] == -1.0f)
      //       break;
      //     printf("|%6.2f", shared_pt[idx]);
      //   }
      //   printf("\n");
      //   printf("Clusters Associations:\n");
      //   for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
      //     if (sorted_indices[idx] == -1)
      //       break;
      //     printf("|%6d", mask[sorted_indices[idx]]);
      //   }
      //   printf("\n");
      //   printf("Clusters Sizes:\n");
      //   for (auto idx = 1; idx < 12; idx++) {
      //     int accu = 0;
      //     for (auto i = 0; i < SHARED_MEM_BLOCK; i++) {
      //       if (mask[i] == idx)
      //         accu += 1;
      //     }
      //     printf("|%6d", accu);
      //   }
      //   printf("\n");
      //   printf("\n");
      // }
    }
  }
};

void Clustering::Cluster(Queue& queue, PuppiCollection& data) {
  uint32_t threads_per_block = ThreadsPerBlockUpperBound(128);
  uint32_t blocks_per_grid = divide_up_by(data.const_view().bx().size(), threads_per_block);        
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, ClusteringKernel{}, data.view());
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
