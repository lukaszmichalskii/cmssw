// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Clustering.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

class SeededConeClusteringKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView data, uint32_t clusters_num) const {
    // const uint8_t SHARED_MEM_BLOCK = 128;
    // auto& sorted_indices = alpaka::declareSharedVar<int[SHARED_MEM_BLOCK], __COUNTER__>(acc);
    // auto& shared_pt = alpaka::declareSharedVar<float[SHARED_MEM_BLOCK], __COUNTER__>(acc); 
    // auto& mask = alpaka::declareSharedVar<uint32_t[SHARED_MEM_BLOCK], __COUNTER__>(acc);
    // auto& cluster = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    // auto& clusters_ptrack = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    // // define grid dimensions
    // uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
    // for (uint32_t block_idx: independent_groups(acc, grid_dim)) {
    //   // fill shared mem (EOF flags)
    //   if (once_per_block(acc)) {
    //     for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
    //       sorted_indices[idx] = -1;
    //       shared_pt[idx] = -1.0f;
    //       mask[idx] = 0;
    //       cluster = 1;
    //       clusters_ptrack = 0;
    //     }
    //   }
    //   alpaka::syncBlockThreads(acc);

    //   // bind range to hw block
    //   uint32_t begin = data.offsets()[block_idx];
    //   uint32_t end = data.offsets()[block_idx + 1];
    //   if (end == 0xFFFFFFFF)
    //     continue;
    //   if (end - begin == 0)
    //     continue;

    //   // define block dimensions
    //   uint32_t block_dim = end - begin;
    //   // fill shared mem
    //   for (uint32_t tid : independent_group_elements(acc, block_dim)) {
    //     uint32_t thread_idx = tid + begin; // global index
    //     sorted_indices[tid] = tid;
    //     shared_pt[tid] = data.pt()[thread_idx];
    //   }
    //   alpaka::syncBlockThreads(acc);

    //   // odd-even sort algorithm
    //   for (uint32_t i = 0; i < block_dim; i++) {
    //     for (uint32_t tid : independent_group_elements(acc, block_dim - 1)) {
    //       if (tid + 1 < block_dim) {
    //         if ((i % 2 == 0 && tid % 2 == 0) || (i % 2 == 1 && tid % 2 == 1)) {
    //           if (shared_pt[tid] < shared_pt[tid + 1]) {
    //             swap(acc, shared_pt[tid], shared_pt[tid + 1]);
    //             swap(acc, sorted_indices[tid], sorted_indices[tid + 1]);
    //           }
    //         }
    //       }
    //       // sync tree
    //       alpaka::syncBlockThreads(acc);
    //     }
    //   }

    //   // seeded cone clustering
    //   if (once_per_block(acc)) {
    //     for (uint32_t idx = 0; idx < block_dim; idx++) {
    //       if (cluster > clusters_num)
    //         break;
    //       uint32_t seed_idx = sorted_indices[idx]; // seed at highest pt
    //       if (mask[seed_idx] != 0)
    //         continue;
    //       mask[seed_idx] = cluster;
    //       alpaka::atomicAdd(acc, &clusters_ptrack, 1);
    //       if (clusters_ptrack == static_cast<int>(end-begin))
    //         break;
          
    //       // seeded cone
    //       for (uint32_t it = 0; it < block_dim; it++) {
    //         uint32_t pidx = sorted_indices[it];
    //         if (mask[pidx] != 0)
    //           continue;
    //         auto cone_constraint = DeltaR2(acc, data, seed_idx+begin, pidx+begin);
    //         if (cone_constraint > 0.4)
    //           continue;
    //         mask[pidx] = cluster;
    //         alpaka::atomicAdd(acc, &clusters_ptrack, 1);
    //       }
    //       if (clusters_ptrack == static_cast<int>(end-begin))
    //         break;
    //       alpaka::atomicAdd(acc, &cluster, static_cast<uint32_t>(1));
    //     }
    //     alpaka::atomicAdd(acc, &data.clusters_density(), static_cast<uint32_t>(clusters_ptrack));
    //   }

    //   // fill cluster association
    //   if (once_per_block(acc)) {
    //     for (uint32_t idx = 0; idx < block_dim - 1; idx++) {
    //       if (sorted_indices[idx] == -1)
    //         continue;
    //       uint32_t thread_idx = sorted_indices[idx] + begin; // global index
    //       data.cluster_association()[thread_idx] = mask[sorted_indices[idx]];
    //     }
    //   }

    //   // debug logs
    //   if (once_per_block(acc)) {
    //     bool flag = true;
    //     for (auto idx = 0; idx < SHARED_MEM_BLOCK - 1; idx++) {
    //       auto curr = shared_pt[idx];
    //       auto next = shared_pt[idx+1];
    //       if (next <= curr) 
    //         continue;
    //       flag = false;
    //       break;
    //     }
    //     assert(flag);

    //     uint32_t max = 0, max_id = 0;
    //     for (uint32_t idx = 0; idx < data.bx().size(); idx++) {
    //       auto begin = data.offsets()[idx];
    //       auto end = data.offsets()[idx+1];
    //       if (end == 0xFFFFFFFF)
    //         break;
    //       if (end - begin > max) {
    //         max = end - begin;
    //         max_id = idx;
    //       }
    //     }
    //     if (block_idx == max_id) {
    //       printf("\nSorted %s: %d -> (%d, %d) [%d]\n", flag == true ? "OK" : "FAILED", block_idx, begin, end, end-begin);
    //       printf("indices:\n");
    //       for (uint32_t idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
    //         if (sorted_indices[idx] == -1)
    //           break;
    //         printf("|%8d", sorted_indices[idx]);
    //       }
    //       printf("\n");
    //       printf("pt:\n");
    //       for (uint32_t idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
    //         if (shared_pt[idx] == -1.0f)
    //           break;
    //         printf("|%8.2f", shared_pt[idx]);
    //       }
    //       printf("\n\n");
    //       printf("Clusters Associations Local Map:\n");
    //       for (uint32_t idx = 0; idx < SHARED_MEM_BLOCK; idx++) {
    //         if (sorted_indices[idx] == -1)
    //           break;
    //         printf("|%2d -> %2d", sorted_indices[idx], data.cluster_association()[sorted_indices[idx]+begin]);
    //       }
    //       printf("\n\n");
    //       printf("Clusters Sizes (%d):\n", clusters_num);
    //       int gaccu = 0;
    //       for (uint32_t idx = 1; idx < clusters_num; idx++) {
    //         int accu = 0;
    //         for (uint32_t i = 0; i < block_dim; i++) {
    //           if (data.cluster_association()[i+begin] == idx)
    //             accu += 1;
    //         }
    //         gaccu += accu;
    //         printf("|%8d", accu);
    //       }
    //       printf("\n\n");
    //       printf("Block Clustering Density: %d -> %d (%.0f%s)\n", end-begin, gaccu, 100.0f * gaccu / (end-begin), "%");
    //     }
    //   }
    // }
  }
};

void SeededConeClustering::Cluster(Queue& queue, PuppiCollection const& data, uint32_t clusters_num) {
  uint32_t threads_per_block = ThreadsPerBlockUpperBound(128);
  uint32_t blocks_per_grid = data.view().bx().size();        
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, SeededConeClusteringKernel{}, data.view(), clusters_num);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
