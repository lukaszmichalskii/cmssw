#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2Kernels.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  int max(Queue& queue, const int* data, const size_t size) {
    int num_clusters = 0;

    uint32_t threads_per_block = 64;
    uint32_t blocks_per_grid = divide_up_by(size, threads_per_block);      
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    auto extent = Vec<alpaka::DimInt<1>>{1};
    auto device_buffer = alpaka::allocAsyncBuf<int, Idx>(queue, extent);
    alpaka::exec<Acc1D>(
        queue, 
        grid, 
        [] ALPAKA_FN_ACC(Acc1D const &acc, const int* data, const size_t size, int* max_v) {
          for (int32_t thread_idx : uniform_elements(acc, size)) {
            alpaka::atomicMax(acc, max_v, data[thread_idx]);
          }
        },
        data, 
        size,
        device_buffer.data());
    auto host_buffer = createView(cms::alpakatools::host(), &num_clusters, extent);
    alpaka::memcpy(queue, host_buffer, device_buffer); 
    return num_clusters;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels