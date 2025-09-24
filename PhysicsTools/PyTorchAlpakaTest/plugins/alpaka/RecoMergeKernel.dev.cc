#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/RecoMergeKernel.h"

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/Nvtx.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace cms::alpakatools;
  using namespace torchportabletest;

  void merge(Queue &queue,
             ReconstructionDeviceCollection &collection,
             const ClassificationDeviceCollection &classification,
             const RegressionDeviceCollection &regression) {
    Nvtx kernel_range("merge()");

    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = collection.view().metadata().size();
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const &acc,
                         ReconstructionDeviceCollection::View reco_view,
                         ClassificationDeviceCollection::ConstView cls_view,
                         RegressionDeviceCollection::ConstView reg_view) {
          for (int32_t thread_idx : uniform_elements(acc, reco_view.metadata().size())) {
            reco_view[thread_idx].merged() = cls_view[thread_idx].c1() * reg_view[thread_idx].reco_pt() +
                                             cls_view[thread_idx].c2() * reg_view[thread_idx].reco_pt();
          }
        },
        collection.view(),
        classification.const_view(),
        regression.const_view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest