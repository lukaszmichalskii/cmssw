#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/PuppiDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestPuppiCollection.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::test_puppi_collection {

  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiSoAView view, int value) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view[idx] = {
          static_cast<uint16_t>(value), 
          static_cast<uint32_t>(value), 
          static_cast<float>(value), 
          static_cast<float>(value), 
          static_cast<float>(value), 
          static_cast<float>(value), 
          static_cast<float>(value), 
          static_cast<float>(value), 
          static_cast<int16_t>(value), 
          static_cast<uint8_t>(value)
        };
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiSoAView view, int value) const {
      for (uint32_t idx : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view[idx].bx() == static_cast<uint16_t>(value));
        ALPAKA_ASSERT_ACC(view[idx].offsets() == static_cast<uint32_t>(value));
        ALPAKA_ASSERT_ACC(view[idx].pt() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].eta() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].phi() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].z0() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].dxy() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].puppiw() == static_cast<float>(value));
        ALPAKA_ASSERT_ACC(view[idx].pdgId() == static_cast<int16_t>(value));
        ALPAKA_ASSERT_ACC(view[idx].quality() == static_cast<uint8_t>(value));
      }
    }
  };

  void LaunchKernels(PuppiSoAView view, Queue& queue, size_t threads_ct) {
    int value = 32;
    uint32_t threads_per_block = static_cast<uint32_t>(threads_ct);
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(view.metadata().size(), threads_per_block);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, view, value);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, view, value);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test_puppi_collection