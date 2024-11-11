#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestL1ScoutingSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiConstants.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::test_l1_scouting_soa {

  using namespace cms::alpakatools;

  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiSoAView view, int value) const {
      for (int32_t idx : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        view[idx].pt() = static_cast<float>(value);
        view[idx].eta() = static_cast<float>(value);
        view[idx].phi() = static_cast<float>(value);
        view[idx].z0() = static_cast<float>(value);
        view[idx].dxy() = static_cast<float>(value);
        view[idx].puppiw() = static_cast<float>(value);
        view[idx].pdgId() = static_cast<int16_t>(value);
        view[idx].quality() = static_cast<uint8_t>(value);
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiSoAView view, int value) const {
      for (uint32_t idx : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
        ALPAKA_ASSERT_ACC(view.bx().size() == constants::BX_ARRAY_SIZE);
        ALPAKA_ASSERT_ACC(view.offsets().size() == constants::OFFSETS_ARRAY_SIZE);
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
    uint32_t threads_per_block = static_cast<uint32_t>(threads_ct);
    uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(view.metadata().size(), threads_per_block);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, view, VALUE);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, view, VALUE);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test_l1_scouting_soa