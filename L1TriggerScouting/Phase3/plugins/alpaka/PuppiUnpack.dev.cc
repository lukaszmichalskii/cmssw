// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "PuppiUnpack.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class FillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::View view, int value) const {
      for (int32_t idx : uniform_elements(acc, view.metadata().size())) {
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

  void PuppiUnpack::Fill(Queue& queue, PuppiCollection& collection, int value) const {
    uint32_t items = 64;
    uint32_t groups = divide_up_by(collection->metadata().size(), items);
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view(), value);
  }

  class AssertKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView view, int value) const {
      for (int32_t idx : uniform_elements(acc, view.metadata().size())) {
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

  void PuppiUnpack::Assert(Queue& queue, PuppiCollection const& collection, int value) const {
    auto workDiv = make_workdiv<Acc1D>(1, 32);
    alpaka::exec<Acc1D>(queue, workDiv, AssertKernel{}, collection.const_view(), value);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
