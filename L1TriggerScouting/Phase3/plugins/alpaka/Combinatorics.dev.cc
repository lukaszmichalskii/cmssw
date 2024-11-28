// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "Combinatorics.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

PuppiCollection Combinatorics::Combinatorial(Queue& queue, PuppiCollection const& raw_data) const {
  return PuppiCollection(1, queue);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
