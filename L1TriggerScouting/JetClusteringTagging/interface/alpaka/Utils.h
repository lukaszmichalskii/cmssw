// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#pragma once
#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

static PlatformHost kPlatform;
static DevHost kDeviceHost = alpaka::getDevByIdx(kPlatform, 0);

inline ALPAKA_FN_HOST uint32_t ThreadsPerBlockUpperBound(uint32_t val) {
  if (val <= 0)
    return 1;
  return std::pow(2, std::ceil(std::log2(val)));
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE
