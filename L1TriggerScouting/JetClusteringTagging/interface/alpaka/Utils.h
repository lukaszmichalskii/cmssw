// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Utils_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Utils_h

#include "alpaka/alpaka.hpp"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

static PlatformHost kPlatform;
static DevHost kDeviceHost = alpaka::getDevByIdx(kPlatform, 0);

template <typename T>
inline auto CopyToDevice(Queue &queue, std::vector<T>& data) {
  Vec<alpaka::DimInt<1>> extent(data.size());
  auto device_buffer = alpaka::allocAsyncBuf<T, Idx>(queue, extent);
  auto host_buffer = createView(kDeviceHost, data, extent); // alpaka::View can be used instead of alpaka::Buf
  alpaka::memcpy(queue, device_buffer, host_buffer);  
  alpaka::wait(queue);
  return device_buffer;
}

inline ALPAKA_FN_HOST uint32_t ThreadsPerBlockUpperBound(uint32_t val) {
  if (val <= 0)
    return 1;
  return std::pow(2, std::ceil(std::log2(val)));
}

template<typename TAcc, typename T>
inline ALPAKA_FN_ACC void swap(TAcc const& acc, T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template<typename TAcc>
inline ALPAKA_FN_ACC float DeltaR2(TAcc const& acc, PuppiCollection::ConstView data, uint32_t i, uint32_t j) {
  float p1 = data.phi()[i];
  float p2 = data.phi()[j];

  float e1 = data.eta()[i];
  float e2 = data.eta()[j];

  float dp = std::abs(p1 - p2);
  if (dp > M_PI)
    dp -= 2 * 3.14159265358979323846;
  return (e1 - e2) * (e1 - e2) + dp * dp;
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Utils_h
