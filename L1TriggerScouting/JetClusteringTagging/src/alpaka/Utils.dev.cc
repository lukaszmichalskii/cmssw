// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

// PlatformHost kPlatform;
// DevHost kDeviceHost = alpaka::getDevByIdx(kPlatform, 0);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
