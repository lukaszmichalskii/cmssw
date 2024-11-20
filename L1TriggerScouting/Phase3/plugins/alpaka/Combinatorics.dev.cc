// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "Combinatorics.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
