#ifndef DataFormats_L1ScoutingSoA_test_alpaka_TestPuppiCollection_h
#define DataFormats_L1ScoutingSoA_test_alpaka_TestPuppiCollection_h

#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::test_puppi_collection {

  void LaunchKernels(PuppiSoAView view, Queue& queue, size_t threads_ct);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test_puppi_collection

#endif  // DataFormats_L1ScoutingSoA_test_alpaka_TestPuppiCollection_h