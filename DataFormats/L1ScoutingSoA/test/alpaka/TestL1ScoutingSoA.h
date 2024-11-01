#ifndef DataFormats_L1ScoutingSoA_test_alpaka_TestL1ScoutingSoA_h
#define DataFormats_L1ScoutingSoA_test_alpaka_TestL1ScoutingSoA_h

#include "DataFormats/L1ScoutingSoA/interface/PuppiStruct.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHost.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/Puppi.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::test_l1_scouting_soa {

  constexpr int VALUE = 32;

  void LaunchKernels(PuppiSoAView view, Queue& queue, size_t threads_ct);
  void LaunchKernel(Puppi::Product* data, Queue& queue, size_t threads_ct);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test_l1_scouting_soa

#endif  // DataFormats_L1ScoutingSoA_test_alpaka_TestL1ScoutingSoA_h