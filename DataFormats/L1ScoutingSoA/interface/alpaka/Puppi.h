#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_Puppi_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_Puppi_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHost.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiDevice.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiStruct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using Puppi =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, PuppiHost, PuppiDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(Puppi, PuppiHost);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_Puppi_h