#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_JetCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_JetCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using JetCollection =
    std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, JetHostCollection, JetDeviceCollection<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(JetCollection, JetHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_JetCollection_h