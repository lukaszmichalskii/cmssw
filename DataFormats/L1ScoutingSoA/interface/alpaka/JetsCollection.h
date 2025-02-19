#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_JetsCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_JetsCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetsHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetsDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using JetsCollection =
    std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, JetsHostCollection, JetsDeviceCollection<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(JetsCollection, JetsHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_JetsCollection_h