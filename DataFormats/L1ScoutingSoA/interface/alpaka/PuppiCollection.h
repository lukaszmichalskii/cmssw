#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_PuppiCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_PuppiCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using PuppiCollection =
    std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, PuppiHostCollection, PuppiDeviceCollection<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(PuppiCollection, PuppiHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_PuppiCollection_h