#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_ClustersCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_ClustersCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ClustersCollection =
    std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ClustersHostCollection, ClustersDeviceCollection<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(ClustersCollection, ClustersHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_ClustersCollection_h