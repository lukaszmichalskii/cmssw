#ifndef DataFormats_L1ScoutingSoA_interface_ClustersDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_ClustersDeviceCollection_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/ClustersSoA.h"

template <typename TDev>
using ClustersDeviceCollection = PortableDeviceCollection<ClustersSoA, TDev>;

#endif  // DataFormats_L1ScoutingSoA_interface_ClustersDeviceCollection_h