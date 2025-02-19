#ifndef DataFormats_L1ScoutingSoA_interface_JetsDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_JetsDeviceCollection_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetsSoA.h"

template <typename TDev>
using JetsDeviceCollection = PortableDeviceCollection<JetsSoA, TDev>;

#endif  // DataFormats_L1ScoutingSoA_interface_JetsDeviceCollection_h