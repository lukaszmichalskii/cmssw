#ifndef DataFormats_L1ScoutingSoA_interface_JetDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_JetDeviceCollection_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetSoA.h"

template <typename TDev>
using JetDeviceCollection = PortableDeviceCollection<JetSoA, TDev>;

#endif  // DataFormats_L1ScoutingSoA_interface_JetDeviceCollection_h