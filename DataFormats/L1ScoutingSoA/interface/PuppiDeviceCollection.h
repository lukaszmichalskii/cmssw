#ifndef DataFormats_L1ScoutingSoA_interface_PuppiDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_PuppiDeviceCollection_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiDevice.h"

template <typename TDev>
using PuppiDeviceCollection = PortableDeviceCollection<PuppiSoA, TDev>;

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiDeviceCollection_h