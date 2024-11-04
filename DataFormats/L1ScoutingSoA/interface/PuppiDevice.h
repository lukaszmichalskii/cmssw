#ifndef DataFormats_L1ScoutingSoA_interface_PuppiDevice_h
#define DataFormats_L1ScoutingSoA_interface_PuppiDevice_h

#include "DataFormats/Portable/interface/PortableDeviceObject.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiStruct.h"

template <typename TDev>
using PuppiDevice = PortableDeviceObject<PuppiStruct, TDev>;

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiDevice_h