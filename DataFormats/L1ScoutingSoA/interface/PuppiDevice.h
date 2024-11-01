#ifndef DataFormats_L1ScoutingSoA_interface_PuppiDevice_h
#define DataFormats_L1ScoutingSoA_interface_PuppiDevice_h

#include "DataFormats/Portable/interface/PortableDeviceObject.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiStruct.h"

template <typename TDev>
class PuppiDevice : public PortableDeviceObject<PuppiStruct, TDev> {

public:
  PuppiDevice() = default;

  explicit PuppiDevice(TDev const &device)
    : PortableDeviceObject<PuppiStruct, TDev>(device) {}

  template <typename TQueue>
  explicit PuppiDevice(TQueue queue)
    : PortableDeviceObject<PuppiStruct, TDev>(queue) {}

};

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiDevice_h