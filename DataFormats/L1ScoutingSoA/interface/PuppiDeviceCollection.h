#ifndef DataFormats_L1ScoutingSoA_interface_PuppiDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_PuppiDeviceCollection_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"

template <typename TDev>
class PuppiDeviceCollection : public PortableDeviceCollection<PuppiSoA, TDev> {

public:
  PuppiDeviceCollection() = default;

  explicit PuppiDeviceCollection(int32_t size, TDev const &device)
    : PortableDeviceCollection<PuppiSoA, TDev>(size, device) {}

  template <typename TQueue>
  explicit PuppiDeviceCollection(int32_t size, TQueue queue)
    : PortableDeviceCollection<PuppiSoA, TDev>(size, queue) {}
    
};

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiDeviceCollection_h