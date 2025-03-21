#ifndef DataFormats_PyTorchTest_interface_torch_alpaka_device_h
#define DataFormats_PyTorchTest_interface_torch_alpaka_device_h

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/PyTorchTest/interface/torch_alpaka_layout.h"

template <typename TDev>
using SimpleCollectionDevice = PortableDeviceCollection<SimpleSoA, TDev>;

#endif  // DataFormats_PyTorchTest_interface_torch_alpaka_device_h