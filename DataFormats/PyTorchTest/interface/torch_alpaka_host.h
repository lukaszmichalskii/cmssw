#ifndef DataFormats_PyTorchTest_interface_torch_alpaka_host_h
#define DataFormats_PyTorchTest_interface_torch_alpaka_host_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PyTorchTest/interface/torch_alpaka_layout.h"

using SimpleCollectionHost = PortableHostCollection<SimpleSoA>;

#endif  // DataFormats_PyTorchTest_interface_torch_alpaka_host_h