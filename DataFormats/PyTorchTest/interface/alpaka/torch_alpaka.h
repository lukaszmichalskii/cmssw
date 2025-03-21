#ifndef DataFormats_PyTorchTest_interface_alpaka_torch_alpaka_h
#define DataFormats_PyTorchTest_interface_alpaka_torch_alpaka_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PyTorchTest/interface/torch_alpaka_device.h"
#include "DataFormats/PyTorchTest/interface/torch_alpaka_host.h"
#include "DataFormats/PyTorchTest/interface/torch_alpaka_layout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

using SimpleCollection =
  std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, SimpleCollectionHost, SimpleCollectionDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SimpleCollection, SimpleCollectionHost);

#endif  // DataFormats_PyTorchTest_interface_alpaka_torch_alpaka_h