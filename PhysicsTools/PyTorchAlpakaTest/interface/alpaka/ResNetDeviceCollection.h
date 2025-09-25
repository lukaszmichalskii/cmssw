#ifndef PhysicsTools_PyTorchAlpakaTest_interface_alpaka_ResNetDeviceCollection_h
#define PhysicsTools_PyTorchAlpakaTest_interface_alpaka_ResNetDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/ResNetHostCollection.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/ResNetSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace torchportabletest {

    using namespace ::torchportabletest;

    using ImageDeviceCollection = PortableCollection<Image>;
    using LogitsDeviceCollection = PortableCollection<Logits>;

  }  // namespace torchportabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::ImageDeviceCollection,
                                      torchportabletest::ImageHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::LogitsDeviceCollection,
                                      torchportabletest::LogitsHostCollection);

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_alpaka_ResNetDeviceCollection_h
