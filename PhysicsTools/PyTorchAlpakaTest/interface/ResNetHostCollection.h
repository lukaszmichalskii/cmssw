#ifndef PhysicsTools_PyTorchAlpakaTest_interface_ResNetHostCollection_h
#define PhysicsTools_PyTorchAlpakaTest_interface_ResNetHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/ResNetSoA.h"

namespace torchportabletest {

  using ImageHostCollection = PortableHostCollection<Image>;
  using LogitsHostCollection = PortableHostCollection<Logits>;

}  // namespace torchportabletest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_ResNetHostCollection_h