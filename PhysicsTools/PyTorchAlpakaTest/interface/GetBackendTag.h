#ifndef PhysicsTools_PyTorchAlpakaTest_interface_GetBackendTag_h
#define PhysicsTools_PyTorchAlpakaTest_interface_GetBackendTag_h

#include "FWCore/Utilities/interface/InputTag.h"

namespace torchtest {

  inline edm::InputTag getBackendTag(edm::InputTag const& tag) {
    return edm::InputTag(tag.label(), "backend", tag.process());
  }

}  // namespace torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_GetBackendTag_h