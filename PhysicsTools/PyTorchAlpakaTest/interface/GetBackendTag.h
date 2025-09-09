#ifndef PhysicsTools_PyTorchAlpakaTest_interface_GetBackendTag_h
#define PhysicsTools_PyTorchAlpakaTest_interface_GetBackendTag_h

#include "alpaka/alpaka.hpp"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaInterface/interface/Backend.h"

namespace torchtest {

  inline edm::InputTag getBackendTag(edm::InputTag const& tag) {
    return edm::InputTag(tag.label(), "backend", tag.process());
  }
  
}  // namespace torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_GetBackendTag_h