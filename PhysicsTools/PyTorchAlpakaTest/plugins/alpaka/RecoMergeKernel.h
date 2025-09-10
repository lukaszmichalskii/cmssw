#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_RecoMergeKernel_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_RecoMergeKernel_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace torchportabletest;

  void merge(Queue& queue, ReconstructionDeviceCollection& collection, const ClassificationDeviceCollection& classification, const RegressionDeviceCollection& regression);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_RecoMergeKernel_h