#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_RandomCollectionFillingKernel_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_RandomCollectionFillingKernel_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  void randomFillParticleCollection(Queue& queue, torchportabletest::ParticleDeviceCollection& particles);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_RandomCollectionFillingKernel_h