#ifndef DataFormats_PyTorchTest_interface_PyTorchTestHostCollections_h
#define DataFormats_PyTorchTest_interface_PyTorchTestHostCollections_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PyTorchTest/interface/PyTorchTestSoA.h"

namespace torchportabletest {

  using ParticleCollectionHost = PortableHostCollection<ParticleSoA>;
  using ClassificationCollectionHost = PortableHostCollection<ClassificationSoA>;
  using RegressionCollectionHost = PortableHostCollection<RegressionSoA>;

}  // namespace torchportabletest

#endif  // DataFormats_PyTorchTest_interface_PyTorchTestHostCollections_h
