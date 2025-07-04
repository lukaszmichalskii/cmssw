#ifndef DataFormats_PyTorchTest_interface_PyTorchTestCollections_h
#define DataFormats_PyTorchTest_interface_PyTorchTestCollections_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PyTorchTest/interface/PyTorchTestHostCollections.h"
#include "DataFormats/PyTorchTest/interface/PyTorchTestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::torchportabletest {

  /**
   * make the names from the top-level `torchportabletest` namespace visible for unqualified lookup
   * inside the `ALPAKA_ACCELERATOR_NAMESPACE::torchportabletest` namespace
   */
  using namespace ::torchportabletest;

  using ParticleCollection = PortableCollection<ParticleSoA>;
  using ClassificationCollection = PortableCollection<ClassificationSoA>;
  using RegressionCollection = PortableCollection< RegressionSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchportabletest

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::ParticleCollection, torchportabletest::ParticleCollectionHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::ClassificationCollection,
                                      torchportabletest::ClassificationCollectionHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::RegressionCollection, torchportabletest::RegressionCollectionHost);

#endif  // DataFormats_PyTorchTest_interface_PyTorchTestCollections_h
