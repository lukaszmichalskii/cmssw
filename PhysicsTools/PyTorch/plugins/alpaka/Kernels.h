#ifndef PhysicsTools_PyTorch_plugins_alpaka_Kernels_h
#define PhysicsTools_PyTorch_plugins_alpaka_Kernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/PyTorchTest/interface/alpaka/PyTorchTestCollections.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /**
   * @class Kernels
   * @brief Utility class containing helper functions to run simple Alpaka kernels for testing or validation.
   *
   * This class provides simple device-side functionality for modifying and verifying
   * collections of structured SoA data, such as particles, classification outputs, and regressions.
   */
  class Kernels {
  public:
    void FillParticleCollection(Queue &queue, torchportabletest::ParticleCollection &data, float value);
    void AssertCombinatorics(Queue &queue, torchportabletest::ParticleCollection &data, float value);
    void AssertClassification(Queue &queue, torchportabletest::ClassificationCollection &data);
    void AssertRegression(Queue &queue, torchportabletest::RegressionCollection &data);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // PhysicsTools_PyTorch_plugins_alpaka_Kernels_h
