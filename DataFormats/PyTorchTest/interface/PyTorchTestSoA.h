#ifndef DataFormats_PyTorchTest_interface_PyTorchTestSoA_h
#define DataFormats_PyTorchTest_interface_PyTorchTestSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace torchportabletest {

  GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))
  using ParticleSoA = ParticleLayout<>;

  GENERATE_SOA_LAYOUT(ClassificationLayout, SOA_COLUMN(float, c1), SOA_COLUMN(float, c2))
  using ClassificationSoA = ClassificationLayout<>;

  GENERATE_SOA_LAYOUT(RegressionLayout, SOA_COLUMN(float, reco_pt))
  using RegressionSoA = RegressionLayout<>;

}  // namespace torchportabletest

#endif  // DataFormats_PyTorchTest_interface_PyTorchTestSoA_h
