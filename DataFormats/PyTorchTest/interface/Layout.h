#ifndef DATA_FORMATS__PYTORCH_TEST__INTERFACE__LAYOUT_H_
#define DATA_FORMATS__PYTORCH_TEST__INTERFACE__LAYOUT_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace torchportable {

  // using Tensor = edm::StdArray<float, 224 * 224 * 3>;
  using ChannelMatrix = Eigen::Matrix<float, 224, 224>;
  GENERATE_SOA_LAYOUT(ResNetInputLayout,
    SOA_EIGEN_COLUMN(ChannelMatrix, r),
    SOA_EIGEN_COLUMN(ChannelMatrix, g),
    SOA_EIGEN_COLUMN(ChannelMatrix, b))
  using ResNetInputSoA = ResNetInputLayout<>;

  using FeatureMap = Eigen::Vector<float, 1000>;
  GENERATE_SOA_LAYOUT(ResNetOutputLayout,
    SOA_EIGEN_COLUMN(FeatureMap, probs))
  using ResNetOutputSoA = ResNetOutputLayout<>;

  GENERATE_SOA_LAYOUT(InputLayout,
    SOA_COLUMN(float, f1), 
    SOA_COLUMN(float, f2), 
    SOA_COLUMN(float, f3), 
    SOA_COLUMN(float, f4),
    SOA_COLUMN(float, f5),
    SOA_COLUMN(float, f6),
    SOA_COLUMN(float, f7),
    SOA_COLUMN(float, f8),
    SOA_COLUMN(float, f9),
    SOA_COLUMN(float, f10),
    SOA_COLUMN(float, f11),
    SOA_COLUMN(float, f12),
    SOA_COLUMN(float, f13),
    SOA_COLUMN(float, f14),
    SOA_COLUMN(float, f15),
    SOA_COLUMN(float, f16))
  using InputSoA = InputLayout<>;

  GENERATE_SOA_LAYOUT(OutputLayout,
    SOA_COLUMN(float, c1), 
    SOA_COLUMN(float, c2),
    SOA_COLUMN(float, c3),
    SOA_COLUMN(float, c4),
    SOA_COLUMN(float, c5),
    SOA_COLUMN(float, c6),
    SOA_COLUMN(float, c7),
    SOA_COLUMN(float, c8),
    SOA_COLUMN(float, c9),
    SOA_COLUMN(float, c10))
  using OutputSoA = OutputLayout<>;

  GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))
  using ParticleSoA = ParticleLayout<>;

  GENERATE_SOA_LAYOUT(ClassificationLayout, SOA_COLUMN(float, c1), SOA_COLUMN(float, c2))
  using ClassificationSoA = ClassificationLayout<>;

  GENERATE_SOA_LAYOUT(RegressionLayout, SOA_COLUMN(float, reco_pt))
  using RegressionSoA = RegressionLayout<>;

}  // namespace torchportable

#endif  // DATA_FORMATS__PYTORCH_TEST__INTERFACE__LAYOUT_H_
