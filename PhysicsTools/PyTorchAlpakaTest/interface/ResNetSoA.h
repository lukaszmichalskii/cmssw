#ifndef PhysicsTools_PyTorchAlpakaTest_interface_ResNetSoA_h
#define PhysicsTools_PyTorchAlpakaTest_interface_ResNetSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace torchportabletest {

  using ColorChannel = Eigen::Matrix<float, 224, 224>;
  GENERATE_SOA_LAYOUT(ImageLayout,
    SOA_EIGEN_COLUMN(ColorChannel, r),
    SOA_EIGEN_COLUMN(ColorChannel, g),
    SOA_EIGEN_COLUMN(ColorChannel, b))
  using Image = ImageLayout<>;

  using LogitsType = Eigen::Vector<float, 1000>;
  GENERATE_SOA_LAYOUT(LogitsLayout,
    SOA_EIGEN_COLUMN(LogitsType, logits))
  using Logits = LogitsLayout<>;

}  // namespace torchportabletest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_ResNetSoA_h