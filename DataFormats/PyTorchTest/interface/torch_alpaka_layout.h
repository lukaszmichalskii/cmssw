#ifndef DataFormats_PyTorchTest_interface_torch_alpaka_layout_h
#define DataFormats_PyTorchTest_interface_torch_alpaka_layout_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(SimpleSoALayout,
  SOA_COLUMN(float, x)
  // SOA_COLUMN(float, x),
  // SOA_COLUMN(float, y),
  // SOA_COLUMN(float, z)
)

using SimpleSoA = SimpleSoALayout<>;

#endif  // DataFormats_PyTorchTest_interface_torch_alpaka_layout_h