#ifndef DataFormats_L1ScoutingSoA_interface_ClustersSoA_h
#define DataFormats_L1ScoutingSoA_interface_ClustersSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"


GENERATE_SOA_LAYOUT(ClustersSoALayout,
  SOA_COLUMN(uint8_t, associations),
  SOA_SCALAR(uint32_t, density)
)

using ClustersSoA = ClustersSoALayout<>;
using ClustersSoAView = ClustersSoA::View;
using ClustersSoAConstView = ClustersSoA::ConstView;

#endif  // DataFormats_L1ScoutingSoA_interface_ClustersSoA_h
