#ifndef DataFormats_L1ScoutingSoA_interface_JetsSoA_h
#define DataFormats_L1ScoutingSoA_interface_JetsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"


GENERATE_SOA_LAYOUT(JetsSoALayout,
  SOA_COLUMN(float, jet)
)

using JetsSoA = JetsSoALayout<>;
using JetsSoAView = JetsSoA::View;
using JetsSoAConstView = JetsSoA::ConstView;

#endif  // DataFormats_L1ScoutingSoA_interface_JetsSoA_h
