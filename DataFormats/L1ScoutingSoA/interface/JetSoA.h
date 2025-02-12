#ifndef DataFormats_L1ScoutingSoA_interface_JetSoA_h
#define DataFormats_L1ScoutingSoA_interface_JetSoA_h

#include <array>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

using JetAttr = edm::StdArray<float, 12>;
using ClusterSpan = edm::StdArray<uint32_t, 12+1>;

GENERATE_SOA_LAYOUT(JetSoALayout,
  SOA_SCALAR(JetAttr, pt),
  SOA_SCALAR(JetAttr, eta),
  SOA_SCALAR(JetAttr, phi),
  SOA_SCALAR(ClusterSpan, cluster_span),
  SOA_COLUMN(float, constituent_pt), 
  SOA_COLUMN(float, constituent_eta),
  SOA_COLUMN(float, constituent_phi)
)

using JetSoA = JetSoALayout<>;
using JetSoAView = JetSoA::View;
using JetSoAConstView = JetSoA::ConstView;

#endif  // DataFormats_L1ScoutingSoA_interface_JetSoA_h
