#ifndef DataFormats_L1ScoutingSoA_interface_CLUEsteringSoA_h
#define DataFormats_L1ScoutingSoA_interface_CLUEsteringSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace l1sc {

  GENERATE_SOA_LAYOUT(CLUEsteringLayout,
                      SOA_COLUMN(uint32_t, cluster),
                      SOA_COLUMN(uint32_t, is_seed))

  using CLUEsteringSoA = CLUEsteringLayout<>;
  using CLUEsteringSoAView = CLUEsteringSoA::View;
  using CLUEsteringSoAConstView = CLUEsteringSoA::ConstView;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_CLUEsteringSoA_h