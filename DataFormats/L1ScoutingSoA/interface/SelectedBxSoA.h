#ifndef DataFormats_L1ScoutingSoA_interface_SelectedBxSoA_h
#define DataFormats_L1ScoutingSoA_interface_SelectedBxSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace l1sc {

  GENERATE_SOA_LAYOUT(SelectedBxLayout, SOA_COLUMN(uint32_t, bx))

  using SelectedBxSoA = SelectedBxLayout<>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_SelectedBxSoA_h