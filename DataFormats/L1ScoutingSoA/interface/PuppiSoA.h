#ifndef DataFormats_L1ScoutingSoA_interface_PuppiSoA_h
#define DataFormats_L1ScoutingSoA_interface_PuppiSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(PuppiSoALayout,
  SOA_COLUMN(uint16_t, bx),
  SOA_COLUMN(uint32_t, offsets),
  SOA_COLUMN(float, pt), 
  SOA_COLUMN(float, eta),
  SOA_COLUMN(float, phi),
  SOA_COLUMN(float, z0),
  SOA_COLUMN(float, dxy),
  SOA_COLUMN(float, puppiw),
  SOA_COLUMN(int16_t, pdgId),
  SOA_COLUMN(uint8_t, quality)
)

using PuppiSoA = PuppiSoALayout<>;
using PuppiSoAView = PuppiSoA::View;
using PuppiSoAConstView = PuppiSoA::ConstView;

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiSoA_h
