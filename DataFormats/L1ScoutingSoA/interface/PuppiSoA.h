#ifndef DataFormats_L1ScoutingSoA_interface_PuppiSoA_h
#define DataFormats_L1ScoutingSoA_interface_PuppiSoA_h

#include <array>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiConstants.h"

/**< Bunch crossings and offsets has fixed size. */
using BxArray = edm::StdArray<uint16_t, constants::BX_ARRAY_SIZE>;
using OffsetsArray = edm::StdArray<uint32_t, constants::OFFSETS_ARRAY_SIZE>;

GENERATE_SOA_LAYOUT(PuppiSoALayout,
  SOA_SCALAR(BxArray, bx),
  SOA_SCALAR(OffsetsArray, offsets),
  SOA_COLUMN(float, pt), 
  SOA_COLUMN(float, eta),
  SOA_COLUMN(float, phi),
  SOA_COLUMN(float, z0),
  SOA_COLUMN(float, dxy),
  SOA_COLUMN(float, puppiw),
  SOA_COLUMN(int16_t, pdgId),
  SOA_COLUMN(uint8_t, quality),
  SOA_COLUMN(uint32_t, selection)
)

using PuppiSoA = PuppiSoALayout<>;
using PuppiSoAView = PuppiSoA::View;
using PuppiSoAConstView = PuppiSoA::ConstView;

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiSoA_h
