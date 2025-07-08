#ifndef DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_SOA_H
#define DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_SOA_H

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace l1sc {

  GENERATE_SOA_LAYOUT(PFCandidateLayout,
    SOA_COLUMN(float, pt),
    SOA_COLUMN(float, eta),
    SOA_COLUMN(float, phi),
    SOA_COLUMN(float, z0),
    SOA_COLUMN(float, dxy),
    // SOA_COLUMN(float, puppiw),
    // SOA_COLUMN(float, quality),
    SOA_COLUMN(int16_t, pdgid)
  )

  using PFCandidateSoA = PFCandidateLayout<>;
  using PFCandidateSoAView = PFCandidateSoA::View;
  using PFCandidateSoAConstView = PFCandidateSoA::ConstView;

} // namespace l1sc

#endif  // DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_SOA_H