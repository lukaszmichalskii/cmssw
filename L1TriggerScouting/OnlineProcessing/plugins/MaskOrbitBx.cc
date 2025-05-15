#include "L1TriggerScouting/OnlineProcessing/interface/MaskOrbitBx.h"

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef MaskOrbitBx<l1ScoutingRun3::Muon> MaskOrbitBxScoutingMuon;
typedef MaskOrbitBx<l1ScoutingRun3::Jet> MaskOrbitBxScoutingJet;
typedef MaskOrbitBx<l1ScoutingRun3::EGamma> MaskOrbitBxScoutingEGamma;
typedef MaskOrbitBx<l1ScoutingRun3::Tau> MaskOrbitBxScoutingTau;
typedef MaskOrbitBx<l1ScoutingRun3::BxSums> MaskOrbitBxScoutingBxSums;
typedef MaskOrbitBx<l1ScoutingRun3::BMTFStub> MaskOrbitBxScoutingBMTFStub;

DEFINE_FWK_MODULE(MaskOrbitBxScoutingMuon);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingJet);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingEGamma);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingTau);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingBxSums);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingBMTFStub);
