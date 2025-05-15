#include "L1TriggerScouting/OnlineProcessing/interface/MaskOrbitBx.h"

#include "DataFormats/L1TParticleFlow/interface/L1ScoutingPuppi.h"
#include "DataFormats/L1TParticleFlow/interface/L1ScoutingTkEm.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef MaskOrbitBx<l1Scouting::Puppi> MaskOrbitBxScoutingPuppi;
typedef MaskOrbitBx<l1Scouting::TkEm> MaskOrbitBxScoutingTkEm;
typedef MaskOrbitBx<l1Scouting::TkEle> MaskOrbitBxScoutingTkEle;

DEFINE_FWK_MODULE(MaskOrbitBxScoutingPuppi);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingTkEm);
DEFINE_FWK_MODULE(MaskOrbitBxScoutingTkEle);