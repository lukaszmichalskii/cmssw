import FWCore.ParameterSet.Config as cms

dimuStruct = cms.EDProducer("ScPhase2TrackerMuonDiMuDemo",
    src = cms.InputTag("scPhase2TrackerMuonRawToDigiStruct"),
)

zdeeStruct = cms.EDProducer("ScPhase2TkEmDarkPhotonDiEle",
    srcTkEm = cms.InputTag("scPhase2TkEmRawToDigiStruct")
)

