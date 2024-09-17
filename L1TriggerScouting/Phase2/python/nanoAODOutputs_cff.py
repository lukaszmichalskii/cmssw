import FWCore.ParameterSet.Config as cms

scPhase2PuppiStructToTable = cms.EDProducer("ScPuppiToOrbitFlatTable",
    src = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    name = cms.string("L1Puppi"),
    doc = cms.string("L1Puppi candidates from Correlator Layer 2"),
)

scPhase2PuppiMaskedStructToTable = scPhase2PuppiStructToTable.clone(
    src = "scPhase2PuppiMasked"
)

scPhase2TkEmStructToTable = cms.EDProducer("ScTkEmToOrbitFlatTable",
    src = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    name = cms.string("L1TkEm"),
    doc = cms.string("L1TkEm candidates"),
)

scPhase2TkEmMaskedStructToTable = scPhase2TkEmStructToTable.clone(
    src = "scPhase2TkEmMasked"
)

scPhase2TkEleStructToTable = cms.EDProducer("ScTkEleToOrbitFlatTable",
    src = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    name = cms.string("L1TkEle"),
    doc = cms.string("L1TkEle candidates"),
)

scPhase2TkEleMaskedStructToTable = scPhase2TkEleStructToTable.clone(
    src = "scPhase2TkEleMasked"
)

tableProducersTask = cms.Task(
    scPhase2PuppiStructToTable,
    scPhase2TkEmStructToTable,
    scPhase2TkEleStructToTable,
)

maskedTableProducersTask = cms.Task(
    scPhase2PuppiMaskedStructToTable,
    scPhase2TkEmMaskedStructToTable,
    scPhase2TkEleMaskedStructToTable,
)

scPhase2NanoAll = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string("all.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    outputCommands = cms.untracked.vstring("drop *", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2PuppiStructToTable_*_*", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEmStructToTable_*_*", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEleStructToTable_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)

scPhase2PuppiNanoSelected = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string("selected.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    selectedBx = cms.InputTag("scPhase2SelectedBXs","SelBx"),
    outputCommands = cms.untracked.vstring("drop *",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2PuppiMaskedStructToTable_*_*",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEmMaskedStructToTable_*_*",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEleMaskedStructToTable_*_*",
        "keep *_scPhase2SelectedBXs_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)