import FWCore.ParameterSet.Config as cms
from IOPool.Input.modules import PoolSource
from L1TriggerScouting.TauTagging.modules import (
    l1sc_L1TScPhase2PFCandidatesAoSToSoA_alpaka,
)

process = cms.Process("L1ScoutingPhase2TauTagging")

# enable multithreading
process.options.numberOfThreads = 1
process.options.numberOfStreams = 1

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# samples
process.source = PoolSource(
    fileNames = [
        "file:/afs/cern.ch/user/l/lmichals/private/CMSSW/CMSSW_15_1_0_pre3/src/L1TriggerScouting/TauTagging/data/l1tPFCandidatesOnly.root"
    ]
)

process.maxEvents.input = 10
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options.wantSummary = False


# convert PFJets to SoA format
process.PFCandidatesAoSToSoA = l1sc_L1TScPhase2PFCandidatesAoSToSoA_alpaka(
    src = cms.InputTag("l1tLayer1Extended", "PF", "L1Dump")
)


# schedule the modules
process.path = cms.Path(
    process.PFCandidatesAoSToSoA
)

process.schedule = cms.Schedule(
    process.path
)
