import FWCore.ParameterSet.Config as cms
from IOPool.Input.modules import PoolSource
from L1TriggerScouting.TauTagging.options_cff import args
from L1TriggerScouting.TauTagging.modules import (
    l1sc_L1TScPhase2PFCandidatesAoSToSoA_alpaka,
    l1sc_L1TScPhase2CLUETaus_alpaka,
    l1sc_L1TScPhase2DirectInference_alpaka
)


args.parseArguments()
process = cms.Process("L1TScPhase2TauTagging")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# logging configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

# process a limited number of events
process.source = PoolSource(
    fileNames = [
      "file:" + cms.FileInPath("L1TriggerScouting/TauTagging/data/l1tPFCandidatesTest.root").value()
    ]
)

# setup chain configs
# PFCandidates
process.PFCandidatesAoSToSoA = l1sc_L1TScPhase2PFCandidatesAoSToSoA_alpaka(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    src = cms.InputTag("l1tLayer1Extended", "PF", "L1Dump"),
    verbose = cms.untracked.bool(args.verbose),
    verboseLevel = cms.untracked.int32(args.verboseLevel)
)
# CLUEstering
process.CLUETaus = l1sc_L1TScPhase2CLUETaus_alpaka(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    src = "PFCandidatesAoSToSoA",
    wrapCoords = cms.bool(args.wrapCoords),
    verbose = cms.untracked.bool(args.verbose),
    verboseLevel = cms.untracked.int32(args.verboseLevel)
)
# ML inference
process.DirectInference = l1sc_L1TScPhase2DirectInference_alpaka(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    srcPFCandidates = "PFCandidatesAoSToSoA",
    srcCLUETaus = "CLUETaus",
    verbose = cms.untracked.bool(args.verbose),
    verboseLevel = cms.untracked.int32(args.verboseLevel)
)

# schedule the modules
process.path = cms.Path(
    process.PFCandidatesAoSToSoA +
    process.CLUETaus + 
    process.DirectInference
)

# do not needed - framework will run path automatically if there is only one 
process.schedule = cms.Schedule(
    process.path
)