import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPyTorchService")

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.PyTorchService = {}

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 0 )
)