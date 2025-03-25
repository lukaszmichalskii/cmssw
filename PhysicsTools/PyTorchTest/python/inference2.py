import FWCore.ParameterSet.Config as cms

Inference2Struct = cms.EDProducer('Inference2@alpaka',
  input = cms.InputTag('DataLoaderStruct'),
  alpaka = cms.untracked.PSet(),
)