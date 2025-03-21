import FWCore.ParameterSet.Config as cms

DataLoaderStruct = cms.EDProducer('DataLoader@alpaka',
  alpaka = cms.untracked.PSet(),
)