import FWCore.ParameterSet.Config as cms

DecoderNodeStruct = cms.EDProducer('DecoderNode@alpaka',
  src = cms.InputTag('rawDataCollector'),
  fedIDs = cms.vuint32(),
  alpaka = cms.vstring(),
)
