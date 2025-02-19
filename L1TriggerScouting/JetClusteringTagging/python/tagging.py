import FWCore.ParameterSet.Config as cms

TaggingNodeStruct = cms.EDProducer('TaggingNode@alpaka',
  data = cms.InputTag('DecoderNode'),
  clusters = cms.InputTag('ClusteringNode'),
  model = cms.FileInPath('model.onnx'),
  backend = cms.string('serial_sync'),
  alpaka = cms.vstring(),
)
