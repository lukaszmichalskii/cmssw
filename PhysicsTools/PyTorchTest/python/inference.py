import FWCore.ParameterSet.Config as cms

InferenceStruct = cms.EDProducer('Inference@alpaka',
  input = cms.InputTag('DataLoaderStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)