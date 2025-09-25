import FWCore.ParameterSet.Config as cms


ScMLBenchmarkJitStruct = cms.EDProducer('ScMLBenchmarkJit@alpaka',
  batchSize = cms.uint32(1),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

ScMLBenchmarkAotStruct = cms.EDProducer('ScMLBenchmarkAot@alpaka',
  batchSize = cms.uint32(1),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

ScMLBenchmarkResNetJitStruct = cms.EDProducer('ScMLBenchmarkResNetJit@alpaka',
  batchSize = cms.uint32(1),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

ScMLBenchmarkResNetAotStruct = cms.EDProducer('ScMLBenchmarkResNetAot@alpaka',
  batchSize = cms.uint32(1),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

