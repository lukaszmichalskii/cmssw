import FWCore.ParameterSet.Config as cms

ClusteringNodeStruct = cms.EDProducer('ClusteringNode@alpaka',
  src = cms.InputTag('DecoderNodeStruct'),
  clustersNum = cms.uint32(1),
  alpaka = cms.vstring(),
)
