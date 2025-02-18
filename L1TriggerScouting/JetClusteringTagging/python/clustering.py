import FWCore.ParameterSet.Config as cms

ClusteringNodeStruct = cms.EDProducer('ClusteringNode@alpaka',
  data = cms.InputTag('DecoderNodeStruct'),
  clustersNum = cms.uint32(5),
  alpaka = cms.vstring(),
)
