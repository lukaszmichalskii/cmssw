import FWCore.ParameterSet.Config as cms

scPhase2PuppiRawToDigiCPUStruct = cms.EDProducer('ScPhase2PuppiRawToDigiCPU',
  src = cms.InputTag('rawDataCollector'),
  fedIDs = cms.vuint32(),
)

goodOrbitsByNBX = cms.EDFilter("GoodOrbitNBxSelector",
    unpackers = cms.VInputTag(
                    cms.InputTag("scPhase2PuppiRawToDigiCPUStruct"),
                ),
    nbxMin = cms.uint32(3564)
)

s_unpackers = cms.Sequence(
   scPhase2PuppiRawToDigiCPUStruct +
   goodOrbitsByNBX 
)