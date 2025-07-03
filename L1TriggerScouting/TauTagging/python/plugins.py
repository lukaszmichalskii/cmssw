import FWCore.ParameterSet.Config as cms


L1TScPhase2PFCandidatesRawToDigiPlugin = cms.EDProducer(
  'L1TScPhase2PFCandidatesRawToDigi@alpaka',
  alpaka = cms.untracked.PSet(),
  linksIds = cms.vuint32(),
  src = cms.InputTag('src'),
)