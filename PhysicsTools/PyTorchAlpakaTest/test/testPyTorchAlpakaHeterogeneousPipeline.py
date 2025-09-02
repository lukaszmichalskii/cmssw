import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchAlpakaTest.options_cff import args
from PhysicsTools.PyTorchAlpakaTest.modules import (
    torchtest_HeterogeneousCollectionProducer_alpaka,
    torchtest_CollectionAnalyzer
)
  
args.parseArguments()
process = cms.Process("testPyTorchAlpakaHeterogeneousPipeline")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source
process.source = cms.Source("EmptySource")

# print a message every event
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# setup chain configs
process.HeterogeneousCollectionProducer = torchtest_HeterogeneousCollectionProducer_alpaka(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
process.CollectionAnalyzer = torchtest_CollectionAnalyzer(
    particles = 'HeterogeneousCollectionProducer'
)

# schedule the modules
process.path = cms.Path(
    process.HeterogeneousCollectionProducer + 
    process.CollectionAnalyzer
)