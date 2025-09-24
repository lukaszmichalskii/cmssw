import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchAlpakaTest.options_cff import args
from PhysicsTools.PyTorchAlpakaTest.modules import (
    torchtest_PortableCollectionProducer_alpaka,
    torchtest_ParticleClassificationProducer_alpaka,
    torchtest_ParticleRegressionProducer_alpaka,
    torchtest_ReconstructionMergeProducer_alpaka,
    torchtest_CollectionAnalyzer
)
  
args.parseArguments()
process = cms.Process("testPyTorchAlpakaHeterogeneousPipeline")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# logging
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.PyTorchService = {}

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# load the PyTorchService, disable internal multithreading
process.PyTorchService = cms.Service("PyTorchService")

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source
process.source = cms.Source("EmptySource")

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = True

# setup chain configs
process.PortableCollectionProducer = torchtest_PortableCollectionProducer_alpaka(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    verbose = cms.untracked.bool(args.verbose)
)
process.ParticleClassificationPortableProducer = torchtest_ParticleClassificationProducer_alpaka(
    model = cms.FileInPath(args.classificationJit),
    particles = 'PortableCollectionProducer',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    verbose = cms.untracked.bool(args.verbose)
)
process.ParticleRegressionPortableProducer = torchtest_ParticleRegressionProducer_alpaka(
    model = cms.FileInPath(args.regressionJit),
    particles = 'PortableCollectionProducer',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    verbose = cms.untracked.bool(args.verbose)
)
process.ReconstructionMergePortableProducer = torchtest_ReconstructionMergeProducer_alpaka(
    classification = 'ParticleClassificationPortableProducer',
    regression = 'ParticleRegressionPortableProducer',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    verbose = cms.untracked.bool(args.verbose)
)
process.CollectionAnalyzer = torchtest_CollectionAnalyzer(
    particles = 'PortableCollectionProducer',
    classification = 'ParticleClassificationPortableProducer',
    regression = 'ParticleRegressionPortableProducer',
    reconstruction = 'ReconstructionMergePortableProducer',
    verbose = cms.untracked.bool(args.verbose)
)

# schedule the modules
process.path = cms.Path(
    process.PortableCollectionProducer + 
    process.ParticleRegressionPortableProducer +
    process.ParticleClassificationPortableProducer +
    process.ReconstructionMergePortableProducer +
    process.CollectionAnalyzer
)