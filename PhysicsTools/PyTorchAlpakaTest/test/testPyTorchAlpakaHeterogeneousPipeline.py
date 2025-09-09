import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchAlpakaTest.options_cff import args
from PhysicsTools.PyTorchAlpakaTest.modules import (
    torchtest_PortableCollectionProducer_alpaka,
    # torchtest_PortableJitClassificationInferenceProducer_alpaka,
    # torchtest_PortableJitRegressionInferenceProducer_alpaka,
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
process.PyTorchService = cms.Service("PyTorchService")

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source
process.source = cms.Source("EmptySource")

# print a message every event
# process.MessageLogger.cerr.FwkReport.reportEvery = 100

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# setup chain configs
process.PortableCollectionProducer = torchtest_PortableCollectionProducer_alpaka(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
# process.PortableJitClassificationInferenceProducer = torchtest_PortableJitClassificationInferenceProducer_alpaka(
#     model = cms.FileInPath(args.classificationJit),
#     particles = 'PortableCollectionProducer',
#     alpaka = cms.untracked.PSet(
#         backend = cms.untracked.string(args.backend)
#     ),
# )
# process.PortableJitRegressionInferenceProducer = torchtest_PortableJitRegressionInferenceProducer_alpaka(
#     model = cms.FileInPath(args.regressionJit),
#     particles = 'PortableCollectionProducer',
#     alpaka = cms.untracked.PSet(
#         backend = cms.untracked.string(args.backend)
#     ),
# )
process.CollectionAnalyzer = torchtest_CollectionAnalyzer(
    particles = 'PortableCollectionProducer',
    # classification = 'PortableJitClassificationInferenceProducer',
    # regression = 'PortableJitRegressionInferenceProducer'
)

# schedule the modules
process.path = cms.Path(
    process.PortableCollectionProducer + 
    # process.PortableJitRegressionInferenceProducer +
    # process.PortableJitClassificationInferenceProducer +
    process.CollectionAnalyzer
)