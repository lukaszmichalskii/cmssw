import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchAlpakaTest.options_cff import parse_args
from PhysicsTools.PyTorchAlpakaTest.modules import (
    torchtest_DataSource_alpaka,
    torchtest_InspectionSink
)

args = parse_args()
process = cms.Process("PyTorchAlpakaHeterogeneousTest")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# logging
process.MessageLogger.PyTorchService = {}

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")
process.PyTorchService = cms.Service("PyTorchService")

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source
process.source = cms.Source("EmptySource")

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options.wantSummary = True

# setup chain configs
process.path = cms.Path()
process.DataSource = torchtest_DataSource_alpaka(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    environment = cms.untracked.int32(args.environment)
)
process.path += process.DataSource
if "SimpleNet" in args.only:
    from PhysicsTools.PyTorchAlpakaTest.modules import torchtest_SimpleNet_alpaka
    process.SimpleNet = torchtest_SimpleNet_alpaka(
        model = cms.FileInPath(args.simpleNet),
        particles = 'DataSource',
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string("serial_sync")  # force serial backend to emulate heterogeneous pipeline
        ),
        environment = cms.untracked.int32(args.environment)
    )
    process.path += process.SimpleNet
if "MultiHeadNet" in args.only:
    from PhysicsTools.PyTorchAlpakaTest.modules import torchtest_MultiHeadNet_alpaka
    process.MultiHeadNet = torchtest_MultiHeadNet_alpaka(
        model = cms.FileInPath(args.multiHeadNet),
        particles = 'DataSource',
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        environment = cms.untracked.int32(args.environment)
    )
    process.path += process.MultiHeadNet
if "MaskedNet" in args.only:
    from PhysicsTools.PyTorchAlpakaTest.modules import torchtest_MaskedNet_alpaka
    process.MaskedNet = torchtest_MaskedNet_alpaka(
        model = cms.FileInPath(args.maskedNet),
        particles = 'DataSource',
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        environment = cms.untracked.int32(args.environment)
    )
    process.path += process.MaskedNet
if "TinyResNet" in args.only:
    from PhysicsTools.PyTorchAlpakaTest.modules import torchtest_TinyResNet_alpaka
    process.TinyResNet = torchtest_TinyResNet_alpaka(
        model = cms.FileInPath(args.tinyResNet),
        images = 'DataSource',
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        environment = cms.untracked.int32(args.environment)
    )
    process.path += process.TinyResNet
process.InspectionSink = torchtest_InspectionSink(
    particles = 'DataSource',
    simple_net = 'SimpleNet',
    masked_net = 'MaskedNet',
    multi_head_net = 'MultiHeadNet',
    images = 'DataSource',
    resnet18 = 'TinyResNet',
    environment = cms.untracked.int32(args.environment)
)
process.path += process.InspectionSink
