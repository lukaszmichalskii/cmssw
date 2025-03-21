import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchTest.arguments import arguments
  
arguments.parseArguments()
process = cms.Process("TestTorchAlpakaPipeline")

# enable multithreading
process.options.numberOfThreads = arguments.nThreads if arguments.nThreads > 0 else 1 
process.options.numberOfStreams = arguments.nStreams if arguments.nStreams > 0 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

# process a limited number of events
process.maxEvents.input = arguments.nEvents if arguments.nEvents > 0 else 1 

# empty source 
process.source = cms.Source("EmptySource")

# print a message every event
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# load pipeline chain
process.load("PhysicsTools.PyTorchTest.data_loader")
process.load("PhysicsTools.PyTorchTest.inference")

# setup chain configs
process.DataLoader = process.DataLoaderStruct.clone(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(arguments.backend)
    ),
)
process.Inference = process.InferenceStruct.clone(
    input = cms.InputTag('DataLoader'),
    modelPath = cms.FileInPath(arguments.modelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(arguments.backend)
    ),
)
# process.MessageLogger.cerr.DataLoader = cms.untracked.PSet()
# process.MessageLogger.cerr.Inference = cms.untracked.PSet()

# schedule the modules
process.path = cms.Path(
    process.DataLoader + 
    process.Inference
)

process.schedule = cms.Schedule(
    process.path
)