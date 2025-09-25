import FWCore.ParameterSet.Config as cms
from L1TriggerScouting.Phase3.options import args
  
args.parseArguments()
process = cms.Process("BenchmarkML")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source 
process.source = cms.Source("EmptySource")

# print a message every event
process.MessageLogger.cerr.FwkReport.reportEvery = 1050

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# load pipeline chain
process.load("L1TriggerScouting.Phase3.modules")

# setup chain configs
process.ScMLBenchmarkJit = process.ScMLBenchmarkJitStruct.clone(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    modelPath = cms.FileInPath(args.modelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
modelPath = args.modelPathCpu
if args.backend == "cuda_async":
    modelPath = args.modelPathCuda
process.ScMLBenchmarkAot = process.ScMLBenchmarkAotStruct.clone(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    modelPath = cms.FileInPath(modelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)


process.ScMLBenchmarkResNetJit = process.ScMLBenchmarkResNetJitStruct.clone(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    modelPath = cms.FileInPath(args.resNetModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
resNetModelPath = args.resNetModelPathCpu
if args.backend == "cuda_async":
    resNetModelPath = args.resNetModelPathCuda
process.ScMLBenchmarkResNetAot = process.ScMLBenchmarkResNetAotStruct.clone(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    modelPath = cms.FileInPath(resNetModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)

# schedule the modules
process.path = cms.Path(
    # process.ScMLBenchmarkJit
    # process.ScMLBenchmarkAot
    process.ScMLBenchmarkResNetJit
    # process.ScMLBenchmarkResNetAot
)

process.schedule = cms.Schedule(
    process.path
)
