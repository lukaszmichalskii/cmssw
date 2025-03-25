import FWCore.ParameterSet.VarParsing as VarParsing

arguments = VarParsing.VarParsing("analysis")

arguments.register(
    "nThreads",
    8,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW threads"
)

arguments.register(
    "nStreams",
    1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW streams"
)

arguments.register(
    "nEvents",
    2,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of events to process"
)

arguments.register(
    "backend",
    "",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Accelerator backend: serial_sync, cuda_async, or rocm_async"
)

arguments.register(
    "modelPath",
    "PhysicsTools/PyTorchTest/models/model.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "PyTorch model"
)
