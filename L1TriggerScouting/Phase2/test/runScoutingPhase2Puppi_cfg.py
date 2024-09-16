from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

options = VarParsing.VarParsing ('analysis')
options.register ('runNumber',
                  37,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('lumiNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('daqSourceMode',
                  'ScoutingPhase2', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "DAQ source data mode")

options.register ('broker',
                  'none', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Broker: 'none' or 'hostname:port'")

options.register ('buBaseDir',
                  '/dev/shm/ramdisk', # default value
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('buNumStreams',
                  [], # default value
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of input streams (i.e. files) used simultaneously for each BU directory")

options.register ('timeslices',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of timeslices")

options.register ('tmuxPeriod',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Time multiplex period")

options.register ('puppiStreamIDs',
                  [], # default value
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Stream IDs for the Puppi inputs")

options.register ('fuBaseDir',
                  '/dev/shm/data', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fffBaseDir',
                  '/dev/shm', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "FFF base directory")

options.register ('numThreads',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW streams")

options.register ('run',
                  'both', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "'inclusive', 'selected', 'both' (default).")

options.register ('analyses',
                  [], # default value
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "analyses: any list of 'w3pi'.")

options.register ('prescaleInclusive',
                  100, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Prescale factor for the inclusive stream.")

options.register ('puppiMode',
                  'struct', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "puppi mode to run: struct is the default, and the only one for which everything is implemented; others are candidate, soa, all, fast")
                 
options.register ('outMode',
                  'none', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "output (none, nanoSelected, nanoInclusive, nanoBoth)")
                   
options.register ('outFile',
                  "NanoOutput.root",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Sub lumisection number to process")

options.register ('task',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Task index (used for json outputs)")


options.parseArguments()
if options.buNumStreams == []:
    options.buNumStreams.append(1)
analyses = options.analyses if options.analyses else ["w3pi"]
if options.puppiMode not in ("struct","candidate", "soa", "all", "fast"):
    raise RuntimeError("Unsupported puppiMode %r" %options.puppiMode)

process = cms.Process("SCPU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

print()

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

if len(options.buNumStreams) != len(options.buBaseDir):
    raise RuntimeError("Mismatch between buNumStreams (%d) and buBaseDirs (%d)" % (len(options.buNumStreams), len(options.buBaseDir)))

if options.puppiStreamIDs == []:
    puppiStreamIDs = list(range(sum(options.buNumStreams))) # take all 
else:
    puppiStreamIDs = options.puppiStreamIDs

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(options.broker != "none"),
    fileBrokerHostFromCfg = cms.untracked.bool(False),
    fileBrokerHost = cms.untracked.string(options.broker.split(":")[0] if options.broker != "none" else "htcp40.cern.ch"),
    fileBrokerPort = cms.untracked.string(options.broker.split(":")[1] if options.broker != "none" else "8080"),
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.buBaseDir[0]),
    buBaseDirsAll = cms.untracked.vstring(*options.buBaseDir),
    buBaseDirsNumStreams = cms.untracked.vint32(*options.buNumStreams),
    directorIsBU = cms.untracked.bool(False),
)
process.FastMonitoringService = cms.Service("FastMonitoringService")

process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string(f'resources.{os.uname()[1]}.{options.task}.json')
#process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )

fuDir = options.fuBaseDir+("/run%06d" % options.runNumber)
buDirs = [b+("/run%06d" % options.runNumber) for b in options.buBaseDir]
for d in [fuDir, options.fuBaseDir] + buDirs + options.buBaseDir:
  if not os.path.isdir(d):
    os.makedirs(d)

process.source = cms.Source("DAQSource",
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(2 * 1024),
    eventChunkSize = cms.untracked.uint32(2 * 1024),
    maxChunkSize = cms.untracked.uint32(4 * 1024),
    numBuffers = cms.untracked.uint32(4),
    maxBufferedFiles = cms.untracked.uint32(4),
    fileListMode = cms.untracked.bool(options.broker == "none"),
    fileNames = cms.untracked.vstring(
        buDirs[0] + "/" + "run%06d_ls%04d_index%06d_stream00.raw" % (options.runNumber, options.lumiNumber, 1),
    )
)
os.system("touch " + buDirs[0] + "/" + "fu.lock")

## test pluging
scPhase2PuppiRawToDigi = cms.EDProducer('ScPhase2PuppiRawToDigi',
  src = cms.InputTag('rawDataCollector'),
  fedIDs = cms.vuint32(*puppiStreamIDs),
  runCandidateUnpacker = cms.bool(False),
  runStructUnpacker = cms.bool(False),
  runSOAUnpacker = cms.bool(False),
)
process.scPhase2PuppiRawToDigiCandidate = scPhase2PuppiRawToDigi.clone(
    runCandidateUnpacker = True
)
process.scPhase2PuppiRawToDigiStruct = scPhase2PuppiRawToDigi.clone(
    runCandidateUnpacker = False,
    runStructUnpacker = True
)
process.scPhase2PuppiRawToDigiSOA = scPhase2PuppiRawToDigi.clone(
    runCandidateUnpacker = False,
    runSOAUnpacker = True
)

process.goodOrbitsByNBX = cms.EDFilter("GoodOrbitNBxSelector",
    unpackers = cms.VInputTag(cms.InputTag("scPhase2PuppiRawToDigiStruct")),
    nbxMin = cms.uint32(3564 * options.timeslices // options.tmuxPeriod)
)

process.w3piCandidate = cms.EDProducer("ScPhase2PuppiW3PiDemo",
    src = cms.InputTag("scPhase2PuppiRawToDigiCandidate"),
    runCandidate = cms.bool(True),
    runStruct = cms.bool(False),
    runSOA = cms.bool(False)
)

process.w3piStruct = cms.EDProducer("ScPhase2PuppiW3PiDemo",
    src = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    runCandidate = cms.bool(False),
    runStruct = cms.bool(True),
    runSOA = cms.bool(False)
)

process.w3piSOA = cms.EDProducer("ScPhase2PuppiW3PiDemo",
    src = cms.InputTag("scPhase2PuppiRawToDigiSOA"),
    runCandidate = cms.bool(False),
    runStruct = cms.bool(False),
    runSOA = cms.bool(True)
)

process.scPhase2SelectedBXs =  cms.EDFilter("FinalBxSelector",
    analysisLabels = cms.VInputTag([cms.InputTag(f"{a}Struct", "selectedBx") for a in analyses]),
)

process.scPhase2PuppiMasked = cms.EDProducer("MaskOrbitBxScoutingPuppi",
    dataTag = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    selectBxs = cms.InputTag("scPhase2SelectedBXs","SelBx"),
)


process.scPhase2PuppiStructToTable = cms.EDProducer("ScPuppiToOrbitFlatTable",
    src = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    name = cms.string("L1Puppi"),
    doc = cms.string("L1Puppi candidates from Correlator Layer 2"),
)

process.scPhase2PuppiMaskedStructToTable = process.scPhase2PuppiStructToTable.clone(
    src = "scPhase2PuppiMasked"
)


from FWCore.Modules.preScaler_cfi import preScaler
process.prescaleInclusive = preScaler.clone(prescaleFactor = options.prescaleInclusive)

process.p_inclusive = cms.Path(
  process.scPhase2PuppiRawToDigiStruct +
  process.goodOrbitsByNBX +
  process.prescaleInclusive +
  process.scPhase2PuppiStructToTable
)

analysisModules = [getattr(process,f"{a}Struct") for a in analyses]
process.s_analyses = cms.Sequence(sum(analysisModules[1:], analysisModules[0]))

process.p_selected = cms.Path(
  process.scPhase2PuppiRawToDigiStruct +
  process.goodOrbitsByNBX +
  process.s_analyses +
  process.scPhase2SelectedBXs +
  process.scPhase2PuppiMasked + 
  process.scPhase2PuppiMaskedStructToTable
)

# Additional paths for benchmarking different data structures

process.p_candidate = cms.Path(
  process.scPhase2PuppiRawToDigiCandidate +
  process.w3piCandidate
)

process.p_soa = cms.Path(
  process.scPhase2PuppiRawToDigiSOA +
  process.w3piSOA
)

process.p_all = cms.Path(
  process.scPhase2PuppiRawToDigiCandidate +
  process.scPhase2PuppiRawToDigiStruct +
  process.scPhase2PuppiRawToDigiSOA +
  process.scPhase2PuppiStructToTable +
  process.w3piCandidate +
  process.w3piStruct +
  process.w3piSOA
)

process.p_fast = cms.Path(
  process.scPhase2PuppiRawToDigiStruct +
  process.scPhase2PuppiRawToDigiSOA +
  process.scPhase2PuppiStructToTable +
  process.w3piStruct +
  process.w3piSOA
)


process.scPhase2NanoAll = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string(options.outFile.replace(".root","")+".inclusive.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p_inclusive')),
    outputCommands = cms.untracked.vstring("drop *", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2PuppiStructToTable_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)

process.scPhase2PuppiNanoSelected = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string(options.outFile.replace(".root","")+".selected.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p_selected')),
    selectedBx = cms.InputTag("scPhase2SelectedBXs","SelBx"),
    outputCommands = cms.untracked.vstring("drop *",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2PuppiMaskedStructToTable_*_*",
        "keep *_w3piStruct_*_*",
        "keep *_scPhase2SelectedBXs_*_*"
        ),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)

sched = [ process.p_inclusive, process.p_selected ]
if options.puppiMode != "struct":
    options.sched = [ getattr(process, "p_" + options.puppiMode)]

process.o_nanoInclusive = cms.EndPath(process.scPhase2NanoAll)
process.o_nanoSelected = cms.EndPath(process.scPhase2PuppiNanoSelected)
process.o_nanoBoth = cms.EndPath(process.scPhase2NanoAll + process.scPhase2PuppiNanoSelected)

sched = [ process.p_inclusive, process.p_selected ]
if options.run == "inclusive": sched = [ process.p_inclusive ]
if options.run == "selected": sched = [ process.p_selected ]

if options.outMode != "none":
  sched.append(getattr(process, "o_"+options.outMode))
process.schedule = cms.Schedule(*sched)
