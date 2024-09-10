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
                  "2", # default value = 2: puppi, tkem
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

options.register ('tkEmStreamIDs',
                  [], # default value
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Stream IDs for the TkEm inputs")

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
                  "analyses: any list of 'w3pi', 'wdsg'.")

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


options.parseArguments()
analyses = options.analyses if options.analyses else ["w3pi", "wdsg", "wpig"]
print(f"Analyses set to {analyses}")

process = cms.Process("SCPU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

if len(options.buNumStreams) != len(options.buBaseDir):
    raise RuntimeError("Mismatch between buNumStreams (%d) and buBaseDirs (%d)" % (len(options.buNumStreams), len(options.buBaseDir)))

if options.puppiStreamIDs == [] and options.tkEmStreamIDs ==  []:
    nStreamsTot = sum(options.buNumStreams)
    puppiStreamIDs = list(range(nStreamsTot//2)) # take first half 
    tkEmStreamIDs = list(range(nStreamsTot//2, nStreamsTot)) # take second half 
else:
    puppiStreamIDs = options.puppiStreamIDs
    tkEmStreamIDs = options.tkEmStreamIDs

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
process.scPhase2PuppiRawToDigiStruct = cms.EDProducer('ScPhase2PuppiRawToDigi',
  src = cms.InputTag('rawDataCollector'),
  fedIDs = cms.vuint32(*puppiStreamIDs),
)

process.scPhase2TkEmRawToDigiStruct = cms.EDProducer('ScPhase2TkEmRawToDigi',
  src = cms.InputTag('rawDataCollector'),
  fedIDs = cms.vuint32(*tkEmStreamIDs),
)

process.goodOrbitsByNBX = cms.EDFilter("GoodOrbitNBxSelector",
    unpackers = cms.VInputTag(
                    cms.InputTag("scPhase2PuppiRawToDigiStruct"),
                    cms.InputTag("scPhase2TkEmRawToDigiStruct"),
                ),
    nbxMin = cms.uint32(3564 * options.timeslices // options.tmuxPeriod)
)

process.w3piStruct = cms.EDProducer("ScPhase2PuppiW3PiDemo",
    src = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    runCandidate = cms.bool(False),
    runStruct = cms.bool(True),
    runSOA = cms.bool(False)
)

process.wdsgStruct = cms.EDProducer("ScPhase2PuppiWDsGammaDemo",
    src = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    src2 = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    runStruct = cms.bool(True)
)

process.wpigStruct = cms.EDProducer("ScPhase2PuppiWPiGammaDemo",
    src = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    src2 = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    runStruct = cms.bool(True)
)

process.scPhase2SelectedBXs =  cms.EDFilter("FinalBxSelector",
    analysisLabels = cms.VInputTag([cms.InputTag(f"{a}Struct", "selectedBx") for a in analyses]),
)

process.scPhase2PuppiMasked = cms.EDProducer("MaskOrbitBxScoutingPuppi",
    dataTag = cms.InputTag("scPhase2PuppiRawToDigiStruct"),
    selectBxs = cms.InputTag("scPhase2SelectedBXs","SelBx"),
)

process.scPhase2TkEmMasked = cms.EDProducer("MaskOrbitBxScoutingTkEm",
    dataTag = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    selectBxs = cms.InputTag("scPhase2SelectedBXs","SelBx"),
)

process.scPhase2TkEleMasked = cms.EDProducer("MaskOrbitBxScoutingTkEle",
    dataTag = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
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

process.scPhase2TkEmStructToTable = cms.EDProducer("ScTkEmToOrbitFlatTable",
    src = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    name = cms.string("L1TkEm"),
    doc = cms.string("L1TkEm candidates"),
)

process.scPhase2TkEmMaskedStructToTable = process.scPhase2TkEmStructToTable.clone(
    src = "scPhase2TkEmMasked"
)

process.scPhase2TkEleStructToTable = cms.EDProducer("ScTkEleToOrbitFlatTable",
    src = cms.InputTag("scPhase2TkEmRawToDigiStruct"),
    name = cms.string("L1TkEle"),
    doc = cms.string("L1TkEle candidates"),
)

process.scPhase2TkEleMaskedStructToTable = process.scPhase2TkEleStructToTable.clone(
    src = "scPhase2TkEleMasked"
)

process.s_unpackers = cms.Sequence(
  process.scPhase2PuppiRawToDigiStruct +
  process.scPhase2TkEmRawToDigiStruct +
  process.goodOrbitsByNBX
)

process.p_inclusive = cms.Path(
  process.s_unpackers +
  process.scPhase2PuppiStructToTable +
  process.scPhase2TkEmStructToTable +
  process.scPhase2TkEleStructToTable
)

analysisModules = [getattr(process,f"{a}Struct") for a in analyses]
process.s_analyses = cms.Sequence(sum(analysisModules[1:], analysisModules[0]))

process.p_selected = cms.Path(
  process.s_unpackers + 
  process.s_analyses +
  process.scPhase2SelectedBXs +
  process.scPhase2PuppiMasked + 
  process.scPhase2TkEmMasked + 
  process.scPhase2TkEleMasked + 
  process.scPhase2PuppiMaskedStructToTable +
  process.scPhase2TkEmMaskedStructToTable +
  process.scPhase2TkEleMaskedStructToTable
)

process.scPhase2NanoAll = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string(options.outFile.replace(".root","")+".inclusive.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p_inclusive')),
    outputCommands = cms.untracked.vstring("drop *", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2PuppiStructToTable_*_*", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEmStructToTable_*_*", 
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEleStructToTable_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)

process.scPhase2PuppiNanoSelected = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string(options.outFile.replace(".root","")+".selected.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p_selected')),
    selectedBx = cms.InputTag("scPhase2SelectedBXs","SelBx"),
    outputCommands = cms.untracked.vstring("drop *",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2PuppiMaskedStructToTable_*_*",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEmMaskedStructToTable_*_*",
        "keep l1ScoutingRun3OrbitFlatTable_scPhase2TkEleMaskedStructToTable_*_*",
        "keep *_w3piStruct_*_*",
        "keep *_wdsgStruct_*_*",
        "keep *_wpigStruct_*_*",
        "keep *_scPhase2SelectedBXs_*_*"
        ),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("LZ4"),
)
process.o_nanoInclusive = cms.EndPath(process.scPhase2NanoAll)
process.o_nanoSelected = cms.EndPath(process.scPhase2PuppiNanoSelected)
process.o_nanoBoth = cms.EndPath(process.scPhase2NanoAll + process.scPhase2PuppiNanoSelected)

sched = [ process.p_inclusive, process.p_selected ]
if options.run == "inclusive": sched = [ process.p_inclusive ]
if options.run == "selected": sched = [ process.p_selected ]

if options.outMode != "none":
  sched.append(getattr(process, "o_"+options.outMode))
process.schedule = cms.Schedule(*sched)
