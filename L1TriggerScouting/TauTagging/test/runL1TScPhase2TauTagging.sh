#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/TauTagging/test/runL1TScPhase2TauTagging.py"
DATA=/eos/home-l/lmichals/W3Pi/data/raw/

if [ "$#" != "3" ]; then
    die "Need exactly 3 arguments: 1st ('cpu', 'cuda', or 'rocm'), 2nd ('num_events'), 3rd ('run_number'), got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda', or 'rocm'; got '$1'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" runNumber=$3 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 numberOfEvents=$2 backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" runNumber=$3 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 numberOfEvents=$2 backend=cuda_async
elif [ "${TARGET}" == "rocm" ]; then
  echo "Running ROCm test"
  cmsRun "${SCRIPT}" runNumber=$3 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 numberOfEvents=$2 backend=rocm_async
fi