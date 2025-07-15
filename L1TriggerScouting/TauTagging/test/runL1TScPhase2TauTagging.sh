#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="${LOCALTOP}/src/L1TriggerScouting/TauTagging/test/runL1TScPhase2TauTagging.py"

if [ "$#" != "2" ]; then
    die "Need exactly 2 arguments: 1st ('cpu', 'cuda', or 'rocm'), 2nd ('num_events'), got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda', or 'rocm'; got '$1'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" backend=serial_sync numberOfEvents=$2 debug=True
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" backend=cuda_async numberOfEvents=$2 debug=True
elif [ "${TARGET}" == "rocm" ]; then
  echo "Running ROCm test"
  cmsRun "${SCRIPT}" backend=rocm_async numberOfEvents=$2 debug=True
fi