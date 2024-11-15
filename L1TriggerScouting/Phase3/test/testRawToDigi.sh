#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

TEST_DIR=L1TriggerScouting/Phase3/test

if [ "$#" != "2" ]; then
    die "Need exactly 2 arguments ('cpu', 'cuda', or 'rocm') and ('num_events'), got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda', or 'rocm'; got '$1'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
    echo "Running CPU-only test"
    cmsRun ${TEST_DIR}/testRawToDigi_cfg.py runNumber=34 buBaseDir=/mnt/ngt/ramdisk_00/lmichals/raw fuBaseDir=/mnt/ngt/ramdisk_00/lmichals/raw buNumStreams=1 maxEvents=$2 backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
    echo "Running CUDA test"
    cmsRun ${TEST_DIR}/testRawToDigi_cfg.py runNumber=34 buBaseDir=/mnt/ngt/ramdisk_00/lmichals/raw fuBaseDir=/mnt/ngt/ramdisk_00/lmichals/raw buNumStreams=1 maxEvents=$2 backend=cuda_async
elif [ "${TARGET}" == "rocm" ]; then
    echo "Running ROCm test"
    cmsRun ${TEST_DIR}/testRawToDigi_cfg.py runNumber=30 buBaseDir=/mnt/ngt/ramdisk_00/lmichals/raw fuBaseDir=/mnt/ngt/ramdisk_00/lmichals/raw buNumStreams=1 maxEvents=$2 backend=rocm_async
fi