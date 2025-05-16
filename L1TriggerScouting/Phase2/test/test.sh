#!/bin/bash

# 27K	run000037
# 128M	run000034
# 486M	run000036
# 1.6G	run000035
# 2.9G	run000030
# 5.0G	total

function die { echo Failed $1: status $2 ; exit $2 ; }

TEST_DIR=L1TriggerScouting/Phase2/test
EXECUTABLE=runScoutingPhase2Puppi_cfg.py
DATA=/eos/home-l/lmichals/W3Pi/data/raw/

if [ "$#" != "3" ]; then
    die "Need exactly 3 arguments: 1st ('cpu'), 2nd ('num_events'), 3rd ('run_number'), got $#" 1
fi
if [[ "$1" =~ ^(cpu)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu'; got '$1'" 1
fi


if [ "${TARGET}" == "cpu" ]; then
    echo "Running CPU-only test"
    cmsRun ${TEST_DIR}/${EXECUTABLE} runNumber=$3 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 maxEvents=$2
fi