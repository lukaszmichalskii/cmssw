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
DATA=~/private/data/raw/Data/raw/

if [ "$#" != "2" ]; then
    die "Need exactly 2 arguments: 1st ('num_events'), 2nd ('run_number'), got $#" 1
fi

echo "Running CPU-only test"
cmsRun ${TEST_DIR}/${EXECUTABLE} runNumber=$2 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 maxEvents=$1
