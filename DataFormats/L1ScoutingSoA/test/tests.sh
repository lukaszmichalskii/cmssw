#!/bin/bash

TEST_DIR=DataFormats/L1ScoutingSoA/test

cd ${TEST_DIR} && scram b unittests && cd ../../..
