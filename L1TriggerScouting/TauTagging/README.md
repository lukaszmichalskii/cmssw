# CLUETaus + MultiScaleTransformer project 

Instruction to setup and run pipeline tests:
```bash
# create workspace
cmsrel CMSSW_15_1_0_pre3
cd CMSSW_15_1_0_pre3/src
cmsenv

# checkout branch
git cms-checkout-topic -u lukaszmichalskii:ml@l1

# add CLUEstering as external lib
git clone git@github.com:cms-patatrack/CLUEstering.git
cd CLUEstering && git submodule update --init && cd ..

# build
scram b -j $(nproc)
```

To run standalone tests for different backends change configuration [options](python/options_cff.py) first (if customization needed):
```bash
# cpu backend
cmsRun L1TriggerScouting/TauTagging/test/runL1TScPhase2TauTagging.py backend=serial_sync 
# cuda backend
cmsRun L1TriggerScouting/TauTagging/test/runL1TScPhase2TauTagging.py backend=cuda_async 
# rocm backend
cmsRun L1TriggerScouting/TauTagging/test/runL1TScPhase2TauTagging.py backend=rocm_async 
```