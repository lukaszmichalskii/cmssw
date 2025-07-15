# CLUETaus + MultiScaleTransformer project 

Instruction to setup and run pipeline tests:
```bash
# create workspace
cmsrel CMSSW_15_1_0_pre3
cd CMSSW_15_1_0_pre3/src
cmsenv

# checkout branch
git cms-checkout-topic -u lukaszmichalskii:ml@l1

# build
scram b -j $(nproc)
```

To run standalone tests for different backends change configuration [options](python/options_cff.py) first (if customization needed):
```bash
# cpu backend
scram b runtests_L1ScoutingPhase2TauTaggingSerialSync -v 
# cuda backend
scram b runtests_L1ScoutingPhase2TauTaggingCudaAsync -v 
# rocm backend
scram b runtests_L1ScoutingPhase2TauTaggingROCmAsync -v 
```