```bash
# workspace
cmsrel CMSSW_15_1_0_pre3
cd CMSSW_15_1_0_pre3/src
cmsenv

# setup clustering lib (external)
git clone git@github.com:cms-patatrack/CLUEstering.git
cd CLUEstering && git submodule update --init && cd ..

# build and test
scram b -j $(nproc)
scram b runtests
```