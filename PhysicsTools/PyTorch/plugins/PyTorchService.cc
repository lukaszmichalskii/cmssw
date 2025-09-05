#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "PhysicsTools/PyTorch/interface/ThreadingGuard.h"

class PyTorchService {
public:
  PyTorchService(const edm::ParameterSet& config, edm::ActivityRegistry& registry) {
    registry.watchPreGlobalBeginRun(this, &PyTorchService::preGlobalBeginRun);
  };
  ~PyTorchService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void preGlobalBeginRun(edm::GlobalContext const&);
};

// __________________________________________________________________________________________________________________
// IMPLEMENTATION

void PyTorchService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("PyTorchService", desc);
  descriptions.setComment("Set single threading for pytorch after the module is loaded.");
}

void PyTorchService::preGlobalBeginRun(edm::GlobalContext const&) { cms::torch::set_threading_guard(); }

DEFINE_FWK_SERVICE(PyTorchService);