#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "PhysicsTools/PyTorch/interface/DisableThreading.h"

class PyTorchService {
public:
  PyTorchService(const edm::ParameterSet& config, edm::ActivityRegistry& registry) { 
    registry.watchPreGlobalBeginRun(this, &PyTorchService::preGlobalBeginRun); 
  };

  ~PyTorchService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.add("PyTorchService", desc);
    descriptions.setComment("Disable internal PyTorch threading model.");
  }

  void preGlobalBeginRun(edm::GlobalContext const&) {
    edm::LogInfo("PyTorchService") << "Disabling PyTorch internal threading model."
      "All Torch CPU based operations will run single-threaded.";          
    cms::torch::disableThreading();
  }
};

DEFINE_FWK_SERVICE(PyTorchService);