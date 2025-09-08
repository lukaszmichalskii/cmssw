#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "PhysicsTools/PyTorch/interface/DisableThreading.h"

using namespace cms::torch;

class PyTorchThreadingService {
public:
  PyTorchThreadingService(const edm::ParameterSet& config, edm::ActivityRegistry& registry)
    : verbose_(config.getUntrackedParameter<bool>("verbose", false)) {
    registry.watchPreGlobalBeginRun(this, &PyTorchThreadingService::preGlobalBeginRun);
  };
  ~PyTorchThreadingService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("verbose", false);
    descriptions.add("PyTorchThreadingService", desc);
    descriptions.setComment("Disable internal PyTorch threading model.");
  }

  void preGlobalBeginRun(edm::GlobalContext const&) {
    if (verbose_) {
      std::cout << "XXX" << std::endl;
      edm::LogInfo("PyTorchThreadingService") 
          << "Disabling PyTorch internal threading model."
            "All CPU based operations will run single-threaded.";
    }
    disableThreading();
  }

private:
  const bool verbose_;
};

DEFINE_FWK_SERVICE(PyTorchThreadingService);