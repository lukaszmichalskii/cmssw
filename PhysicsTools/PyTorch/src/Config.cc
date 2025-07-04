#include "PhysicsTools/PyTorch/interface/Config.h"
#include "FWCore/Utilities/interface/Exception.h"


namespace cms::torch {

  ::torch::jit::script::Module load(const std::string &model_path) {
    try {
      return ::torch::jit::load(model_path);
    } catch (const c10::Error &e) {
      cms::Exception ex("ModelLoadingError");
      ex << "Error loading the model: " << e.what();
      ex.addContext("Calling cms::torch::load(const std::string&)");
      throw ex;
    }
  }

}  // namespace cms::torch