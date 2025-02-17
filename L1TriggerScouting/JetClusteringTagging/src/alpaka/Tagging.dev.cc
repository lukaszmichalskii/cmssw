// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Tagging.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

Tagging::Tagging() {
  tensorflow::setLogging("3");
  const std::string model = "/afs/cern.ch/user/l/lmichals/private/CMSSW_14_0_12/src/L1TriggerScouting/JetClusteringTagging/plugins/alpaka/graph.pb";
  // load the graph definition
  printf("Compiling tensorflow:: model %s\n", model.c_str());
  graph_def_ = tensorflow::loadGraphDef(model);
  tensorflow::Options options {tensorflow::Backend::cpu};
  // tensorflow::SessionCache cache(model, options);
  session_ = tensorflow::createSession(graph_def_, options);

  printf("--------------------------------------------\n");
  printf("Layers:\n");
  for (const auto& node : graph_def_->node()) {
    printf("%s\n", node.name().c_str());
  }
  printf("\n");
  printf("--------------------------------------------\n");
}

Tagging::Tagging(const std::string &model) {
  tensorflow::setLogging("3");

  // load the graph definition
  printf("Compiling tensorflow:: model %s\n", model.c_str());
  graph_def_ = tensorflow::loadGraphDef(model);
  session_ = tensorflow::createSession(graph_def_);

  printf("--------------------------------------------\n");
  printf("Layers:\n");
  for (const auto& node : graph_def_->node()) {
    printf("%s\n", node.name().c_str());
  }
  printf("\n");
  printf("--------------------------------------------\n");
}

Tagging::~Tagging() {
  tensorflow::closeSession(session_);
  delete graph_def_;
  graph_def_ = nullptr;
}

void Tagging::Tag(Queue& queue, PuppiCollection& data) {
  // dummy inputs
  tensorflow::Tensor inputs(tensorflow::DT_FLOAT, { 1, 10 });
  for (size_t i = 0; i < 10; i++) {
      inputs.matrix<float>()(0, i) = float(i);
  }
  
  // inference
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(session_, { { "inputs", inputs } }, { "sequential_1/output_1/Softmax" }, &outputs);

  for (auto output : outputs) {
    printf("Probs: ");
    for (size_t i = 0; i < 3; i++) {
      printf("%.4f ", output.matrix<float>()(0, i));
    }
    printf("\n");
  }
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
