// #ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h
// #define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h

// #include <string>
// #include <memory>
// #include <alpaka/alpaka.hpp>
// #include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// #include "PhysicsTools/TensorFlow/interface/TensorFlow.h"


// namespace ALPAKA_ACCELERATOR_NAMESPACE {

// class Tagging {
// public:
//   Tagging();
//   Tagging(const std::string &model);
//   ~Tagging();
//   void Tag(Queue& queue, PuppiCollection& data);
// private:
//   tensorflow::GraphDef* graph_def_;
//   tensorflow::Session* session_;
// };

// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// #endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h