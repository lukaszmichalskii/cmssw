#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h

// libs
#include <string>
#include <memory>
#include <alpaka/alpaka.hpp>
// typedefs
#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/JetsCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// heterogeneous
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// inference runtime
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Tagging {
public:
  Tagging(const std::string &model, const std::string &backend);
  void Tag(Queue& queue, PuppiCollection const& data, ClustersCollection const& clusters, JetsCollection& jets);
  
private:
  std::string model_;
  std::string backend_;
  std::unique_ptr<Ort::Env> env_ = nullptr;
  std::unique_ptr<Ort::Session> session_ = nullptr;
  std::unique_ptr<Ort::RunOptions> options_ = nullptr;

  std::unique_ptr<Ort::MemoryInfo> device_mem_allocator_info_ = nullptr;

  std::vector<std::string> input_node_strings_;
  std::vector<const char*> input_node_names_;
  std::map<std::string, std::vector<int64_t>> input_node_dims_;

  std::vector<std::string> output_node_strings_;
  std::vector<const char*> output_node_names_;
  std::map<std::string, std::vector<int64_t>> output_node_dims_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Tagging_h