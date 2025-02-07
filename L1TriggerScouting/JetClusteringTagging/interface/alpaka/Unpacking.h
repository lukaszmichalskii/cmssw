#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Unpacking_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Unpacking_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Unpacking {
public:
  void Unpack(Queue& queue, std::vector<uint64_t>& headers, std::vector<uint64_t>& data, PuppiCollection& collection);

private:
  void UnpackHeaders(Queue& queue, std::vector<uint64_t>& headers, PuppiCollection& collection) const;
  void UnpackData(Queue& queue, std::vector<uint64_t>& data, PuppiCollection& collection) const;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Unpacking_h