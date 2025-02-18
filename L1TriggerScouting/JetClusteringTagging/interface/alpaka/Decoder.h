#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Unpacking_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Unpacking_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {


class Decoder {
public:
  void Decode(Queue& queue, std::vector<uint64_t>& headers, std::vector<uint64_t>& data, PuppiCollection& collection);

private:
  void DecodeHeaders(Queue& queue, WorkDiv<Dim1D>& grid, std::vector<uint64_t>& headers, PuppiCollection& collection) const;
  void DecodeData(Queue& queue, WorkDiv<Dim1D>& grid, std::vector<uint64_t>& data, PuppiCollection& collection) const;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Unpacking_h