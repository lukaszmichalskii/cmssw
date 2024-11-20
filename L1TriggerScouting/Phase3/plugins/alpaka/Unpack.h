#ifndef L1TriggerScouting_Phase3_plugins_alpaka_Unpack_h
#define L1TriggerScouting_Phase3_plugins_alpaka_Unpack_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Unpack {
  public:
    void ProcessHeaders(Queue& queue, std::vector<uint64_t>& headers, PuppiCollection& collection) const;
    void ProcessData(Queue& queue, std::vector<uint64_t>& data, PuppiCollection& collection) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_Unpack_h