#ifndef L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h
#define L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Isolation {
public:
  PuppiCollection Isolate(Queue& queue, PuppiCollection const& data) const;

private:
  template<typename T, typename U, typename Tc>
  void Filter(Queue& queue, PuppiCollection::ConstView const_view, T* mask, U* charge, Tc* int_cut_ct, Tc* high_cut_ct) const;
  template<typename T, typename Tc>
  void EstimateSize(Queue &queue, T* mask, uint32_t const size, Tc* accumulator) const;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h