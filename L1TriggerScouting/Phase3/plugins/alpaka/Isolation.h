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
  // template<typename T, typename U, typename Tc>
  // void Filter(Queue& queue, PuppiCollection::ConstView const_view, uint32_t begin, uint32_t end, T* __restrict__ mask, U* __restrict__ charge, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct) const;

  // template<typename T, typename Tc>
  // void EstimateSize(Queue &queue, T* __restrict__ mask, uint32_t begin, uint32_t end, Tc* __restrict__ accumulator) const;

  // template<typename T, typename U, typename Tc, typename Tf>
  // void Combinatorics(
  //   Queue& queue, PuppiCollection::ConstView const_view, 
  //   uint32_t begin, uint32_t end, 
  //   T* __restrict__ mask, U* __restrict__ charge, 
  //   Tc* __restrict__ pions_num, Tc* __restrict__ int_cut_ct, Tc* __restrict__ high_cut_ct, Tf* __restrict__ best_score
  // ) const;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_Isolation_h