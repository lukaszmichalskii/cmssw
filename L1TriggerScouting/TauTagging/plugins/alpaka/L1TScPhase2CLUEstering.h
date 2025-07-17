#ifndef L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2CLUEstering_h
#define L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2CLUEstering_h

#include <alpaka/alpaka.hpp>

#include "CLUEstering/CLUEstering.hpp"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/CLUEsteringCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  using namespace ::l1sc;

  class L1TScPhase2CLUEstering {
  public:
    explicit L1TScPhase2CLUEstering(float dc, float rhoc, float dm);

    void run(Queue& queue);
    void bindInputs(PFCandidateCollection& pf_candidates);
    void bindOutputs(CLUEsteringCollection& clue_collection);
    void numberOfPoints(uint32_t n_points);
    void setWrappedCoords(const std::array<uint8_t, kDims>& coords);

  private:
    std::unique_ptr<clue::Clusterer<kDims>> clue_algo_;
    const FlatKernel kernel_{0.5f};
    float* input_buffer_ = nullptr;
    uint32_t* output_buffer_ = nullptr;
    uint32_t n_points_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

#endif  // L1TriggerScouting_TauTagging_plugins_alpaka_L1TScPhase2CLUEstering_h