#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2CLUEstering.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  L1TScPhase2CLUEstering::L1TScPhase2CLUEstering(float dc, float rhoc, float dm) 
      : clue_algo_(std::make_unique<clue::Clusterer<kDims>>(dc, rhoc, dm)) {}

  void L1TScPhase2CLUEstering::run(Queue& queue) {
    // std::vector<float>& coords = {1, 2, 3, 4};
    // std::vector<int>& results;
    // clue::PointsHost<1> h_points(queue, nTracks, coords, results);
    // auto points = clue::PointsDevice<kDims, Device>(queue, n_points_, input_buffer_, output_buffer_);
    // clue_algo_->make_clusters(points, kernel_, queue, 64);
  }

  void L1TScPhase2CLUEstering::bindInputs(PFCandidateCollection& pf_candidates) {
    // ptr to first element in memory -> span up to n_points
    // cover: eta, phi, pt
    auto* mem_ptr = pf_candidates.view().eta();
    input_buffer_ = mem_ptr;
  }

  void L1TScPhase2CLUEstering::bindOutputs(CLUEsteringCollection& clue_collection) {
    // ptr to first element in memory -> span up to n_points
    // cover: cluster, is_seed
    auto* mem_ptr = clue_collection.view().cluster();
    output_buffer_ = mem_ptr;
  }

  void L1TScPhase2CLUEstering::numberOfPoints(uint32_t n_points) {
    n_points_ = n_points;
  }

  void L1TScPhase2CLUEstering::setWrappedCoords(const std::array<uint8_t, kDims>& coords) {
    clue_algo_->setWrappedCoordinates(coords);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc