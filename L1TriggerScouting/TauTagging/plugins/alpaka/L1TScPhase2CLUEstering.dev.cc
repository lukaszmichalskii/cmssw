#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2CLUEstering.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  L1TScPhase2CLUEstering::L1TScPhase2CLUEstering(float dc, float rhoc, float dm, bool wrap_coords)
      : dc_(dc), rhoc_(rhoc), dm_(dm), wrap_coords_(wrap_coords) {}

  void L1TScPhase2CLUEstering::run(Queue& queue,
                                   PFCandidateCollection& pf_candidates,
                                   CLUEsteringCollection& clue_collection) {
    uint32_t n_points = pf_candidates.view().metadata().size();

    auto* coords_ptr = pf_candidates.view().eta();
    auto* weights_ptr = pf_candidates.view().pt();
    auto* cluster_indices_ptr = clue_collection.view().cluster();
    auto* is_seed_ptr = clue_collection.view().is_seed();

    auto points_device =
        clue::PointsDevice<kDims, Device>(queue, n_points, coords_ptr, weights_ptr, cluster_indices_ptr, is_seed_ptr);
    
    auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);    
    if (wrap_coords_)
      clue_algo.setWrappedCoordinates({{0, 1}});
    clue_algo.make_clusters(points_device, FlatKernel{0.5f}, queue, /** block_size = */ 64);

    // uint32_t n_points = pf_candidates.view().metadata().size();

    // auto pf_host_collection = PFCandidateHostCollection(n_points, queue);
    // auto clue_host_collection = CLUEsteringHostCollection(n_points, queue);

    // alpaka::memcpy(queue, pf_host_collection.buffer(), pf_candidates.buffer());
    // alpaka::memcpy(queue, clue_host_collection.buffer(), clue_collection.buffer());
    // // alpaka::wait(queue);

    // auto* coords_host_ptr = pf_host_collection.view().eta();
    // auto* weights_host_ptr = pf_host_collection.view().pt();
    // auto* cluster_indices_host_ptr = clue_host_collection.view().cluster();
    // auto* is_seed_host_ptr = clue_host_collection.view().is_seed();

    // auto points_host = clue::PointsHost<kDims>(
    //     queue, n_points, coords_host_ptr, weights_host_ptr, cluster_indices_host_ptr, is_seed_host_ptr);
    // auto points_device =
    //     clue::PointsDevice<kDims, Device>(queue, n_points);
    // auto clue_algo = clue::Clusterer<kDims>(queue, dc_, rhoc_, dm_);  
    // if (wrap_coords_)
    //   clue_algo.setWrappedCoordinates({{0, 1}});
    // clue_algo.make_clusters(points_host, points_device, FlatKernel{0.5f}, queue, /** block_size = */ 64);
    // alpaka::memcpy(queue, clue_collection.buffer(), clue_host_collection.buffer());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc