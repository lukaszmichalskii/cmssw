#include <alpaka/alpaka.hpp>
#include "CLUEstering/CLUEstering.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  int runClueTest() {
    Platform platform;
    std::vector<Device> devices = ::alpaka::getDevs(platform);
    const auto& device = devices[0];
    Queue queue{device};
    std::vector<float> coords = {// x, y
                                 0.1,
                                 0.1,
                                 0.1,
                                 0.1,
                                 0.1,
                                 0.1,
                                 5.1,
                                 5.1,
                                 8.1,
                                 8.1,
                                 // weights
                                 1.0,
                                 1.0,
                                 1.0,
                                 1.0,
                                 1.0};
    std::vector<int> results = {// cluster index
                                0,
                                0,
                                0,
                                0,
                                0,
                                // is_seed
                                0,
                                0,
                                0,
                                0,
                                0};

    std::cout << "Clusters: ";
    for (std::size_t i = 0; i < results.size(); ++i) {
      std::cout << results[i] << " ";
      if (i == results.size() / 2 - 1) {
        std::cout << "\n";
        std::cout << "Is seed: ";
      }
    }

    auto* coords_ptr = coords.data();
    auto* results_ptr = results.data();

    clue::PointsHost<2> h_points(queue, results.size() / 2, coords_ptr, results_ptr);
    clue::PointsDevice<2, Device> d_points(queue, results.size() / 2);
    clue::Clusterer<2> algo(queue, 0.1, 0.1, 0.1);

    const std::size_t block_size{64};
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, block_size);
    alpaka::wait(queue);

    std::cout << "\n-----------------------------\n";

    std::cout << "Clusters: ";
    for (std::size_t i = 0; i < results.size(); ++i) {
      std::cout << results[i] << " ";
      if (i == results.size() / 2 - 1) {
        std::cout << "\n";
        std::cout << "Is seed: ";
      }
    }

    std::cout << "\n";
    return 0;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  ALPAKA_ACCELERATOR_NAMESPACE::runClueTest();
  return 0;
}