#include <cstdlib>
#include <iostream>
#include <chrono>
#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestL1ScoutingSoA.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

constexpr size_t SIZE = 100;
constexpr size_t THREADS_PER_BLOCK = 128;

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  for (const auto& device : devices) {
    std::cout << "\n\tRunning on: " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    // Inner scope to deallocate memory before destroying the stream
    {
      // Instantiate on device. Allocation on device is done automatically
      PuppiCollection collection(SIZE, queue);

      const auto start = std::chrono::high_resolution_clock::now();
      test_l1_scouting_soa::LaunchKernels(collection.view(), queue, THREADS_PER_BLOCK);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      std::cout << "\tthreads_per_block = " << THREADS_PER_BLOCK << "; duration = " << duration.count() << " ns" << std::endl;

      // Instantiate on host. Destination memory for data to be copied.
      PuppiHostCollection h_collection(collection.view().metadata().size(), queue);
      alpaka::memcpy(queue, h_collection.buffer(), collection.const_buffer());
      alpaka::wait(queue);

      // Puppi collection on device
      for (int i = 0; i < h_collection.view().metadata().size(); ++i) {
        std::cout << "\tPuppi collection on device:\n\t";
        std::cout << h_collection.view().bx().size() << "; ";
        std::cout << h_collection.view().offsets().size() << "; ";
        std::cout << h_collection.view()[i].pt() << "; ";
        std::cout << h_collection.view()[i].eta() << "; ";
        std::cout << h_collection.view()[i].phi() << "; ";
        std::cout << h_collection.view()[i].z0() << "; ";
        std::cout << h_collection.view()[i].dxy() << "; ";
        std::cout << h_collection.view()[i].puppiw() << "; ";
        std::cout << h_collection.view()[i].pdgId() << "; ";
        std::cout << h_collection.view()[i].quality() << std::endl;
        break; // one line only for debugging
      }
    }
  }
  return EXIT_SUCCESS;
}