#include <cstdlib>
#include <iostream>
#include <chrono>
#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestPuppiCollection.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

constexpr size_t SIZE = 100;

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
      // Instantiate tracks on device. PortableDeviceCollection allocates
      // SoA on device automatically.
      PuppiCollection collection(SIZE, queue);

      std::vector<int> threads_per_block = {32, 64, 128, 256, 512, 1024};
      for (const auto& threads_ct : threads_per_block) {
        const auto start = std::chrono::high_resolution_clock::now();
        test_puppi_collection::LaunchKernels(collection.view(), queue, threads_ct);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "\tthreads_ct = " << threads_ct << "; duration = " << duration.count() << " ns" << std::endl;
      }

      // Instantate tracks on host. This is where the data will be
      // copied to from device.
      PuppiHostCollection puppi(collection.view().metadata().size(), queue);
      alpaka::memcpy(queue, puppi.buffer(), collection.const_buffer());
      alpaka::wait(queue);
    }
  }
  return EXIT_SUCCESS;
}