#ifndef PhysicsTools_PyTorchAlpakaTest_interface_Common_h
#define PhysicsTools_PyTorchAlpakaTest_interface_Common_h

#include <fmt/format.h>
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"

namespace torchtest {

  constexpr int32_t kMaxView = 5;

  void printParticleCollection(const torchportabletest::ParticleHostCollection& collection, cms::alpakatools::Backend collection_backend, const edm::Event& event) {
    constexpr auto line = "+-------+---------+---------+---------+\n";
    const auto size = collection.view().metadata().size();
    fmt::memory_buffer buffer;

    // Header message
    fmt::format_to(std::back_inserter(buffer),
                  "[DEBUG] ParticleCollection[{}] ({}, {}):\n",
                  size,
                  cms::alpakatools::toString(collection_backend),
                  event.id().event());

    fmt::format_to(std::back_inserter(buffer), "{}", line);
    fmt::format_to(std::back_inserter(buffer),
                  "| {:>5} | {:>7} | {:>7} | {:>7} |\n",
                  "index", "pt", "eta", "phi");
    fmt::format_to(std::back_inserter(buffer), "{}", line);

    // Table rows (preview)
    for (int32_t i = 0; i < std::min<int32_t>(kMaxView, size); ++i) {
      const auto& view = collection.const_view()[i];
      fmt::format_to(std::back_inserter(buffer),
                    "| {:5d} | {:7.2f} | {:7.2f} | {:7.2f} |\n",
                    static_cast<int>(i), view.pt(), view.eta(), view.phi());
    }

    // Ellipsis row if truncated
    if (size > kMaxView) {
      fmt::format_to(std::back_inserter(buffer),
                    "| {:>5} | {:>7} | {:>7} | {:>7} |\n",
                    "...", "...", "...", "...");
    }

    fmt::format_to(std::back_inserter(buffer), "{}", line);
    fmt::print("{}", fmt::to_string(buffer));
  }

  void printRegressionCollection(const torchportabletest::RegressionHostCollection& collection, cms::alpakatools::Backend collection_backend, const edm::Event& event) {
    constexpr auto line = "+-------+---------+\n";
    const auto size = collection.view().metadata().size();
    fmt::memory_buffer buffer;

    // Header message
    fmt::format_to(std::back_inserter(buffer),
                  "[DEBUG] RegressionCollection[{}] ({}, {}):\n",
                  size,
                  cms::alpakatools::toString(collection_backend),
                  event.id().event());

    fmt::format_to(std::back_inserter(buffer), "{}", line);
    fmt::format_to(std::back_inserter(buffer),
                  "| {:>5} | {:>7} |\n",
                  "index", "reco_pt");
    fmt::format_to(std::back_inserter(buffer), "{}", line);

    // Table rows (preview)
    for (int32_t i = 0; i < std::min<int32_t>(kMaxView, size); ++i) {
      const auto& view = collection.const_view()[i];
      fmt::format_to(std::back_inserter(buffer),
                    "| {:5d} | {:7.2f} |\n",
                    static_cast<int>(i), view.reco_pt());
    }

    // Ellipsis row if truncated
    if (size > kMaxView) {
      fmt::format_to(std::back_inserter(buffer),
                    "| {:>5} | {:>7} |\n",
                    "...", "...");
    }

    fmt::format_to(std::back_inserter(buffer), "{}", line);
    fmt::print("{}", fmt::to_string(buffer));
  }

}  // namespace torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_Common_h