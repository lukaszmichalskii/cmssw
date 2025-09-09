#include <cassert>

#include <fmt/format.h>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace torchtest {

  inline edm::InputTag getBackendTag(edm::InputTag const& tag) {
    return edm::InputTag(tag.label(), "backend", tag.process());
  }

  using namespace torchportabletest;

  class CollectionAnalyzer : public edm::stream::EDAnalyzer<> {
  public:
    CollectionAnalyzer(const edm::ParameterSet& params)
        : particles_token_{consumes(params.getParameter<edm::InputTag>("particles"))},
          particles_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("particles")))},
          classification_token_{consumes(params.getParameter<edm::InputTag>("classification"))},
          classification_backend_{
              consumes(getBackendTag(params.getParameter<edm::InputTag>("classification")))},
          regression_token_{consumes(params.getParameter<edm::InputTag>("regression"))},
          regression_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("regression")))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("particles");
      desc.add<edm::InputTag>("classification");
      desc.add<edm::InputTag>("regression");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::Event const& event, edm::EventSetup const&) override {
      auto const& particle_collection = event.get(particles_token_);
      auto const particle_collection_backend = static_cast<cms::alpakatools::Backend>(event.get(particles_backend_));

      auto const& classififcation_collection = event.get(classification_token_);
      auto const classifiacation_collection_backend =
          static_cast<cms::alpakatools::Backend>(event.get(classification_backend_));

      auto const& regression_collection = event.get(regression_token_);
      auto const regression_collection_backend = static_cast<cms::alpakatools::Backend>(event.get(regression_backend_));

      printParticleCollection(particle_collection, particle_collection_backend, event);
      printClassificationCollection(classififcation_collection, classifiacation_collection_backend, event);
      printRegressionCollection(regression_collection, regression_collection_backend, event);
    }

  private:
    const edm::EDGetTokenT<ParticleHostCollection> particles_token_;
    const edm::EDGetTokenT<unsigned short> particles_backend_;

    const edm::EDGetTokenT<ClassificationHostCollection> classification_token_;
    const edm::EDGetTokenT<unsigned short> classification_backend_;

    const edm::EDGetTokenT<RegressionHostCollection> regression_token_;
    const edm::EDGetTokenT<unsigned short> regression_backend_;

    const int32_t kMaxView = 5;

    void printParticleCollection(const ParticleHostCollection& collection, cms::alpakatools::Backend collection_backend, const edm::Event& event) {
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
      fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} | {:>7} | {:>7} |\n", "index", "pt", "eta", "phi");
      fmt::format_to(std::back_inserter(buffer), "{}", line);

      // Table rows (preview)
      for (int32_t i = 0; i < std::min<int32_t>(kMaxView, size); ++i) {
        const auto& view = collection.const_view()[i];
        fmt::format_to(std::back_inserter(buffer),
                      "| {:5d} | {:7.2f} | {:7.2f} | {:7.2f} |\n",
                      static_cast<int>(i),
                      view.pt(),
                      view.eta(),
                      view.phi());
      }

      // Ellipsis row if truncated
      if (size > kMaxView) {
        fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} | {:>7} | {:>7} |\n", "...", "...", "...", "...");
      }

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::print("{}", fmt::to_string(buffer));
    }

    void printRegressionCollection(const RegressionHostCollection& collection, cms::alpakatools::Backend collection_backend, const edm::Event& event) {
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
      fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} |\n", "index", "reco_pt");
      fmt::format_to(std::back_inserter(buffer), "{}", line);

      // Table rows (preview)
      for (int32_t i = 0; i < std::min<int32_t>(kMaxView, size); ++i) {
        const auto& view = collection.const_view()[i];
        fmt::format_to(std::back_inserter(buffer), "| {:5d} | {:7.2f} |\n", static_cast<int>(i), view.reco_pt());
      }

      // Ellipsis row if truncated
      if (size > kMaxView) {
        fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} |\n", "...", "...");
      }

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::print("{}", fmt::to_string(buffer));
    }

    void printClassificationCollection(const ClassificationHostCollection& collection, cms::alpakatools::Backend collection_backend, const edm::Event& event) {
      constexpr auto line = "+-------+-------+-------+\n";
      const auto size = collection.view().metadata().size();
      fmt::memory_buffer buffer;

      // Header message
      fmt::format_to(std::back_inserter(buffer),
                    "[DEBUG] ClassificationCollection[{}] ({}, {}):\n",
                    size,
                    cms::alpakatools::toString(collection_backend),
                    event.id().event());

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>5} | {:>5} |\n", "index", "c1", "c2");
      fmt::format_to(std::back_inserter(buffer), "{}", line);

      // Table rows (preview)
      for (int32_t i = 0; i < std::min<int32_t>(kMaxView, size); ++i) {
        const auto& view = collection.const_view()[i];
        fmt::format_to(
            std::back_inserter(buffer), "| {:5d} | {:5.2f} | {:5.2f} |\n", static_cast<int>(i), view.c1(), view.c2());
      }

      // Ellipsis row if truncated
      if (size > kMaxView) {
        fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>5} | {:>5} |\n", "...", "...", "...");
      }

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::print("{}", fmt::to_string(buffer));
    }
  };

}  // namespace torchtest

DEFINE_FWK_MODULE(torchtest::CollectionAnalyzer);