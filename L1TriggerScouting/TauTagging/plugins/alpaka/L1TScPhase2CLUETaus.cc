#include "DataFormats/L1ScoutingSoA/interface/alpaka/CLUEsteringCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2CLUEstering.h"
#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2Kernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * @class L1TScPhase2CLUETaus
   * @brief Produces CLUEstering results: clusters lists + seeds indices
   */
  class L1TScPhase2CLUETaus : public stream::EDProducer<> {
  public:
    explicit L1TScPhase2CLUETaus(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    void logDebugMessage(const CLUEsteringHostCollection &clue_collection) const;

    const device::EDGetToken<PFCandidateCollection> pf_candidates_token_;
    const device::EDPutToken<CLUEsteringCollection> cluestering_token_;
    const edm::EDPutTokenT<int> num_clusters_token_;
    std::unique_ptr<L1TScPhase2CLUEstering> clustering_;
    const bool verbose_;
    const int verbose_level_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2CLUETaus::L1TScPhase2CLUETaus(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("src"))},
        cluestering_token_{produces()},
        num_clusters_token_{produces()},
        verbose_(params.getUntrackedParameter<bool>("verbose")),
        verbose_level_(params.getUntrackedParameter<int>("verboseLevel")) {
    clustering_ = std::make_unique<L1TScPhase2CLUEstering>(static_cast<float>(params.getParameter<double>("dc")),
                                                           static_cast<float>(params.getParameter<double>("rhoc")),
                                                           static_cast<float>(params.getParameter<double>("dm")),
                                                           params.getParameter<bool>("wrapCoords"));
  }

  /**
   * Execute the logic of the module.
   */
  void L1TScPhase2CLUETaus::produce(device::Event &event, const device::EventSetup &event_setup) {
    // get collection from device memory space (implicit copy done by framework)
    const auto &pf_candidates = event.get(pf_candidates_token_);
    const auto n_points = pf_candidates.const_view().metadata().size();

    // allocate buffer for CLUEstering results
    auto clue_collection = CLUEsteringCollection(n_points, event.queue());

    // run CLUEstering
    clustering_->run(event.queue(), const_cast<PFCandidateCollection &>(pf_candidates), clue_collection);
    int num_clusters = kernels::max(event.queue(), clue_collection.const_view().cluster(), n_points);

    // debug info
    if (verbose_) {
      auto clue_host_collection = CLUEsteringHostCollection(clue_collection.view().metadata().size(), event.queue());
      alpaka::memcpy(event.queue(), clue_host_collection.buffer(), clue_collection.buffer());
      alpaka::wait(event.queue());
      logDebugMessage(clue_host_collection);
    }

    // move clustering results to event storage
    event.emplace(cluestering_token_, std::move(clue_collection));
    event.emplace(num_clusters_token_, std::move(num_clusters));

    // log info
    std::cout << "[INFO] l1sc::L1TScPhase2CLUETaus: OK" << std::endl;
  }

  /**
   * Define parameters for the module.
   */
  void L1TScPhase2CLUETaus::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.add<double>("dc", kDc);
    desc.add<double>("rhoc", kRhoc);
    desc.add<double>("dm", kDm);
    desc.add<bool>("wrapCoords", kWrapCoords);
    desc.addUntracked<bool>("verbose", false);
    desc.addUntracked<int>("verboseLevel", 0);
    descriptions.addWithDefaultLabel(desc);
  }

  /**
   * Log CLUEstering results to stdout
   */
  void L1TScPhase2CLUETaus::logDebugMessage(const CLUEsteringHostCollection &clue_collection) const {
    const auto size = clue_collection.const_view().metadata().size();
    fmt::print("[DEBUG] l1sc::L1TScPhase2CLUETaus: CLUEstering results:\n", size);
    // table header
    const std::string separator = "+-------+---------+---------+";
    fmt::print("{}\n", separator);
    fmt::print("| {:>5} | {:>7} | {:>7} |\n", "index", "cluster", "is_seed");
    fmt::print("{}\n", separator);
    // log collection
    auto span = (size > 10) ? 10 : size;
    if (verbose_level_ == 1)
      span = size;
    for (int32_t idx = 0; idx < span; ++idx) {
      const auto &view = clue_collection.const_view()[idx];
      fmt::print("| {:5d} | {:7d} | {:7d} |\n", idx, view.cluster(), view.is_seed());
    }
    // log tail
    if (span < size) {
      fmt::print("| {:>5} | {:>7} | {:>7} |\n", "...", "...", "...");
    }
    fmt::print("{}\n", separator);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2CLUETaus);