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

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * @class L1TScPhase2CLUETaus
   * @brief Produces CLUEstering results: clusters lists + seeds indices
   */
  class L1TScPhase2CLUETaus : public stream::EDProducer<> {
  public:
    L1TScPhase2CLUETaus(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<PFCandidateCollection> pf_candidates_token_;
    const device::EDPutToken<CLUEsteringCollection> cluestering_token_;
    std::unique_ptr<L1TScPhase2CLUEstering> clustering_;
    const bool debug_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2CLUETaus::L1TScPhase2CLUETaus(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("src"))},
        cluestering_token_{produces()},
        debug_(params.getUntrackedParameter<bool>("debug")) {
    clustering_ = std::make_unique<L1TScPhase2CLUEstering>(static_cast<float>(params.getParameter<double>("dc")),
                                                           static_cast<float>(params.getParameter<double>("rhoc")),
                                                           static_cast<float>(params.getParameter<double>("dm")));
  }

  void L1TScPhase2CLUETaus::produce(device::Event &event, const device::EventSetup &event_setup) {
    // get collection from device memory space (implicit copy done by framework)
    const auto &pf_candidates = event.get(pf_candidates_token_);
    const auto n_points = pf_candidates.const_view().metadata().size();

    // allocate buffer for CLUEstering results
    auto clue_collection = CLUEsteringCollection(n_points, event.queue());
    clue_collection.zeroInitialise(event.queue());

    // clustering_->bindInputs(const_cast<PFCandidateCollection&>(pf_candidates));
    // clustering_->bindOutputs(clue_collection);
    // clustering_->numberOfPoints(n_points);
    // clustering_->setWrappedCoords({{0, 1}});
    clustering_->run(event.queue(), const_cast<PFCandidateCollection &>(pf_candidates), clue_collection);

    if (debug_) {
      auto clue_size = clue_collection.const_view().metadata().size();
      auto clue_host_collection = CLUEsteringHostCollection(clue_size, event.queue());
      alpaka::memcpy(event.queue(), clue_host_collection.buffer(), clue_collection.buffer());
      alpaka::wait(event.queue());
      fmt::print("[DEBUG] l1sc::L1TScPhase2CLUETaus: CLUEstering results:\n", clue_size);

      // table header
      const std::string separator = "+-------+---------+---------+";
      fmt::print("{}\n", separator);
      fmt::print("| {:>5} | {:>7} | {:>7} |\n", "index", "cluster", "is_seed");
      fmt::print("{}\n", separator);

      // log head of collection (10 records at most)
      auto span = (clue_size > 1000) ? 1000 : clue_size;
      for (int32_t idx = 0; idx < span; ++idx) {
        const auto &view = clue_host_collection.const_view()[idx];
        fmt::print("| {:5d} | {:7d} | {:7d} |\n", idx, view.cluster(), view.is_seed());
      }

      // log tail if collection size is larger than 10
      if (span < clue_size) {
        fmt::print("| {:>5} | {:>7} | {:>7} |\n", "...", "...", "...");
      }

      fmt::print("{}\n", separator);
    }

    // move converted soa to event storage
    event.emplace(cluestering_token_, std::move(clue_collection));

    // log info
    std::cout << "[INFO] l1sc::L1TScPhase2CLUETaus: OK" << std::endl;
  }

  /**
   * Define parameters for the module.
   * 
   */
  void L1TScPhase2CLUETaus::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.add<double>("dc", kDc);
    desc.add<double>("rhoc", kRhoc);
    desc.add<double>("dm", kDm);
    desc.addUntracked<bool>("debug", false);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2CLUETaus);