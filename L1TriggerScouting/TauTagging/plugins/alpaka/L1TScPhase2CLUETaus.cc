#include "CLUEstering/CLUEstering.hpp"
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
    std::unique_ptr<clue::Clusterer<kDims>> clue_algo_;
    const bool debug_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2CLUETaus::L1TScPhase2CLUETaus(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("src"))},
        cluestering_token_{produces()},
        debug_(params.getUntrackedParameter<bool>("debug")) {
    // instantiate CLUEstering
    clue_algo_ = std::make_unique<clue::Clusterer<kDims>>(
        static_cast<float>(params.getParameter<double>("dc")), 
        static_cast<float>(params.getParameter<double>("rhoc")),
        static_cast<float>(params.getParameter<double>("dm")));
  }

  void L1TScPhase2CLUETaus::produce(device::Event &event, const device::EventSetup &event_setup) {
    // get collection from device memory space (implicit copy done by framework)
    const auto& pf_candidates = event.get(pf_candidates_token_);
    const auto n_points = pf_candidates.const_view().metadata().size();

    // allocate buffer for CLUEstering results
    auto clue_collection = CLUEsteringCollection(n_points, event.queue());
    clue_collection.zeroInitialise(event.queue());

    // // extract device pointers? 
    // auto coords_ptr = reinterpret_cast<std::byte*>(const_cast<float*>(pf_candidates.const_view().eta()));
    // auto weights_ptr = reinterpret_cast<std::byte*>(const_cast<float*>(pf_candidates.const_view().pt()));
    // auto clusters_ptr = reinterpret_cast<std::byte*>(clue_collection.view().clusters());
    // auto seeds_ptr = reinterpret_cast<std::byte*>(clue_collection.view().seeds());

    // // wrap preallocated memory
    // auto clue_data = clue::PointsDevice<kDims, Device>(
    //     event.queue(), n_points, coords_ptr, weights_ptr, clusters_ptr, seeds_ptr);

    // // use default kernel
    // const auto kernel = FlatKernel{0.5};   

    // // set phi coordinate to be circular (wrapped)
    // auto coords_wrap = std::array<uint8_t, kDims>{{0, 1}};
    // clue_algo_->setWrappedCoordinates(coords_wrap);

    // // run clustering
    // clue_algo_->make_clusters(clue_data, kernel, event.queue(), 64);

    if (debug_) {
      auto clue_size = clue_collection.const_view().metadata().size();
      auto clue_host_collection = CLUEsteringHostCollection(clue_size, event.queue());
      alpaka::memcpy(event.queue(), clue_host_collection.buffer(), clue_collection.buffer());
      fmt::print("[DEBUG] l1sc::L1TScPhase2CLUETaus: CLUEstering results:\n",
                 clue_size);

      // table header
      const std::string separator = "+-------+---------+---------+";
      fmt::print("{}\n", separator);
      fmt::print("| {:>5} | {:>7} | {:>7} |\n",
                 "index",
                 "cluster",
                 "is_seed");
      fmt::print("{}\n", separator);

      // log head of collection (10 records at most)
      auto span = (clue_size > 10) ? 10 : clue_size;
      for (int32_t idx = 0; idx < span; ++idx) {
        const auto &view = clue_host_collection.const_view()[idx];
        fmt::print("| {:5d} | {:7d} | {:7d} |\n",
                   idx,
                   view.cluster(),
                   view.is_seed());
      }

      // log tail if collection size is larger than 10
      if (span < clue_size) {
        fmt::print("| {:>5} | {:>7} | {:>7} |\n",
                   "...",
                   "...",
                   "...");
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