#include "DataFormats/L1ScoutingSoA/interface/alpaka/ClustersDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/TauTagging/plugins/alpaka/CLUEsteringAlgo.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  class CLUETaus : public stream::EDProducer<> {
  public:
    explicit CLUETaus(const edm::ParameterSet &params) 
        : EDProducer<>(params),
          pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("pf"))},
          cluestering_token_{produces()},
          num_clusters_token_{produces()},
          clustering_(static_cast<float>(params.getParameter<double>("dc")),
                      static_cast<float>(params.getParameter<double>("rhoc")),
                      static_cast<float>(params.getParameter<double>("dm")),
                      params.getParameter<bool>("wrapCoords")),
          environment_{static_cast<Environment>(params.getUntrackedParameter<int>("environment"))} {}

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      // get collection from device memory space (implicit copy done by framework)
      const auto &pf = event.get(pf_candidates_token_);
      const auto n_points = pf.const_view().metadata().size();

    // allocate buffer
    auto clusters = ClustersDeviceCollection(n_points, event.queue());

    // run CLUEstering algo
    clustering_.run(event.queue(), pf, clusters);

    // move clustering results to event storage
    event.emplace(cluestering_token_, std::move(clusters));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pf");
      desc.add<double>("dc");
      desc.add<double>("rhoc");
      desc.add<double>("dm");
      desc.add<bool>("wrapCoords");
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // get device pf data 
    const device::EDGetToken<PFCandidateDeviceCollection> pf_candidates_token_;
    // put device clustering data
    const device::EDPutToken<ClustersDeviceCollection> cluestering_token_;
    const edm::EDPutTokenT<int> num_clusters_token_;
    // algorithm
    const kernels::CLUEsteringAlgo clustering_;
    // debug / test
    const Environment environment_;
  };

  // /**
  //  * Log CLUEstering results to stdout
  //  */
  // void CLUETaus::logDebugMessage(const CLUEsteringHostCollection &clue_collection) const {
  //   const auto size = clue_collection.const_view().metadata().size();
  //   fmt::print("[DEBUG] l1sc::CLUETaus: CLUEstering results:\n", size);
  //   // table header
  //   const std::string separator = "+-------+---------+---------+";
  //   fmt::print("{}\n", separator);
  //   fmt::print("| {:>5} | {:>7} | {:>7} |\n", "index", "cluster", "is_seed");
  //   fmt::print("{}\n", separator);
  //   // log collection
  //   auto span = (size > 10) ? 10 : size;
  //   if (verbose_level_ == 1)
  //     span = size;
  //   for (int32_t idx = 0; idx < span; ++idx) {
  //     const auto &view = clue_collection.const_view()[idx];
  //     fmt::print("| {:5d} | {:7d} | {:7d} |\n", idx, view.cluster(), view.is_seed());
  //   }
  //   // log tail
  //   if (span < size) {
  //     fmt::print("| {:>5} | {:>7} | {:>7} |\n", "...", "...", "...");
  //   }
  //   fmt::print("{}\n", separator);
  // }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::CLUETaus);