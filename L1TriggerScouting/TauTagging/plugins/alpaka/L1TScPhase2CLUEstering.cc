#include "CLUEstering/CLUEstering.hpp"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/OrbitEventIndexMapCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
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
   * @class L1TScPhase2CLUEstering
   * @brief Produces CLUEstering results: clusters lists + seeds indices
   */
  class L1TScPhase2CLUEstering : public stream::EDProducer<> {
  public:
    L1TScPhase2CLUEstering(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<PFCandidateCollection> pf_candidates_token_;
    const device::EDGetToken<OrbitEventIndexMapCollection> orbit_association_map_token_;
    const device::EDPutToken<PFCandidateCollection> clusters_token_;
    std::chrono::steady_clock::time_point t_start_, t_end_;
    std::unique_ptr<clue::Clusterer<kDims>> clue_algo_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2CLUEstering::L1TScPhase2CLUEstering(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_token_(consumes(params.getParameter<edm::InputTag>("src"))),
        orbit_association_map_token_(consumes(params.getParameter<edm::InputTag>("src"))),
        clusters_token_{produces()} {
    clue_algo_ = std::make_unique<clue::Clusterer<kDims>>(
        static_cast<float>(params.getParameter<double>("dc")), 
        static_cast<float>(params.getParameter<double>("rhoc")),
        static_cast<float>(params.getParameter<double>("dm")));
  }

  void L1TScPhase2CLUEstering::produce(device::Event &event, const device::EventSetup &event_setup) {
    // timestamp
    t_start_ = std::chrono::steady_clock::now();

    // get raw data input
    auto& pf_candidates = event.get(pf_candidates_token_);
    auto& orbit_association_map = event.get(orbit_association_map_token_);
    auto tmp = PFCandidateCollection(1, event.queue());
    event.emplace(clusters_token_, std::move(tmp));

    // explicit device sync (only for time measurements)
    alpaka::wait(event.queue());

    // timestamp
    t_end_ = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end_ - t_start_).count();

    // log info
    std::cout << "OK - L1TScPhase2CLUEstering [" << elapsed << " us]" << std::endl;
  }

  /**
   * Define parameters for the module.
   * 
   */
  void L1TScPhase2CLUEstering::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.add<double>("dc", kDc);
    desc.add<double>("rhoc", kRhoc);
    desc.add<double>("dm", kDm);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2CLUEstering);