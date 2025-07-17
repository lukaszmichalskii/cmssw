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

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * @class L1TScPhase2Jets
   * @brief Produces Jet representation for clustered PF candidates used in ML inference
   */
  class L1TScPhase2Jets : public stream::EDProducer<> {
  public:
    explicit L1TScPhase2Jets(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<PFCandidateCollection> pf_candidates_token_;
    const device::EDGetToken<CLUEsteringCollection> cluestering_token_;
    const bool debug_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2Jets::L1TScPhase2Jets(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("srcPFCandidates"))},
        cluestering_token_{consumes(params.getParameter<edm::InputTag>("srcCLUETaus"))},
        debug_(params.getUntrackedParameter<bool>("debug")) {}

  /**
   * Execute the logic of the module.
   */
  void L1TScPhase2Jets::produce(device::Event &event, const device::EventSetup &event_setup) {
    // get collections
    const auto &pf_candidates = event.get(pf_candidates_token_);
    const auto &clue_collection = event.get(cluestering_token_);

    if (debug_) {
    }

    // log info
    std::cout << "[INFO] l1sc::L1TScPhase2Jets: OK" << std::endl;
  }

  /**
   * Define parameters for the module
   */
  void L1TScPhase2Jets::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("srcPFCandidates");
    desc.add<edm::InputTag>("srcCLUETaus");
    desc.addUntracked<bool>("debug", false);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2Jets);