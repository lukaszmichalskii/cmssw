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
#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2JetConcatenation.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * @class L1TScPhase2DirectInference
   * @brief Produces Jet representation for clustered PF candidates used in ML inference
   */
  class L1TScPhase2DirectInference : public stream::EDProducer<> {
  public:
    explicit L1TScPhase2DirectInference(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<PFCandidateCollection> pf_candidates_token_;
    const device::EDGetToken<CLUEsteringCollection> cluestering_token_;
    const bool verbose_;
    const int verbose_level_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2DirectInference::L1TScPhase2DirectInference(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_token_{consumes(params.getParameter<edm::InputTag>("srcPFCandidates"))},
        cluestering_token_{consumes(params.getParameter<edm::InputTag>("srcCLUETaus"))},
        verbose_(params.getUntrackedParameter<bool>("verbose")),
        verbose_level_(params.getUntrackedParameter<int>("verboseLevel")) {}

  /**
   * Execute the logic of the module.
   */
  void L1TScPhase2DirectInference::produce(device::Event &event, const device::EventSetup &event_setup) {
    // get collections
    const auto &pf_candidates = event.get(pf_candidates_token_);
    const auto &clue_collection = event.get(cluestering_token_);

    kernels::concatenate(event.queue(), pf_candidates, clue_collection);

    if (verbose_) {
    }

    // log info
    std::cout << "[INFO] l1sc::L1TScPhase2DirectInference: OK" << std::endl;
  }

  /**
   * Define parameters for the module
   */
  void L1TScPhase2DirectInference::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("srcPFCandidates");
    desc.add<edm::InputTag>("srcCLUETaus");
    desc.addUntracked<bool>("verbose", false);
    desc.addUntracked<int>("verboseLevel", 0);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2DirectInference);