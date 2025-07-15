#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  using namespace ::l1sc;

  /**
   * @class L1TScPhase2PFCandidatesAoSToSoA
   * @brief Produces PFCandidateCollection (PortableCollection)
   */
  class L1TScPhase2PFCandidatesAoSToSoA : public stream::EDProducer<> {
  public:
    L1TScPhase2PFCandidatesAoSToSoA(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const edm::EDGetTokenT<std::vector<l1t::PFCandidate>> pf_candidates_aos_token_;
    const edm::EDPutTokenT<PFCandidateHostCollection> pf_candidates_soa_token_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2PFCandidatesAoSToSoA::L1TScPhase2PFCandidatesAoSToSoA(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_aos_token_{consumes(params.getParameter<edm::InputTag>("src"))}, 
        pf_candidates_soa_token_{produces()} {}

  void L1TScPhase2PFCandidatesAoSToSoA::produce(device::Event &event, const device::EventSetup &event_setup) {
    // grab PF candidates
    const auto& pf_candidates_aos = event.get(pf_candidates_aos_token_);

    // allocate buffer to store converted soa
    auto pf_candidates_soa = PFCandidateHostCollection(pf_candidates_aos.size(), event.queue());

    // convert aos to soa
    // for (size_t idx = 0; idx < pf_candidates_aos.size(); ++idx) {
    //   const auto& pf_candidate = pf_candidates_aos[idx];
    //   pf_candidates_soa.view()[i] = {
    //       static_cast<float>(pf_candidate.hwEta()), 
    //       static_cast<float>(pf_candidate.hwPhi()), 
    //       static_cast<float>(pf_candidate.hwPt()),
    //       static_cast<float>(pf_candidate.z0()),
    //       static_cast<float>(pf_candidate.dxy()),
    //       static_cast<int16_t>(pf_candidate.pdgId())};
    // }

    // move converted soa to event storage
    event.emplace(pf_candidates_soa_token_, std::move(pf_candidates_soa));
    // log info
    std::cout << "OK - L1TScPhase2PFCandidatesAoSToSoA" << std::endl;
  }

  /**
   * Define parameters for the module.
   */
  void L1TScPhase2PFCandidatesAoSToSoA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2PFCandidatesAoSToSoA);