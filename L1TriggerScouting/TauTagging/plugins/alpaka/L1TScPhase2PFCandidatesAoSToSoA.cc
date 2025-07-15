#include <iomanip>

#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
    const bool debug_;
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2PFCandidatesAoSToSoA::L1TScPhase2PFCandidatesAoSToSoA(const edm::ParameterSet &params)
      : EDProducer<>(params),
        pf_candidates_aos_token_{consumes(params.getParameter<edm::InputTag>("src"))},
        pf_candidates_soa_token_{produces()},
        debug_(params.getUntrackedParameter<bool>("debug")) {}

  void L1TScPhase2PFCandidatesAoSToSoA::produce(device::Event &event, const device::EventSetup &event_setup) {
    // grab PF candidates
    const auto &pf_candidates_aos = event.get(pf_candidates_aos_token_);

    // allocate buffer to store converted soa
    auto pf_candidates_soa = PFCandidateHostCollection(pf_candidates_aos.size(), event.queue());

    // convert aos to soa
    for (size_t idx = 0; idx < pf_candidates_aos.size(); ++idx) {
      const auto &pf_candidate = pf_candidates_aos[idx];
      pf_candidates_soa.view()[idx] = {static_cast<float>(pf_candidate.hwEta() * 3.14f / 720.0f),
                                       static_cast<float>(pf_candidate.hwPhi() * 3.14f / 720.0f),
                                       static_cast<float>(pf_candidate.hwPt() * 0.25f),
                                       static_cast<float>(pf_candidate.z0()),
                                       static_cast<float>(pf_candidate.dxy()),
                                       static_cast<int16_t>(pf_candidate.id())};  // TODO: map it to real pdgid later
    }

    // debug log to stdout
    if (debug_) {
      const auto soa_size = pf_candidates_soa.const_view().metadata().size();
      fmt::print("[DEBUG] l1sc::L1TScPhase2PFCandidatesAoSToSoA: Converted PFCandidateCollection[{}] (AoS -> SoA):\n",
                 soa_size);

      // table header
      const std::string separator = "+--------------+---------+---------+---------+---------+---------+---------+";
      fmt::print("{}\n", separator);
      fmt::print("| {:>12} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} |\n",
                 "PFCandidate",
                 "pt",
                 "eta",
                 "phi",
                 "z0",
                 "dxy",
                 "pdgid");
      fmt::print("{}\n", separator);

      // log head of collection (10 records at most)
      auto span = (soa_size > 10) ? 10 : soa_size;
      for (int32_t idx = 0; idx < span; ++idx) {
        const auto &view = pf_candidates_soa.const_view()[idx];
        fmt::print("| {:12d} | {:7.2f} | {:7.2f} | {:7.2f} | {:7.2f} | {:7.2f} | {:7d} |\n",
                   idx,
                   view.pt(),
                   view.eta(),
                   view.phi(),
                   view.z0(),
                   view.dxy(),
                   view.pdgid());
      }

      // log tail if collection size is larger than 10
      if (span < soa_size) {
        fmt::print("| {:>12} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} |\n",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...",
                   "...");
      }

      fmt::print("{}\n", separator);
    }

    // move converted soa to event storage
    event.emplace(pf_candidates_soa_token_, std::move(pf_candidates_soa));
    // log info
    std::cout << "[INFO] l1sc::L1TScPhase2PFCandidatesAoSToSoA: OK" << std::endl;
  }

  /**
   * Define parameters for the module.
   */
  void L1TScPhase2PFCandidatesAoSToSoA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.addUntracked<bool>("debug", false);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2PFCandidatesAoSToSoA);