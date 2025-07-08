#ifndef L1_TRIGGER_SCOUTING__TAU_TAGGING__PLUGINS__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_H
#define L1_TRIGGER_SCOUTING__TAU_TAGGING__PLUGINS__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_H

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/TauTagging/interface/alpaka/L1TScPhase2PFCandidatesRawToDigiKernels.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * @class L1TScPhase2PFCandidatesRawToDigi
   * @brief Produces PFCandidateCollection (PortableCollection)
   */
  class L1TScPhase2PFCandidatesRawToDigi : public stream::EDProducer<> {
    public:
      L1TScPhase2PFCandidatesRawToDigi(const edm::ParameterSet &params);

      void produce(device::Event &event, const device::EventSetup &event_setup) override;
      static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    private:
      const edm::EDGetTokenT<SDSRawDataCollection> raw_data_token_;  // raw data
      const device::EDPutToken<PFCandidateCollection> pf_candidates_token_;  // PF candidates
      const std::vector<uint32_t> links_ids_;  // front-end devices stream links
      std::chrono::high_resolution_clock::time_point t_start_, t_end_;  // timestamps
      std::array<data_t, kOrbitSize> h_data_{};  // headers 64-bit words
      std::vector<data_t> pf_data_{};  // payload 64-bit words

      void collectBuffers(const SDSRawDataCollection &raw_data);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

#endif  // L1_TRIGGER_SCOUTING__TAU_TAGGING__PLUGINS__ALPAKA__L1TSC_PHASE2_PF_CANDIDATES_RAW_TO_DIGI_H