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
      const device::EDPutToken<OrbitEventIndexMapCollection> orbit_association_map_token_;  // orbit association map
      const std::vector<uint32_t> links_ids_;  // front-end devices stream links
      std::chrono::steady_clock::time_point t_start_, t_end_;  // timestamps
      std::array<data_t, kOrbitSize> h_data_{};  // headers 64-bit words
      std::vector<data_t> pf_data_{};  // payload 64-bit words
      std::unique_ptr<kernels::L1TScPhase2RawToDigiKernels> raw_to_digi_kernels_;  // kernels for decoding

      void collectBuffers(const SDSRawDataCollection &raw_data);
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2PFCandidatesRawToDigi::L1TScPhase2PFCandidatesRawToDigi(const edm::ParameterSet &params)
      : EDProducer<>(params),
        raw_data_token_{consumes(params.getParameter<edm::InputTag>("src"))},
        pf_candidates_token_{produces()},
        orbit_association_map_token_{produces()},
        links_ids_(params.getParameter<std::vector<uint32_t>>("linksIds")),
        raw_to_digi_kernels_(std::make_unique<kernels::L1TScPhase2RawToDigiKernels>()) {}

  void L1TScPhase2PFCandidatesRawToDigi::produce(
      device::Event &event, 
      const device::EventSetup &event_setup) {
    // intialize device constant memory -> called only once
    raw_to_digi_kernels_->initialize(event.queue()); 

    // timestamp
    t_start_ = std::chrono::steady_clock::now();    

    // get raw data input
    auto raw_data = event.getHandle(raw_data_token_);  
    
    // preprocess header -> payload
    collectBuffers(*raw_data);

    // orbit event index association map
    auto map_size = links_ids_.size() * kOrbitSize + 1;
    auto orbit_association_map = OrbitEventIndexMapCollection(map_size, event.queue());
    kernels::associateOrbitEventIndex(event.queue(), h_data_.data(), orbit_association_map);

    // pf candidates data
    auto pf_candidates = PFCandidateCollection(pf_data_.size(), event.queue());  
    kernels::rawToDigi(event.queue(), pf_data_.data(), pf_candidates);
    kernels::printPFCandidateCollection(event.queue(), pf_candidates);

    // store data in the event
    event.emplace(orbit_association_map_token_, std::move(orbit_association_map));
    event.emplace(pf_candidates_token_, std::move(pf_candidates));

    // explicit device sync (only for time measurements)
    alpaka::wait(event.queue());

    // timestamp
    t_end_ = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end_ - t_start_).count();

    // log info
    std::cout << "OK - L1TScPhase2PFCandidatesRawToDigi [" << elapsed << " us]" << std::endl;    
  }

  /**
   * Define parameters for the module.
   * 
   */
  void L1TScPhase2PFCandidatesRawToDigi::fillDescriptions(
      edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<uint32_t>>("linksIds");
    desc.add<edm::InputTag>("src");
    descriptions.addWithDefaultLabel(desc);
  }

  /**
   * 
   */
  void L1TScPhase2PFCandidatesRawToDigi::collectBuffers(
      const SDSRawDataCollection &raw_data) {
    pf_data_.clear();  // reset payload buffer

    size_t h_idx = 0;
    for (auto &link_id : links_ids_) {
      const auto &link = raw_data.FEDData(link_id);
      const auto chunk_begin = reinterpret_cast<const data_t*>(link.data());
      const auto chunk_end = reinterpret_cast<const data_t*>(link.data() + link.size());

      for (auto ptr = chunk_begin; ptr < chunk_end;) {
        // skip empty words
        if (*ptr == 0) {  
          ++ptr;
          continue;
        }

        h_data_[h_idx++] = *ptr;  // store header
        auto chunk_size = (*ptr) & 0xFFF;  // unpack chunk size
        ++ptr; // move to the next word

        const size_t payload = chunk_end - ptr;  // calculate payload size
        const size_t copy_count = std::min<size_t>(chunk_size, payload);  // block size

        // skip if no trailing payload
        if (copy_count == 0)
          continue;

        pf_data_.insert(pf_data_.end(), ptr, ptr + copy_count);  // copy payload
        ptr += copy_count;  // move to the next word
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2PFCandidatesRawToDigi);