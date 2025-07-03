#include "L1TriggerScouting/TauTagging/plugins/alpaka/L1TScPhase2PFCandidatesRawToDigi.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  L1TScPhase2PFCandidatesRawToDigi::L1TScPhase2PFCandidatesRawToDigi(const edm::ParameterSet &params)
      : EDProducer<>(params),
        raw_data_token_{consumes<SDSRawDataCollection>(params.getParameter<edm::InputTag>("src"))},
        pf_candidates_token_{produces()},
        links_ids_(params.getParameter<std::vector<uint32_t>>("linksIds")) {
    raw_to_digi_kernels_ = std::make_unique<l1sc::L1TScPhase2PFCandidatesRawToDigiKernels>();
  }

  void L1TScPhase2PFCandidatesRawToDigi::produce(
      device::Event &event, 
      const device::EventSetup &event_setup) {
    t_start_ = std::chrono::high_resolution_clock::now();    
    auto raw_data = event.getHandle(raw_data_token_);  
    
    collectBuffers(*raw_data);

    auto pf_candidates = l1sc::PFCandidateCollection(pf_data_.size(), event.queue());  
    // pf_candidates.zeroInitialise(event.queue());
    raw_to_digi_kernels_->RawToDigi(event.queue(), pf_candidates);    
    event.emplace(pf_candidates_token_, std::move(pf_candidates));
    alpaka::wait(event.queue());

    t_end_ = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end_ - t_start_).count();
    std::cout << "OK - L1TScPhase2PFCandidatesRawToDigi [" << elapsed << " us]" << std::endl;    
  }

  void L1TScPhase2PFCandidatesRawToDigi::fillDescriptions(
      edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<uint32_t>>("linksIds");
    desc.add<edm::InputTag>("src");
    descriptions.addWithDefaultLabel(desc);
  }

  void L1TScPhase2PFCandidatesRawToDigi::collectBuffers(
      const SDSRawDataCollection &raw_data) {
    pf_data_.clear();  // reset payload buffer

    size_t h_idx = 0;
    for (auto &link_id : links_ids_) {
      const auto &link = raw_data.FEDData(link_id);
      const auto chunk_begin = reinterpret_cast<const l1sc::data_t*>(link.data());
      const auto chunk_end = reinterpret_cast<const l1sc::data_t*>(link.data() + link.size());

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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(L1TScPhase2PFCandidatesRawToDigi);