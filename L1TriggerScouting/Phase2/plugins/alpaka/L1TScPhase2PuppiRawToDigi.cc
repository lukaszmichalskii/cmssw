#include <queue>
#include <vector>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/DeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/DeviceObject.h"
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
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/Phase2/interface/alpaka/L1TScPhase2PuppiRawToDigiKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  struct BxData {
    unsigned int bx;
    const data_t* payload_begin;
    size_t payload_size;
    const data_t* header_ptr;
  };

  struct BxDataComparator {
    bool operator()(const BxData &t, const BxData &u) const {
      return t.bx > u.bx;  // min-heap by bx
    }
  };

  using MinHeap = std::priority_queue<BxData, std::vector<BxData>, BxDataComparator>;

  using namespace ::l1sc;

  /**
   * @class L1TScPhase2PuppiRawToDigi
   * @brief Produces PuppiDeviceCollection (PortableCollection)
   */
  class L1TScPhase2PuppiRawToDigi : public stream::EDProducer<> {
  public:
    L1TScPhase2PuppiRawToDigi(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const edm::EDGetTokenT<SDSRawDataCollection> raw_data_token_;                     // raw data
    const device::EDPutToken<PuppiDeviceCollection> puppi_token_;                     // PUPPI candidates
    const device::EDPutToken<NbxMapDeviceCollection> nbx_map_token_;                  // orbit association map
    const device::EDPutToken<BxCounterDeviceObject> bx_ct_token_;
    const std::vector<uint32_t> links_ids_;                                           // front-end devices stream links
    std::vector<data_t> h_data_{};                                                    // headers 64-bit words
    std::vector<data_t> p_data_{};                                                    // payload 64-bit words
    std::unique_ptr<kernels::L1TScPhase2PuppiRawToDigiKernels> raw_to_digi_kernels_;  // kernels for decoding
    const bool verbose_;                                                              // verbose output

    void collectBuffers(const SDSRawDataCollection &raw_data);
  };

  // __________________________________________________________________________________________________________________
  // IMPLEMENTATION

  L1TScPhase2PuppiRawToDigi::L1TScPhase2PuppiRawToDigi(const edm::ParameterSet &params)
      : EDProducer<>(params),
        raw_data_token_{consumes(params.getParameter<edm::InputTag>("src"))},
        puppi_token_{produces()},
        nbx_map_token_{produces()},
        bx_ct_token_{produces()},
        links_ids_(params.getParameter<std::vector<uint32_t>>("linksIds")),
        raw_to_digi_kernels_(std::make_unique<kernels::L1TScPhase2PuppiRawToDigiKernels>()),
        verbose_(params.getUntrackedParameter<bool>("verbose")) {}

  void L1TScPhase2PuppiRawToDigi::produce(device::Event &event, const device::EventSetup &event_setup) {
    auto timestamp = std::chrono::steady_clock::now();
    // intialize device constant memory -> called only once
    raw_to_digi_kernels_->initialize(event.queue());

    // get raw data input
    auto raw_data = event.getHandle(raw_data_token_);

    // preprocess header -> payload
    collectBuffers(*raw_data);
    auto nbx = static_cast<int32_t>(h_data_.size());

    // nbx index map
    std::array<int32_t, 2> const sizes{{nbx, nbx + 1}};
    auto nbx_map = NbxMapDeviceCollection(sizes, event.queue());
    nbx_map.zeroInitialise(event.queue());
    kernels::associateNbxEventIndex(event.queue(), h_data_.data(), nbx_map);

    // pf candidates data
    auto puppi = PuppiDeviceCollection(p_data_.size(), event.queue());
    kernels::rawToDigi(event.queue(), p_data_.data(), puppi);

    // debug log to stdout
    // if (verbose_) {
    //   fmt::print("[DEBUG] l1sc::L1TScPhase2PuppiRawToDigi: OK (event: {})\n", event.id().event());
    // }

    auto bx_ct_host = BxCounterHostObject(event.queue(), nbx);
    auto bx_ct = BxCounterDeviceObject(event.queue());
    alpaka::memcpy(event.queue(), bx_ct.buffer(), bx_ct_host.buffer());

    // store data in the event
    event.emplace(nbx_map_token_, std::move(nbx_map));
    event.emplace(puppi_token_, std::move(puppi));
    event.emplace(bx_ct_token_, std::move(bx_ct));

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - timestamp);
    // if (verbose_) {
    fmt::print("[DEBUG] l1sc::L1TScPhase2PuppiRawToDigi: OK {} us\n", elapsed.count());
    // }
  }

  void L1TScPhase2PuppiRawToDigi::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<uint32_t>>("linksIds");
    desc.add<edm::InputTag>("src");
    desc.addUntracked<bool>("verbose", false);
    descriptions.addWithDefaultLabel(desc);
  }

  void L1TScPhase2PuppiRawToDigi::collectBuffers(const SDSRawDataCollection &raw_data) {
    p_data_.clear();
    h_data_.clear();

    MinHeap min_heap;
    // readout data from links breadth-first (order not guaranteed)
    for (size_t link_idx = 0; link_idx < links_ids_.size(); ++link_idx) {
        auto link_id = links_ids_[link_idx];
        const auto &link = raw_data.FEDData(link_id);
        const auto chunk_begin = reinterpret_cast<const data_t *>(link.data());
        const auto chunk_end = reinterpret_cast<const data_t *>(link.data() + link.size());

        for (auto ptr = chunk_begin; ptr < chunk_end;) {
            if (*ptr == 0) { ++ptr; continue; }  // skip empty words
 
            unsigned int bx = ((*ptr) >> 12) & 0xFFF;  // unpack bx number
            auto chunk_size = (*ptr) & 0xFFF;          // unpack chunk size
            ++ptr;                                     // move past header

            const size_t payload = chunk_end - ptr;                           // calculate payload size
            const size_t copy_count = std::min<size_t>(chunk_size, payload);  // block size
            if (copy_count == 0) continue;                                    // skip if no trailing payload

            min_heap.push(BxData{
                .bx = bx,
                .payload_begin = ptr,
                .payload_size = copy_count,
                .header_ptr = ptr - 1,
            });

            ptr += copy_count;  // move to the next word
        }
    }

    // restore bx order (size of heap at max 3564 not that heavy to track)
    while (!min_heap.empty()) {
        const auto &bx_data = min_heap.top();
        h_data_.push_back(*(bx_data.header_ptr));                                                            // store header
        p_data_.insert(p_data_.end(), bx_data.payload_begin, bx_data.payload_begin + bx_data.payload_size);  // copy payload
        min_heap.pop();
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

DEFINE_FWK_ALPAKA_MODULE(l1sc::L1TScPhase2PuppiRawToDigi);