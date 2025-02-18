#include "DecoderNode.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {


DecoderNode::DecoderNode(edm::ParameterSet const& params)
  : host_token_{consumes<DataStream>(params.getParameter<edm::InputTag>("data"))},
    device_token_{produces()},
    front_end_devices_(params.getParameter<std::vector<uint32_t>>("fedIDs")) {}

DecoderNode::~DecoderNode() = default;


void DecoderNode::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto data = event.getHandle(host_token_);
  auto decoded_data = Decode(event.queue(), *data);
  event.emplace(device_token_, std::move(decoded_data));
}


void DecoderNode::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data");
  desc.add<std::vector<uint32_t>>("fedIDs");
  descriptions.addWithDefaultLabel(desc);
}


PuppiCollection DecoderNode::Decode(Queue &queue, const DataStream &data) {
  auto t1 = std::chrono::high_resolution_clock::now();

  std::vector<uint64_t> headers_buffer, buffer;
  // memory scan
  for (uint32_t &fdevice : front_end_devices_) {
    const auto &src = data.FEDData(fdevice);
    const auto chunk_begin = reinterpret_cast<const uint64_t*>(src.data());
    const auto chunk_end = reinterpret_cast<const uint64_t*>(src.data() + src.size());

    for (auto ptr = chunk_begin; ptr != chunk_end;) {
      if (*ptr == 0)
      continue;

      headers_buffer.insert(headers_buffer.end(), ptr, ptr + 1);
      auto chunk_size = (*ptr) & 0xFFF;
      ptr++;
      buffer.insert(buffer.end(), ptr, ptr + chunk_size);
      ptr += chunk_size;
    }
  }

  // decoding
  auto decoded_data = PuppiCollection(buffer.size(), queue);
  decoder_.Decode(queue, headers_buffer, buffer, decoded_data);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Decoder: OK [" << duration.count() << " us]" << std::endl;
  std::cout << "-------------------------------------" << std::endl;

  return decoded_data;
}


} // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(DecoderNode);
