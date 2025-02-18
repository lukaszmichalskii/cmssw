#ifndef L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Decoder_h
#define L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Decoder_h

// libs
#include <cstdint>
#include <alpaka/alpaka.hpp>
// fw core
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// heterogeneous
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// typedefs
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
// decoding
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Decoder.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using DataStream = SDSRawDataCollection; /**< alias */

/**
* Takes as input raw data collection.
* Heterogenous decoding of scouting raw data stream.
* Implicit device memory allocation and transfers with reduced copy operations.
* The product stays on device memory and can be automatically transferred to host if needed.
*
* @brief Raw data stream decoding plugin.
*/
class DecoderNode : public stream::EDProducer<> {

public:
  DecoderNode(const edm::ParameterSet& params);
  ~DecoderNode() override;

  /**
  * @brief cmssw callback for node
  */
  void produce(device::Event& event, const device::EventSetup& event_setup) override;

  /**
  * @brief Declare parameters for node
  */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /**
  * @brief Decode raw data stream @see <<DataStream>>
  */
  PuppiCollection Decode(Queue &queue, const DataStream &data);

  // tokens
  edm::EDGetTokenT<DataStream> host_token_; /**< host data */
  device::EDPutToken<PuppiCollection> device_token_; /**< device data */

  std::vector<uint32_t> front_end_devices_{}; /**< fed identifiers */
  Decoder decoder_{}; /**< bytes stream decoder */

  // stats
  std::chrono::high_resolution_clock::time_point start_, end_; /**< timestamps */
};

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_JetClusteringTagging_plugins_alpaka_Decoder_h
