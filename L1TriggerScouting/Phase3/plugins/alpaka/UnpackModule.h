#ifndef L1TriggerScouting_Phase3_plugins_alpaka_UnpackModule_h
#define L1TriggerScouting_Phase3_plugins_alpaka_UnpackModule_h

#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"

#include "Unpack.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

/**
 * @class UnpackModule
 * @brief Producer of Puppi struct-of-array for CPU, CUDA, ROCm architectures
 * 
 * Takes as input raw data collection.
 * Compose device memory layout for Puppi collection.
 * Transfer data from host to device and launch decoding pipeline kernels.
 * The product stays on device memory and can be automatically transferred to host if needed.
 */
class UnpackModule : public stream::EDProducer<> {

public:
  // Constructor & destructor
  UnpackModule(const edm::ParameterSet& params);
  ~UnpackModule() override = default;

  // Virtual methods
  void produce(device::Event& event, const device::EventSetup& event_setup) override;
  // void beginStream(edm::StreamID) override;
  // void endStream() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Tokens to read/write
  edm::EDGetTokenT<SDSRawDataCollection> raw_token_;  /**< host product */
  device::EDPutToken<PuppiCollection> token_;  /**< device product */

  int bunch_crossing_ = 0;  /**< bunch crossing counter */
  std::vector<unsigned int> fed_ids_;  /**< fed identifiers */

  // Pipeline utility methods
  Unpack utils_;

  // Pipeline methods
  template<typename T> 
  std::tuple<std::vector<T>, std::vector<T>> MemoryScan(const SDSRawDataCollection &raw_data);
  PuppiCollection UnpackCollection(Queue &queue, const SDSRawDataCollection &raw_data);

  // Debugging helpers
  std::chrono::high_resolution_clock::time_point start_, end_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif // L1TriggerScouting_Phase3_plugins_alpaka_UnpackModule_h
