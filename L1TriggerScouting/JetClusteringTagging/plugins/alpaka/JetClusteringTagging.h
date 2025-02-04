#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class JetClusteringTagging : public stream::EDProducer<> {

public:
  // Constructor & destructor
  JetClusteringTagging(const edm::ParameterSet& params);
  ~JetClusteringTagging() override = default;

  // Virtual methods
  void produce(device::Event& event, const device::EventSetup& event_setup) override;
  void beginStream(edm::StreamID stream) override;
  void endStream() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<SDSRawDataCollection> raw_token_; 
  std::chrono::high_resolution_clock::time_point start_stamp_, end_stamp_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE