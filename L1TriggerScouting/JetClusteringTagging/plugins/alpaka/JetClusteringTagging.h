#include <alpaka/alpaka.hpp>
#include <chrono>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"

#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Unpacking.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Clustering.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Tagging.h"


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
  void unpacking(Queue &queue, const SDSRawDataCollection &raw_data);
  void clustering(Queue &queue);
  void tagging(Queue &queue);

  Unpacking unpacking_;
  Clustering clustering_;
  Tagging tagging_;

  PuppiCollection data_;
  edm::EDGetTokenT<SDSRawDataCollection> raw_token_; 
  std::chrono::high_resolution_clock::time_point start_stamp_, end_stamp_;
  int bunch_crossing_ = 0;  
  std::vector<uint32_t> fed_ids_;
  uint32_t clusters_num_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE