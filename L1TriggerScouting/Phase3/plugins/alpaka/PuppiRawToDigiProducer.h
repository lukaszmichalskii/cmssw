#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/L1ScoutingPuppi.h"

#include "PuppiUnpack.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class PuppiRawToDigiProducer : public stream::EDProducer<> {

public:
  PuppiRawToDigiProducer(edm::ParameterSet const& config);
  ~PuppiRawToDigiProducer() override = default;

  void produce(device::Event& event, device::EventSetup const& event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::chrono::high_resolution_clock::time_point Tick();
  void Summary(const long &duration);
  void LogSeparator();

  edm::EDGetTokenT<SDSRawDataCollection> raw_token_;
  std::vector<unsigned int> fed_ids_;
  device::EDPutToken<PuppiCollection> token_;
  int bunch_crossing_ = 0;

  // implementation of the algorithm
  PuppiUnpack unpacker_;
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
