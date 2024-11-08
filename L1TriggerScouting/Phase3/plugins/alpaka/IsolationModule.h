#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"

#include "Isolation.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

/**
 * @class IsolationModule
 * @brief Filter out irrelevant particles
 * 
 * Puppi collection is filtered based on particle type and isolation
 * The output is reduced collection that passes the criteria.
 * Operated on data that is on device memory or transfer automaticaly if needed.
 * The product stays on device memory (also can be automatically transferred on demand).
 */
class IsolationModule : public stream::EDProducer<> {

public:
  // Constructor & destructor
  IsolationModule(const edm::ParameterSet& params);
  ~IsolationModule() override = default;

  // Virtual methods
  void produce(device::Event& event, const device::EventSetup& event_setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Tokens to read/write
  const device::EDGetToken<PuppiCollection> raw_token_;
  device::EDPutToken<PuppiCollection> token_;

  // Pipeline kernels
  Isolation isolation_;

  // Pipeline methods
  PuppiCollection Isolate(Queue &queue, const PuppiCollection &raw_collection);

  // Debugging helpers
  std::chrono::high_resolution_clock::time_point Tick();
  void Summary(const long &duration);
  void LogSeparator();
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
