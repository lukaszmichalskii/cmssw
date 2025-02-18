// #include <alpaka/alpaka.hpp>
// #include <chrono>

// #include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// #include "DataFormats/FEDRawData/interface/FEDRawData.h"
// #include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
// #include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"

// #include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Unpacking.h"
// #include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Clustering.h"
// // #include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Tagging.h"

// #include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

// // using namespace cms::Ort;


// namespace ALPAKA_ACCELERATOR_NAMESPACE {

// class JetClusteringTagging : public stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {

// public:
//   // Constructor & destructor
//   JetClusteringTagging(const edm::ParameterSet& params, const cms::Ort::ONNXRuntime *onnx_runtime);
//   ~JetClusteringTagging() override = default;

//   // Virtual methods
//   void produce(device::Event& event, const device::EventSetup& event_setup) override;
//   void beginStream(edm::StreamID stream) override;
//   void endStream() override;
//   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

//   // Inference
//   static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &params);
//   static void globalEndJob(const cms::Ort::ONNXRuntime *onnx_runtime);

// private:
//   void unpacking(Queue &queue, const SDSRawDataCollection &raw_data);
//   void clustering(Queue &queue);
//   void tagging(Queue &queue);

//   Unpacking unpacking_;
//   Clustering clustering_;
//   // Tagging tagging_;

//   PuppiCollection data_;
//   edm::EDGetTokenT<SDSRawDataCollection> raw_token_; 
//   std::chrono::high_resolution_clock::time_point start_stamp_, end_stamp_;
//   int bunch_crossing_ = 0;  
//   std::vector<uint32_t> fed_ids_;
//   uint32_t clusters_num_;

//   std::vector<std::string> input_names_ = {"inputs"};
//   std::vector<std::vector<int64_t>> input_shapes_;
//   cms::Ort::FloatArrays model_data_; // each stream hosts its own data
// };

// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE