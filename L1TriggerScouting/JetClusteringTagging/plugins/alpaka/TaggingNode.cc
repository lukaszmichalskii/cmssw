#include "L1TriggerScouting/JetClusteringTagging/plugins/alpaka/TaggingNode.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/JetsCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

TaggingNode::TaggingNode(edm::ParameterSet const& params)
  : device_data_token_(consumes(params.getParameter<edm::InputTag>("data"))),
    device_clusters_token_(consumes(params.getParameter<edm::InputTag>("clusters"))),
    model_(params.getParameter<edm::FileInPath>("model").fullPath()),
    backend_(params.getParameter<std::string>("backend")) {
  
  // set up session options and backend
  // Ort::SessionOptions sess_opts;
  // sess_opts.SetIntraOpNumThreads(1);
  // if (backend_ == "cuda_async") { 
  //   OrtCUDAProviderOptions options;
  //   sess_opts.AppendExecutionProvider_CUDA(options);
  // }

  // // initialize onnx runtime
  // env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TaggingNodeOnnxRt");
  // session_ = std::make_unique<Ort::Session>(env_, model_.c_str(), sess_opts);

  // // allocator handle
  // Ort::AllocatorWithDefaultOptions allocator;

  // // get input names and shapes
  // size_t num_input_nodes = session_->GetInputCount();
  // input_node_strings_.resize(num_input_nodes);
  // input_node_names_.resize(num_input_nodes);
  // input_node_dims_.clear();

  // for (size_t i = 0; i < num_input_nodes; i++) {
  //   // get input node names
  //   std::string input_name(session_->GetInputNameAllocated(i, allocator).get());
  //   input_node_strings_[i] = input_name;
  //   input_node_names_[i] = input_node_strings_[i].c_str();

  //   // get input shapes
  //   auto type_info = session_->GetInputTypeInfo(i);
  //   auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

  //   input_node_dims_[input_name] = tensor_info.GetShape();
  // }

  // // get output names and shapes
  // size_t num_output_nodes = session_->GetOutputCount();
  // output_node_strings_.resize(num_output_nodes);
  // output_node_names_.resize(num_output_nodes);
  // output_node_dims_.clear();

  // for (size_t i = 0; i < num_output_nodes; i++) {
  //   // get output node names
  //   std::string output_name(session_->GetOutputNameAllocated(i, allocator).get());
  //   output_node_strings_[i] = output_name;
  //   output_node_names_[i] = output_node_strings_[i].c_str();

  //   // get output node types
  //   auto type_info = session_->GetOutputTypeInfo(i);
  //   auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  //   output_node_dims_[output_name] = tensor_info.GetShape();

  //   // the 0th dim depends on the batch size
  //   output_node_dims_[output_name].at(0) = -1;
  // }
}

void TaggingNode::produce(device::Event& event, device::EventSetup const& event_setup) {
  auto t1 = std::chrono::high_resolution_clock::now();

  [[maybe_unused]] auto const& data = event.get(device_data_token_);
  [[maybe_unused]] auto const& clusters = event.get(device_clusters_token_);
  auto jets = JetsCollection(10, event.queue());

  // Prepare input tensor from device memory
  // Ort::MemoryInfo memory_info_cuda(backend_ == "cuda_async" ? "CudaPinned" : "Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
  // std::array<int64_t, 2> input_shape{{1, 10}};
  // auto ort_value = Ort::Value::CreateTensor<float>(
  //     memory_info_cuda, jets.view().jet(), 10, input_shape.data(), input_shape.size());
  // std::cout << "Tensor allocated" << std::endl;
  // auto* tensor_data = ort_value.GetTensorMutableData<float>();
  // std::cout << "Tensor values: ";
  // for (int i = 0; i < 10; ++i) {
  //     std::cout << tensor_data[i] << " ";  // Print each element in the tensor
  // }
  // std::cout << std::endl;

  // Get the output data (on device memory)
  // float* output_data = output_tensors[0].GetTensorMutableData<float>();

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Tagging (" << backend_ << "): OK [" << duration.count() << " us]" << std::endl;
  std::cout << "-------------------------------------" << std::endl;
}  

void TaggingNode::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data");
  desc.add<edm::InputTag>("clusters");
  desc.add<edm::FileInPath>("model");
  desc.add<std::string>("backend");
  descriptions.addWithDefaultLabel(desc);
}

void TaggingNode::beginStream(edm::StreamID) {
  std::cout << "=====================================" << std::endl;
  start_stamp_ = std::chrono::high_resolution_clock::now();
}

void TaggingNode::endStream() {
  end_stamp_ = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_stamp_ - start_stamp_);
  std::cout << "JetClusteringTagging (" << duration.count() << " ms)" << std::endl;
  std::cout << "=====================================" << std::endl;
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TaggingNode);
