// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Tagging.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

Tagging::Tagging(const std::string &model, const std::string &backend) : model_(model), backend_(backend) {
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  std::cout << "Available Execution Providers:" << std::endl;
  for (const auto& provider : providers) {
      std::cout << "- " << provider << std::endl;
  }
      
  // set up session options and backend
  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(1);
  if (backend_ == "cuda_async") { 
      OrtCUDAProviderOptions options;
      options.device_id = 0;
      options.arena_extend_strategy = 0;
      options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      options.do_copy_in_default_stream = 1;
      sess_opts.AppendExecutionProvider_CUDA(options);
  }

  // initialize onnx runtime
  options_ = std::make_unique<Ort::RunOptions>(nullptr);
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TaggingNodeOnnxRt");
  session_ = std::make_unique<Ort::Session>(*env_, model_.c_str(), sess_opts);

  // allocator handle
  Ort::AllocatorWithDefaultOptions allocator;

  // get input names and shapes
  size_t num_input_nodes = session_->GetInputCount();
  input_node_strings_.resize(num_input_nodes);
  input_node_names_.resize(num_input_nodes);
  input_node_dims_.clear();

  std::cout << "Inputs: " << std::endl;
  for (size_t i = 0; i < num_input_nodes; i++) {
      // get input node names
      std::string input_name(session_->GetInputNameAllocated(i, allocator).get());
      std::cout << input_name << std::endl;
      input_node_strings_[i] = input_name;
      input_node_names_[i] = input_node_strings_[i].c_str();

      // get input shapes
      auto type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      input_node_dims_[input_name] = tensor_info.GetShape();
  }

  // get output names and shapes
  size_t num_output_nodes = session_->GetOutputCount();
  output_node_strings_.resize(num_output_nodes);
  output_node_names_.resize(num_output_nodes);
  output_node_dims_.clear();

  std::cout << "Outputs: " << std::endl;
  for (size_t i = 0; i < num_output_nodes; i++) {
      // get output node names
      std::string output_name(session_->GetOutputNameAllocated(i, allocator).get());
      std::cout << output_name << std::endl;
      output_node_strings_[i] = output_name;
      output_node_names_[i] = output_node_strings_[i].c_str();

      // get output node types
      auto type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      output_node_dims_[output_name] = tensor_info.GetShape();

      // the 0th dim depends on the batch size
      output_node_dims_[output_name].at(0) = -1;
  }
}


class DummyInputKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, JetsCollection::View data) const {
    if (once_per_grid(acc)) {
      for (int32_t idx = 0; idx < data.metadata().size(); idx++) {
        data.jet()[idx] = 0.0f;
        data.classification()[idx] = 0.0f;
        data.pt_regression()[idx] = 0.0f;
      }

      printf("Input:\n");
      for (int32_t i = 0; i < 10; i++) {
        printf("%.2f ", data.jet()[i]);
      }
      printf("...\n");
    }
  }
};


void Tagging::Tag(Queue& queue, PuppiCollection const& data, ClustersCollection const& clusters, JetsCollection& jets) {
  // uint32_t batch_size = data.const_view().bx().size();
  uint32_t batch_size = 100;

  // fill data
  uint32_t threads_per_block = 1024;
  uint32_t blocks_per_grid = divide_up_by(jets.const_view().metadata().size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, DummyInputKernel{}, jets.view());
  alpaka::wait(queue);

  // input
  float* jet_data = jets.view().jet();
  Ort::MemoryInfo device_mem(backend_ == "cuda_async" ? "Cuda" : "Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
  std::array<int64_t, 3> input_shape{{batch_size, 16, 20}};
  auto input_tensor = Ort::Value::CreateTensor<float>(
    device_mem, jet_data, batch_size * 16 * 20, input_shape.data(), input_shape.size());
  assert(input_tensor.IsTensor());

  // std::cout << "Input Tensor: ";
  // float* input_data = input_tensor.GetTensorMutableData<float>();
  // for (size_t i = 0; i < 16 * 20; i++) {
  //   input_data[i] = i * 0.01f;
  //   printf("%.2f ", input_data[i]);
  // }
  // std::cout << std::endl;

  // outputs
  Ort::MemoryInfo cpu_mem("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  // jet class
  std::array<int64_t, 2> output_shape_jet{{batch_size, 8}};
  std::vector<float> jet_output_data(batch_size * 8, 0.0f);
  auto output_tensor_jet = Ort::Value::CreateTensor<float>(
    cpu_mem, jet_output_data.data(), batch_size * 8, output_shape_jet.data(), output_shape_jet.size());
  assert(output_tensor_jet.IsTensor());
  // pt regression
  std::array<int64_t, 2> output_shape_pt{{batch_size, 1}};
  std::vector<float> pt_output_data(batch_size * 1, 0.0f);
  auto output_tensor_pt = Ort::Value::CreateTensor<float>(
    cpu_mem, pt_output_data.data(), batch_size * 1, output_shape_pt.data(), output_shape_pt.size());
  assert(output_tensor_pt.IsTensor());

  // bindings
  Ort::IoBinding io_binding{*session_};
  io_binding.BindInput("inputs", input_tensor);
  io_binding.BindOutput("jet_class_output", output_tensor_jet);
  io_binding.BindOutput("pt_output", output_tensor_pt);

  // inference
  auto tm1 = std::chrono::high_resolution_clock::now();
  session_->Run(*options_, io_binding);
  auto tm2 = std::chrono::high_resolution_clock::now();
  auto d = std::chrono::duration_cast<std::chrono::microseconds>(tm2 - tm1);
  std::cout << "Session Run: " << d.count() << " us" << std::endl;

  // debug
  std::cout << "Class probs: ";
  for (size_t i = 0; i < 8; i++)
    printf("%.2f ", jet_output_data[i]);
  printf("\n");
  printf("Pt regression: %.2f\n", pt_output_data[0]);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
