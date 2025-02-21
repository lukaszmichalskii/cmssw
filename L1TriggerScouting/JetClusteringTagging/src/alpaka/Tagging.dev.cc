// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Utils.h"
#include "L1TriggerScouting/JetClusteringTagging/interface/alpaka/Tagging.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

struct CudaMemoryDeleter {
  explicit CudaMemoryDeleter(Ort::Allocator* alloc) {
    alloc_ = alloc;
  }
  void operator()(void* ptr) const {
    alloc_->Free(ptr);
  }

  Ort::Allocator* alloc_;
};

Tagging::Tagging(const std::string &model, const std::string &backend) : model_(model), backend_(backend) {
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  std::cout << "Available Execution Providers:" << std::endl;
  for (const auto& provider : providers) {
      std::cout << "- " << provider << std::endl;
  }
      
  // set up session options and backend
  Ort::SessionOptions sess_opts;
  // sess_opts.SetIntraOpNumThreads(1);
  sess_opts.SetExecutionMode(ORT_SEQUENTIAL);
  if (backend_ == "cuda_async") { 
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
    cuda_options.arena_extend_strategy = 1;
    cuda_options.do_copy_in_default_stream = 1;
    sess_opts.AppendExecutionProvider_CUDA(cuda_options);
  }

  // initialize onnx runtime
  options_ = std::make_unique<Ort::RunOptions>(nullptr);
  // options_->SetProfilingEnabled(true);
  // options_->SetProfilingFile("onnx_profile.json");

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "TaggingNodeOnnxRt");
  // session_ = std::make_unique<Ort::Session>(*env_, model_.c_str(), sess_opts);

  // allocator handle
  // Ort::AllocatorWithDefaultOptions cpu_allocator;
  // device_mem_allocator_info_ = std::make_unique<Ort::MemoryInfo>(
  //   backend_ == "cuda_async" ? "Cuda" : "Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
  // Ort::Allocator device_allocator(*session_, *device_mem_allocator_info_);
  // auto allocator_info = device_allocator.GetInfo();
  // assert(*device_mem_allocator_info_ == allocator_info);

  // // get input names and shapes
  // size_t num_input_nodes = session_->GetInputCount();
  // input_node_strings_.resize(num_input_nodes);
  // input_node_names_.resize(num_input_nodes);
  // input_node_dims_.clear();

  // std::cout << "Input Layers:" << std::endl;
  // for (size_t i = 0; i < num_input_nodes; i++) {
  //   // get input node names
  //   std::string input_name(session_->GetInputNameAllocated(i, cpu_allocator).get());
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

  // std::cout << "Output Layers:" << std::endl;
  // for (size_t i = 0; i < num_output_nodes; i++) {
  //   // get output node names
  //   std::string output_name(session_->GetOutputNameAllocated(i, cpu_allocator).get());
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

      printf("Inputs: ");
      for (int32_t i = 0; i < 10; i++) {
        printf("%.2f ", data.jet()[i]);
      }
      printf("...\n");
    }
  }
};


class DummyOutputKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, JetsCollection::View data, float* classes, float* regression) const {
    if (once_per_grid(acc)) {
      printf("Classes probs: ");
      for (size_t i = 0; i < 8; i++)
        printf("%.2f ", classes[i]);
      printf("\n");
      printf("Regression: %.2f\n", regression[0]);
    }
  }
};


void Tagging::Tag(Queue& queue, PuppiCollection const& data, ClustersCollection const& clusters, JetsCollection& jets) {
  uint32_t batch_size = 1;

  // fill data
  uint32_t threads_per_block = 1024;
  uint32_t blocks_per_grid = divide_up_by(jets.const_view().metadata().size(), threads_per_block);      
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, DummyInputKernel{}, jets.view());
  alpaka::wait(queue);

  // backend_ = "serial";

  Ort::SessionOptions session_options;
  if (backend_ == "cuda_async")
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
  Ort::Session session(*env_, model_.c_str(), session_options);

  Ort::MemoryInfo info_cuda(backend_ == "cuda_async" ? "Cuda" : "Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session, info_cuda);
  auto allocator_info = cuda_allocator.GetInfo();
  assert(info_cuda == allocator_info);

  const std::array<int64_t, 2> x_shape = {{batch_size, 10}};
  std::array<float, 10> x_values = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}};
  auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(
    cuda_allocator.Alloc(x_values.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
  assert(input_data.get() != nullptr);
  if (backend_ == "cuda_async") {
    cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice); 
  } else {
    memcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size());
  }

  Ort::Value bound_x = Ort::Value::CreateTensor(
    info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(), x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {{batch_size, 3}};
  const std::array<float, 3> expected_y = {{1.0f, 2.0f, 3.0f}};
  auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(
    cuda_allocator.Alloc(expected_y.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
  assert(output_data.get() != nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(
    info_cuda, reinterpret_cast<float*>(output_data.get()), expected_y.size(),  expected_y_shape.data(), expected_y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("inputs", bound_x);
  binding.BindOutput("outputs", bound_y);

  auto tm1 = std::chrono::high_resolution_clock::now();
  session.Run(Ort::RunOptions(), binding);
  auto tm2 = std::chrono::high_resolution_clock::now();
  auto d = std::chrono::duration_cast<std::chrono::microseconds>(tm2 - tm1);
  std::cout << "Session Run: " << d.count() << " us" << std::endl;

  std::array<float, 3> y_values_0;
  if (backend_ == "cuda_async") {
    cudaMemcpy(y_values_0.data(), output_data.get(), sizeof(float) * y_values_0.size(), cudaMemcpyDeviceToHost);
  } else {
    memcpy(y_values_0.data(), output_data.get(), sizeof(float) * y_values_0.size());
  }

  std::cout << "y_values_0: ";
  for (uint32_t i = 0; i < y_values_0.size(); ++i) {
    std::cout << y_values_0[i] << " ";
  }
  std::cout << std::endl;

  // input
  // float* input = jets.view().jet();
  // std::array<int64_t, 3> input_shape{{batch_size, 16, 20}};
  // auto input_tensor = Ort::Value::CreateTensor<float>(
  //   *device_mem_allocator_info_, 
  //   input, 
  //   batch_size * 16 * 20, 
  //   input_shape.data(), 
  //   input_shape.size());
  // assert(input_tensor.IsTensor());

  // // outputs
  // float* d_jet_output_data = jets.view().classification();
  // float* d_pt_output_data = jets.view().pt_regression();

  // // jet class
  // std::array<int64_t, 2> output_shape_jet{{batch_size, 8}};
  // auto output_tensor_jet = Ort::Value::CreateTensor<float>(
  //   *device_mem_allocator_info_, 
  //   d_jet_output_data, 
  //   batch_size * 8, 
  //   output_shape_jet.data(), 
  //   output_shape_jet.size());
  // assert(output_tensor_jet.IsTensor());

  // // pt regression
  // std::array<int64_t, 2> output_shape_pt{{batch_size, 1}};
  // auto output_tensor_pt = Ort::Value::CreateTensor<float>(
  //   *device_mem_allocator_info_, 
  //   d_pt_output_data, 
  //   batch_size * 1, 
  //   output_shape_pt.data(), 
  //   output_shape_pt.size());
  // assert(output_tensor_pt.IsTensor());

  // // bindings
  // Ort::IoBinding io_binding{*session_};
  // io_binding.BindInput("inputs", input_tensor);
  // io_binding.BindOutput("jet_class_output", output_tensor_jet);
  // io_binding.BindOutput("pt_output", output_tensor_pt);

  // // inference
  // auto tm1 = std::chrono::high_resolution_clock::now();
  // session_->Run(*options_, io_binding);
  // auto tm2 = std::chrono::high_resolution_clock::now();
  // auto d = std::chrono::duration_cast<std::chrono::microseconds>(tm2 - tm1);
  // std::cout << "Session Run: " << d.count() << " us" << std::endl;

  // debug
  // alpaka::exec<Acc1D>(queue, grid, DummyOutputKernel{}, jets.view(), d_jet_output_data, d_pt_output_data);
  // alpaka::wait(queue);
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
