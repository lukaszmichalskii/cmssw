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
  sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  if (backend_ == "cuda_async") { 
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
    cuda_options.arena_extend_strategy = 1;
    cuda_options.do_copy_in_default_stream = 1;
    sess_opts.AppendExecutionProvider_CUDA(cuda_options);
  }

  // initialize onnx runtime
  options_ = std::make_unique<Ort::RunOptions>(nullptr);
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "TaggingNodeOnnxRt");
  session_ = std::make_unique<Ort::Session>(*env_, model_.c_str(), sess_opts);

  // allocator handle
  Ort::AllocatorWithDefaultOptions cpu_allocator;
  device_mem_allocator_info_ = std::make_unique<Ort::MemoryInfo>(
    backend_ == "cuda_async" ? "Cuda" : "Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator device_allocator(*session_, *device_mem_allocator_info_);
  auto allocator_info = device_allocator.GetInfo();
  assert(*device_mem_allocator_info_ == allocator_info);

  // get input names and shapes
  size_t num_input_nodes = session_->GetInputCount();
  input_node_strings_.resize(num_input_nodes);
  input_node_names_.resize(num_input_nodes);
  input_node_dims_.clear();

  std::cout << "Input Layers:" << std::endl;
  for (size_t i = 0; i < num_input_nodes; i++) {
    // get input node names
    std::string input_name(session_->GetInputNameAllocated(i, cpu_allocator).get());
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

  std::cout << "Output Layers:" << std::endl;
  for (size_t i = 0; i < num_output_nodes; i++) {
    // get output node names
    std::string output_name(session_->GetOutputNameAllocated(i, cpu_allocator).get());
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

class ZeroFillKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, JetsCollection::View data) const {
    for (uint32_t thread_idx : uniform_elements(acc, data.metadata().size())) {
      data.jet()[thread_idx] = 0.0f;
      data.classification()[thread_idx] = 0.0f;
      data.pt_regression()[thread_idx] = 0.0f;
    }
  }
};

class PreprocessKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, PuppiCollection::ConstView raw_data, 
      ClustersCollection::ConstView clusters, JetsCollection::View data) const {

    // const uint8_t SHARED_MEM_BLOCK = 128;
    // auto& jet_pt = alpaka::declareSharedVar<float[12], __COUNTER__>(acc);
    // auto& jet_eta = alpaka::declareSharedVar<float[12], __COUNTER__>(acc);
    // auto& jet_phi = alpaka::declareSharedVar<float[12], __COUNTER__>(acc);

    // uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
    // for (uint32_t block_idx: independent_groups(acc, grid_dim)) {
    //   // bind range to hw block
    //   uint32_t begin = raw_data.offsets()[block_idx];
    //   uint32_t end = raw_data.offsets()[block_idx + 1];
    //   if (end == 0xFFFFFFFF)
    //     continue;
    //   if (end - begin == 0)
    //     continue;

    //   // define block dimensions
    //   uint32_t block_dim = end - begin;
    //   // fill shared mem
    //   for (uint32_t tid : independent_group_elements(acc, block_dim)) {
        
    //   }
    // }
  }
};


class DummyInputKernel {
public:
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, JetsCollection::View data) const {
    if (once_per_grid(acc)) {
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
  uint32_t batch_size = 100;

  // fill data
  uint32_t threads_per_block = ThreadsPerBlockUpperBound(128);
  uint32_t blocks_per_grid = data.const_view().bx().size();
  auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
  alpaka::exec<Acc1D>(queue, grid, ZeroFillKernel{}, jets.view());
  alpaka::exec<Acc1D>(queue, grid, PreprocessKernel{}, data.const_view(), clusters.const_view(), jets.view());
  alpaka::exec<Acc1D>(queue, grid, DummyInputKernel{}, jets.view());
  alpaka::wait(queue);

  // std::array<int64_t, 3> in_shape{{batch_size, 16, 20}};
  // void *input_device;
  // cudaMalloc(&input_device, batch_size * 16 * 20 * sizeof(float));
  // auto input_tensor = Ort::Value::CreateTensor(
  //     *device_mem_allocator_info_, input_device, batch_size * 16 * 20 * sizeof(float),
  //     in_shape.data(), in_shape.size(),
  //     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  // std::array<int64_t, 2> out_pt_shape{{batch_size, 1}};
  // void *pt_device;
  // cudaMalloc(&pt_device, batch_size * 1 * sizeof(float));
  // auto pt_tensor = Ort::Value::CreateTensor(
  //     *device_mem_allocator_info_, pt_device, batch_size * 1 * sizeof(float),
  //     out_pt_shape.data(), out_pt_shape.size(),
  //     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  // std::array<int64_t, 2> out_class_shape{{batch_size, 8}};
  // void *class_device;
  // cudaMalloc(&class_device, batch_size * 8 * sizeof(float));
  // auto class_tensor = Ort::Value::CreateTensor(
  //     *device_mem_allocator_info_, class_device, batch_size * 8 * sizeof(float),
  //     out_class_shape.data(), out_class_shape.size(),
  //     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  // std::vector<Ort::Value> inputs;
  // inputs.push_back(std::move(input_tensor));
  
  // std::vector<Ort::Value> outputs;
  // outputs.push_back(std::move(class_tensor));
  // outputs.push_back(std::move(pt_tensor));


  std::array<int64_t, 2> in_shape{{batch_size, 10}};
  // cuda
  // void *input_device;
  // cudaMalloc(&input_device, batch_size * 10 * sizeof(float));
  // cpu 
  // auto input_device = malloc(batch_size * 10 * sizeof(float));
  auto* input_device = jets.view().jet();
  auto input_tensor = Ort::Value::CreateTensor(
      *device_mem_allocator_info_, input_device, batch_size * 10 * sizeof(float),
      in_shape.data(), in_shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  std::array<int64_t, 2> out_shape{{batch_size, 3}};
  // cuda
  // void *out_device;
  // cudaMalloc(&out_device, batch_size * 3 * sizeof(float));
  // cpu
  // auto out_device = malloc(batch_size * 3 * sizeof(float));
  auto* out_device = jets.view().classification();
  auto output_tensor = Ort::Value::CreateTensor(
      *device_mem_allocator_info_, out_device, batch_size * 3 * sizeof(float),
      out_shape.data(), out_shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  // usual run
  // std::vector<Ort::Value> inputs;
  // inputs.push_back(std::move(input_tensor));
  // std::vector<Ort::Value> outputs;
  // outputs.push_back(std::move(output_tensor));
  // session_->Run(
  //   *options_, input_node_names_.data(), inputs.data(), inputs.size(), output_node_names_.data(), outputs.data(), outputs.size());

  // io binding
  Ort::IoBinding io_bindings{*session_};
  io_bindings.BindInput("inputs", input_tensor);
  io_bindings.BindOutput("outputs", output_tensor);
  session_->Run(*options_, io_bindings);

  
  // // input
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
