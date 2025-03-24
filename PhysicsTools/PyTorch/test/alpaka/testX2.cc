#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include <sys/prctl.h>
#include "../testBase.h"
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "PhysicsTools/PyTorch/interface/common.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using namespace cms::torch_alpaka_tools;
using namespace cms::torch_alpaka_common;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Input SOA
GENERATE_SOA_LAYOUT(SoAPositionTemplate, SOA_COLUMN(float, x))

using SoAPosition = SoAPositionTemplate<>;
using SoAPositionView = SoAPosition::View;
using SoAPositionConstView = SoAPosition::ConstView;

// Output SOA
GENERATE_SOA_LAYOUT(SoAResultTemplate, SOA_COLUMN(float, x))

using SoAResult = SoAResultTemplate<>;
using SoAResultView = SoAResult::View;

class testSOAToTorch : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testSOAToTorch);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSOAToTorch);

std::string testSOAToTorch::pyScript() const { return "create_x2_model.py"; }

// Build Tensor, run model and fill output pointer with result
template <typename SOA_Input, typename SOA_Output>
void run(torch::Device device,
         torch::jit::script::Module model,
         ModelMetadata metadata,
         std::byte* input,
         std::byte* output) {
  std::vector<torch::jit::IValue> input_tensor = Converter<SOA_Input>::convert_input(metadata, device, input);

  Converter<SOA_Output>::convert_output(metadata, device, output) = model.forward(input_tensor).toTensor();
}

void testSOAToTorch::test() {
  std::cout << "ALPAKA Platform info:" << std::endl;
  int idx = 0;
  try {
    for (;;) {
      alpaka::Platform<alpaka::DevCpu> platformHost;
      alpaka::DevCpu host = alpaka::getDevByIdx(platformHost, idx);
      std::cout << "Host[" << idx++ << "]:   " << alpaka::getName(host) << std::endl;
    }
  } catch (...) {
  }
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  for (const auto& d : alpakaDevices) {
    std::cout << "Device[" << idx++ << "]:   " << alpaka::getName(d) << std::endl;
  }
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  std::cout << "Will create torch device with type=" << kDeviceType
            << " and native handle=" << alpakaDevice.getNativeHandle() << std::endl;
  torch::Device torchDevice(kDeviceType, alpakaDevice.getNativeHandle());

  // Number of elements
  const std::size_t batch_size = 10;

  std::vector<float> input{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
//   float result_check[4][2] = {{2.3, -0.5}, {6.6, 3.0}, {2.5, -4.9}, {4.4, 1.3}};

  // Create and fill needed portable collections
  PortableHostCollection<SoAPosition> positionHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAPosition, Device> positionCollection(batch_size, alpakaDevice);
  SoAPositionView& positionCollectionView = positionHostCollection.view();

  for (size_t i = 0; i < batch_size; i++) {
    positionCollectionView.x()[i] = input[i];
  }
  alpaka::memcpy(queue, positionCollection.buffer(), positionHostCollection.buffer());

  PortableHostCollection<SoAResult> resultHostCollection(batch_size, cms::alpakatools::host());
  PortableCollection<SoAResult, Device> resultCollection(batch_size, alpakaDevice);

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string model_path = dataPath_ + "/model_x2.pt";
    model = torch::jit::load(model_path);
    model.to(torchDevice);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }

  // Call function to build tensor and run model
  InputMetadata inputMask(torch::kFloat, 1);
  ModelMetadata mask(batch_size, inputMask, OutputMetadata(torch::kFloat, 1));

  run<SoAPosition, SoAResult>(
      torchDevice, model, mask, positionCollection.buffer().data(), resultCollection.buffer().data());

  // Compare if values are the same as for python script
  alpaka::memcpy(queue, resultHostCollection.buffer(), resultCollection.buffer());
  alpaka::wait(queue);

  SoAResultView& resultView = resultHostCollection.view();

  std::cout << "Output Matrix:" << std::endl;
  for (size_t i = 0; i < batch_size; i++) {
    std::cout << resultView.x()[i] << std::endl;
    // CPPUNIT_ASSERT(std::abs(resultView.x()[i] - result_check[i][0]) <= 1.0e-05);
  }
}
