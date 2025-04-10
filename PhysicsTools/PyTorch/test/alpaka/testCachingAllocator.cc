#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/testBase.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace torch_alpaka;

class testCachingAllocator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCachingAllocator);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

 public:
  void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testCachingAllocator);


void testCachingAllocator::test() {
  // alpaka setup
  Platform platform;
  std::vector<Device> alpakaDevices = alpaka::getDevs(platform);
  const auto& alpakaHost = alpaka::getDevByIdx(alpaka_common::PlatformHost(), 0u);
  CPPUNIT_ASSERT(alpakaDevices.size());
  const auto& alpakaDevice = alpakaDevices[0];
  Queue queue{alpakaDevice};

  auto wrapper = std::make_shared<TorchAllocatorWrapper>(queue);
  auto alloc_fn = [wrapper](size_t size, int device, cudaStream_t stream) -> void* {
    return wrapper->allocate(size, device, stream);
  };
  auto free_fn = [wrapper](void* ptr, size_t size, int device, cudaStream_t stream) {
    wrapper->deallocate(ptr, size, device, stream);
  };
  auto custom_allocator = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(alloc_fn, free_fn);
  torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(custom_allocator);
}
