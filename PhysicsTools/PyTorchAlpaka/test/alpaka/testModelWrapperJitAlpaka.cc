#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Nvtx.h"
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModelJit.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/GetDevice.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
static c10::cuda::CUDAStream default_stream{c10::cuda::getCurrentCUDAStream()};
#endif

void cacheStream() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  default_stream = c10::cuda::getCurrentCUDAStream();
#endif
}

void restoreStream() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  c10::cuda::setCurrentCUDAStream(default_stream);
#endif
}

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;
  using namespace cms::torch::alpakatools;

  class TestModelWrapperJitAlpaka : public ::torchtest::testTorchBase {
  public:
    std::string script() const override;
    void testCtorFromDevice();
    void testCtorFromQueue();
    void testMoveToDeviceFromAlpakaDevice();
    void testMoveToDeviceFromAlpakaQueue();
    void testAsyncExecution();

  private:
    CPPUNIT_TEST_SUITE(TestModelWrapperJitAlpaka);
    CPPUNIT_TEST(testCtorFromDevice);
    CPPUNIT_TEST(testCtorFromQueue);
    CPPUNIT_TEST(testMoveToDeviceFromAlpakaDevice);
    CPPUNIT_TEST(testMoveToDeviceFromAlpakaQueue);
    CPPUNIT_TEST(testAsyncExecution);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelWrapperJitAlpaka);

  std::string TestModelWrapperJitAlpaka::script() const { return "testExportLinearDnn.py"; }

  void TestModelWrapperJitAlpaka::testCtorFromDevice() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModelJit(m_path, dev);

    CPPUNIT_ASSERT_EQUAL(getDevice(dev), m.device());
  }

  void TestModelWrapperJitAlpaka::testCtorFromQueue() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];
    Queue queue{dev};

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModelJit(m_path, queue);

    CPPUNIT_ASSERT_EQUAL(getDevice(queue), m.device());
  }

  void TestModelWrapperJitAlpaka::testMoveToDeviceFromAlpakaDevice() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModelJit(m_path);
    m.to(dev);

    CPPUNIT_ASSERT_EQUAL(getDevice(dev), m.device());
  }

  void TestModelWrapperJitAlpaka::testMoveToDeviceFromAlpakaQueue() {
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];
    Queue queue{dev};

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModelJit(m_path);
    m.to(queue);

    CPPUNIT_ASSERT_EQUAL(getDevice(queue), m.device());
  }

  void TestModelWrapperJitAlpaka::testAsyncExecution() {
    Nvtx range("testAsyncExecutionModel");

    // setup alpaka queue
    const auto& devices = cms::alpakatools::devices<Platform>();
    CPPUNIT_ASSERT(!devices.empty());
    const auto& dev = devices[0];
    Queue queue{dev};

    // load model
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = AlpakaModelJit(m_path);

    // prepare input buffers
    Nvtx inbuf("inputBuffers");
    auto inputs = std::vector<::torch::IValue>();
    inputs.push_back(::torch::randn({batch_size_, 3}, getDevice(queue)));
    inbuf.end();

    cacheStream();

    // guard scope, restores when goes out of scope
    // all operations should be scheduled on provided queue.
    {
      QueueGuard<Queue> guard(queue);
      // async model load and inference check
      Nvtx exec1("execInExternalStream");
      Nvtx mmove("modelMoveToDevice");
      m.to(queue);
      mmove.end();

      for (uint32_t i = 0; i < 10; ++i) {
        Nvtx iter(("forwardPass:" + std::to_string(i)).c_str());
        auto out = m.forward(inputs);
        iter.end();
      }
      exec1.end();
    }

    // TODO: guard does not restore previous state
    // consider adding reset with caching of previous stream
    restoreStream();

    // operations should be restored and scheduled on default stream
    Nvtx exec2("execInDefaultStream");
    for (uint32_t i = 0; i < 10; ++i) {
      Nvtx iter(("forwardPass:" + std::to_string(i)).c_str());
      auto out = m.forward(inputs);
      iter.end();
    }
    exec2.end();

    range.end();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest