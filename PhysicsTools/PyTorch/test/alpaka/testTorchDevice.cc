#include "PhysicsTools/PyTorch/interface/common.h"
#include "PhysicsTools/PyTorch/interface/model.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

void assert_device(const torch::Device &m_dev, const torch::Device &dev) {
  std::cout << "device: " << 
   dev.type() << ":" << static_cast<int>(dev.index()) << "=" << 
   m_dev.type() << ":" << static_cast<int>(m_dev.index()) << std::endl;
  assert(m_dev == dev);
}

int main() {
  auto devices = cms::alpakatools::devices<Platform>();
  for (auto dev : devices) {
    auto model = cms::torch_alpaka::Model("src/PhysicsTools/PyTorch/test/alpaka/model_x2.pt");
    assert_device(model.device(), cms::torch_alpaka_tools::device(dev));
  }
  return 0;
}