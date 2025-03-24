#ifndef PhysicsTools_PyTorchTest_plugins_alpaka_kernels_h
#define PhysicsTools_PyTorchTest_plugins_alpaka_kernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/PyTorchTest/interface/alpaka/torch_alpaka.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class Kernels {
 public:
  void FillSimpleCollection(Queue &queue, SimpleCollection &data, float value);
};

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // PhysicsTools_PyTorchTest_plugins_alpaka_kernels_h