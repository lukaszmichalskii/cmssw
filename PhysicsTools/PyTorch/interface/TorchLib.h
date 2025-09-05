#ifndef PhysicsTools_PyTorch_interface_TorchLib_h
#define PhysicsTools_PyTorch_interface_TorchLib_h

// TODO: find a better way to resolve PyTorch and ROOT's `ClassDef` macro clash
#ifdef ClassDef
#undef ClassDef
#endif

#include <torch/script.h>
#include <torch/torch.h>

#endif  // PhysicsTools_PyTorch_interface_TorchLib_h