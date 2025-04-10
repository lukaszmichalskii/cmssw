#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_

#include <alpaka/alpaka.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"
#include "HeterogeneousCore/AlpakaInterface/interface/getDeviceCachingAllocator.h"


namespace torch_alpaka {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CUDA;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#else
#error "Could not define the torch device type."
#endif

namespace tools {

inline torch::Device device(const ALPAKA_ACCELERATOR_NAMESPACE::Device &dev) {
  return torch::Device(kTorchDeviceType, dev.getNativeHandle());
}

inline torch::Device device(const ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
  return torch::Device(kTorchDeviceType, alpaka::getDev(queue).getNativeHandle());
}

inline torch::jit::script::Module load(const std::string &model_path) {
  try {
    return torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    throw std::runtime_error("Error loading the model: " + std::string(e.what()));
  }
}

inline std::string queue_hash(const ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
  std::stringstream repr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  auto stream = c10::cuda::getStreamFromExternal(
  queue.getNativeHandle(), device(queue).index());
  repr << "0x" << std::hex << stream.stream();
  return repr.str();
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  return "0x0";
#endif
  repr << "0x" << std::hex << std::hash<std::thread::id>{}(std::this_thread::get_id());
  return repr.str();
}

inline std::string current_stream_hash(const ALPAKA_ACCELERATOR_NAMESPACE::Queue &queue) {
  std::stringstream repr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  const auto dev = tools::device(queue);
  auto stream = c10::cuda::getCurrentCUDAStream(dev.index());
  repr << "0x" << std::hex << stream.stream();
  return repr.str();
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  return "0x0";
#endif
  repr << "0x" << std::hex << std::hash<std::thread::id>{}(std::this_thread::get_id());
  return repr.str();
}
     
}  // namespace torch_alpaka::tools

constexpr auto Byte = torch::kByte;
constexpr auto Char = torch::kChar;
constexpr auto Short = torch::kShort;
constexpr auto Int = torch::kInt;
constexpr auto Long = torch::kLong;
constexpr auto UInt16 = torch::kUInt16;
constexpr auto UInt32 = torch::kUInt32;
constexpr auto UInt64 = torch::kUInt64;
constexpr auto Half = torch::kHalf;
constexpr auto Float = torch::kFloat;
constexpr auto Double = torch::kDouble;

template <typename TQueue>
inline void set_guard(const TQueue &queue);


// template <typename TDev, typename TQueue>
// class TorchAllocatorWrapper{
//  public:
//   TorchAllocatorWrapper(TQueue queue) {
//     queue_ = queue;
//     allocator_ = cms::alpakatools::getDeviceCachingAllocator<TDev, TQueue>(alpaka::getDev(queue));
//   }

//   void* allocate(size_t size, int device_id, cudaStream_t) {
//     return allocator_->allocate(size, queue_);
//   }

//   void deallocate(void *ptr, size_t size, int device_id, cudaStream_t) {
//     allocator_->free(ptr);
//   }
  
//  private:
//   cms::alpakatools::CachingAllocator<TDev, TQueue> &allocator_;
//   TQueue queue_;
// };

  

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

template <>
inline void set_guard(const alpaka_cuda_async::Queue &queue) {
  const auto dev = tools::device(queue);
  auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), dev.index());
  c10::cuda::setCurrentCUDAStream(stream);
  cudaError_t err = cudaSetDevice(dev.index());
  if (err != cudaSuccess) {
    std::cerr << "CUDA set device failed: " << cudaGetErrorString(err) << std::endl;
  }
}

class TorchAllocatorWrapper{
 public:
  TorchAllocatorWrapper(alpaka_cuda_async::Queue queue) 
      : queue_(queue), 
        allocator_(cms::alpakatools::getDeviceCachingAllocator<
          alpaka_cuda_async::Device, alpaka_cuda_async::Queue>(alpaka::getDev(queue))) {}

  void* allocate(size_t size, int device_id, cudaStream_t) {
    return allocator_.allocate(size, queue_);
  }

  void deallocate(void *ptr, size_t size, int device_id, cudaStream_t) {
    allocator_.free(ptr);
  }
  
 private:
  alpaka_cuda_async::Queue queue_;
  cms::alpakatools::CachingAllocator<alpaka_cuda_async::Device, alpaka_cuda_async::Queue> &allocator_;
};

#elif ALPAKA_ACC_GPU_HIP_ENABLED

template <>
inline void set_guard(const alpaka_rocm_async::Queue &queue) {}

#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED 

template <>
inline void set_guard(const alpaka_serial_sync::Queue &queue) {
  [[maybe_unused]] static bool initialized = [] {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    return true;
  }();
}

#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

template <>
inline void set_guard(const alpaka_tbb_async::Queue &queue) {
  [[maybe_unused]] static bool initialized = [] {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    return true;
  }();
}

#else
#error "Automatic backend detection failed."
#endif

template <typename TBuf>
torch::Tensor toTensor(TBuf &alpakaBuff) {
  static_assert(1 == alpaka::Dim<TBuf>::value, "Current support limited to 1-dimension buffers");
  // Torch types are defined in (pytorch/torch/csrc/api/include/)torch/types.h
  // https://discuss.pytorch.org/t/mapping-a-template-type-to-a-scalartype/53174 to the rescue 😅
  // Apparently we are limited to signed types except for uint8_t.
  // Current master in gitlab supports more types. (see AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS macro in
  // c10/core/ScalarType.h
  auto options =
      torch::TensorOptions()
          .dtype(torch::CppTypeToScalarType<typename std::remove_reference<decltype(*alpakaBuff.data())>::type>::value)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
          .device(c10::DeviceType::CUDA, alpaka::getDev(alpakaBuff).getNativeHandle())
#endif
          .pinned_memory(true);
  //    std::cout << "data type=" << typeid(*alpakaBuff.data()).name() << std::endl;
  //    std::cout << "data sizeof element=" << (unsigned)sizeof(*alpakaBuff.data()) << std::endl;
  //    std::cout << "buff extent product=" << alpaka::getExtentProduct(alpakaBuff) << std::endl;
  //    std::cout << "getExtends(buff)=[";
  //    for (auto s : alpaka::getExtents(alpakaBuff)) {
  //      std::cout << s << ", ";
  //    }
  //    std::cout << "]" << std::endl;
  return torch::from_blob(alpakaBuff.data(), {alpaka::getExtents(alpakaBuff)[0]}, options);
}

}  // namespace torch_alpaka

#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_
