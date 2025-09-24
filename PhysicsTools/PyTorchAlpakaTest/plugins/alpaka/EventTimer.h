#ifndef PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_EventTimer_h
#define PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_EventTimer_h

#include <chrono>
#include <fmt/format.h>
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "PhysicsTools/PyTorchAlpaka/interface/QueueGuard.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace cms::torch::alpakatools;

  using Clock_t = std::chrono::steady_clock;

  inline std::string formatDevice(Device device) {
    std::string dev_str;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    dev_str = "cuda:";
#elif ALPAKA_ACC_GPU_HIP_ENABLED
    dev_str = "rocm:";
#else
    dev_str = "cpu";
    return dev_str;
#endif

    return dev_str + std::to_string(device.getNativeHandle());
  }

  class EventTimer {
  public:
    EventTimer(const std::string& label, const device::Event& event, bool verbose)
        : timestamp_{Clock_t::now()}, event_{event}, verbose_{verbose} {
      msg_ += fmt::format("[DEBUG] OK - {} [event: {}, stream: {}, device: {}, queue: {}] ",
                          label,
                          event.id().event(),
                          static_cast<int>(event.streamID().value()),
                          formatDevice(event.device()),
                          QueueHash<Queue>::alpakaQueue(event.queue()));
    }

    ~EventTimer() {
      if (verbose_) {
        alpaka::wait(event_.queue());  // explicit synchronization

        auto end = Clock_t::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - timestamp_).count();
        msg_ += fmt::format("({} us)\n", duration);
        fmt::print("{}", msg_);
      }
    }

  private:
    Clock_t::time_point timestamp_;
    const device::Event& event_;
    bool verbose_;
    std::string msg_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_plugins_alpaka_EventTimer_h