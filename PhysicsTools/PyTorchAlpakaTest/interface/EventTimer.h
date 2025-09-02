#ifndef PhysicsTools_PyTorchAlpakaTest_interface_EventTimer_h
#define PhysicsTools_PyTorchAlpakaTest_interface_EventTimer_h

#include <chrono>
#include <fmt/format.h>
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {
  
  using Clock_t = std::chrono::steady_clock;

  class EventTimer {
  public:
    EventTimer(const std::string& label, const device::Event& event)
        : timestamp_{Clock_t::now()}, event_{event} {
      msg_ += fmt::format("[DEBUG] OK - {} [{}] ", label, event.id().event());
    }

    ~EventTimer() {
      alpaka::wait(event_.queue());  // explicit synchronization

      auto end = Clock_t::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - timestamp_).count();
      msg_ += fmt::format("({} us)\n", duration);
      fmt::print("{}", msg_);
    }

  private:
    Clock_t::time_point timestamp_; 
    const device::Event& event_; 
    std::string msg_;  
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_EventTimer_h