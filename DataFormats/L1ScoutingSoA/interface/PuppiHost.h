#ifndef DataFormats_L1ScoutingSoA_interface_PuppiHost_h
#define DataFormats_L1ScoutingSoA_interface_PuppiHost_h

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiStruct.h"

class PuppiHost : public PortableHostObject<PuppiStruct> {
public:
  PuppiHost() = default;

  template <typename TQueue>
  explicit PuppiHost(TQueue queue)
    : PortableHostObject<PuppiStruct>(queue) {}
    
};

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiHost_h