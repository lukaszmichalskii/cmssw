#ifndef DataFormats_L1ScoutingSoA_interface_Struct_h
#define DataFormats_L1ScoutingSoA_interface_Struct_h

#include <cstdint>

namespace l1sc {

  struct W3PiTriplet {
    uint32_t i;
    uint32_t j;
    uint32_t k;
  };

  struct BxCounter {
    uint32_t nbx;
  };

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_Struct_h