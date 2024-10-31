#ifndef DataFormats_L1ScoutingSoA_interface_PuppiStruct_h
#define DataFormats_L1ScoutingSoA_interface_PuppiStruct_h

#include <cstdint>

struct PuppiStruct {
  uint16_t bx;
  uint32_t offsets;
  float pt;
  float eta;
  float phi;
  float z0;
  float dxy;
  float puppiw;
  int16_t pdgId;
  uint8_t quality;
};

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiStruct_h