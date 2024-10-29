#ifndef DataFormats_L1ScoutingSoA_interface_PuppiHostCollection_h
#define DataFormats_L1ScoutingSoA_interface_PuppiHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"

class PuppiHostCollection : public PortableHostCollection<PuppiSoA> {
public:
  PuppiHostCollection() = default;

  template <typename TQueue>
  explicit PuppiHostCollection(int32_t size, TQueue queue)
    : PortableHostCollection<PuppiSoA>(size, queue) {}

};

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiHostCollection_h