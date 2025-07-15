#ifndef DataFormats_L1ScoutingSoA_interface_CLUEsteringHostCollection_h
#define DataFormats_L1ScoutingSoA_interface_CLUEsteringHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/CLUEsteringSoA.h"

namespace l1sc {

  using CLUEsteringHostCollection = PortableHostCollection<CLUEsteringSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_CLUEsteringHostCollection_h