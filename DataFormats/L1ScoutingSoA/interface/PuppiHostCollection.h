#ifndef DataFormats_L1ScoutingSoA_interface_PuppiHostCollection_h
#define DataFormats_L1ScoutingSoA_interface_PuppiHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/PuppiHost.h"

using PuppiHostCollection = PortableHostCollection<PuppiSoA>;

#endif  // DataFormats_L1ScoutingSoA_interface_PuppiHostCollection_h