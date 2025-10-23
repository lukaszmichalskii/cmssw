#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_JetDeviceCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_JetDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/JetHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/IndexSoA.h"
#include "DataFormats/L1ScoutingSoA/interface/OffsetsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  // make the names from the top-level `l1sc` namespace visible for unqualified lookup
  // inside the `ALPAKA_ACCELERATOR_NAMESPACE::l1sc` namespace
  using namespace ::l1sc;

  using JetDeviceCollection = PortableCollection2<IndexSoA, OffsetsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::JetDeviceCollection, l1sc::JetHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_JetDeviceCollection_h