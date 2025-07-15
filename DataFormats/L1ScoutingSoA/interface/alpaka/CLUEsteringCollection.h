#ifndef DataFormats_L1ScoutingSoA_interface_alpaka_CLUEsteringCollection_h
#define DataFormats_L1ScoutingSoA_interface_alpaka_CLUEsteringCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/CLUEsteringHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/CLUEsteringSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * make the names from the top-level `l1sc` namespace visible for unqualified lookup
   * inside the `ALPAKA_ACCELERATOR_NAMESPACE::l1sc` namespace
   */
  using namespace ::l1sc;

  using CLUEsteringCollection = PortableCollection<CLUEsteringSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::CLUEsteringCollection, l1sc::CLUEsteringHostCollection);

#endif  // DataFormats_L1ScoutingSoA_interface_alpaka_CLUEsteringCollection_h