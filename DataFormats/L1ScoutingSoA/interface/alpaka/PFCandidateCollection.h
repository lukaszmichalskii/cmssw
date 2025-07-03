#ifndef DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__ALPAKA__PF_CANDIDATE_COLLECTION_H
#define DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__ALPAKA__PF_CANDIDATE_COLLECTION_H

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc {

  /**
   * make the names from the top-level `l1sc` namespace visible for unqualified lookup
   * inside the `ALPAKA_ACCELERATOR_NAMESPACE::l1sc` namespace
   */
  using namespace ::l1sc;
  using ::l1sc::PFCandidateDeviceCollection;
  using ::l1sc::PFCandidateHostCollection;

  using PFCandidateCollection = std::conditional_t<
      std::is_same_v<Device, alpaka::DevCpu>, 
      PFCandidateHostCollection, 
      PFCandidateDeviceCollection<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(l1sc::PFCandidateCollection, l1sc::PFCandidateHostCollection);

#endif  // DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__ALPAKA__PF_CANDIDATE_COLLECTION_H