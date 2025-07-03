#ifndef DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_DEVICE_COLLECTION_H
#define DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_DEVICE_COLLECTION_H

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateSoA.h"

namespace l1sc {

  template <typename TDev>
  using PFCandidateDeviceCollection = PortableDeviceCollection<PFCandidateSoA, TDev>;

}  // namespace l1sc

#endif  // DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_DEVICE_COLLECTION_H