#ifndef DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_HOST_COLLECTION_H
#define DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_HOST_COLLECTION_H

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/PFCandidateSoA.h"

namespace l1sc {

  using PFCandidateHostCollection = PortableHostCollection<PFCandidateSoA>;

}  // namespace l1sc

#endif  // DATA_FORMATS__L1_SCOUTING_SOA__INTERFACE__PF_CANDIDATE_HOST_COLLECTION_H