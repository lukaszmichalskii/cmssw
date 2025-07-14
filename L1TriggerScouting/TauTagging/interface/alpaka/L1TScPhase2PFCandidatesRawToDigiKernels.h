#ifndef L1TriggerScouting_TauTagging_interface_alpaka_L1TScPhase2PFCandidatesRawToDigiKernels_h
#define L1TriggerScouting_TauTagging_interface_alpaka_L1TScPhase2PFCandidatesRawToDigiKernels_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/OrbitEventIndexMapCollection.h"
#include "DataFormats/L1ScoutingSoA/interface/alpaka/PFCandidateCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "L1TriggerScouting/TauTagging/interface/L1TScPhase2Common.h"
#include "L1TriggerScouting/TauTagging/interface/alpaka/L1TScPhase2BitsEncoding.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  ALPAKA_STATIC_ACC_MEM_CONSTANT alpaka::DevGlobal<Acc1D, const int16_t[8]> kPdgid;
  ALPAKA_STATIC_ACC_MEM_CONSTANT alpaka::DevGlobal<Acc1D, const float> kPi720;

  class L1TScPhase2RawToDigiKernels {
  public:
    L1TScPhase2RawToDigiKernels() = default;
    explicit L1TScPhase2RawToDigiKernels(Queue& queue);

    void initialize(Queue& queue);

  private:
    static std::once_flag init_flag_;
  };

  void printPFCandidateCollection(Queue& queue, PFCandidateCollection& pf_candidates);
  void rawToDigi(Queue& queue, data_t* pf_data, PFCandidateCollection& pf_candidates);
  void associateOrbitEventIndex(Queue& queue, data_t* h_data, OrbitEventIndexMapCollection& orbit_association_map);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_TauTagging_interface_alpaka_L1TScPhase2PFCandidatesRawToDigiKernels_h