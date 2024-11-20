#ifndef L1TriggerScouting_Phase3_plugins_alpaka_Utils_h
#define L1TriggerScouting_Phase3_plugins_alpaka_Utils_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/alpaka/PuppiCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::utils {

using namespace cms::alpakatools;

template<typename TAcc>
ALPAKA_FN_ACC float Energy(TAcc const& acc, float pt, float eta, float mass);

template<typename TAcc>
ALPAKA_FN_ACC float MassInvariant(TAcc const& acc, PuppiCollection::ConstView data, uint32_t i, uint32_t j, uint32_t k);

template<typename TAcc>
ALPAKA_FN_ACC float DeltaPhi(TAcc const& acc, float phi1, float phi2);

template<typename TAcc>
ALPAKA_FN_ACC bool AngularSeparation(TAcc const& acc, PuppiCollection::ConstView data, uint32_t pidx, uint32_t idx);

template<typename TAcc>
ALPAKA_FN_ACC bool ConeIsolation(TAcc const& acc, PuppiCollection::ConstView data, uint32_t thread_idx, uint32_t span_begin, uint32_t span_end);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::utils

#endif  // L1TriggerScouting_Phase3_plugins_alpaka_Utils_h