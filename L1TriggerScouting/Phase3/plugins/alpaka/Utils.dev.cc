// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "Utils.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::utils {

using namespace cms::alpakatools;

template<typename TAcc>
ALPAKA_FN_ACC float Energy(TAcc const& acc, float pt, float eta, float mass) {
  float pz = pt * alpaka::math::sinh(acc, eta);
  float p = alpaka::math::sqrt(acc, pt * pt + pz * pz);
  return alpaka::math::sqrt(acc, p * p + mass * mass);
}

template<typename TAcc>
ALPAKA_FN_ACC float MassInvariant(TAcc const& acc, PuppiCollection::ConstView data, uint32_t i, uint32_t j, uint32_t k) {
  auto px1 = data.pt()[i] * alpaka::math::cos(acc, data.phi()[i]);
  auto py1 = data.pt()[i] * alpaka::math::sin(acc, data.phi()[i]);
  auto pz1 = data.pt()[i] * alpaka::math::sinh(acc, data.eta()[i]);
  auto e1 = Energy(acc, data.pt()[i], data.eta()[i], 0.1396);

  auto px2 = data.pt()[j] * alpaka::math::cos(acc, data.phi()[j]);
  auto py2 = data.pt()[j] * alpaka::math::sin(acc, data.phi()[j]);
  auto pz2 = data.pt()[j] * alpaka::math::sinh(acc, data.eta()[j]);
  auto e2 = Energy(acc, data.pt()[j], data.eta()[j], 0.1396);

  auto px3 = data.pt()[k] * alpaka::math::cos(acc, data.phi()[k]);
  auto py3 = data.pt()[k] * alpaka::math::sin(acc, data.phi()[k]);
  auto pz3 = data.pt()[k] * alpaka::math::sinh(acc, data.eta()[k]);
  auto e3 = Energy(acc, data.pt()[k], data.eta()[k], 0.1396);

  auto t_energy = e1 + e2 + e3;
  auto t_px = px1 + px2 + px3;
  auto t_py = py1 + py2 + py3;
  auto t_pz = pz1 + pz2 + pz3;

  auto t_momentum = t_px * t_px + t_py * t_py + t_pz * t_pz;
  auto invariant_mass = t_energy * t_energy - t_momentum;

  return invariant_mass > 0 ? alpaka::math::sqrt(acc, invariant_mass) : 0.0f;
}

template<typename TAcc>
ALPAKA_FN_ACC float DeltaPhi(TAcc const& acc, float phi1, float phi2) {
  const float M_PI_CONST = 3.14159265358979323846;
  const float M_2_PI_CONST = 2.0 * M_PI_CONST;
  auto r = alpaka::math::fmod(acc, phi2 - phi1, M_2_PI_CONST);
  if (r < -M_PI_CONST)
    return r + M_2_PI_CONST;
  if (r > M_PI_CONST)
    return r - M_2_PI_CONST;
  return r;
}

template<typename TAcc>
ALPAKA_FN_ACC bool AngularSeparation(TAcc const& acc, PuppiCollection::ConstView data, uint32_t pidx, uint32_t idx) {
  static constexpr float ang_sep_lower_bound = 0.5 * 0.5;
  float delta_eta = data.eta()[pidx] - data.eta()[idx];
  float delta_phi = DeltaPhi(acc, data.phi()[pidx], data.phi()[idx]);
  float ang_sep = delta_eta * delta_eta + delta_phi * delta_phi;
  if (ang_sep < ang_sep_lower_bound)
    return false;
  return true;
}

template<typename TAcc>
ALPAKA_FN_ACC bool ConeIsolation(TAcc const& acc, PuppiCollection::ConstView data, uint32_t thread_idx, uint32_t span_begin, uint32_t span_end) {
  const float min_threshold = 0.01 * 0.01;
  const float max_threshold = 0.25 * 0.25; 
  const float max_isolation_threshold = 2.0;

  float accumulated = 0.0f;
  for (auto idx = span_begin; idx < span_end; idx++) {
    if (thread_idx == idx) 
      continue;
    auto delta_eta = data.eta()[thread_idx] - data.eta()[idx];
    auto delta_phi = DeltaPhi(acc, data.phi()[thread_idx], data.phi()[idx]);
    
    float th_value = delta_eta * delta_eta + delta_phi * delta_phi;
    if (th_value >= min_threshold && th_value <= max_threshold) {
      accumulated += data.pt()[idx];
    }
  }
  return accumulated <= max_isolation_threshold * data.pt()[thread_idx];
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::utils
