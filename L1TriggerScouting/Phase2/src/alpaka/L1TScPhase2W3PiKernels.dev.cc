#include "L1TriggerScouting/Phase2/interface/alpaka/L1TScPhase2W3PiKernels.h"

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  using namespace cms::alpakatools;

  // TODO: move utilities to header file
  ALPAKA_FN_HOST uint32_t make_threads_per_block(uint32_t val) {
    if (val <= 0)
      return 1;
    return std::pow(2, std::ceil(std::log2(val)));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC void printBits(TAcc const& acc, uint64_t bits) {
    for (int i = std::numeric_limits<uint64_t>::digits - 1; i >= 0; --i) {
      printf("%llu", (bits >> i) & 1ULL);
      if (i % 8 == 0)
        printf(" ");
    }
    printf("\n");
  }

  template <typename TAcc>
  ALPAKA_FN_ACC int8_t charge(TAcc const& acc, int16_t cls) {
    return alpaka::math::abs(acc, static_cast<int>(cls)) == 11 ? (cls > 0 ? -1 : +1) : (cls > 0 ? +1 : -1);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC float energy(TAcc const& acc, float pt, float eta, float mass) {
    float pz = pt * alpaka::math::sinh(acc, eta);
    float p = alpaka::math::sqrt(acc, pt * pt + pz * pz);
    return alpaka::math::sqrt(acc, p * p + mass * mass);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC float massInvariant(
      TAcc const& acc, PuppiDeviceCollection::ConstView puppi, uint32_t i, uint32_t j, uint32_t k) {
    auto px1 = puppi.pt()[i] * alpaka::math::cos(acc, puppi.phi()[i]);
    auto py1 = puppi.pt()[i] * alpaka::math::sin(acc, puppi.phi()[i]);
    auto pz1 = puppi.pt()[i] * alpaka::math::sinh(acc, puppi.eta()[i]);
    auto e1 = energy(acc, puppi.pt()[i], puppi.eta()[i], 0.1396);

    auto px2 = puppi.pt()[j] * alpaka::math::cos(acc, puppi.phi()[j]);
    auto py2 = puppi.pt()[j] * alpaka::math::sin(acc, puppi.phi()[j]);
    auto pz2 = puppi.pt()[j] * alpaka::math::sinh(acc, puppi.eta()[j]);
    auto e2 = energy(acc, puppi.pt()[j], puppi.eta()[j], 0.1396);

    auto px3 = puppi.pt()[k] * alpaka::math::cos(acc, puppi.phi()[k]);
    auto py3 = puppi.pt()[k] * alpaka::math::sin(acc, puppi.phi()[k]);
    auto pz3 = puppi.pt()[k] * alpaka::math::sinh(acc, puppi.eta()[k]);
    auto e3 = energy(acc, puppi.pt()[k], puppi.eta()[k], 0.1396);

    auto t_energy = e1 + e2 + e3;
    auto t_px = px1 + px2 + px3;
    auto t_py = py1 + py2 + py3;
    auto t_pz = pz1 + pz2 + pz3;

    auto t_momentum = t_px * t_px + t_py * t_py + t_pz * t_pz;
    auto invariant_mass = t_energy * t_energy - t_momentum;

    return invariant_mass > 0 ? alpaka::math::sqrt(acc, invariant_mass) : 0.0f;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC float deltaPhi(TAcc const& acc, float phi1, float phi2) {
    const float M_2_PI_CONST = 2.0 * alpaka::math::constants::pi;
    auto r = alpaka::math::fmod(acc, phi2 - phi1, M_2_PI_CONST);
    if (r < -alpaka::math::constants::pi)
      return r + M_2_PI_CONST;
    if (r > alpaka::math::constants::pi)
      return r - M_2_PI_CONST;
    return r;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC bool angularSeparation(TAcc const& acc,
                                       PuppiDeviceCollection::ConstView puppi,
                                       uint32_t pidx,
                                       uint32_t idx) {
    static constexpr float ang_sep_lower_bound = 0.5 * 0.5;
    float delta_eta = puppi.eta()[pidx] - puppi.eta()[idx];
    float delta_phi = deltaPhi(acc, puppi.phi()[pidx], puppi.phi()[idx]);
    float ang_sep = delta_eta * delta_eta + delta_phi * delta_phi;
    if (ang_sep < ang_sep_lower_bound)
      return false;
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC bool coneIsolation(TAcc const& acc,
                                   PuppiDeviceCollection::ConstView puppi,
                                   uint32_t thread_idx,
                                   uint32_t span_begin,
                                   uint32_t span_end) {
    const float min_threshold = 0.01 * 0.01;
    const float max_threshold = 0.25 * 0.25;
    const float max_isolation_threshold = 2.0;

    float accumulated = 0.0f;
    for (auto idx = span_begin; idx < span_end; idx++) {
      if (thread_idx == idx)
        continue;
      auto delta_eta = puppi.eta()[thread_idx] - puppi.eta()[idx];
      auto delta_phi = deltaPhi(acc, puppi.phi()[thread_idx], puppi.phi()[idx]);

      float th_value = delta_eta * delta_eta + delta_phi * delta_phi;
      if (th_value >= min_threshold && th_value <= max_threshold) {
        accumulated += puppi.pt()[idx];
      }
    }
    return accumulated <= max_isolation_threshold * puppi.pt()[thread_idx];
  }

  // TODO: - define traits for shared memory to be queried dynamically from hardware device
  //       - is it possible to vectorize innermost loops to reduce register pressure?
  //       - use cuts everywhere
  class W3PiKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  PuppiDeviceCollection::ConstView puppi,
                                  OffsetsSoA::ConstView offsets,
                                  SelectionBitmaskDeviceCollection::View selection_bitmask,
                                  BufferDevice::View bx_mask,
                                  const W3PiAlgoParams* params,
                                  PortableCounter* w3pi_triplets_ct) const {
      // shared memory
      const uint8_t SHARED_MEM_BLOCK = 128;
      auto& bit_field = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& size = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& mask = alpaka::declareSharedVar<int[SHARED_MEM_BLOCK], __COUNTER__>(acc);
      auto& best_score = alpaka::declareSharedVar<float, __COUNTER__>(acc);

      if (once_per_block(acc)) {
        size = 0;
        bit_field = 0;
        best_score = 0.0f;
        for (auto idx = 0; idx < SHARED_MEM_BLOCK; idx++)
          mask[idx] = 0;
      }

      uint32_t grid_dim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
      for (uint32_t block_idx : independent_groups(acc, grid_dim)) {
        uint32_t begin = offsets.offsets()[block_idx];
        uint32_t end = offsets.offsets()[block_idx + 1];
        if (end == 0xFFFFFFFF)
          continue;
        if (end - begin == 0)
          continue;
        uint32_t block_dim = end - begin;
        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          auto thread_idx = tid + begin;  // global index
          auto cls = alpaka::math::abs(acc, static_cast<int>(puppi.pdgid()[thread_idx]));
          if (cls != 211 && cls != 11)
            continue;
          auto pt = puppi.pt()[thread_idx];
          if (pt < params->pT_min)
            continue;
          alpaka::atomicExch(acc, &mask[tid], 1);
        }

        alpaka::syncBlockThreads(acc);

        if (once_per_block(acc)) {
          int local_size = 0;
          for (auto idx = 0; idx < SHARED_MEM_BLOCK; ++idx)
            if (mask[idx] == 1)
              ++local_size;
          size = local_size;
          if (size < 3)
            continue;
        }

        for (uint32_t tid : independent_group_elements(acc, block_dim)) {
          auto thread_idx = tid + begin;  // global index
          if (mask[tid] == 0)
            continue;
          if (puppi.pt()[thread_idx] < params->pT_max)
            continue;
          if (!coneIsolation(acc, puppi, thread_idx, begin, end))
            continue;
          for (uint32_t i = 0; i < block_dim; i++) {
            auto global_i_idx = i + begin;
            if (mask[i] == 0)
              continue;
            if (global_i_idx == thread_idx || puppi.pt()[global_i_idx] < params->pT_int)
              continue;
            if (puppi.pt()[global_i_idx] > puppi.pt()[thread_idx] ||
                (puppi.pt()[global_i_idx] == puppi.pt()[thread_idx] && global_i_idx < thread_idx))
              continue;
            if (!angularSeparation(acc, puppi, thread_idx, global_i_idx))
              continue;
            for (uint32_t j = 0; j < block_dim; j++) {
              auto global_j_idx = j + begin;
              if (mask[j] == 0)
                continue;
              if (global_j_idx == thread_idx || global_j_idx == global_i_idx ||
                  puppi.pt()[global_i_idx] < params->pT_min)
                continue;
              if (puppi.pt()[global_j_idx] > puppi.pt()[thread_idx] ||
                  (puppi.pt()[global_j_idx] == puppi.pt()[thread_idx] && global_j_idx < thread_idx))
                continue;
              if (puppi.pt()[global_j_idx] > puppi.pt()[global_i_idx] ||
                  (puppi.pt()[global_j_idx] == puppi.pt()[global_i_idx] && global_j_idx < global_i_idx))
                continue;
              if (alpaka::math::abs(acc,
                                    static_cast<int>(charge(acc, puppi.pdgid()[thread_idx]) +
                                                     charge(acc, puppi.pdgid()[global_i_idx]) +
                                                     charge(acc, puppi.pdgid()[global_j_idx]))) != 1)
                continue;
              auto mass = massInvariant(acc, puppi, thread_idx, global_i_idx, global_j_idx);
              if (mass < params->invariant_mass_lower_bound || mass > params->invariant_mass_upper_bound)
                continue;
              if (angularSeparation(acc, puppi, thread_idx, global_j_idx) &&
                  angularSeparation(acc, puppi, global_i_idx, global_j_idx)) {
                if (coneIsolation(acc, puppi, global_i_idx, begin, end) &&
                    coneIsolation(acc, puppi, global_j_idx, begin, end)) {
                  // atomic then shift to avoid race conditions
                  int bit_shift = alpaka::atomicAdd(acc, &bit_field, 1);
                  uint64_t bit_mask = (1ull << bit_shift);
                  // encode combinatorial part of triplets (from LSB to MSB)
                  alpaka::atomicOr(acc, &selection_bitmask.bits()[thread_idx], bit_mask);
                  alpaka::atomicOr(acc, &selection_bitmask.bits()[global_i_idx], bit_mask);
                  alpaka::atomicOr(acc, &selection_bitmask.bits()[global_j_idx], bit_mask);
                  // printf("i: %u -> ", thread_idx);
                  // printBits(acc, selection_bitmask.bits()[thread_idx]);
                  // printf("j: %u -> ", global_i_idx);
                  // printBits(acc, selection_bitmask.bits()[global_i_idx]);
                  // printf("k: %u -> ", global_j_idx);
                  // printBits(acc, selection_bitmask.bits()[global_j_idx]);
                  // printf("\n");

                  // mask to recover selected bxs
                  alpaka::atomicExch(acc, &bx_mask.value()[block_idx], static_cast<uint32_t>(1));
                  // counter to allocate table for decoded triplets
                  alpaka::atomicAdd(acc, &w3pi_triplets_ct->value, 1);
                }
              }
            }
          }
        }
      }
    }
  };

  std::tuple<SelectedBxDeviceCollection, W3PiDeviceTable> runW3Pi(Queue& queue,
                                                                  const PuppiDeviceCollection& puppi,
                                                                  const BxLookupDeviceCollection& bx_lookup,
                                                                  const W3PiAlgoParams* params) {
    auto bx_mask = BufferDevice(bx_lookup.const_view().metadata().size(), queue);
    bx_mask.zeroInitialise(queue);

    auto selection_bitmask = SelectionBitmaskDeviceCollection(puppi.const_view().metadata().size(), queue);
    selection_bitmask.zeroInitialise(queue);
    auto w3pi_triplets_ct = CounterDevice(queue);
    w3pi_triplets_ct.zeroInitialise(queue);

    uint32_t threads_per_block = make_threads_per_block(128);
    uint32_t blocks_per_grid = bx_lookup.const_view<BxIndexSoA>().metadata().size();
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        W3PiKernel{},
                        puppi.const_view(),
                        bx_lookup.const_view<OffsetsSoA>(),
                        selection_bitmask.view(),
                        bx_mask.view(),
                        params,
                        w3pi_triplets_ct.data());

    // TODO: optimize this later
    auto nbx_ct = CounterDevice(queue);
    nbx_ct.zeroInitialise(queue);
    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, BufferDevice::ConstView mask, PortableCounter* ct) {
          for (int32_t thread_idx : uniform_elements(acc, mask.metadata().size())) {
            if (mask.value()[thread_idx] == 1)
              alpaka::atomicAdd(acc, &ct->value, 1);
          }
        },
        bx_mask.const_view(),
        nbx_ct.data());

    // TODO: find a better way to return the requested data:
    //   - mask might be sufficient instead of compressed indices
    //   - triplets table can be recover later without filling kernel
    auto w3pi_triplets_ct_host = CounterHost(queue);
    auto nbx_ct_host = CounterHost(queue);
    alpaka::memcpy(queue, w3pi_triplets_ct_host.buffer(), w3pi_triplets_ct.buffer());
    alpaka::memcpy(queue, nbx_ct_host.buffer(), nbx_ct.buffer());
    // explicit synchronization in production code should be avoided
    alpaka::wait(queue);

    // allocate buffers for compacted bx and triplets
    auto selected_bxs = SelectedBxDeviceCollection(nbx_ct_host.data()->value, queue);
    auto w3pi_table = W3PiDeviceTable(w3pi_triplets_ct_host.data()->value, queue);

    // return empty buffers
    if (nbx_ct_host.data()->value == 0) {
      return std::make_tuple(std::move(selected_bxs), std::move(w3pi_table));
    }

    // grid dims can be tuned for performance
    uint32_t threads_per_block_inclusive_scan = 1024;
    uint32_t blocks_per_grid_inclusive_scan =
        cms::alpakatools::divide_up_by(bx_mask.view().metadata().size(), threads_per_block_inclusive_scan);
    auto grid_inclusive_scan =
        cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid_inclusive_scan, threads_per_block_inclusive_scan);

    // inclusive scan to recover compressed selected bx indices
    auto prefix_sum = BufferDevice(bx_lookup.const_view().metadata().size(), queue);
    auto pc = alpaka::allocAsyncBuf<int32_t, Idx>(queue, Vec1D{1});
    alpaka::memset(queue, pc, 0x00);
    alpaka::exec<Acc1D>(queue,
                        grid_inclusive_scan,
                        cms::alpakatools::multiBlockPrefixScan<uint32_t>{},
                        bx_mask.view().value(),
                        prefix_sum.view().value(),
                        bx_lookup.view().metadata().size(),
                        blocks_per_grid_inclusive_scan,
                        pc.data(),
                        alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
    alpaka::wait(queue);

    // compress mask [0 0 1 0 1 0 ... BX] into selected bxs only e.g. [124, ... 2456].
    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc,
                         BufferDevice::ConstView mask,
                         BufferDevice::ConstView prefixsum,
                         SelectedBxDeviceCollection::View compressed) {
          for (uint32_t thread_idx : uniform_elements(acc, mask.metadata().size())) {
            if (mask.value()[thread_idx] == 1) {
              uint32_t dest_idx = prefixsum.value()[thread_idx] - 1;  // inclusive scan!
              compressed.bx()[dest_idx] = thread_idx;
            }
          }
        },
        bx_mask.const_view(),
        prefix_sum.const_view(),
        selected_bxs.view());

    // grid dims can be tuned for performance
    uint32_t threads_per_block_fill_table = 32;
    uint32_t blocks_per_grid_fill_table =
        cms::alpakatools::divide_up_by(selected_bxs.view().metadata().size(), threads_per_block_fill_table);
    auto grid_fill_table =
        cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid_fill_table, threads_per_block_fill_table);

    auto table_pos = CounterDevice(queue);
    table_pos.zeroInitialise(queue);

    // kernel to fill w3pi table
    alpaka::exec<Acc1D>(
        queue,
        grid_fill_table,
        [] ALPAKA_FN_ACC(Acc1D const& acc,
                         PuppiDeviceCollection::ConstView puppi,
                         SelectedBxDeviceCollection::ConstView selected_bxs,
                         SelectionBitmaskDeviceCollection::ConstView bitmask,
                         OffsetsSoA::ConstView offsets,
                         W3PiDeviceTable::View w3pi_table,
                         PortableCounter* table_pos) {
          for (uint32_t thread_idx : uniform_elements(acc, selected_bxs.metadata().size())) {
            uint32_t bx_idx = selected_bxs.bx()[thread_idx];
            uint32_t begin = offsets.offsets()[bx_idx];
            uint32_t end = offsets.offsets()[bx_idx + 1];
            if (end <= begin || end == 0xFFFFFFFF)
              continue;

            // assumed 64bit words will not be exceeded at any time
            for (int bit = 0; bit < std::numeric_limits<uint64_t>::digits; ++bit) {
              int found = 0;
              uint32_t i = 0u, j = 0u, k = 0u;
              for (uint32_t p = begin; p < end; ++p) {
                if ((bitmask.bits()[p] & (1ull << bit)) != 0ull) {
                  ++found;
                  if (found == 1)
                    i = p;
                  else if (found == 2)
                    j = p;
                  else if (found == 3) {
                    k = p;
                    break;
                  }
                }
              }

              // reject combinations with less than 3 particles
              // continue to search for rest of combination
              if (found < 3)
                continue;

              // assert to be sure encoding is correct and bits are set correctly
              ALPAKA_ASSERT_ACC(found == 3);

              // fill table
              int32_t pos = alpaka::atomicAdd(acc, &table_pos->value, 1);
              w3pi_table.i()[pos] = i;
              w3pi_table.j()[pos] = j;
              w3pi_table.k()[pos] = k;
            }
          }
        },
        puppi.const_view(),
        selected_bxs.const_view(),
        selection_bitmask.const_view(),
        bx_lookup.const_view<OffsetsSoA>(),
        w3pi_table.view(),
        table_pos.data());

    return std::make_tuple(std::move(selected_bxs), std::move(w3pi_table));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels