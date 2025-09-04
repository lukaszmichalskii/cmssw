#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/RandomCollectionFillingKernel.h"

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/alpaka/Nvtx.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  using namespace cms::alpakatools;

  void randomFillParticleCollection(Queue& queue, torchportabletest::ParticleDeviceCollection& particles) {
    Nvtx kernel_range("randomFillParticleCollection()");

    uint32_t threads_per_block = 1024;
    uint32_t blocks_per_grid = particles.view().metadata().size();
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, torchportabletest::ParticleDeviceCollection::View particles_view) {
          for (int32_t thread_idx : uniform_elements(acc, particles_view.metadata().size())) {
            auto rnd_gen = alpaka::rand::engine::createDefault(acc, 43, thread_idx);
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
            particles_view[thread_idx].pt() = dist(rnd_gen);
            particles_view[thread_idx].eta() = dist(rnd_gen);
            particles_view[thread_idx].phi() = dist(rnd_gen);
          }
        },
        particles.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest