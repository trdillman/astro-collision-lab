import numpy as np
import pytest

from astro_collision_lab.galaxy_tree import GalaxyTreeRuntime
from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutConfig, BarnesHutRuntime


@pytest.mark.parametrize("ordered", [True, False])
def test_galaxy_sort_preserves_ties_padding_and_trajectory(ordered, device="cpu"):
    wp = pytest.importorskip("warp")

    class Context:
        def initialize(self):
            wp.init()
            self.wp = wp

    context = Context()
    context.device = device
    n = 63
    rng = np.random.default_rng(478)
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    positions[::3] = 0
    active = np.ones(n, dtype=np.int32)
    active[2::9] = 0
    ids = np.arange(-n, n, 2, dtype=np.int64)
    bodies = BodySet(positions=positions, velocities=np.zeros_like(positions),
                     masses=np.full(n, 1/n, np.float32), radii=np.zeros(n, np.float32),
                     active=active, stable_ids=ids if ordered else rng.permutation(ids))
    config = BarnesHutConfig(capacity=n, count=n, gravitational_constant=1.,
                             dt=.0001, softening=.02)
    reference = BarnesHutRuntime(context, config)
    candidate = GalaxyTreeRuntime(context, config)
    try:
        for runtime in (reference, candidate):
            runtime.load_bodies(bodies)
        assert ("radix_keys" in candidate._arrays) == ordered
        for _ in range(3):
            for key in ("sorted_indices", "sorted_keys", "accelerations", "positions", "velocities"):
                np.testing.assert_array_equal(candidate._arrays[key].numpy(),
                                              reference._arrays[key].numpy())
            reference.step()
            candidate.step()
    finally:
        candidate.close()
        reference.close()
