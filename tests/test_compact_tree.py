"""Compact storage must preserve the live tree's force and integration contract."""
from dataclasses import replace

import pytest
from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutConfig
from astro_collision_lab.compact_tree import (
    CompactBarnesHutRuntime,
    compact_tree_config,
    grown_node_capacity,
    reachable_node_bound,
)


def config(count=32, depth=16):
    return BarnesHutConfig(capacity=count, count=count, gravitational_constant=1.0,
                          dt=0.0001, softening=0.01, max_depth=depth,
                          stack_capacity=1 + 7 * depth)


def test_initial_configuration_admits_large_depth_sixteen_without_reserving_stride():
    original = config(1_000_000)
    compact = compact_tree_config(original)
    compact.validate()
    assert compact.node_capacity == 17
    assert replace(compact, node_capacity=0) == original


def test_reachable_bound_accounts_for_long_single_child_chains():
    assert reachable_node_bound(8, 16, 8) == 1
    assert reachable_node_bound(9, 16, 8) == 25
    assert reachable_node_bound(1_000_000, 16, 8) == 2_148_559


def test_capacity_reuses_allocation_and_fails_closed_at_budget():
    assert grown_node_capacity(100, 99) == 100
    assert grown_node_capacity(17, 101) == 101
    assert grown_node_capacity(101, 102) >= 102
    with pytest.raises(ValueError, match='9000000'):
        grown_node_capacity(17, 9_000_001)


@pytest.mark.parametrize('free_nodes,expected_capacity', [(110, 101), (100, None)])
def test_growth_uses_exact_requirement_when_only_spare_headroom_exceeds_gpu_budget(
        free_nodes, expected_capacity):
    from types import SimpleNamespace

    allocations = []
    old_array = SimpleNamespace(dtype='float32')

    class FakeWarp:
        def get_device(self, device):
            return SimpleNamespace(is_cuda=True,
                                   free_memory=64*1024*1024 + free_nodes*124)
        def zeros(self, count, *, dtype, device):
            allocations.append(count)
            return SimpleNamespace(dtype=dtype)

    runtime = CompactBarnesHutRuntime(SimpleNamespace(device='fake:0'), config())
    runtime._wp = FakeWarp()
    runtime._total_nodes = 100
    runtime._compact_growths = 1
    runtime._arrays = dict.fromkeys(['node_mass', 'walk_links', 'walk_cells', 'walk_mass_diagonal', 'walk_off_diagonal'], old_array)
    if expected_capacity is None:
        with pytest.raises(MemoryError, match='additional GPU bytes'):
            runtime._grow_node_pool(101)
        assert not allocations
        assert runtime.total_nodes == 100
        assert all(array is old_array for array in runtime._arrays.values())
    else:
        runtime._grow_node_pool(101)
        assert runtime.total_nodes == expected_capacity
        assert allocations and set(allocations) == {expected_capacity}


@pytest.mark.parametrize('count,depth', [(8, 16), (32, 9), (32, 16)])
@pytest.mark.parametrize('wide,quadrupoles', [(False, True), (True, True), (False, False), (True, False)])
def test_compact_cpu_runtime_preserves_direct_force_and_updates(count, depth, wide, quadrupoles):
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.nbody import Precision
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet

    class CPUContext:
        device = 'cpu'
        def initialize(self):
            wp.init()
            self.wp = wp

    rng = np.random.default_rng(312)
    positions = rng.normal(size=(count, 3)).astype(np.float32)
    bodies = BodySet(positions=positions, velocities=np.zeros_like(positions),
                     masses=np.full(count, 1/count, np.float32),
                     radii=np.zeros(count, np.float32), active=np.ones(count, np.int32),
                     stable_ids=np.arange(count, dtype=np.int64))
    runtime = CompactBarnesHutRuntime(CPUContext(), replace(config(count, depth), theta=0.01,
        quadrupoles=quadrupoles, precision=Precision.FLOAT64 if wide else Precision.FLOAT32))
    try:
        runtime.load_bodies(bodies)
        accel = runtime._arrays['accelerations'].numpy()
        delta = positions[None, :, :] - positions[:, None, :]
        distance = np.sum(delta*delta, axis=2) + 0.01**2
        expected = (delta * (1/count) / distance[:, :, None]**1.5).sum(axis=1)
        np.testing.assert_allclose(accel, expected, rtol=2e-5, atol=2e-6)
        status = runtime.storage_status()
        assert status['actual_nodes'] <= status['allocated_nodes'] < 1 + count*depth
        diag = runtime.diagnostics()
        assert diag.node_count == status['actual_nodes']
        assert diag.leaf_count >= 1
        runtime.step()
        assert runtime.step_index == 1
        assert np.isfinite(runtime._arrays['positions'].numpy()).all()
    finally:
        runtime.close()


@pytest.mark.parametrize('case', ['coincident', 'clustered', 'tail', 'inactive', 'all_inactive'])
@pytest.mark.parametrize('depth', [9, 16])
def test_compact_cpu_matches_original_tree_for_edge_topologies(case, depth):
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
    from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutRuntime

    class CPUContext:
        device = 'cpu'
        def initialize(self):
            wp.init()
            self.wp = wp

    count = 40
    rng = np.random.default_rng(341)
    positions = rng.normal(size=(count, 3)).astype(np.float32)
    if case == 'coincident':
        positions[:] = 0
    elif case == 'clustered':
        positions[:32] *= 0.00001
    elif case == 'tail':
        positions[:32] *= 0.001
        positions[32:, 0] += np.linspace(3, 30, 8)
    active = np.ones(count, np.int32)
    if case == 'inactive':
        active[4:] = 0
    elif case == 'all_inactive':
        active[:] = 0
    bodies = BodySet(positions=positions, velocities=np.zeros_like(positions),
                     masses=np.full(count, 1/count, np.float32), radii=np.zeros(count, np.float32),
                     active=active, stable_ids=np.arange(count, dtype=np.int64))
    original = BarnesHutRuntime(CPUContext(), config(count, depth))
    compact = CompactBarnesHutRuntime(CPUContext(), config(count, depth))
    try:
        original.load_bodies(bodies)
        compact.load_bodies(bodies)
        np.testing.assert_allclose(compact._arrays['accelerations'].numpy(),
                                   original._arrays['accelerations'].numpy(), rtol=4e-5, atol=2e-5)
        before_capacity = compact.total_nodes
        compact.rebuild_tree()
        assert compact.total_nodes == before_capacity
        original.step()
        compact.step()
        np.testing.assert_allclose(compact._arrays['positions'].numpy(),
                                   original._arrays['positions'].numpy(), rtol=4e-5, atol=2e-6)
        assert not compact.diagnostics().overflow
    finally:
        original.close()
        compact.close()


def test_pruned_prefix_mapping_cannot_alias_an_unrelated_internal_node():
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet

    class CPUContext:
        device = 'cpu'
        def initialize(self):
            wp.init()
            self.wp = wp

    count = 14
    bodies = BodySet(positions=np.zeros((count, 3), np.float32),
                     velocities=np.zeros((count, 3), np.float32),
                     masses=np.ones(count, np.float32), radii=np.zeros(count, np.float32),
                     active=np.ones(count, np.int32), stable_ids=np.arange(count, dtype=np.int64))
    runtime = CompactBarnesHutRuntime(CPUContext(), config(count, 4))
    try:
        runtime.load_bodies(bodies)
        keys = np.full(runtime.sort_size, np.iinfo(np.int64).max, np.int64)
        keys[:count] = [0]*9 + [512, 513, 514, 515, 516]
        wp.copy(runtime._arrays['sorted_keys'], wp.array(keys, dtype=wp.int64, device='cpu'))
        runtime._record_topology_moments()
        runtime._record_walk()
        runtime._raise_tree_overflow()
        assert runtime.storage_status()['actual_nodes'] == 6
        assert runtime._arrays['walk_node_counts'].numpy().sum() == 6
        mapping = runtime._arrays['slot_nodes'].numpy().reshape(5, count)
        assert (mapping[2:, 9:] == -1).all()
        links = runtime._arrays['walk_links'].numpy()[:6]
        assert len({(int(link[0]), int(link[3]) >> 24) for link in links}) == 6
    finally:
        runtime.close()


def test_topology_growth_preserves_particle_arrays_and_retains_the_new_pool():
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet

    class CPUContext:
        device = 'cpu'
        def initialize(self):
            wp.init()
            self.wp = wp

    count = 128
    bodies = BodySet(positions=np.zeros((count, 3), np.float32),
                     velocities=np.zeros((count, 3), np.float32),
                     masses=np.full(count, 1/count, np.float32), radii=np.zeros(count, np.float32),
                     active=np.ones(count, np.int32), stable_ids=np.arange(count, dtype=np.int64))
    runtime = CompactBarnesHutRuntime(CPUContext(), config(count))
    try:
        runtime.load_bodies(bodies)
        before = runtime.total_nodes
        particle_pointers = {name: runtime._arrays[name].ptr
                             for name in ['positions', 'velocities', 'accelerations']}
        positions = np.random.default_rng(491).normal(size=(count, 3)).astype(np.float32)
        wp.copy(runtime._arrays['positions'], wp.array(positions, dtype=wp.vec3, device='cpu'))
        runtime.rebuild_tree()
        assert runtime.total_nodes > before
        grown = runtime.total_nodes
        assert particle_pointers == {name: runtime._arrays[name].ptr for name in particle_pointers}
        runtime.rebuild_tree()
        assert runtime.total_nodes == grown
        wp.copy(runtime._arrays['positions'], wp.zeros(count, dtype=wp.vec3, device='cpu'))
        runtime.rebuild_tree()
        assert runtime.total_nodes == grown
        assert runtime.step_index == 0
    finally:
        runtime.close()



@pytest.mark.parametrize('count', [40, 1000])
@pytest.mark.parametrize('case', ['random', 'disk', 'coincident', 'clustered', 'inactive'])
@pytest.mark.parametrize('wide,quadrupoles', [(False, True), (True, True), (False, False)])
def test_compact_walk_is_bit_identical_to_the_vendor_walk(count, case, wide, quadrupoles):
    """The add-on walk reorders loads only; every force must equal the vendor's exactly."""
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.nbody import Precision
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
    from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutRuntime

    class CPUContext:
        device = 'cpu'
        def initialize(self):
            wp.init()
            self.wp = wp

    rng = np.random.default_rng(7000 + count)
    positions = rng.normal(size=(count, 3)).astype(np.float32)
    if case == 'disk':
        radius = rng.exponential(1.0, count)
        angle = rng.uniform(0, 2 * np.pi, count)
        positions = np.stack([radius * np.cos(angle), radius * np.sin(angle),
                              rng.normal(0, 0.05, count)], axis=1).astype(np.float32)
    elif case == 'coincident':
        positions[:] = 0
    elif case == 'clustered':
        positions[:count // 2] *= 0.00001
    active = np.ones(count, np.int32)
    if case == 'inactive':
        active[rng.random(count) < 0.4] = 0
    masses = rng.uniform(0.5, 1.5, count).astype(np.float32) / count
    bodies = BodySet(positions=positions, velocities=np.zeros_like(positions), masses=masses,
                     radii=np.zeros(count, np.float32), active=active,
                     stable_ids=np.arange(count, dtype=np.int64))
    cfg = replace(config(count, 16), quadrupoles=quadrupoles,
                  precision=Precision.FLOAT64 if wide else Precision.FLOAT32)
    original = BarnesHutRuntime(CPUContext(), cfg)
    compact = CompactBarnesHutRuntime(CPUContext(), cfg)
    try:
        original.load_bodies(bodies)
        launched = []
        real_launch = wp.launch

        def spy(kernel, *a, **kw):
            launched.append(getattr(kernel, 'key', kernel))
            return real_launch(kernel, *a, **kw)

        wp.launch = spy
        try:
            compact.load_bodies(bodies)
        finally:
            wp.launch = real_launch
        assert compact._require_compact().acceleration_walk.key in launched
        assert compact._require_kernels().tree_acceleration_walk.key not in launched
        # Reference: the vendor walk kernel over the very same compact walk list.
        arrays = compact._arrays
        reference = wp.zeros_like(arrays['accelerations'])
        diag = wp.zeros_like(arrays['diag'])
        wp.launch(compact._require_kernels().tree_acceleration_walk, dim=count, inputs=[
            arrays['sorted_indices'], count, cfg.gravitational_constant, cfg.softening,
            cfg.theta, cfg.max_depth, cfg.leaf_capacity, compact.total_nodes,
            int(cfg.quadrupoles), arrays['walk_bodies'], arrays['walk_offsets'],
            arrays['walk_links'], arrays['walk_cells'], arrays['walk_mass_diagonal'],
            arrays['walk_off_diagonal'], reference, diag], device='cpu')
        np.testing.assert_array_equal(arrays['accelerations'].numpy(), reference.numpy())
        walked = compact._arrays['diag'].numpy()
        assert int(diag.numpy()[4]) > 0 or not active.any()
        assert not walked[3]
        # And the compact tree still agrees with the vendor tree to rounding.
        np.testing.assert_allclose(arrays['accelerations'].numpy(),
                                   original._arrays['accelerations'].numpy(),
                                   rtol=2e-4 if not wide else 1e-12, atol=1e-6 if not wide else 1e-14)
    finally:
        original.close()
        compact.close()

@pytest.mark.parametrize('case', ['random', 'coincident', 'ties', 'inactive', 'all_inactive'])
@pytest.mark.parametrize('count,depth', [(1, 16), (37, 16), (300, 9), (1000, 16)])
def test_compact_sort_reproduces_the_vendor_order_without_a_bitonic_network(case, count, depth):
    """Morton key, then stable id: the vendor bitonic order, from one stable radix sort."""
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
    from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutRuntime

    class CPUContext:
        device = 'cpu'
        def initialize(self):
            wp.init()
            self.wp = wp

    rng = np.random.default_rng(9100 + count + depth)
    positions = rng.normal(size=(count, 3)).astype(np.float32)
    if case == 'coincident':
        positions[:] = 0.25
    elif case == 'ties':
        positions = rng.integers(0, 3, size=(count, 3)).astype(np.float32)
    active = np.ones(count, np.int32)
    if case == 'inactive':
        active[rng.random(count) < 0.5] = 0
    elif case == 'all_inactive':
        active[:] = 0
    # Stable ids deliberately disagree with index order, so ties expose the tie-break.
    stable_ids = rng.permutation(count * 3)[:count].astype(np.int64)

    def bodies(ids):
        return BodySet(positions=positions, velocities=np.zeros_like(positions),
                       masses=np.full(count, 1 / count, np.float32),
                       radii=np.zeros(count, np.float32), active=active, stable_ids=ids)

    original = BarnesHutRuntime(CPUContext(), config(count, depth))
    compact = CompactBarnesHutRuntime(CPUContext(), config(count, depth))
    try:
        for ids in (stable_ids, stable_ids[::-1].copy()):
            original.load_bodies(bodies(ids))
            compact.load_bodies(bodies(ids))
            for name in ('sorted_keys', 'sorted_indices', 'body_keys'):
                np.testing.assert_array_equal(compact._arrays[name].numpy(),
                                              original._arrays[name].numpy(), err_msg=name)
            launched = []
            kernels = compact._require_kernels()
            real_launch = wp.launch

            def spy(kernel, *a, launched=launched, real_launch=real_launch, **kw):
                launched.append(kernel)
                return real_launch(kernel, *a, **kw)

            wp.launch = spy
            try:
                compact.rebuild_tree()
            finally:
                wp.launch = real_launch
            assert kernels.bitonic_step not in launched
            original.rebuild_tree()
            for name in ('sorted_keys', 'sorted_indices'):
                np.testing.assert_array_equal(compact._arrays[name].numpy(),
                                              original._arrays[name].numpy(), err_msg=name)
        snapshot = original.snapshot() if hasattr(original, 'snapshot') else None
        if snapshot is not None:
            compact.restore_snapshot(snapshot)
            compact.rebuild_tree()
            original.rebuild_tree()
            np.testing.assert_array_equal(compact._arrays['sorted_indices'].numpy(),
                                          original._arrays['sorted_indices'].numpy())
    finally:
        original.close()
        compact.close()
