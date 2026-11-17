"""Compact storage must preserve the live tree's force and integration contract."""
from dataclasses import replace

import pytest

from astro_collision_lab.compact_tree import (
    CompactBarnesHutRuntime,
    compact_tree_config,
    grown_node_capacity,
    reachable_node_bound,
)
from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutConfig


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


@pytest.mark.parametrize("depth,ordered,radix", [(4, True, False), (16, False, False), (16, True, True)])
def test_compact_sort_reuses_flags_and_maps_only_reachable_prefixes(depth, ordered, radix):
    import numpy as np
    wp = pytest.importorskip("warp")
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet

    class CPUContext:
        device = "cpu"
        def initialize(self):
            wp.init()
            self.wp = wp

    count = 91
    rng = np.random.default_rng(55)
    positions = rng.normal(size=(count, 3)).astype(np.float32)
    positions[:9] = positions[0]
    positions[9:17] = positions[9]
    ids = rng.permutation(count).astype(np.int64) if not ordered else np.arange(count, dtype=np.int64)
    active = np.ones(count, np.int32)
    active[::11] = 0
    bodies = BodySet(positions=positions, velocities=np.zeros_like(positions), masses=np.ones(count),
                     radii=np.zeros(count), active=active, stable_ids=ids)
    rt = CompactBarnesHutRuntime(CPUContext(), config(count, depth))
    try:
        rt.load_bodies(bodies)
        scratch = rt._arrays["compact_flags"].ptr
        rt.rebuild_tree()
        assert rt._arrays["compact_flags"].ptr == scratch
        assert rt._compact_radix == radix
        assert "radix_keys" not in rt._arrays and "radix_indices" not in rt._arrays
        arrays = rt._arrays
        sorted_keys = arrays["sorted_keys"].numpy()[:count]
        sorted_ids = arrays["sorted_indices"].numpy()[:count]
        keys = arrays["body_keys"].numpy()
        expected_order = sorted(np.flatnonzero(active), key=lambda i: (keys[i], ids[i]))
        np.testing.assert_array_equal(sorted_ids[:len(expected_order)], expected_order)
        table = arrays["slot_nodes"].numpy().reshape(depth+1, count)
        node = 0
        for level in range(1, depth+1):
            shift = 3*(depth-level)
            prefixes = sorted_keys[:len(expected_order)] >> shift
            parents, counts = np.unique(prefixes >> 3, return_counts=True)
            sizes = dict(zip(parents, counts))
            expected = np.full(count, -1, np.int32)
            for prefix in np.unique(prefixes):
                if sizes[prefix >> 3] > rt.config.leaf_capacity:
                    node += 1
                    expected[np.flatnonzero(prefixes == prefix)] = node
            np.testing.assert_array_equal(table[level], expected)
        assert rt.storage_status()["actual_nodes"] == node + 1
    finally:
        rt.close()


@pytest.mark.parametrize('free_bytes,expected_capacity,packed', [
    (110*128, 101, True), (101*124, 101, False), (100*124, None, False),
])
def test_growth_uses_exact_requirement_when_only_spare_headroom_exceeds_gpu_budget(
        free_bytes, expected_capacity, packed):
    from types import SimpleNamespace

    allocations = []
    old_array = SimpleNamespace(dtype='float32')

    class FakeWarp:
        vec4 = object()

        def get_device(self, device):
            return SimpleNamespace(is_cuda=True,
                                   free_memory=64*1024*1024 + free_bytes)
        def zeros(self, count, *, dtype, device):
            allocations.append(count)
            return SimpleNamespace(dtype=dtype, ptr=1024)

        def array(self, **view):
            return SimpleNamespace(**view)

    runtime = CompactBarnesHutRuntime(SimpleNamespace(device='fake:0'), config())
    runtime._wp = FakeWarp()
    runtime._total_nodes = 100
    runtime._compact_growths = 1
    runtime._arrays = {name: old_array for name in [
        'node_mass', 'walk_links', 'walk_cells', 'walk_mass_diagonal', 'walk_off_diagonal']}
    runtime._arrays['walk_storage'] = old_array
    if expected_capacity is None:
        with pytest.raises(MemoryError, match='additional GPU bytes'):
            runtime._grow_node_pool(101)
        assert not allocations
        assert runtime.total_nodes == 100
        assert all(array is old_array for array in runtime._arrays.values())
    else:
        runtime._grow_node_pool(101)
        assert runtime.total_nodes == expected_capacity
        if not packed:
            assert 'walk_storage' not in runtime._arrays
            assert allocations == [expected_capacity] * 5
            return
        assert allocations == [expected_capacity, (expected_capacity, 4)]
        owner = runtime._arrays['walk_storage']
        for index, name in enumerate(('walk_links', 'walk_cells', 'walk_mass_diagonal', 'walk_off_diagonal')):
            view = runtime._arrays[name]
            assert (view.ptr, view.strides, view.shape) == (owner.ptr + 16*index, (64,), (expected_capacity,))


@pytest.mark.parametrize('count,depth', [(8, 16), (32, 9), (32, 16)])
@pytest.mark.parametrize('wide,quadrupoles', [(False, True), (True, True), (False, False), (True, False)])
def test_compact_cpu_runtime_preserves_direct_force_and_updates(count, depth, wide, quadrupoles):
    wp = pytest.importorskip('warp')
    import numpy as np
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
    from astro_collision_lab._vendor.astro_warp.product.nbody import Precision

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
    from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutRuntime
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet

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
