"""Explicit validation readbacks for parent-owned, serialized CUDA qualification.

Inside the existing Blender process, load this module with importlib.util and call
``run_compact_tree_probe(existing_device_context)``. It does not launch Blender,
change a scene, or retain a runtime. Array readbacks here are validation evidence;
the production adapter reads only the four-byte occupancy total while building.
"""
from __future__ import annotations

from dataclasses import asdict


def run_compact_tree_probe(device_context):
    import numpy as np
    from astro_collision_lab.compact_tree import CompactBarnesHutRuntime
    from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutConfig, BarnesHutRuntime
    from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet

    device_context.initialize()
    wp = device_context.wp
    device = wp.get_device(device_context.device)
    if not device.is_cuda:
        raise ValueError('GPU qualification requires the actual CUDA device')
    rows = []
    count = 96
    for depth in (9, 16):
        for case in ('separated', 'coincident', 'clustered', 'tail', 'inactive', 'all_inactive'):
            rng = np.random.default_rng(913)
            positions = rng.normal(size=(count, 3))
            if case == 'coincident':
                positions[:] = 0
            elif case == 'clustered':
                positions[:80] *= 0.00001
            elif case == 'tail':
                positions[:80] *= 0.001
                positions[80:, 0] += np.linspace(3, 50, 16)
            active = np.ones(count, np.int32)
            if case == 'inactive':
                active[5:] = 0
            elif case == 'all_inactive':
                active[:] = 0
            bodies = BodySet(positions=positions, velocities=rng.normal(0, 0.001, size=(count, 3)),
                             masses=np.full(count, 1/count), radii=np.zeros(count), active=active,
                             stable_ids=np.arange(count, dtype=np.int64))
            config = BarnesHutConfig(capacity=count, count=count, gravitational_constant=1.0,
                                     dt=0.0001, theta=0.5, softening=0.01, max_depth=depth,
                                     stack_capacity=1 + 7*depth, leaf_capacity=8)
            original = BarnesHutRuntime(device_context, config)
            compact = CompactBarnesHutRuntime(device_context, config)
            try:
                original.load_bodies(bodies)
                compact.load_bodies(bodies)
                before = compact.storage_status()
                lhs = compact._arrays['accelerations'].numpy()
                rhs = original._arrays['accelerations'].numpy()
                np.testing.assert_allclose(lhs, rhs, rtol=8e-5, atol=3e-5)
                max_force_difference = float(np.max(np.abs(lhs-rhs)))
                original.step(2)
                compact.step(2)
                lhs = compact._arrays['positions'].numpy()
                rhs = original._arrays['positions'].numpy()
                np.testing.assert_allclose(lhs, rhs, rtol=8e-5, atol=3e-6)
                diag = compact.diagnostics()
                if diag.overflow or diag.capacity_overflow:
                    raise AssertionError('compact tree overflowed during the CUDA parity probe')
                status = compact.storage_status()
                actual_pool_bytes = sum(array.size * wp.types.type_size_in_bytes(array.dtype)
                    for name, array in compact._arrays.items()
                    if (name.startswith('node_') and name != 'node_cursor') or
                    name in ('walk_links', 'walk_cells', 'walk_mass_diagonal', 'walk_off_diagonal'))
                if actual_pool_bytes != status['node_storage_bytes']:
                    raise AssertionError('reported node storage does not match allocated arrays')
                rows.append(dict(case=case, depth=depth, count=count,
                                 max_force_difference=max_force_difference,
                                 max_position_difference=float(np.max(np.abs(lhs-rhs))),
                                 initial_storage=before, storage=status,
                                 allocated_pool_bytes=actual_pool_bytes,
                                 diagnostics=asdict(diag)))
            finally:
                original.close()
                compact.close()
    return dict(passed=True, device=str(device), device_name=device.name,
                warp_version=wp.__version__, precision='float32', cases=rows,
                limits='Small deterministic CUDA parity probes; no large-count performance claim.')
