"""Compare the CUDA walk with a previous float32 kernel on identical tree inputs.

Run under gated Blender with the galaxy_scene_profile environment and pass
-- <previous galaxy_walk.py> <output.json>. Only the force kernel is replaced.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.append(os.environ["ACL_PROFILE_DEPENDENCY"])
sys.path.insert(0, os.environ["ACL_PROFILE_EXTENSION"])
import warp as wp

from astro_collision_lab._vendor.astro_warp.product.runtime_foundation import DeviceContext
from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutConfig
from astro_collision_lab.galaxy_tree import GalaxyTreeRuntime
from astro_collision_lab.galaxy_walk import tree_acceleration_walk


def check(reference, count, inactive, damage):
    context = DeviceContext()
    config = BarnesHutConfig(count=count, capacity=count, dt=.001,
                             gravitational_constant=1., softening=.03)
    runtime = GalaxyTreeRuntime(context, config)
    p = np.random.default_rng(491).normal(size=(count, 3)).astype(np.float32)
    p[::7] = p[0]
    active = np.ones(count, np.int32)
    active[:inactive] = 0
    bodies = BodySet(positions=p, velocities=np.zeros_like(p),
                     masses=np.full(count, 1/count, np.float32), radii=np.zeros(count, np.float32),
                     active=active, stable_ids=np.arange(count, dtype=np.int64))
    try:
        runtime.load_bodies(bodies)
        launch = wp.launch
        arguments = {}

        def build_layout(kernel, *args, **kwargs):
            if kernel is tree_acceleration_walk:
                arguments.update(kwargs)
            else:
                launch(kernel, *args, **kwargs)

        with patch.object(wp, "launch", build_layout):
            runtime._record_walk()
        assert arguments["block_dim"] == runtime.walk_block_dim
        arrays = runtime._arrays
        if damage:
            links = arrays["walk_links"].numpy()
            used = int(arrays["walk_offsets"].numpy()[count])
            # Corrupt one deep cell so some targets fail while others finish.
            deep = np.flatnonzero((links[:used, 3] >> 24) >= 5)
            links[deep[len(deep)//2], 3] = -1
            arrays["walk_links"].assign(links)
        outputs = []
        poison = np.full_like(p, 91234.)
        for kernel in (reference, tree_acceleration_walk):
            arrays["accelerations"].assign(poison)
            arrays["diag"].zero_()
            launch(kernel, **arguments)
            a = arrays["accelerations"].numpy()
            assert np.isfinite(a).all() and np.all(np.any(a[active != 0] != poison[active != 0], axis=1))
            np.testing.assert_array_equal(a[active == 0], poison[active == 0])
            outputs.append((a, arrays["diag"].numpy()))
        for expected, actual in zip(*outputs):
            np.testing.assert_array_equal(expected, actual)
        result, diag = outputs[1]
        assert bool(diag[3]) == damage
        if damage:
            zero = np.all(result[active != 0] == 0, axis=1)
            assert zero.any() and not zero.all()
        return dict(count=count, inactive=inactive, injected_fault=damage, interactions=int(diag[4]))
    finally:
        runtime.close()
        context.close()


if __name__ == "__main__":
    source, output = map(Path, sys.argv[sys.argv.index("--") + 1:])
    spec = importlib.util.spec_from_file_location("acl_serial_walk_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    wp.config.kernel_cache_dir = os.environ["ACL_KERNEL_CACHE"]
    cases = [(1, 0, False), (31, 0, False), (32, 0, False), (33, 0, False),
             (63, 5, False), (65, 65, False), (4097, 1, True),
             (100003, 7, False), (400000, 0, False)]
    output.write_text(json.dumps([check(module.tree_acceleration_walk, *case) for case in cases],
                                 indent=2))
