"""Compare CUDA galaxy forces at identical evolved states in a gated Blender process.

Pass a galaxy_scene_profile JSON as the first argument after --, and a report path
as the second. Use the same extension and dependency environment as that probe.
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.append(os.environ["ACL_PROFILE_DEPENDENCY"])
sys.path.insert(0, os.environ["ACL_PROFILE_EXTENSION"])
import warp as wp

from astro_collision_lab._vendor.astro_warp.product import tree
from astro_collision_lab.galaxy_walk import tree_acceleration_walk
from astro_collision_lab.simulation import AstroCollisionSimulation, BodySpec, SimulationConfig


def check(sim):
    runtime = sim.runtime
    kernels = runtime._require_kernels()
    assert kernels.tree_acceleration_walk is tree_acceleration_walk
    reference = tree._kernels(wp, runtime.config.precision)
    graphs = {}
    block = runtime.walk_block_dim
    try:
        for label, kernel, threads in (("candidate", tree_acceleration_walk, block),
                                       ("reference", reference.tree_acceleration_walk, None)):
            runtime._kernels = replace(kernels, tree_acceleration_walk=kernel)
            runtime.walk_block_dim = threads
            with wp.ScopedCapture(device=runtime.device_context.device) as capture:
                runtime._record_accelerations()
            graphs[label] = capture.graph
    finally:
        runtime._kernels = kernels
        runtime.walk_block_dim = block
    results = []
    for step in range(121):
        if step in (0, 1, 8, 32, 64, 120):
            forces = {}
            counts = {}
            # Restore candidate accelerations last, for the next integration step.
            for label in ("reference", "candidate"):
                wp.capture_launch(graphs[label])
                forces[label] = runtime._arrays["accelerations"].numpy().astype(np.float64)
                flags = runtime._arrays["diag"].numpy()
                assert flags[3] == flags[7] == 0
                counts[label] = int(flags[4])
            assert counts["candidate"] == counts["reference"] > 0
            error = np.linalg.norm(forces["candidate"] - forces["reference"], axis=1)
            scale = np.linalg.norm(forces["reference"], axis=1)
            relative = error / np.maximum(scale, 1e-12)
            rms = float(np.linalg.norm(error) / np.linalg.norm(scale))
            assert np.isfinite(relative).all() and rms < 1e-6 and relative.max() < 1e-4
            results.append(dict(step=step, interactions=counts["candidate"], relative_rms=rms,
                                max_relative=float(relative.max()), max_absolute=float(error.max())))
        if step < 120:
            sim.step()
    return results


if __name__ == "__main__":
    scene_path, report_path = map(Path, sys.argv[sys.argv.index("--") + 1:])
    scene = json.loads(scene_path.read_text())
    wp.config.kernel_cache_dir = os.environ["ACL_KERNEL_CACHE"]
    sim = AstroCollisionSimulation(SimulationConfig(**scene["config"]))
    try:
        sim.initialize([BodySpec(**body) for body in scene["bodies"]])
        report_path.write_text(json.dumps(check(sim), indent=2))
    finally:
        sim.close()
