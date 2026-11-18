"""Headless Galaxy step profiler (diagnostic, not acceptance).

Run inside Blender's Python so the bundled Warp is the one that ships:
  blender -b --factory-startup --python tests/validation/galaxy_phase_profile.py -- \
      --count 800000 --depth 16 --storage COMPACT --report out.json

Reports synchronized whole-step time, then a second pass where every
runtime ``_record_*`` phase is bracketed by device synchronization, with the
number of Warp launches and host<->device copies issued inside each phase.
Phase timings serialize the GPU and are only for attribution.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "extension"))
sys.path.insert(0, str(ROOT / "tests" / "validation"))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=800000)
parser.add_argument("--depth", type=int, default=16)
parser.add_argument("--storage", default="COMPACT")
parser.add_argument("--frames", type=int, default=40)
parser.add_argument("--phase-frames", type=int, default=10)
parser.add_argument("--report", required=True)
args = parser.parse_args(argv)

import warp as wp
from astro_collision_lab.simulation import AstroCollisionSimulation, SimulationConfig
from solver_probe import probe_scene

# probe_scene validates its default (STRIDED) config, which rejects 800K at depth 16;
# defer validation to the real storage/depth config below.
_validate = SimulationConfig.validate
SimulationConfig.validate = lambda self: None
try:
    config, bodies = probe_scene("GALAXY", args.count)
finally:
    SimulationConfig.validate = _validate
config = dataclasses.replace(config, tree_max_depth=args.depth, tree_storage=args.storage)
config.validate()
sim = AstroCollisionSimulation(config)


def sync():
    wp.synchronize_device(sim.device_context.device)


t = time.perf_counter()
sim.initialize(bodies)
sync()
device = sim.device_context.device
init_s = time.perf_counter() - t


def find_runtime(root, seen=None, depth=0):
    seen = seen or set()
    if id(root) in seen or depth > 4:
        return None
    seen.add(id(root))
    if any(k.__name__.endswith("BarnesHutRuntime") for k in type(root).__mro__):
        return root
    for value in list(getattr(root, "__dict__", {}).values()):
        if hasattr(value, "__dict__") and not isinstance(value, type):
            hit = find_runtime(value, seen, depth + 1)
            if hit is not None:
                return hit
    return None


runtime = find_runtime(sim)
for _ in range(5):
    sim.step()
sync()
whole = []
for _ in range(args.frames):
    t = time.perf_counter()
    sim.step()
    sync()
    whole.append((time.perf_counter() - t) * 1000)

phases = defaultdict(lambda: {"ms": 0.0, "launches": 0, "copies": 0, "calls": 0})
stack = []
counters = {"launches": 0, "copies": 0}
_launch, _copy = wp.launch, wp.copy
_numpy = wp.array.numpy


kernel_ms = defaultdict(lambda: {"ms": 0.0, "launches": 0})


def counting(fn, key):
    def wrapped(*a, **kw):
        counters[key] += 1
        return fn(*a, **kw)
    return wrapped


def timed_launch(kernel, *a, **kw):
    counters["launches"] += 1
    sync()
    t = time.perf_counter()
    try:
        return _launch(kernel, *a, **kw)
    finally:
        sync()
        k = kernel_ms[getattr(kernel, "key", str(kernel))]
        k["ms"] += (time.perf_counter() - t) * 1000
        k["launches"] += 1


wp.launch = timed_launch
wp.copy = counting(_copy, "copies")
wp.array.numpy = counting(_numpy, "copies")
cls = type(runtime)
names = sorted({n for k in cls.__mro__ for n in vars(k) if n.startswith("_record_")})
originals = {n: getattr(cls, n) for n in names}


def bracket(name, fn):
    def wrapped(self, *a, **kw):
        sync()
        before = dict(counters)
        stack.append(name)
        t = time.perf_counter()
        try:
            return fn(self, *a, **kw)
        finally:
            sync()
            p = phases[name]
            p["ms"] += (time.perf_counter() - t) * 1000
            p["launches"] += counters["launches"] - before["launches"]
            p["copies"] += counters["copies"] - before["copies"]
            p["calls"] += 1
            stack.pop()
    return wrapped


for n in names:
    setattr(cls, n, bracket(n, originals[n]))
phased = []
try:
    for _ in range(args.phase_frames):
        before = dict(counters)
        t = time.perf_counter()
        sim.step()
        sync()
        phased.append({"ms": (time.perf_counter() - t) * 1000,
                       "launches": counters["launches"] - before["launches"],
                       "copies": counters["copies"] - before["copies"]})
finally:
    for n in names:
        setattr(cls, n, originals[n])
    wp.launch, wp.copy, wp.array.numpy = _launch, _copy, _numpy

def walk_ab():
    """Add-on walk vs vendor walk on the same compact walk list (diagnostic readback)."""
    if not hasattr(runtime, "_require_compact"):
        return None
    arrays, cfg = runtime._arrays, runtime.config
    out = {}
    for name, kernel in (("vendor", runtime._require_kernels().tree_acceleration_walk),
                         ("addon", runtime._require_compact().acceleration_walk)):
        acc = wp.zeros_like(arrays["accelerations"])
        diag = wp.zeros_like(arrays["diag"])
        times = []
        for _ in range(7):
            diag.zero_()
            sync()
            t = time.perf_counter()
            _launch(kernel, dim=cfg.count, inputs=[
                arrays["sorted_indices"], cfg.count, cfg.gravitational_constant, cfg.softening,
                cfg.theta, cfg.max_depth, cfg.leaf_capacity, runtime.total_nodes,
                int(cfg.quadrupoles), arrays["walk_bodies"], arrays["walk_offsets"],
                arrays["walk_links"], arrays["walk_cells"], arrays["walk_mass_diagonal"],
                arrays["walk_off_diagonal"], acc, diag], device=device)
            sync()
            times.append((time.perf_counter() - t) * 1000)
        out[name] = {"ms_median": round(statistics.median(times), 3),
                     "interactions": int(diag.numpy()[4]), "overflow": int(diag.numpy()[3]),
                     "acc": acc.numpy()}
    import numpy as np
    va, aa = out["vendor"].pop("acc"), out["addon"].pop("acc")
    scale = np.maximum(np.linalg.norm(va, axis=1), 1e-30)
    rel = np.linalg.norm(aa - va, axis=1) / scale
    out["identical_fraction"] = float(np.mean(np.all(aa == va, axis=1)))
    out["max_relative_difference"] = float(rel.max())
    out["theta"], out["leaf_capacity"], out["quadrupoles"] = cfg.theta, cfg.leaf_capacity, cfg.quadrupoles
    out["precision"] = str(cfg.precision)
    return out


walk_check = walk_ab()

per_frame = {k: {m: round(v[m] / args.phase_frames, 3) for m in ("ms", "launches", "copies", "calls")}
             for k, v in sorted(phases.items(), key=lambda kv: -kv[1]["ms"])}
whole_sorted = sorted(whole)
report = {
    "schema": "acl-galaxy-phase-profile", "count": args.count, "depth": args.depth,
    "storage": args.storage, "warp": wp.__version__, "device": str(device),
    "runtime_class": type(runtime).__name__ if runtime is not None else None,
    "storage_status": runtime.storage_status() if hasattr(runtime, "storage_status") else None,
    "init_s": round(init_s, 3),
    "step_ms": {"median": round(statistics.median(whole), 3),
                "p95": round(whole_sorted[int(.95 * (len(whole) - 1))], 3),
                "min": round(min(whole), 3), "fps_equiv": round(1000 / statistics.median(whole), 2)},
    "phased_step_ms_median": round(statistics.median(p["ms"] for p in phased), 3),
    "phased_launches_per_step": phased[-1]["launches"], "phased_copies_per_step": phased[-1]["copies"],
    "phases_per_step": per_frame,
    "walk_ab": walk_check,
    "kernels_per_step": {k: {"ms": round(v["ms"] / args.phase_frames, 3),
                             "launches": round(v["launches"] / args.phase_frames, 2)}
                         for k, v in sorted(kernel_ms.items(), key=lambda kv: -kv[1]["ms"])},
}
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print("PROFILE", json.dumps({k: report[k] for k in ("step_ms", "phased_step_ms_median",
                                                    "phased_launches_per_step", "walk_ab")}))
