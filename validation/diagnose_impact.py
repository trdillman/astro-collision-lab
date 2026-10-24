"""Untimed full-state diagnostic for impact stability and numerical safeguards."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extension"))
from solver_probe import probe_scene
from astro_collision_lab.simulation import AstroCollisionSimulation
import numpy as np
import warp as wp

wp.config.log_level = wp.LOG_WARNING
cfg, bodies = probe_scene("IMPACT", 100000)
sim = AstroCollisionSimulation(cfg)
rows = []
try:
    sim.initialize(bodies)
    solver = sim.runtime._require_solver()
    for frame in range(7):
        wp.synchronize_device(sim.device_context.device)
        ch = sim.display_channels()
        x = ch["positions"].numpy()
        v = ch["velocities"].numpy()
        row = {
            "frame": frame, "time": sim.physical_time,
            "max_radius": float(np.linalg.norm(x, axis=1).max()),
            "speed_p50_p95_max": np.percentile(np.linalg.norm(v, axis=1), [50, 95, 100]).tolist(),
            "overflow": int(solver.neighbor_overflow_total()),
            "speed_clamps": int(solver.speed_clamps()),
            "active_outer_steps": int(solver.active_outer_steps),
            "budget_bound_steps": int(solver.budget_bound_steps),
            "step_ms": sim.stats.get("last_step_ms"),
        }
        for name in ("densities", "pressures", "internal_energies"):
            if name in ch:
                row[name] = np.percentile(ch[name].numpy(), [0, 50, 95, 100]).tolist()
        rows.append(row)
        print(json.dumps(row), flush=True)
        if frame < 6:
            sim.step()
    out = Path("status/impact-diagnostic.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "stats": sim.stats}, indent=2))
finally:
    sim.close()
