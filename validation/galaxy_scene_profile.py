"""Profile the open Galaxy scene through Start and synchronized viewport presentation."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import addon_utils
import bpy


report = Path(os.environ["ACL_PROFILE_REPORT"])
sys.path.append(os.environ["ACL_PROFILE_DEPENDENCY"])
sys.path.insert(0, os.environ["ACL_PROFILE_EXTENSION"])
state = {"scene": bpy.data.filepath, "blender": bpy.app.version_string}
started = None
owner = None


def tick():
    global started, owner
    try:
        if started is None:
            addon = addon_utils.enable("astro_collision_lab")
            if addon is None:
                raise RuntimeError("Could not enable Astro Collision Lab")
            from astro_collision_lab import runtime
            scene = bpy.context.scene
            counts = os.environ.get("ACL_PROFILE_COUNTS")
            if counts and "expected_count" not in state:
                galaxies = sorted((obj for obj in scene.objects if obj.acl_body.is_galaxy),
                                  key=lambda obj: obj.name_full)
                budgets = list(map(int, counts.split(",")))
                for obj, count in zip(galaxies, budgets, strict=True):
                    obj.acl_body.particle_count = count
                state["expected_count"] = sum(budgets)
                return 1.0
            config, bodies, _ = runtime.authored_scene(scene)
            from dataclasses import asdict
            state.update(config=asdict(config), bodies=[asdict(body) for body in bodies],
                         module=addon.__file__)
            if config.mode != "GALAXY" or not bodies:
                raise ValueError("Expected authored galaxy sources in the open scene")
            if config.particle_count != state.get("expected_count", config.particle_count):
                raise ValueError("Authored count does not match requested galaxy budgets")
            owner = runtime.controller
            owner.measurement_sync = True
            started = time.perf_counter()
            if bpy.ops.acl.start() != {"FINISHED"}:
                raise RuntimeError(owner.message)
        rows = owner.measurements
        if owner.error:
            raise RuntimeError(owner.error)
        if len(rows) < 140:
            if time.perf_counter() - started > 240:
                raise TimeoutError(owner.message)
            return 0.05
        if any(r["active_count"] != state["config"]["particle_count"] or
               r["displayed_count"] != r["active_count"] for r in rows):
            raise ValueError("Presented count does not match authored particle budget")
        if [r["revision"] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Expected consecutive rendered revisions starting at 1")
        # Revision zero is drawn before stepping, but is not itself a measurement row.
        # The first row's interval covers revisions zero through one.
        first_draw = (owner.last_presented - started -
                      sum(r["frame_interval_ms"] for r in rows) / 1000)
        state.update(rows=rows[:140], viewport=owner.status()["viewport"],
                     start_to_first_draw_s=first_draw,
                     start_to_first_advanced_draw_s=first_draw + rows[0]["frame_interval_ms"] / 1000,
                     start_through_draw_140_s=first_draw +
                     sum(r["frame_interval_ms"] for r in rows[:140]) / 1000)
    except Exception:
        state["error"] = traceback.format_exc()
    finally:
        if "rows" in state or "error" in state:
            report.write_text(json.dumps(state, indent=2), encoding="utf-8")
            bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(tick, first_interval=1)
