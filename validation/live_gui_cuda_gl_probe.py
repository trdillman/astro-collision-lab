"""Installed-window CUDA/GL proof, including motion, Blender depth and teardown.

This proves only the owned presentation surface, not a complete simulation's
frame rate or absence of CPU timeline caching. Pixel readback is measurement.
"""

from __future__ import annotations

import ctypes as C
import importlib
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from probe_support import (
    configure_environment,
    extension_code_fingerprint,
    import_extension_module,
    resolve_probe_paths,
)

REPORT_FILENAME = "astro_collision_lab_cuda_gl_pixels.json"
_STATE = {}
_KERNEL = None


def _kernel(wp):
    global _KERNEL
    if _KERNEL is None:
        # Global binding is needed by Warp's annotation/code generator.
        globals()["wp"] = wp

        @wp.kernel
        def sentinel(out: wp.array2d(dtype=wp.float32), x: float, z: float):
            out[0, 0] = x
            out[0, 1] = 0.0
            out[0, 2] = z
            out[0, 3] = 0.35
            out[0, 4] = 1.0
            out[0, 5] = 0.0
            out[0, 6] = 0.0
            out[0, 7] = 1.0
            out[0, 8] = 0.0
            out[0, 9] = 0.0
            out[0, 10] = 0.0
            out[0, 11] = 0.0

        _KERNEL = sentinel
    return _KERNEL


def _environment(module_name, repo_root, report_dir, warp_cache_dir):
    paths = resolve_probe_paths(
        repo_root=repo_root, report_dir=report_dir, warp_cache_dir=warp_cache_dir
    )
    configure_environment(paths)
    module = import_extension_module(module_name, mode="installed")
    module.register()
    wp = importlib.import_module("warp")
    wp.config.kernel_cache_dir = str(paths.warp_cache_dir)
    wp.init()
    if wp.__version__ != "1.15.0":
        raise RuntimeError("probe requires the packaged Warp 1.15.0")
    return module, wp, paths


def warm_kernel_cache(*, module_name, repo_root, report_dir, warp_cache_dir):
    _, wp, _ = _environment(module_name, repo_root, report_dir, warp_cache_dir)
    data = wp.empty((1, 12), dtype=wp.float32, device="cuda:0")
    wp.launch(_kernel(wp), dim=1, inputs=[data, 0.0, 0.0], device=data.device)
    wp.synchronize_device(data.device)
    print("CUDA_GL_KERNEL_WARMED")


def _matrix(matrix):
    return tuple(float(matrix[row][col]) for col in range(4) for row in range(4))


def _capture(name):
    import bpy
    import numpy as np

    window, area, region = _STATE["view"]
    path = _STATE["paths"].report_dir / ("cuda-gl-" + name + ".png")
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=str(path))
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    finally:
        bpy.data.images.remove(image)
    # Middle 80% excludes all headers and border decoration; only red is measured.
    left, right = width // 10, width * 9 // 10
    bottom, top = height // 10, height * 9 // 10
    rgb = pixels[bottom:top, left:right, :3]
    red = (rgb[:, :, 0] > 0.3) & (rgb[:, :, 1] < 0.1) & (rgb[:, :, 2] < 0.1)
    yy, xx = np.nonzero(red)
    return {
        "red_pixels": int(red.sum()),
        "centroid_x": float(xx.mean()) if len(xx) else None,
        "centroid_y": float(yy.mean()) if len(yy) else None,
        "image": str(path),
        "image_size": [width, height],
        "crop": [left, bottom, right, top],
    }


def _draw():
    import bpy

    s = _STATE
    if s.get("error") or s.get("closed"):
        return
    if bpy.context.region != s["view"][2]:
        return
    try:
        wp, renderer = s["wp"], s["renderer"]
        if s.get("batch") is None:
            s["batch"] = renderer.CudaGLBatch("cuda:0", 1)
        batch = s["batch"]
        if s["phase"] == "cleanup":
            s["report"]["batch_before_close"] = batch.report()
            batch.close()
            deleted = 0
            for _ in range(20):
                trial = renderer.CudaGLBatch("cuda:0", 1)
                trial.update(lambda out: wp.launch(
                    _kernel(wp), dim=1, inputs=[out, 0.0, 0.0], device=trial.device
                ))
                name, gl = trial.vbo, trial._gl
                if not gl._call("glIsBuffer", C.c_ubyte, (C.c_uint,), name):
                    raise AssertionError("allocated buffer is not live")
                trial.close()
                if gl._call("glIsBuffer", C.c_ubyte, (C.c_uint,), name):
                    raise AssertionError("closed GL buffer remains live")
                deleted += 1
            s["report"]["teardown_cycles"] = deleted
            s["report"]["batch_after_close"] = batch.report()
            s["closed"] = True
            return
        x, z = {"left": (-1.5, 0.0), "right": (1.5, 0.0),
                "behind": (0.0, -2.0), "front": (0.0, 2.0)}[s["phase"]]
        batch.update(lambda out: wp.launch(
            _kernel(wp), dim=1, inputs=[out, x, z], device=batch.device
        ))
        region, rv3d = bpy.context.region, bpy.context.region_data
        batch.draw(1, view_projection=_matrix(rv3d.perspective_matrix),
                   view_matrix=_matrix(rv3d.view_matrix),
                   viewport_size=(region.width, region.height), shaded=True)
        s["draws"] += 1
    except Exception:
        s["error"] = traceback.format_exc()


def _finish():
    import bpy

    s, report = _STATE, _STATE["report"]
    if s.get("handler"):
        bpy.types.SpaceView3D.draw_handler_remove(s["handler"], "WINDOW")
    if s.get("guarded"):
        s["wp"].copy = s["original_copy"]
        s["wp"].array.numpy = s["original_numpy"]
        s["renderer"]._GL._call = s["original_gl_call"]
    if s.get("error"):
        report["exception"] = s["error"]
    p = report.get("pixels", {})
    assertions = {}
    if all(key in p for key in ("left", "right", "behind", "front")):
        assertions = {
            "cuda_written_pixels_visible": p["left"]["red_pixels"] > 100,
            "device_motion_changes_window_pixels": (
                p["right"]["red_pixels"] > 100 and
                p["right"]["centroid_x"] - p["left"]["centroid_x"] > 20
            ),
            "native_mesh_occludes_particle": p["behind"]["red_pixels"] <= 4,
            "front_particle_visible": p["front"]["red_pixels"] > 100,
            "repeated_explicit_teardown": report.get("teardown_cycles") == 20,
            "no_forbidden_particle_transfers": report["forbidden_transfers"] == 0,
            "resource_closed": bool(report.get("batch_after_close", {}).get("closed")),
        }
    report["assertions"] = assertions
    report["pass"] = bool(assertions) and all(assertions.values()) and not s.get("error")
    report["draws"] = s.get("draws", 0)
    path = s["paths"].report_dir / REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CUDA_GL_REPORT=" + str(path), flush=True)
    bpy.ops.wm.quit_blender()


def _setup():
    import bpy
    import gpu
    from mathutils import Quaternion

    s = _STATE
    module, wp, paths = _environment(**s["args"])
    renderer = importlib.import_module(module.__name__ + ".cuda_gl")
    s.update(wp=wp, paths=paths, renderer=renderer)
    s["report"].update(
        blender=bpy.app.version_string, warp=wp.__version__,
        device=wp.get_device("cuda:0").name, gpu_backend=gpu.platform.backend_type_get(),
        code_fingerprint=extension_code_fingerprint(module),
        module_file=str(Path(module.__file__).resolve()), background=bpy.app.background,
    )
    if bpy.app.background or gpu.platform.backend_type_get() != "OPENGL":
        raise RuntimeError("real OpenGL Blender window required")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    s["cube"] = bpy.context.object
    s["cube"].hide_set(True)
    bpy.ops.object.select_all(action="DESELECT")
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    s["view"] = (window, area, region)
    space = area.spaces.active
    space.shading.type = "SOLID"
    space.shading.background_type = "VIEWPORT"
    space.shading.background_color = (0.0, 0.0, 0.0)
    space.overlay.show_overlays = False
    space.show_gizmo = False
    space.show_region_toolbar = space.show_region_ui = False
    view = space.region_3d
    view.view_perspective = "ORTHO"
    view.view_location = (0.0, 0.0, 0.0)
    view.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    view.view_distance = 8.0
    view.update()
    s["original_copy"], s["original_numpy"] = wp.copy, wp.array.numpy
    s["original_gl_call"] = renderer._GL._call

    def deny(message):
        s["report"]["forbidden_transfers"] += 1
        raise AssertionError(message)

    def guarded_copy(dest, src, *args, **kwargs):
        if src.device.is_cuda and dest.device.is_cpu:
            deny("particle CUDA-to-host copy forbidden in viewport proof")
        return s["original_copy"](dest, src, *args, **kwargs)

    def guarded_numpy(array, *args, **kwargs):
        if array.device.is_cuda:
            deny("CUDA array.numpy forbidden in viewport proof")
        return s["original_numpy"](array, *args, **kwargs)

    def guarded_gl(gl, name, result, args, *values):
        if name in ("glBufferSubData", "glGetBufferSubData"):
            deny(name + " forbidden in viewport proof")
        if name == "glBufferData" and values[2] is not None:
            deny("CPU vertex upload forbidden in viewport proof")
        return s["original_gl_call"](gl, name, result, args, *values)

    wp.copy, wp.array.numpy, renderer._GL._call = guarded_copy, guarded_numpy, guarded_gl
    s["guarded"] = True
    s.update(phase="left", phase_draws=0, phase_time=time.monotonic())
    s["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
    area.tag_redraw()


def _tick():
    s = _STATE
    try:
        if not s.get("started"):
            s["started"] = True
            _setup()
        elif s.get("error") or s.get("closed"):
            _finish()
            return None
        elif time.monotonic() - s["begin"] > 180:
            raise TimeoutError("CUDA/GL window proof exceeded 180 seconds")
        elif s["draws"] - s["phase_draws"] >= 2 and time.monotonic()-s["phase_time"] > 0.6:
            phase = s["phase"]
            s["report"]["pixels"][phase] = _capture(phase)
            next_phase = {"left": "right", "right": "behind", "behind": "front",
                          "front": "cleanup"}[phase]
            s.update(phase=next_phase, phase_draws=s["draws"], phase_time=time.monotonic())
            s["cube"].hide_set(next_phase not in ("behind", "front"))
        s["view"][1].tag_redraw()
        return 0.25
    except Exception:
        s["error"] = traceback.format_exc()
        if "paths" not in s:
            s["paths"] = resolve_probe_paths(
                repo_root=s["args"]["repo_root"], report_dir=s["args"]["report_dir"],
                warp_cache_dir=s["args"]["warp_cache_dir"]
            )
        _finish()
        return None


def schedule_gui_run(*, module_name, repo_root, report_dir, warp_cache_dir):
    import bpy

    _STATE.clear()
    _STATE.update(
        args=dict(module_name=module_name, repo_root=repo_root, report_dir=report_dir,
                  warp_cache_dir=warp_cache_dir), begin=time.monotonic(), draws=0,
        report=dict(schema="astro-collision-lab-cuda-gl-pixels-v1", pixels={}, forbidden_transfers=0,
                    generated_at_utc=datetime.now(UTC).isoformat(),
                    scope="owned viewport interop; CPU simulation history is outside this probe"),
    )
    bpy.app.timers.register(_tick, first_interval=2.0)
