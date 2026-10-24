"""One strict device-to-GL presentation path; no simulation array readbacks."""

import time
from types import SimpleNamespace

import bpy
import numpy as np

from .cuda_gl import CudaGLBatch, current_context_key
from .device_appearance import Appearance, DeviceAppearanceWriter, MATERIAL_PALETTE

_HANDLE = None
_OWNER = None
_BATCHES = {}
_RETIRE = set()
_SHUTDOWN = False
_ERROR = ""


def _matrix(value):
    return tuple(float(value[row][col]) for col in range(4) for row in range(4))


def request_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def attach(owner):
    global _HANDLE, _OWNER, _SHUTDOWN, _ERROR
    _OWNER = owner
    _SHUTDOWN = False
    _ERROR = ""
    if _HANDLE is None and not bpy.app.background:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
    request_redraw()


def retire():
    _RETIRE.update(_BATCHES)
    request_redraw()


def shutdown():
    global _SHUTDOWN, _OWNER
    _SHUTDOWN = True
    _OWNER = None
    retire()
    _collect()


def _collect():
    """Delete GL resources only in the owning context on Blender's main thread."""
    global _HANDLE
    context = current_context_key()
    for key in tuple(_RETIRE):
        if key not in _BATCHES:
            _RETIRE.discard(key)
        elif key[0] == context:
            _BATCHES[key]["batch"].close()
            del _BATCHES[key]
            _RETIRE.discard(key)
    if _SHUTDOWN and not _BATCHES and _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, "WINDOW")
        _HANDLE = None


def _palette(owner, metadata):
    colors = []
    sources = getattr(owner, "sources", ())
    for index, name in enumerate(metadata["source_names"]):
        try:
            obj = sources[index] if index < len(sources) else owner.scene.objects.get(name)
            colors.append(tuple(obj.color) if obj is not None else tuple(MATERIAL_PALETTE[index % len(MATERIAL_PALETTE)]))
        except ReferenceError:
            colors.append(tuple(MATERIAL_PALETTE[index % len(MATERIAL_PALETTE)]))
    return tuple(colors)


def _draw():
    global _ERROR
    try:
        _collect()
        owner = _OWNER
        if owner is None or owner.simulation is None or bpy.context.scene != owner.scene:
            return
        if _SHUTDOWN:
            return
        start = time.perf_counter()
        sim = owner.simulation
        channels = sim.display_channels()
        frame = SimpleNamespace(channels=channels, count=sim.count)
        context = current_context_key()
        if not context:
            raise RuntimeError("Select Blender's OpenGL backend for the CUDA/GL viewport")
        key = (context, owner.run_id)
        entry = _BATCHES.get(key)
        if entry is None:
            batch = CudaGLBatch(channels["positions"].device, sim.count)
            try:
                writer = DeviceAppearanceWriter(channels["positions"].device)
                metadata = sim.authoring_metadata()
                source = np.asarray(metadata["source_index"], dtype=np.int64)
                colors = np.asarray(_palette(owner, metadata), dtype=np.float32)[source]
                writer.set_metadata(stable_ids=metadata["stable_ids"], colors=colors,
                                    revision=owner.run_id)
            except Exception:
                batch.close()
                raise
            entry = _BATCHES[key] = {"batch": batch, "writer": writer, "version": None}
        metadata=sim.authoring_metadata()
        palette = _palette(owner, metadata)
        if entry.get("palette")!=palette:
            source=np.asarray(metadata["source_index"],dtype=np.int64)
            entry["writer"].set_metadata(stable_ids=metadata["stable_ids"],colors=np.asarray(palette,dtype=np.float32)[source],revision=(owner.run_id,palette))
            entry["palette"]=palette
            entry["version"]=None
        settings = owner.scene.acl_settings
        look = Appearance(
            size_source="PHYSICAL" if settings.physical_radius else "UNIFORM",
            color_source={"SOURCE": "SOURCE_OBJECT"}.get(settings.color_by, settings.color_by),
            ramp_low=settings.ramp_min, ramp_high=settings.ramp_max,
        )
        version = (owner.revision, look)
        if version != entry["version"]:
            entry["writer"].validate(frame, look)
            entry["batch"].update(lambda mapped: entry["writer"].write(frame, mapped, look))
            entry["version"] = version
        packed = time.perf_counter()
        region, view = bpy.context.region, bpy.context.region_data
        entry["batch"].draw(
            sim.count, view_projection=_matrix(view.perspective_matrix),
            view_matrix=_matrix(view.view_matrix), viewport_size=(region.width, region.height),
            min_diameter_px=settings.particle_size_px, shaded=True,blend_mode=settings.blend_mode,
        )
        if owner.measurement_sync:
            # Acceptance measures completion of GL work; normal playback never glFinish-es.
            entry["batch"]._gl._call("glFinish", None, ())
        finished = time.perf_counter()
        owner.on_presented(owner.revision, sim.count, start, packed, finished)
        _ERROR = ""
    except Exception as error:
        _ERROR = f"{type(error).__name__}: {error}"
        if _OWNER is not None:
            _OWNER.fail(error)


def status():
    return {
        "backend": "CUDA_OPENGL_OWNED_BUFFER", "registered": _HANDLE is not None,
        "last_error": _ERROR, "pending_context_cleanup": len(_RETIRE),
        "batches": [dict(**entry["batch"].report(), packing=entry["writer"].report())
                    for entry in _BATCHES.values()],
    }
