"""Main-thread live controller for the independent add-on.

Each step waits for the preceding revision to reach the viewport. Repeated
redraws never count as newly simulated frames. There is no particle history cache.
"""

import dataclasses
import json
import os
import subprocess
import tempfile
import time
import uuid
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy

from . import viewport
from .simulation import AstroCollisionSimulation, BodySpec, SimulationConfig, validate_scene
from .interaction import (physical_signature, equilibrium_signature, resolve_mode,
                          code_fingerprint, marker_valid, warmup_identity,
                          runtime_identity, same_sources)


def authored_scene(scene, *, validate=True):
    """Extract current inputs; repair tools may inspect invalid sampling first.

    Runtime entry points use the strict default. The sampling repair operator
    alone defers body validation while computing a corrected particle budget.
    """
    from .ui import source_objects

    props = scene.acl_settings
    objects = tuple(source_objects(scene))
    bodies = []
    for obj in objects:
        p = obj.acl_body
        matrix = obj.matrix_world
        rotation = matrix.to_quaternion().to_matrix()
        scale = matrix.to_scale()
        coords = tuple(obj.bound_box)
        extent = [max(c[i] for c in coords)-min(c[i] for c in coords) for i in range(3)]
        dims = [abs(float(scale[i])) * extent[i] for i in range(3)]
        radius = max(dims[:2] if p.kind == "DISK" else dims) * 0.5
        thickness=max(1e-6,dims[2])
        if getattr(p,"native_primitive",False):
            radius=p.radius*max(abs(float(v)) for v in (scale[:2] if p.kind=="DISK" else scale))
            thickness=p.thickness*abs(float(scale[2])) if p.kind=="DISK" else 2*p.radius*abs(float(scale[2]))
        bodies.append(BodySpec(
            name=obj.name, kind=p.kind, particle_count=p.particle_count,
            mass=p.mass, radius=radius, position=tuple(matrix.translation),
            velocity=tuple(p.velocity), spin=tuple(p.spin),
            rotation=tuple(tuple(float(rotation[r][c]) for c in range(3)) for r in range(3)),
            thickness=max(1e-6,thickness), material=p.material,
            galaxy_id=(obj.parent.name_full if obj.parent and obj.parent.acl_body.is_galaxy else ""),
            scale_length=p.scale_length*max(abs(float(scale[0])),abs(float(scale[1]))),
            toomre_q=p.toomre_q, rotation_fraction=p.rotation_fraction,seed=p.seed,
        ))
    collection = props.source_collection
    mode = resolve_mode(props.mode, bodies,
                         bool(collection and collection.get("acl_preset") == "GALAXY"))
    if mode != props.mode:
        props.mode = mode
    count = sum(body.particle_count for body in bodies)
    config = SimulationConfig(
        mode=mode, particle_count=count, gravitational_constant=props.gravitational_constant,
        dt=props.seconds_per_frame, softening=props.softening, theta=props.theta,
        substeps=props.substeps if mode=="GALAXY" else 1,seed=props.seed,
        bulk_modulus=props.bulk_modulus,shear_modulus=props.shear_modulus,
        yield_stress=props.yield_stress,melt_energy=props.melt_energy,
        specific_internal_energy=props.specific_internal_energy,courant_number=props.courant_number,
    )
    if validate:
        validate_scene(config, bodies)
    return config, tuple(bodies), objects


def _cache_dir():
    return Path(os.environ.get("ACL_KERNEL_CACHE", str(Path(tempfile.gettempdir()) / "acl-kc")))


def _signature(config, bodies):
    return physical_signature(config, bodies)


def _editor():
    return sys.modules.get(__package__ + ".authoring")


_RETIRED_WORKERS = []


def _retire_worker(process):
    # Cancellation never waits for a background Blender on the UI thread.
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    _RETIRED_WORKERS.append((process, time.monotonic()))
    if not bpy.app.timers.is_registered(_reap_workers):
        bpy.app.timers.register(_reap_workers, first_interval=.1)


def _reap_workers():
    for process, started in tuple(_RETIRED_WORKERS):
        if process.poll() is not None:
            _RETIRED_WORKERS.remove((process, started))
        elif time.monotonic() - started > 2:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    return .1 if _RETIRED_WORKERS else None


class Controller:
    def __init__(self):
        self.simulation = None
        self.scene = None
        self.playing = False
        self.error = ""
        self.revision = 0
        self.presented_revision = -1
        self.run_id = ""
        self.measurement_sync = False
        self.measurements = []
        self.hidden = []
        self.warming = None
        self.warm_log = None
        self.pending = None
        self.last_presented = None
        self.step_started = None
        self.signature = None
        self.single_step_pending = False
        self.message = "Author primitives, then Start"
        self.intent = "STOPPED"
        self.sources = ()
        self.request_key = None
        self.pending_identity = None
        self.deferred = False
        self.failed = False

    def _initialize(self, scene, config, bodies, objects, play):
        from .dependencies import activate
        activate(required=True)
        import warp as wp

        wp.config.kernel_cache_dir = str(_cache_dir())
        if wp.__version__ != "1.15.0":
            raise RuntimeError("Install the certified Warp 1.15.0 dependency in the isolated extension environment")
        simulation = AstroCollisionSimulation(config)
        try:
            simulation.initialize(bodies)
        except Exception:
            simulation.close()
            raise
        self.simulation = simulation
        self.signature = _signature(config, bodies)
        self.scene, self.playing = scene, play
        self.sources = tuple(objects)
        self.single_step_pending = self.intent == "STEP"
        self.failed = False
        self.error = ""
        self.run_id = uuid.uuid4().hex
        self.revision, self.presented_revision = 0, -1
        self.measurements.clear()
        self.last_presented = None
        for obj in objects:
            if getattr(obj.acl_body,"native_primitive",False):continue
            for layer in scene.view_layers:
                if obj.name not in layer.objects:
                    continue
                self.hidden.append((obj, layer, obj.hide_get(view_layer=layer)))
                obj.hide_set(True, view_layer=layer)
        self.message = "Live" if play else "Paused"
        if config.mode=="GALAXY":scene.acl_settings.authoring_status="Validated isolated model"
        viewport.attach(self)

    def begin(self, context, *, play=True):
        scene = context.scene
        if self.scene is not None and self.scene != scene:
            self.reset()
        self.scene = scene
        # Calls from the debounce queue use the retained intent. Direct callers
        # retain backward compatibility with begin(play=...).
        if self.intent != "STEP" or play:
            self.intent = "PLAY" if play else "PAUSE"
        editor = _editor()
        try:
            if editor is not None:
                editor.synchronize(scene)
            config, bodies, objects = authored_scene(scene)
        except Exception as error:
            self.reset(preserve_intent=True)
            self.error = str(error)
            self.message = "Invalid settings: " + self.error
            scene.acl_settings.authoring_status = self.message
            raise
        signature = _signature(config, bodies)
        if editor is not None:
            editor.accept_snapshot(scene, config, bodies, objects)
        play = self.intent == "PLAY"
        self.single_step_pending = self.intent == "STEP"
        if self.warming is not None and self.pending is not None:
            old_scene, old_config, old_bodies, _objects, _play, marker, owned = self.pending
            if old_scene == scene and _signature(old_config, old_bodies) == signature:
                self.pending = (scene, config, bodies, objects, play, marker, owned)
                _ensure_timer()
                return
        if (self.simulation is not None and not self.failed
                and self.signature == signature and same_sources(self.sources, objects)):
            if not self.playing:
                self.last_presented = None
            self.playing = play
            self.sources = tuple(objects)
            self.message = "Live" if play else "Paused"
            _ensure_timer()
            return
        self.reset(preserve_intent=True)
        self.single_step_pending = self.intent == "STEP"
        # Termination is asynchronous. Keep the UI responsive while ensuring
        # that only one owned preparation worker can use the GPU at a time.
        _reap_workers()
        if _RETIRED_WORKERS:
            self.deferred = True
            self.message = "Waiting for previous preparation to stop"
            scene.acl_settings.authoring_status = self.message
            _ensure_timer()
            return
        from .dependencies import activate
        activate(required=True)
        import warp as wp

        if wp.__version__ != "1.15.0":
            raise RuntimeError("Install the certified Warp 1.15.0 dependency")
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        package = Path(__file__).parent
        # Hash actual code: a developer edit cannot reuse an old passing receipt
        # merely because the installed build manifest has not been regenerated.
        identity = warmup_identity(
            signature=equilibrium_signature(config, bodies) if config.mode == "GALAXY"
                      else signature,
            code_fingerprint=code_fingerprint(package),
            runtime=runtime_identity(wp, bpy.app, __package__),
        )
        key = identity["request_id"]
        self.request_key = key
        self.pending_identity = identity
        marker = cache / (key + ".ready.json")
        if marker_valid(marker, identity):
            self._initialize(scene, config, bodies, objects, play)
        else:
            owned = Path(tempfile.mkdtemp(prefix="acl-warm-"))
            request = owned / "request.json"
            request.write_text(json.dumps({
                "config": dataclasses.asdict(config), "bodies": [dataclasses.asdict(b) for b in bodies],
                "package": str(package), "dependency_path": str(Path(wp.__file__).parent.parent),
                "module_name": __package__, "identity": identity,
                "cache": str(cache), "marker": str(marker),
            }), encoding="utf-8")
            self.warm_log = (owned / "worker.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            for key_name in ("CONFIG", "RESOURCES", "EXTENSIONS", "SCRIPTS", "DATAFILES"):
                path = owned / key_name.lower()
                path.mkdir()
                env["BLENDER_USER_"+key_name] = str(path)
            self.pending = (scene, config, bodies, objects, play, marker, owned)
            try:
                self.warming = subprocess.Popen(
                    [bpy.app.binary_path, "--background", "--factory-startup", "--python",
                     str(package / "warm_worker.py"), "--", str(request)],
                    env=env, stdout=self.warm_log, stderr=subprocess.STDOUT,
                )
            except Exception:
                self.warm_log.close()
                self.warm_log = None
                self.pending = None
                raise
            self.message = "Preparing equilibrium and CUDA kernels" if config.mode == "GALAXY" else "Preparing CUDA kernels"
            scene.acl_settings.authoring_status = self.message
        _ensure_timer()

    def fail(self, error):
        self.error = f"{type(error).__name__}: {error}"
        self.message = "Stopped: " + self.error
        self.playing = False
        self.single_step_pending = False
        self.intent = "STOPPED"
        self.failed = True
        if self.scene is not None:
            self.scene.acl_settings.authoring_status = self.message
        viewport.request_redraw()

    def advance(self):
        if self.simulation is None:
            raise RuntimeError("Initialize the simulation first")
        self.step_started = time.perf_counter()
        self.simulation.step()
        self.revision += 1
        viewport.request_redraw()

    def on_presented(self, revision, count, draw_start, packed, finished):
        if revision <= self.presented_revision:
            return
        if revision > 0:
            interval = None if self.last_presented is None else (finished-self.last_presented)*1000
            self.measurements.append({
                "revision": revision, "active_count": self.simulation.count, "displayed_count": count,
                "physical_time": self.simulation.physical_time,
                "solver_ms": self.simulation.stats.get("last_step_ms"),
                "pack_ms": (packed-draw_start)*1000, "draw_ms": (finished-packed)*1000,
                "frame_interval_ms": interval,
                "end_to_end_ms": None if self.step_started is None else (finished-self.step_started)*1000,
            })
            if len(self.measurements) > 10000:
                del self.measurements[:1000]
        self.last_presented = finished
        self.presented_revision = revision

    def reset(self, *, preserve_intent=False):
        scene, intent = self.scene, self.intent
        self.playing = False
        if self.warming is not None:
            _retire_worker(self.warming)  # Only this controller's owned subprocess.
            self.warming = None
        if self.warm_log is not None:
            self.warm_log.close()
            self.warm_log = None
        self.pending = None
        viewport.retire()
        if self.simulation is not None:
            self.simulation.close()
            self.simulation = None
        for obj, layer, hidden in self.hidden:
            try:
                obj.hide_set(hidden, view_layer=layer)
            except (ReferenceError, RuntimeError):
                pass
        self.hidden.clear()
        self.scene = None
        self.error = ""
        self.revision, self.presented_revision = 0, -1
        self.measurements.clear()
        self.last_presented = self.step_started = None
        self.signature = None
        self.single_step_pending = False
        self.run_id = ""
        self.message = "Reset; edit source objects and Start"
        self.sources = ()
        self.request_key = None
        self.pending_identity = None
        self.deferred = False
        self.failed = False
        self.intent = intent if preserve_intent else "STOPPED"
        self.scene = scene if preserve_intent else None

    def status(self):
        recent = self.measurements[-120:]
        intervals = [m["frame_interval_ms"] for m in recent if m["frame_interval_ms"] is not None]
        rate = 0.0 if not intervals else 1000.0/(sum(intervals)/len(intervals))
        return {"message": self.message, "intent": self.intent, "running": self.playing, "warming": self.warming is not None,
                "particles": 0 if self.simulation is None else self.simulation.count,
                "active_count": 0 if self.simulation is None else self.simulation.count,
                "fps": rate, "physical_time": 0 if self.simulation is None else self.simulation.physical_time,
                "frame": self.revision, "error": self.error,
                "solver": {} if self.simulation is None else self.simulation.stats,
                "viewport": viewport.status()}


controller = Controller()


def _tick():
    try:
        editor = _editor()
        # A source edit must be reconciled before another old-state step or a
        # warmup result is published, including the short preview throttle.
        if controller.scene is not None and editor is not None and editor.is_dirty(controller.scene):
            return .03
        if controller.deferred:
            _reap_workers()
            if _RETIRED_WORKERS:
                return .03
            controller.deferred = False
            controller.begin(SimpleNamespace(scene=controller.scene),
                             play=controller.intent == "PLAY")
            return .03
        if controller.warming is not None:
            result = controller.warming.poll()
            if result is None:
                return .15
            pending = controller.pending
            controller.warming = None
            if controller.warm_log is not None:
                controller.warm_log.close()
                controller.warm_log = None
            scene, config, bodies, _objects, _play, marker, owned = pending
            controller.pending = None
            if result != 0 or not marker_valid(marker, controller.pending_identity):
                raise RuntimeError(f"Preparation failed; see {owned / 'worker.log'}")
            current_config, current_bodies, current_objects = authored_scene(scene)
            if _signature(current_config, current_bodies) != _signature(config, bodies):
                controller.begin(SimpleNamespace(scene=scene), play=controller.intent == "PLAY")
            else:
                controller._initialize(scene, current_config, current_bodies, current_objects,
                                       controller.intent == "PLAY")
        elif (controller.playing or controller.single_step_pending) and controller.simulation is not None:
            config, bodies, _objects = authored_scene(controller.scene)
            if _signature(config, bodies) != controller.signature:
                if editor is not None:
                    editor.changed(None, SimpleNamespace(scene=controller.scene))
                    return .03
                controller.begin(SimpleNamespace(scene=controller.scene), play=controller.intent == "PLAY")
                return .03
            if controller.presented_revision == controller.revision:
                once = controller.single_step_pending
                controller.single_step_pending = False
                controller.advance()
                if once:
                    controller.intent = "PAUSE"
                    controller.playing = False
                    controller.message = "Paused after one simulation step"
        if controller.playing or controller.warming is not None or controller.single_step_pending:
            return .001
        return None
    except Exception as error:
        controller.fail(error)
        return None


def _ensure_timer():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.001)


def _request(context, intent):
    editor = _editor()
    if editor is not None:
        editor.cancel_restart(context.scene)
    if controller.scene is not None and controller.scene != context.scene:
        controller.reset()
    controller.scene = context.scene
    controller.intent = intent
    controller.single_step_pending = intent == "STEP"
    controller.begin(context, play=intent == "PLAY")


def start(context):
    _request(context, "PLAY")


def pause(context):
    editor = _editor()
    if editor is not None:
        editor.pause_restart(None)
    if controller.scene is not None:
        controller.intent = "PAUSE"
    controller.playing = False
    controller.single_step_pending = False
    if controller.pending is not None:
        scene, config, bodies, objects, _play, marker, owned = controller.pending
        controller.pending = (scene, config, bodies, objects, False, marker, owned)
    controller.message = "Paused" if controller.warming is None else "Preparing; will remain paused"
    controller.last_presented = None
    viewport.request_redraw()


def reset(context):
    editor = _editor()
    if editor is not None:
        editor.cancel_restart(None)
    owned_scene = controller.scene
    controller.reset()
    if owned_scene is not None:
        owned_scene.acl_settings.authoring_status = "Preview; press Start to simulate"
    context.scene.acl_settings.authoring_status = "Preview; press Start to simulate"
    viewport.request_redraw()


def step(context):
    _request(context, "STEP")


def shutdown():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    controller.reset()
    viewport.shutdown()


_HISTORY_INTENT = None


def _before_load(_unused):
    global _HISTORY_INTENT
    _HISTORY_INTENT = None
    shutdown()


def _before_history(_unused):
    global _HISTORY_INTENT
    scene = controller.scene
    _HISTORY_INTENT = None if scene is None else (scene.name_full, controller.intent)
    shutdown()


def _after_history(_unused):
    global _HISTORY_INTENT
    saved, _HISTORY_INTENT = _HISTORY_INTENT, None
    if saved is None or saved[1] == "STOPPED":
        return
    scene = bpy.data.scenes.get(saved[0])
    if scene is not None:
        controller.scene, controller.intent = scene, saved[1]
        editor = _editor()
        if editor is not None:
            editor.changed(None, SimpleNamespace(scene=scene))


def _before_save(_unused):
    # Save editable authoring objects; GPU state is deliberately not serialized.
    for obj, layer, hidden in controller.hidden:
        try:
            obj.hide_set(hidden, view_layer=layer)
        except (ReferenceError, RuntimeError):
            pass


def _after_save(_unused):
    if controller.simulation is not None:
        for obj, layer, _hidden in controller.hidden:
            try:
                obj.hide_set(True, view_layer=layer)
            except (ReferenceError, RuntimeError):
                pass


def register_handlers():
    for name, function in (("load_pre", _before_load),
                           ("undo_pre", _before_history), ("redo_pre", _before_history),
                           ("undo_post", _after_history), ("redo_post", _after_history),
                           ("save_pre", _before_save), ("save_post", _after_save)):
        bpy.app.handlers.persistent(function)
        handlers = getattr(bpy.app.handlers, name)
        if function not in handlers:
            handlers.append(function)


def unregister_handlers():
    for name, function in (("load_pre", _before_load),
                           ("undo_pre", _before_history), ("redo_pre", _before_history),
                           ("undo_post", _after_history), ("redo_post", _after_history),
                           ("save_pre", _before_save), ("save_post", _after_save)):
        handlers = getattr(bpy.app.handlers, name)
        if function in handlers:
            handlers.remove(function)
