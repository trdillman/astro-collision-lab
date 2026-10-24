"""Controller lifecycle regressions using the real runtime and no Blender/GPU.

The fakes replace external process, simulation, and display boundaries. Authoring
extraction, restart decisions, pending actions, stepping, timing, and cleanup are
the production controller code. Every replaced module is restored after a test.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid


_RUNTIME = (
    Path(__file__).resolve().parents[1]
    / "extension" / "astro_collision_lab" / "runtime.py"
)


class Matrix:
    translation = (0.0, 0.0, 0.0)

    def to_scale(self):
        return (1.0, 1.0, 1.0)

    def to_quaternion(self):
        return self

    def to_matrix(self):
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class Source:
    def __init__(self, name="Target", *, hidden=False):
        self.name = name
        self.parent = None
        self.matrix_world = Matrix()
        self.bound_box = tuple(
            (x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)
        )
        self.acl_body = types.SimpleNamespace(
            kind="SPHERE", particle_count=100_000, mass=1.0, scale_length=.2,toomre_q=1.5,rotation_fraction=1.,seed=0,
            velocity=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0), material="ROCK",
        )
        self.hidden = hidden

    def hide_get(self, *, view_layer=None):
        return self.hidden

    def hide_set(self, value, *, view_layer=None):
        self.hidden = value


class ViewLayer:
    def __init__(self, scene):
        self.scene = scene

    @property
    def objects(self):
        return {source.name: source for source in self.scene.active_objects}

    def update(self):
        pass


class Timers:
    def __init__(self):
        self.registered = set()

    def is_registered(self, callback):
        return callback in self.registered

    def register(self, callback, **_kwargs):
        self.registered.add(callback)

    def unregister(self, callback):
        self.registered.discard(callback)


class Worker:
    def __init__(self):
        self.result = None
        self.terminated = False

    def poll(self):
        return self.result

    def terminate(self):
        self.terminated = True
        self.result = -1


class ControllerBehavior(unittest.TestCase):
    def setUp(self):
        # Keep NumPy in the module snapshot: real preflight imports it, and the
        # scoped package cleanup must not unload and reinitialize its C module.
        importlib.import_module("numpy")
        self.directory = tempfile.TemporaryDirectory(prefix="acl-controller-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"
        self.simulations = []
        self.workers = []
        self.fail_initialization = False
        self.scene = types.SimpleNamespace(
            name="Controller scene", name_full="Controller scene",
            active_objects=[Source()],
            acl_settings=types.SimpleNamespace(
                source_collection=None, authoring_status="Preview",
                mode="IMPACT", gravitational_constant=1.0,
                seconds_per_frame=0.005, softening=0.03, theta=0.6, substeps=1,seed=0,bulk_modulus=10.,shear_modulus=.5,yield_stress=.02,melt_energy=5.,specific_internal_energy=0.,courant_number=.3,
            ),
        )
        self.scene.view_layers = [ViewLayer(self.scene)]
        self.context = types.SimpleNamespace(scene=self.scene)

        # The package name is private to this test. In particular, importing the
        # runtime never imports the actual extension or its vendored solvers.
        package_name = "_acl_controller_test_" + uuid.uuid4().hex
        package = types.ModuleType(package_name)
        package.__path__ = [str(_RUNTIME.parent)]
        viewport = types.ModuleType(package_name + ".viewport")
        viewport.attach = lambda owner: None
        viewport.retire = lambda: None
        viewport.request_redraw = lambda: None
        viewport.shutdown = lambda: None
        viewport.status = lambda: {"test_boundary": True}
        ui = types.ModuleType(package_name + ".ui")
        ui.source_objects = lambda scene: tuple(scene.active_objects)
        harness = self

        class Simulation:
            def __init__(self, config):
                self.config = config
                self.count = config.particle_count
                self.physical_time = 0.0
                self.steps = 0
                self.initialized = False
                self.closed = False
                self.stats = {"last_step_ms": 1.0}
                harness.simulations.append(self)

            def initialize(self, bodies):
                if harness.fail_initialization:
                    harness.fail_initialization = False
                    raise RuntimeError("injected initialization failure")
                self.bodies = tuple(bodies)
                self.initialized = True

            def step(self):
                if not self.initialized or self.closed:
                    raise RuntimeError("attempted to step an unavailable simulation")
                self.steps += 1
                self.physical_time += self.config.dt

            def close(self):
                self.closed = True

        bpy = types.ModuleType("bpy")
        app = types.ModuleType("bpy.app")
        handlers = types.ModuleType("bpy.app.handlers")
        handlers.persistent = lambda callback: callback
        for field in ("load_pre", "load_post", "save_pre", "save_post", "save_post_fail",
                      "undo_pre", "undo_post", "redo_pre", "redo_post"):
            setattr(handlers, field, [])
        app.handlers = handlers
        app.timers = Timers()
        app.binary_path = "/unused/blender"
        app.version = (5, 1, 0)
        app.build_hash = b"test-build"
        app.background = False
        bpy.app = app
        bpy.context = self.context
        bpy.data = types.SimpleNamespace(scenes=[self.scene])
        warp = types.ModuleType("warp")
        warp.__version__ = "1.15.0"
        warp.__file__ = str(self.root / "warp" / "__init__.py")
        warp.config = types.SimpleNamespace(kernel_cache_dir=None)
        warp.get_device = lambda alias: types.SimpleNamespace(
            name="Test GPU", arch="test-arch", uuid="test-device-uuid")
        override = patch.dict(sys.modules, {
            package_name: package,
            package_name + ".viewport": viewport,
            package_name + ".dependencies": types.SimpleNamespace(activate=lambda **kw: True),
            package_name + ".ui": ui,
            "bpy": bpy, "bpy.app": app, "bpy.app.handlers": handlers,
            "warp": warp,
        })
        override.start()
        self.addCleanup(override.stop)
        # Keep the real descriptor and CPU preflight validation. Only the live
        # GPU simulation boundary is replaced; validation never initializes CUDA.
        simulation = importlib.import_module(package_name + ".simulation")
        simulation.AstroCollisionSimulation = Simulation
        spec = importlib.util.spec_from_file_location(package_name + ".runtime", _RUNTIME)
        self.runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.runtime
        spec.loader.exec_module(self.runtime)
        self.contract = importlib.import_module(package_name + ".interaction")
        self.warp, self.app, self.package_name = warp, app, package_name
        self.runtime._cache_dir = lambda: self.cache
        # Never launch a process; retain real JSON authoring and pending-state
        # handling while using a temporary, test-owned request directory.
        self.runtime.tempfile = types.SimpleNamespace(
            mkdtemp=lambda **kwargs: tempfile.mkdtemp(dir=self.root, **kwargs)
        )
        self.runtime.subprocess = types.SimpleNamespace(Popen=self._spawn, STDOUT=-2)
        self.controller = self.runtime.controller
        self.addCleanup(self.controller.reset)
        self._write_ready_receipt()

    def _spawn(self, *_args, **_kwargs):
        worker = Worker()
        self.workers.append(worker)
        return worker

    def _finish_worker(self):
        marker = self.controller.pending[5]
        marker.write_text(json.dumps(self.contract.ready_marker(self.controller.pending_identity)),
                          encoding="utf-8")
        self.workers[-1].result = 0
        return self.runtime._tick()

    def _write_ready_receipt(self, text=None):
        config, bodies, _objects = self.runtime.authored_scene(self.scene)
        identity = self.contract.warmup_identity(
            signature=self.contract.physical_signature(config, bodies),
            code_fingerprint=self.contract.code_fingerprint(_RUNTIME.parent),
            runtime=self.contract.runtime_identity(self.warp, self.app, self.package_name),
        )
        self.cache.mkdir(parents=True, exist_ok=True)
        marker = self.cache / (identity["request_id"] + ".ready.json")
        marker.write_text(json.dumps(self.contract.ready_marker(identity)) if text is None else text,
                          encoding="utf-8")
        return marker

    def _clear_ready_receipts(self):
        for marker in self.cache.glob("*.ready.json"):
            marker.unlink()

    def _present(self, finished=1.0):
        controller = self.controller
        controller.on_presented(
            controller.revision, controller.simulation.count,
            finished - 0.002, finished - 0.001, finished,
        )

    def test_invalid_marker_cannot_skip_preparation(self):
        for text in ('{"pass": false}', 'truncated-json'):
            with self.subTest(marker=text):
                self._write_ready_receipt(text)
                self.runtime.start(self.context)
                self.assertIsNone(self.controller.simulation)
                self.assertIsNotNone(self.controller.warming)
                self.controller.reset()

    def test_first_single_step_initializes_and_advances_exactly_once(self):
        self.runtime.step(self.context)
        simulation = self.controller.simulation
        self.assertIsNotNone(simulation)
        self.assertEqual(simulation.steps, 0, "initial revision must be displayed first")
        self.runtime._tick()
        self.assertEqual(simulation.steps, 0)
        self._present(.98)
        self.runtime._tick()
        self.assertEqual(simulation.steps, 1)
        self.assertEqual(self.controller.status()["frame"], 1)
        self.assertAlmostEqual(simulation.physical_time, self.scene.acl_settings.seconds_per_frame)
        self.assertFalse(self.controller.playing)
        self._present()
        self.runtime._tick()
        self.assertEqual(simulation.steps, 1)

    def test_first_single_step_survives_asynchronous_warmup(self):
        self._clear_ready_receipts()
        self.runtime.step(self.context)
        self.assertIsNone(self.controller.simulation)
        self.assertEqual(len(self.workers), 1)
        self._finish_worker()
        self.assertIsNotNone(self.controller.simulation, self.controller.error)
        self.assertEqual(self.controller.simulation.steps, 0)
        self._present()
        self.runtime._tick()
        self.assertEqual(self.controller.simulation.steps, 1)
        self.assertFalse(self.controller.playing)

    def test_pause_during_warmup_stays_paused_after_initialization(self):
        self._clear_ready_receipts()
        self.runtime.start(self.context)
        self.assertIsNotNone(self.controller.warming)
        self.runtime.pause(self.context)
        self._finish_worker()
        self.assertIsNotNone(self.controller.simulation, self.controller.error)
        self.assertEqual(self.controller.simulation.steps, 0)
        self.assertFalse(self.controller.playing)
        self._present()
        self.runtime._tick()
        self.assertEqual(self.controller.simulation.steps, 0)
        self.assertFalse(self.workers[0].terminated)

    def test_source_changes_during_warmup_initialize_latest_authored_scene(self):
        self._clear_ready_receipts()
        self.runtime.start(self.context)
        self.scene.active_objects[0].acl_body.mass = 2.5
        self._finish_worker()
        self.assertIsNone(self.controller.simulation,
                          "a receipt for old inputs cannot authorize the latest inputs")
        self.assertEqual(len(self.workers), 2)
        self._finish_worker()
        self.assertIsNotNone(self.controller.simulation, self.controller.error)
        self.assertEqual(self.controller.simulation.bodies[0].mass, 2.5)
        self.assertTrue(self.controller.playing)

    def test_reset_clears_visible_measurements_and_restores_source_visibility(self):
        source = self.scene.active_objects[0]
        self.runtime.step(self.context)
        self._present(.98)
        self.runtime._tick()
        self._present(1.0)
        self.runtime.step(self.context)
        self.runtime._tick()
        self._present(1.02)
        simulation = self.controller.simulation
        previous = self.controller.status()
        self.assertGreater(previous["fps"], 0)
        self.assertEqual(previous["frame"], 2)
        self.assertTrue(source.hidden)
        self.runtime.reset(self.context)
        current = self.controller.status()
        for field in ("particles", "active_count", "fps", "physical_time", "frame"):
            self.assertEqual(current[field], 0, field)
        self.assertFalse(current["running"])
        self.assertEqual(self.controller.measurements, [])
        self.assertFalse(source.hidden)
        self.assertTrue(simulation.closed)

    def test_initialization_failure_can_be_retried_without_resuming_partial_state(self):
        self.fail_initialization = True
        with self.assertRaisesRegex(RuntimeError, "injected initialization failure"):
            self.runtime.start(self.context)
        self.assertIsNone(self.controller.simulation)
        self.assertTrue(self.simulations[0].closed)
        self.assertFalse(self.scene.active_objects[0].hidden)
        self.runtime.start(self.context)
        self.assertEqual(len(self.simulations), 2)
        self.assertIs(self.controller.simulation, self.simulations[1])
        self.assertTrue(self.controller.simulation.initialized)
        self.assertTrue(self.controller.playing)

    def test_new_sources_and_edited_physics_rebuild_but_unchanged_start_resumes(self):
        self.runtime.start(self.context)
        original = self.controller.simulation
        original_source = self.scene.active_objects[0]
        self.runtime.pause(self.context)
        self.runtime.start(self.context)
        self.assertIs(self.controller.simulation, original)

        self.scene.active_objects = [Source("New source group")]
        self._write_ready_receipt()
        self.runtime.start(self.context)
        replacement = self.controller.simulation
        self.assertIsNot(replacement, original)
        self.assertTrue(original.closed)
        self.assertFalse(original_source.hidden)
        self.assertEqual(replacement.bodies[0].name, "New source group")

        self.scene.acl_settings.seconds_per_frame = 0.02
        self._write_ready_receipt()
        self.runtime.start(self.context)
        edited_config = self.controller.simulation
        self.assertIsNot(edited_config, replacement)
        self.assertTrue(replacement.closed)
        self.assertEqual(edited_config.config.dt, 0.02)

        self.scene.active_objects[0].acl_body.mass = 2.0
        self._write_ready_receipt()
        self.runtime.start(self.context)
        self.assertIsNot(self.controller.simulation, edited_config)
        self.assertTrue(edited_config.closed)
        self.assertEqual(self.controller.simulation.bodies[0].mass, 2.0)

    def test_single_step_waits_for_pending_display_before_advancing_again(self):
        self.runtime.step(self.context)
        simulation = self.controller.simulation
        self.assertEqual(simulation.steps, 0)
        self._present(.98)
        self.runtime._tick()
        self.assertEqual(simulation.steps, 1)
        self.runtime.step(self.context)
        self.assertEqual(simulation.steps, 1, "an unpresented revision must not be skipped")
        self.runtime._tick()
        self.assertEqual(simulation.steps, 1)
        self._present()
        self.runtime._tick()
        self.assertEqual(simulation.steps, 2)
        self.assertFalse(self.controller.playing)
        self._present(1.02)
        self.runtime._tick()
        self.assertEqual(simulation.steps, 2, "queued single-step must be consumed exactly once")


if __name__ == "__main__":
    unittest.main()
