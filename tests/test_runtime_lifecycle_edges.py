"""Playback lifecycle regressions with no Blender or GPU acceptance claim.

Workers and Blender data are controlled substitutes. Tests exercise the real
controller, public actions, marker validation and registered history handlers.
"""

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Timers:
    def __init__(self):
        self.registered = set()

    def is_registered(self, function):
        return function in self.registered

    def register(self, function, **_options):
        self.registered.add(function)

    def unregister(self, function):
        self.registered.discard(function)

    def pump(self, rounds=4):
        for _ in range(rounds):
            for function in tuple(self.registered):
                if function() is None:
                    self.registered.discard(function)


class Scene:
    def __init__(self, name="Test", *, name_full=None):
        self.name = name
        self.name_full = name if name_full is None else name_full
        self.acl_settings = types.SimpleNamespace(authoring_status="Preview")
        self.invalidated = False

    def as_pointer(self):
        return id(self)

    def __getattribute__(self, name):
        if (name in {"name", "name_full", "acl_settings", "as_pointer"}
                and object.__getattribute__(self, "invalidated")):
            raise ReferenceError("RNA scene was replaced by history")
        return object.__getattribute__(self, name)


class Worker:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminate = Mock()
        self.kill = Mock()

    def poll(self):
        return self.returncode


class RuntimeLifecycleEdges(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="acl-lifecycle-test-")
        self.owned = Path(self.temporary.name)
        self.bpy = types.ModuleType("bpy")
        self.bpy.app = types.SimpleNamespace(
            timers=Timers(), version=(5, 1, 0), binary_path="owned-blender.exe",
            handlers=types.SimpleNamespace(
                persistent=lambda function: function,
                **{name: [] for name in (
                    "load_pre", "undo_pre", "redo_pre", "undo_post", "redo_post",
                    "save_pre", "save_post",
                )},
            ),
        )
        self.bpy.data = types.SimpleNamespace(scenes={})
        self.viewport = types.ModuleType("astro_collision_lab.viewport")
        for name in ("retire", "request_redraw", "shutdown", "attach"):
            setattr(self.viewport, name, Mock())
        self.viewport.status = dict
        self.editor = types.ModuleType("astro_collision_lab.authoring")
        for name in (
            "cancel_restart", "pause_restart", "synchronize", "accept_snapshot", "changed",
        ):
            setattr(self.editor, name, Mock())
        self.editor.is_dirty = lambda _scene: False
        dependencies = types.ModuleType("astro_collision_lab.dependencies")
        dependencies.activate = Mock()
        warp = types.ModuleType("warp")
        warp.__version__ = "1.15.0"
        warp.__file__ = str(self.owned / "warp" / "__init__.py")
        warp.config = types.SimpleNamespace()
        warp.get_device = lambda _device: types.SimpleNamespace(name="Test CUDA", arch=89)
        self.modules = patch.dict(sys.modules, {
            "bpy": self.bpy,
            "astro_collision_lab.viewport": self.viewport,
            "astro_collision_lab.authoring": self.editor,
            "astro_collision_lab.dependencies": dependencies,
            "warp": warp,
        })
        self.modules.start()
        sys.modules.pop("astro_collision_lab.runtime", None)
        self.runtime = importlib.import_module("astro_collision_lab.runtime")
        self.runtime.viewport = self.viewport
        self.controller = self.runtime.controller
        self.scene = Scene()
        self.context = types.SimpleNamespace(scene=self.scene)
        self.bpy.data.scenes[self.scene.name_full] = self.scene
        from astro_collision_lab.simulation import BodySpec, SimulationConfig

        self.config = SimulationConfig(particle_count=10)
        self.bodies = (BodySpec("Source", particle_count=10),)
        from astro_collision_lab.interaction import ready_marker, warmup_identity

        self.identity = warmup_identity(
            signature=self.runtime._signature(self.config, self.bodies),
            code_fingerprint="test-package-fingerprint",
            runtime=("Blender", (5, 1, 0), "Warp", "1.15.0", "test-device"),
        )
        self.receipt = ready_marker(self.identity)
        self.authored = patch.object(
            self.runtime, "authored_scene", return_value=(self.config, self.bodies, ()),
        )
        self.authored.start()
        self.cache = patch.object(self.runtime, "_cache_dir", return_value=self.owned)
        self.cache.start()
        self.simulation = Mock(count=10, physical_time=0.0, stats={})
        self.simulation_factory = patch.object(
            self.runtime, "AstroCollisionSimulation", return_value=self.simulation,
        )
        self.simulation_factory.start()

    def tearDown(self):
        # A replacement-worker test opens a real log even though Popen is mocked.
        # Release controller-owned resources before Windows removes the fixture.
        self.controller.reset()
        self.simulation_factory.stop()
        self.cache.stop()
        self.authored.stop()
        self.modules.stop()
        sys.modules.pop("astro_collision_lab.runtime", None)
        self.temporary.cleanup()

    def prepare_worker(self, *, play=True, receipt=None, returncode=None):
        marker = self.owned / "ready.json"
        if receipt is not None:
            marker.write_text(json.dumps(receipt), encoding="utf-8")
        controller = self.controller
        controller.scene = self.scene
        controller.intent = "PLAY" if play else "PAUSE"
        controller.warming = Worker(returncode)
        controller.warm_log = Mock()
        controller.pending = (
            self.scene, self.config, self.bodies, (), play, marker, self.owned,
        )
        controller.request_key = self.identity["request_id"]
        controller.pending_identity = self.identity
        return controller.warming, marker

    def make_live(self, scene=None):
        controller = self.controller
        controller.scene = self.scene if scene is None else scene
        controller.simulation = self.simulation
        controller.signature = self.runtime._signature(self.config, self.bodies)
        controller.revision = controller.presented_revision = 3
        controller.intent = "PLAY"
        controller.playing = True

    def test_step_during_play_warmup_runs_once_then_stays_paused(self):
        worker, _marker = self.prepare_worker(receipt=self.receipt)
        self.runtime.step(self.context)
        self.assertFalse(self.controller.pending[4])
        self.assertEqual(self.controller.intent, "STEP")
        worker.returncode = 0
        self.runtime._tick()
        self.simulation.step.assert_not_called()
        self.controller.presented_revision = self.controller.revision
        self.runtime._tick()
        self.simulation.step.assert_called_once()
        self.controller.presented_revision = self.controller.revision
        self.runtime._tick()
        self.runtime._tick()
        self.simulation.step.assert_called_once()
        self.assertFalse(self.controller.playing)
        self.assertFalse(self.controller.single_step_pending)
        self.assertEqual(self.controller.intent, "PAUSE")

    def test_step_from_another_scene_does_not_advance_previous_simulation(self):
        previous_scene = Scene("Previous")
        self.make_live(previous_scene)
        with patch.object(self.controller, "begin") as begin:
            self.runtime.step(self.context)
        self.simulation.step.assert_not_called()
        self.simulation.close.assert_called_once()
        self.assertIs(self.controller.scene, self.scene)
        self.assertEqual(self.controller.intent, "STEP")
        begin.assert_called_once()
        self.assertIs(begin.call_args.args[0].scene, self.scene)

    def test_zero_exit_cannot_publish_failed_stale_or_missing_receipt(self):
        receipts = {
            "failed": {**self.receipt, "pass": False},
            "stale": {**self.receipt, "request_id": "another-request"},
            "old-schema": {**self.receipt, "schema": 0},
            "missing": None,
        }
        for label, receipt in receipts.items():
            with self.subTest(label=label):
                marker = self.owned / "ready.json"
                marker.unlink(missing_ok=True)
                self.prepare_worker(receipt=receipt, returncode=0)
                with patch.object(self.controller, "_initialize") as initialize:
                    self.assertIsNone(self.runtime._tick())
                initialize.assert_not_called()
                self.assertFalse(self.controller.playing)
                self.assertEqual(self.controller.intent, "STOPPED")
                self.assertTrue(self.controller.error)

    def test_replacement_waits_until_cancelled_owned_worker_exits(self):
        previous, _marker = self.prepare_worker()
        replacement = Worker()
        make_directory = tempfile.mkdtemp
        with (
            patch.object(self.runtime.subprocess, "Popen", return_value=replacement) as launch,
            patch.object(self.runtime.tempfile, "mkdtemp",
                         side_effect=lambda **options: make_directory(dir=self.owned, **options)),
        ):
            self.controller.reset(preserve_intent=True)
            previous.terminate.assert_called_once()
            self.runtime.start(self.context)
            self.bpy.app.timers.pump(rounds=2)
            self.assertEqual(launch.call_count, 0, "Replacement must wait for the owned worker")
            previous.returncode = -15
            self.bpy.app.timers.pump(rounds=5)
            self.assertEqual(launch.call_count, 1)
            self.assertIs(self.controller.warming, replacement)

    def test_pause_from_another_scene_pauses_all_queued_requests(self):
        self.make_live()
        other = Scene("Other")
        queued = {self.scene: True, other: True}

        def pause_requests(scene):
            for item in tuple(queued) if scene is None else (scene,):
                if item in queued:
                    queued[item] = False

        self.editor.pause_restart.side_effect = pause_requests
        self.runtime.pause(types.SimpleNamespace(scene=other))
        self.assertFalse(self.controller.playing)
        self.assertEqual(self.controller.intent, "PAUSE")
        self.assertFalse(any(queued.values()))

    def test_reset_from_another_scene_cancels_all_queued_requests(self):
        self.make_live()
        other = Scene("Other")
        queued = {self.scene: True, other: True}

        def cancel_requests(scene):
            if scene is None:
                queued.clear()
            else:
                queued.pop(scene, None)

        self.editor.cancel_restart.side_effect = cancel_requests
        self.runtime.reset(types.SimpleNamespace(scene=other))
        self.assertFalse(queued)
        self.assertIsNone(self.controller.scene)
        self.assertIsNone(self.controller.simulation)
        self.assertEqual(self.controller.intent, "STOPPED")
        self.simulation.close.assert_called_once()

    def history_reacquires_scene(self, operation):
        old_scene = Scene("Short", name_full="Scene.Full.Name")
        replacement = Scene("Short", name_full="Scene.Full.Name")
        self.make_live(old_scene)
        self.controller.sources = (object(),)
        self.runtime.register_handlers()
        for handler in getattr(self.bpy.app.handlers, operation + "_pre"):
            handler(None)
        self.assertIsNone(self.controller.scene)
        self.assertEqual(self.controller.sources, ())
        old_scene.invalidated = True
        self.bpy.data.scenes = {replacement.name_full: replacement}
        self.editor.changed.reset_mock()
        for handler in getattr(self.bpy.app.handlers, operation + "_post"):
            handler(None)
        self.assertIs(self.controller.scene, replacement)
        self.assertEqual(self.controller.intent, "PLAY")
        self.assertEqual(self.controller.sources, ())
        self.editor.changed.assert_called_once()
        self.assertIs(self.editor.changed.call_args.args[1].scene, replacement)
        self.simulation.step.assert_not_called()

    def test_undo_reacquires_scene_by_full_name_after_rna_replacement(self):
        self.history_reacquires_scene("undo")

    def test_redo_reacquires_scene_by_full_name_after_rna_replacement(self):
        self.history_reacquires_scene("redo")

    def test_proxy_visibility_is_restored_in_its_owned_view_layer(self):
        proxy = types.SimpleNamespace(name="Proxy", acl_body=types.SimpleNamespace(native_primitive=False),
                                      hide_get=Mock(return_value=False), hide_set=Mock())
        layer = types.SimpleNamespace(objects={"Proxy": proxy})
        self.scene.view_layers = [layer]
        self.controller.intent = "PLAY"
        self.controller._initialize(self.scene, self.config, self.bodies, (proxy,), True)
        proxy.hide_get.assert_called_once_with(view_layer=layer)
        proxy.hide_set.assert_called_once_with(True, view_layer=layer)
        other = Scene("Other")
        self.runtime.reset(types.SimpleNamespace(scene=other))
        self.assertEqual(proxy.hide_set.call_args.args, (False,))
        self.assertIs(proxy.hide_set.call_args.kwargs["view_layer"], layer)


if __name__ == "__main__":
    unittest.main()
