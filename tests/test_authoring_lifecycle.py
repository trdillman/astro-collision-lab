"""Authoring identity and recovery contracts; no native Blender/GPU claim.

The real flush and preview snapshot paths run against small in-memory scenes.
Only source extraction, Blender services and the simulation owner are replaced.
"""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

# Keep the real numerical runtime outside the temporary module registry below.
import numpy
import pytest

from astro_collision_lab.interaction import physical_signature
from astro_collision_lab.simulation import BodySpec, SimulationConfig


PACKAGE = Path(__file__).resolve().parents[1] / "extension" / "astro_collision_lab"


class Timers:
    def __init__(self):
        self.registered = set()

    def is_registered(self, function):
        return function in self.registered

    def register(self, function, **_options):
        self.registered.add(function)

    def unregister(self, function):
        self.registered.discard(function)


class Object:
    def __init__(self, name, color=(.1, .3, .8, 1.)):
        self.name = name
        self.color = color

    def as_pointer(self):
        return id(self)


class Scene:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.invalid = None
        self.acl_settings = SimpleNamespace(
            authoring_status="Preview", source_collection=object(),
        )

    def as_pointer(self):
        return id(self)


@pytest.fixture
def lifecycle():
    fake_bpy = ModuleType("bpy")
    fake_bpy.app = SimpleNamespace(timers=Timers(), background=True)
    fake_bpy.types = SimpleNamespace(Scene=Scene, Object=Object)
    package_name = "acl_authoring_lifecycle_test"
    package = ModuleType(package_name)
    package.__path__ = [str(PACKAGE)]
    ui = ModuleType(package_name + ".ui")
    ui.source_objects = lambda scene: scene.snapshot[2]
    runtime = ModuleType(package_name + ".runtime")
    runtime._signature = physical_signature
    owner = SimpleNamespace(
        scene=None, intent="STOPPED", simulation=None, signature=None,
        sources=(), error="", message="",
    )

    def reset(*, preserve_intent=False):
        owner.simulation = None
        owner.signature = None
        owner.sources = ()
        owner.error = ""
        if not preserve_intent:
            owner.scene = None
            owner.intent = "STOPPED"

    owner.reset = Mock(side_effect=reset)
    owner.begin = Mock()
    runtime.controller = owner

    def authored_scene(scene):
        if scene.invalid is not None:
            raise ValueError(scene.invalid)
        return scene.snapshot

    runtime.authored_scene = Mock(side_effect=authored_scene)
    viewport = ModuleType(package_name + ".viewport")
    viewport.request_redraw = Mock()
    spec = importlib.util.spec_from_file_location(
        package_name + ".authoring", PACKAGE / "authoring.py",
    )
    module = importlib.util.module_from_spec(spec)
    clock = SimpleNamespace(now=10.)
    modules = {
        "bpy": fake_bpy, package_name: package,
        package_name + ".ui": ui,
        package_name + ".runtime": runtime,
        package_name + ".viewport": viewport,
        package_name + ".authoring": module,
    }
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        with patch.object(module, "synchronize"), patch.object(
            module.time, "monotonic", side_effect=lambda: clock.now,
        ):
            config = SimulationConfig(mode="GALAXY", particle_count=32)
            body = BodySpec("Source A", kind="HALO", particle_count=32)
            source = Object(body.name)
            scene = Scene((config, (body,), (source,)))
            module.accept_snapshot(scene, *scene.snapshot)
            yield SimpleNamespace(
                authoring=module, runtime=runtime, controller=owner,
                clock=clock, scene=scene, source=source, body=body,
            )


def queue_edit(env):
    scene = env.scene
    env.authoring._PENDING[scene.as_pointer()] = (scene, env.clock.now)


def make_live(env, intent="PLAY"):
    owner = env.controller
    owner.scene = env.scene
    owner.intent = intent
    owner.simulation = object()
    owner.signature = physical_signature(*env.scene.snapshot[:2])
    owner.sources = env.scene.snapshot[2]


@pytest.mark.parametrize("change", ["replace_source", "switch_identical_collection"])
def test_replaced_sources_refresh_preview_and_rebuild_owned_physics(lifecycle, change):
    env = lifecycle
    make_live(env)
    new_source = Object("Replacement", (.9, .2, .1, .6))
    config, bodies, _objects = env.scene.snapshot
    env.scene.snapshot = (config, (replace(bodies[0], name=new_source.name),), (new_source,))
    if change == "switch_identical_collection":
        env.scene.acl_settings.source_collection = object()
    queue_edit(env)

    env.authoring.flush()

    entries = env.authoring._PREVIEWS[env.scene.as_pointer()]
    assert set(entries) == {new_source.as_pointer()}
    assert entries[new_source.as_pointer()]["obj"] is new_source
    assert env.controller.simulation is None
    assert env.controller.intent == "PLAY"
    assert env.scene.as_pointer() in env.authoring._RESTARTS
    env.controller.reset.assert_called_once_with(preserve_intent=True)
    env.controller.begin.assert_not_called()


def test_equal_physics_refresh_preserves_pending_rebuild_deadline_and_cached_batch(lifecycle):
    env = lifecycle
    env.controller.scene = env.scene
    env.controller.intent = "PLAY"
    pointer = env.scene.as_pointer()
    deadline = env.clock.now + .35
    env.authoring._RESTARTS[pointer] = (env.scene, deadline)
    entry = env.authoring._PREVIEWS[pointer][env.source.as_pointer()]
    batch = entry["batch"] = object()
    points = entry["points"]
    env.source.color = (.8, .4, .1, 1.)
    queue_edit(env)

    assert env.authoring.flush() is not None

    assert env.authoring._RESTARTS[pointer] == (env.scene, deadline)
    updated = env.authoring._PREVIEWS[pointer][env.source.as_pointer()]
    assert updated["batch"] is batch
    assert updated["points"] is points
    env.controller.begin.assert_not_called()
    env.clock.now = deadline + .01
    env.authoring.flush()
    env.controller.begin.assert_called_once()
    assert env.controller.begin.call_args.kwargs == {"play": True}


@pytest.mark.parametrize("intent", ["PLAY", "PAUSE", "STEP"])
def test_invalid_then_valid_edit_recovers_latest_intent_without_retry_loop(lifecycle, intent):
    env = lifecycle
    make_live(env, intent)
    env.scene.invalid = "invalid authored radius"
    queue_edit(env)

    assert env.authoring.flush() is None

    assert env.controller.scene is env.scene
    assert env.controller.intent == intent
    assert env.controller.simulation is None
    assert env.scene.acl_settings.authoring_status.startswith("Invalid settings:")
    assert env.scene.as_pointer() not in env.authoring._RESTARTS
    calls = env.runtime.authored_scene.call_count
    assert env.authoring.flush() is None
    assert env.runtime.authored_scene.call_count == calls
    env.controller.begin.assert_not_called()

    env.scene.invalid = None
    queue_edit(env)
    assert env.authoring.flush() is not None
    assert env.controller.intent == intent
    assert env.scene.acl_settings.authoring_status.startswith("Preview")
    env.clock.now += .36
    env.authoring.flush()
    env.controller.begin.assert_called_once()
    assert env.controller.begin.call_args.args[0].scene is env.scene
    assert env.controller.begin.call_args.kwargs == {"play": intent == "PLAY"}


def test_empty_source_group_clears_preview_and_retains_recoverable_intent(lifecycle):
    env = lifecycle
    make_live(env)
    env.scene.snapshot = (env.scene.snapshot[0], (), ())
    queue_edit(env)

    assert env.authoring.flush() is None

    assert env.scene.acl_settings.authoring_status.startswith("Preview")
    assert "Invalid" not in env.scene.acl_settings.authoring_status
    assert env.scene.as_pointer() not in env.authoring._PREVIEWS
    assert env.scene.as_pointer() not in env.authoring._SIGNATURES
    assert env.controller.intent == "PLAY"
    assert env.controller.simulation is None
    env.runtime.authored_scene.assert_not_called()
    env.controller.begin.assert_not_called()
