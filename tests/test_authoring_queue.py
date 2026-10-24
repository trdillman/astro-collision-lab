"""Exercise production event routing with Blender and runtime boundaries replaced."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import uuid
import threading


@contextmanager
def api():
    file=Path(__file__).resolve().parents[1]/'extension/astro_collision_lab/authoring.py'
    calls=[];clock=[0.]
    timers=SimpleNamespace(is_registered=lambda f: bool(calls),register=lambda f,**kwargs:calls.append(kwargs))
    name = '_acl_authoring_queue_' + uuid.uuid4().hex
    package = ModuleType(name)
    package.__path__ = [str(file.parent)]
    runtime = ModuleType(name + '.runtime')
    runtime.controller = SimpleNamespace(scene=None, intent='STOPPED')
    bpy = ModuleType('bpy')
    bpy.types = SimpleNamespace(Scene=type('Scene', (), {}), Object=type('Object', (), {}))
    bpy.app = SimpleNamespace(timers=timers)
    bpy.data = SimpleNamespace(scenes=[])
    with patch.dict(sys.modules, {name: package, name + '.ui': ModuleType(name + '.ui'),
                                 name + '.runtime': runtime, 'bpy': bpy}):
        spec = importlib.util.spec_from_file_location(name + '.authoring', file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.time = SimpleNamespace(monotonic=lambda: clock[0])
        yield module, calls, clock, runtime.controller


def test_continuous_drag_cannot_starve_preview_publication():
    with api() as (authoring, calls, clock, _controller):
        scene=SimpleNamespace(as_pointer=lambda:17);context=SimpleNamespace(scene=scene)
        authoring.changed(None,context);deadline=authoring._PENDING[17][1]
        for t in (.01,.03,.05,.10):
            clock[0]=t;authoring.changed(None,context)
        assert authoring._PENDING[17][1]==deadline==.02
        assert len(calls)==1


def test_worker_property_updates_only_signal_until_main_thread_drains():
    with api() as (authoring, calls, clock, _controller):
        class ForbiddenRNA:
            def __getattribute__(self, name):
                raise AssertionError('Worker must not access RNA: ' + name)
        failures=[]
        def callback():
            try:
                authoring.changed(ForbiddenRNA(), ForbiddenRNA())
                authoring.appearance_changed()
            except Exception as exc:
                failures.append(exc)
        worker=threading.Thread(target=callback)
        worker.start();worker.join(timeout=2)
        assert not worker.is_alive()
        assert not failures
        assert not calls
        assert not authoring._PENDING
        assert authoring.is_dirty(ForbiddenRNA())
        with patch.object(authoring, 'reload') as reload, patch.object(authoring, 'appearance_changed') as redraw:
            assert authoring.drain_threaded_updates()==.03
            reload.assert_called_once()
            redraw.assert_called_once()
        assert not authoring._THREADED_PHYSICS.is_set()
        assert not authoring._THREADED_APPEARANCE.is_set()


def test_pause_and_reset_preserve_intent_during_deferred_restart():
    with api() as (authoring, _calls, _clock, controller):
        scene=SimpleNamespace(as_pointer=lambda:17)
        controller.scene, controller.intent = scene, 'PLAY'
        authoring._RESTARTS[17]=(scene,.35)
        authoring.pause_restart(scene)
        assert controller.intent == 'PAUSE'
        assert authoring._RESTARTS[17] == (scene,.35)
        authoring.cancel_restart(scene)
        assert not authoring._RESTARTS
