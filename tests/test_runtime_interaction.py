"""Controller contracts without claiming Blender/CUDA runtime acceptance."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class Timers:
    def __init__(self): self.registered=set()
    def is_registered(self,fn): return fn in self.registered
    def register(self,fn,**kw): self.registered.add(fn)
    def unregister(self,fn): self.registered.discard(fn)

class Scene:
    name='Test'
    name_full='Test'
    def __init__(self): self.acl_settings=types.SimpleNamespace(authoring_status='Preview')
    def as_pointer(self): return id(self)

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.bpy=types.ModuleType('bpy')
        handlers=types.SimpleNamespace(persistent=lambda f:f,**{n:[] for n in ('load_pre','undo_pre','redo_pre','undo_post','redo_post','save_pre','save_post')})
        self.bpy.app=types.SimpleNamespace(timers=Timers(),handlers=handlers)
        self.viewport=types.ModuleType('astro_collision_lab.viewport')
        for n in ('retire','request_redraw','shutdown','attach'):setattr(self.viewport,n,Mock())
        self.viewport.status=lambda:{}
        self.editor=types.ModuleType('astro_collision_lab.authoring')
        for n in ('cancel_restart','pause_restart','synchronize','accept_snapshot','changed'):setattr(self.editor,n,Mock())
        self.editor.is_dirty=lambda s:False
        self.modules=patch.dict(sys.modules,{'bpy':self.bpy,'astro_collision_lab.viewport':self.viewport,'astro_collision_lab.authoring':self.editor})
        self.modules.start()
        sys.modules.pop('astro_collision_lab.runtime',None)
        self.r=importlib.import_module('astro_collision_lab.runtime')
        # Package attributes can retain a module from a previous isolated import.
        self.r.viewport=self.viewport
        self.scene=Scene();self.context=types.SimpleNamespace(scene=self.scene)
        from astro_collision_lab.simulation import SimulationConfig,BodySpec
        self.config=SimulationConfig(particle_count=10)
        self.bodies=(BodySpec('A',particle_count=10),)
        self.authored=patch.object(self.r,'authored_scene',return_value=(self.config,self.bodies,()))
        self.authored.start()
        self.c=self.r.controller
    def tearDown(self):
        self.authored.stop();self.modules.stop();sys.modules.pop('astro_collision_lab.runtime',None)
    def live(self):
        self.c.scene=self.scene;self.c.simulation=Mock(count=10,physical_time=0.,stats={})
        self.c.signature=self.r._signature(self.config,self.bodies)
        self.c.revision=self.c.presented_revision=3
        self.c.intent='PLAY';self.c.playing=True
    def test_step_always_reconciles_current_inputs(self):
        self.live()
        with patch.object(self.c,'begin') as begin:
            self.r.step(self.context)
        begin.assert_called_once()
        self.c.simulation.step.assert_not_called()
        self.assertEqual(self.c.intent,'STEP')
        self.editor.cancel_restart.assert_called_once_with(self.scene)
    def test_start_changes_intent_while_worker_prepares(self):
        self.c.scene=self.scene;self.c.intent='PAUSE';self.c.warming=Mock()
        self.c.pending=(self.scene,self.config,self.bodies,(),False,Path('marker'),Path('owned'))
        self.r.start(self.context)
        self.assertEqual(self.c.intent,'PLAY')
        self.assertTrue(self.c.pending[4])
    def test_pause_cancels_step_and_pending_play(self):
        self.c.scene=self.scene;self.c.intent='STEP';self.c.single_step_pending=True
        self.c.pending=(self.scene,self.config,self.bodies,(),True,Path('marker'),Path('owned'))
        self.r.pause(self.context)
        self.assertEqual(self.c.intent,'PAUSE')
        self.assertFalse(self.c.single_step_pending);self.assertFalse(self.c.pending[4])
    def test_invalid_edit_reset_retains_scene_intent(self):
        self.live();self.c.reset(preserve_intent=True)
        self.assertIs(self.c.scene,self.scene)
        self.assertEqual(self.c.intent,'PLAY')
        self.assertIsNone(self.c.simulation)
        self.assertFalse(self.c.playing)
    def test_explicit_reset_forgets_intent(self):
        self.live();self.r.reset(self.context)
        self.assertEqual(self.c.intent,'STOPPED');self.assertIsNone(self.c.scene)
    def test_dirty_authoring_stops_next_step(self):
        self.live();self.editor.is_dirty=lambda s:True
        self.r._tick()
        self.c.simulation.step.assert_not_called()
    def test_failed_simulation_clears_pending_step(self):
        self.live();self.c.intent='STEP';self.c.single_step_pending=True
        self.c.simulation.step.side_effect=RuntimeError('failed health check')
        self.assertIsNone(self.r._tick())
        self.assertFalse(self.c.single_step_pending)
        self.assertEqual(self.c.intent,'STOPPED')
        self.assertTrue(self.c.error)
    def test_redo_has_runtime_teardown_and_reacquire(self):
        self.r.register_handlers()
        self.assertTrue(self.bpy.app.handlers.redo_pre)
        self.assertTrue(self.bpy.app.handlers.redo_post)
        self.r.unregister_handlers()
        self.assertFalse(self.bpy.app.handlers.redo_pre)
    def test_signature_change_waits_for_authoring_instead_of_stepping(self):
        self.live();self.c.signature='old'
        self.r._tick()
        self.c.simulation.step.assert_not_called()
        self.editor.changed.assert_called_once()

if __name__=='__main__':unittest.main()
