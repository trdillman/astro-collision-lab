"""Run in the isolated installed GUI; public control interruption acceptance."""
import json, time, traceback
from pathlib import Path
import bpy
from astro_collision_lab import runtime

REPORT = Path(__file__).resolve().parents[2] / 'interrupted-controls.json'
state = {'pass': False, 'checks': {}, 'stage': 0}
started = time.monotonic()

def check(name, value):
    state['checks'][name] = bool(value)
    assert value, name

def finish(error=None):
    state['error'] = error
    state['pass'] = error is None and all(state['checks'].values())
    runtime.controller.reset()
    REPORT.write_text(json.dumps(state, indent=2))
    bpy.ops.wm.quit_blender()

def tick():
    try:
        owner = runtime.controller
        if time.monotonic()-started > 240:
            raise TimeoutError('Control lifecycle deadline')
        if owner.error:
            raise RuntimeError(owner.error)
        stage = state['stage']
        if stage == 0:
            bpy.ops.acl.moon_preset()
            bpy.ops.acl.start()
            check('Preparation started', owner.warming is not None)
            bpy.ops.acl.pause()
            check('Pause overrides pending play', owner.intent == 'PAUSE' and not owner.playing)
            state['stage'] = 1
        elif stage == 1 and owner.simulation is not None and owner.presented_revision >= 0:
            check('Prepared state stays paused', owner.revision == 0 and not owner.playing)
            bpy.ops.acl.step()
            state['stage'] = 2
        elif stage == 2 and owner.presented_revision == 1:
            state['pause_at'] = time.monotonic()
            state['stage'] = 3
        elif stage == 3 and time.monotonic()-state['pause_at'] > .5:
            check('Single Step advances exactly once then pauses', owner.revision == 1 and not owner.playing)
            bpy.ops.acl.start()
            state['stage'] = 4
        elif stage == 4 and owner.revision >= 5:
            bpy.context.scene.acl_settings.seconds_per_frame *= 1.02
            state['stage'] = 5
        elif stage == 5 and owner.warming is not None:
            bpy.ops.acl.reset()
            state['reset_at'] = time.monotonic()
            state['stage'] = 6
        elif stage == 6 and time.monotonic()-state['reset_at'] > 1:
            check('Reset during rebuild prevents stale restart', owner.intent == 'STOPPED' and not owner.playing and owner.simulation is None and owner.pending is None)
            finish()
            return None
    except BaseException:
        finish(traceback.format_exc())
        return None
    return .05

bpy.app.timers.register(tick, first_interval=2.)
