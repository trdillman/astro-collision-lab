"""Installed GUI probe for the 2026-11-06 "Galaxy plays after reopen" report.

Replays the reported sequence through public operators: an Impact scene that
was run and reset, then a second Galaxy scene (two 1,024-particle roots):
Start -> velocity edit (auto-resume restart) -> Pause -> Step -> Reset ->
save -> wm.open_mainfile from a timer. After every reopen the controller must
stay STOPPED with no simulation, worker or begin() for the watch window.
"""
import argparse
import importlib
import json
import pathlib
import sys
import time
import traceback

import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--module', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--watch', type=float, default=35.)
parser.add_argument('--timeout', type=float, default=900.)
args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:])
REPORT = pathlib.Path(args.report)
BLEND = REPORT.with_name('Galaxy-Lifecycle.blend')
rt = importlib.import_module(args.module + '.runtime')
state = {'schema': 'acl-reopen-autoplay-probe', 'pass': False, 'checks': {}, 'events': [],
         'cycles': args.rounds, 'watch_seconds': args.watch, 'stage': 'impact_setup', 'cycle': 0}
started = time.monotonic()
mark = {'t': started}
begins = []
_original_begin = type(rt.controller).begin


def traced_begin(self, context, *, play=True):
    begins.append({'t': round(time.monotonic() - started, 3), 'stage': state['stage'],
                   'scene': getattr(context.scene, 'name', None), 'play': play, 'intent': self.intent})
    return _original_begin(self, context, play=play)


type(rt.controller).begin = traced_begin


def log(event, **extra):
    s = rt.controller.status()
    state['events'].append({'t': round(time.monotonic() - started, 3), 'event': event,
                            'intent': s['intent'], 'running': s['running'], 'frame': s.get('frame'),
                            'sim': rt.controller.simulation is not None, **extra})
    REPORT.write_text(json.dumps({**state, 'begins': begins}, indent=2, default=str), encoding='utf-8')


def check(name, value):
    state['checks'][name] = bool(value)
    if not value:
        raise AssertionError(name)


def window():
    return bpy.context.window_manager.windows[0]


def ops(op, **kw):
    win = window()
    with bpy.context.temp_override(window=win, scene=win.scene):
        return op(**kw)


def scene_named(name):
    return bpy.data.scenes[name]


def use(name):
    window().scene = scene_named(name)


def advance(stage):
    state['stage'] = stage
    mark['t'] = time.monotonic()
    log(stage)


def waited(seconds):
    return time.monotonic() - mark['t'] >= seconds


def finish(error=None):
    state['error'] = error
    state['pass'] = error is None and bool(state['checks']) and all(state['checks'].values())
    state['elapsed_seconds'] = round(time.monotonic() - started, 1)
    try:
        ops(bpy.ops.acl.reset)
    except Exception:  # noqa: BLE001 - best-effort cleanup before writing the report
        pass
    type(rt.controller).begin = _original_begin
    REPORT.write_text(json.dumps({**state, 'begins': begins}, indent=2, default=str), encoding='utf-8')
    bpy.ops.wm.quit_blender()


def open_file():
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    return None


def tick():
    try:
        c = rt.controller
        stage = state['stage']
        if time.monotonic() - started > args.timeout:
            raise TimeoutError('probe deadline at stage ' + stage)
        if c.error:
            raise RuntimeError(c.error)
        if stage == 'impact_setup':
            window().scene.name = 'Impact Lifecycle'
            check('Impact preset', ops(bpy.ops.acl.moon_preset) == {'FINISHED'})
            for o in window().scene.acl_settings.source_collection.objects:
                if o.acl_body.enabled:
                    o.acl_body.particle_count = 1024
            window().view_layer.update()
            check('Impact balance', ops(bpy.ops.acl.balance_sampling) == {'FINISHED'})
            window().view_layer.update()
            check('Impact start', ops(bpy.ops.acl.start) == {'FINISHED'})
            advance('impact_live')
        elif stage == 'impact_live' and c.simulation is not None and c.revision > 30:
            ops(bpy.ops.acl.pause)
            advance('impact_paused')
        elif stage == 'impact_paused' and waited(.5):
            ops(bpy.ops.acl.step)
            advance('impact_step')
        elif stage == 'impact_step' and waited(1.):
            ops(bpy.ops.acl.reset)
            advance('galaxy_setup')
        elif stage == 'galaxy_setup' and waited(1.):
            check('New scene', ops(bpy.ops.scene.new, type='NEW') == {'FINISHED'})
            window().scene.name = 'Galaxy Lifecycle'
            check('Galaxy preset', ops(bpy.ops.acl.galaxy_preset) == {'FINISHED'})
            for o in window().scene.acl_settings.source_collection.objects:
                if o.acl_body.is_galaxy:
                    o.acl_body.particle_count = 1024
            window().view_layer.update()
            advance('galaxy_start')
        elif stage == 'galaxy_start' and waited(.5):
            use('Galaxy Lifecycle')
            check(f'Galaxy start {state["cycle"]}', ops(bpy.ops.acl.start) == {'FINISHED'})
            advance('galaxy_live')
        elif stage == 'galaxy_live' and c.simulation is not None and c.playing and c.revision > 200:
            state['run_id'] = c.run_id
            root = next(o for o in window().scene.acl_settings.source_collection.objects if o.acl_body.is_galaxy)
            v = list(root.acl_body.velocity)
            v[0] += .01
            root.acl_body.velocity = v
            window().view_layer.update()
            advance('galaxy_edited')
        elif stage == 'galaxy_edited' and c.simulation is not None and c.playing and c.run_id != state['run_id'] and c.revision > 30:
            check(f'Edit auto-resumed {state["cycle"]}', True)
            ops(bpy.ops.acl.pause)
            advance('galaxy_paused')
        elif stage == 'galaxy_paused' and waited(1.):
            state['paused_frame'] = c.revision
            check(f'Pause holds {state["cycle"]}', not c.playing)
            ops(bpy.ops.acl.step)
            advance('galaxy_step')
        elif stage == 'galaxy_step' and waited(1.):
            check(f'Step once {state["cycle"]}', c.revision == state['paused_frame'] + 1 and not c.playing)
            ops(bpy.ops.acl.reset)
            advance('galaxy_reset')
        elif stage == 'galaxy_reset' and waited(1.5):
            check(f'Reset settled {state["cycle"]}', c.simulation is None and c.intent == 'STOPPED' and c.scene is None)
            check(f'Save {state["cycle"]}', bpy.ops.wm.save_as_mainfile(filepath=str(BLEND)) == {'FINISHED'})
            advance('reopening')
            state['begins_before_open'] = len(begins)
            bpy.app.timers.register(open_file, first_interval=0.)
        elif stage == 'reopening' and bpy.data.filepath and waited(.5):
            advance('watch')
        elif stage == 'watch':
            if c.simulation is not None or c.warming is not None or c.intent != 'STOPPED' or len(begins) > state['begins_before_open']:
                log('UNEXPECTED_AUTOPLAY')
                check(f'Reopen stays stopped {state["cycle"]}', False)
            if waited(args.watch):
                check(f'Reopen stays stopped {state["cycle"]}', True)
                state['cycle'] += 1
                if state['cycle'] >= args.rounds:
                    finish()
                    return None
                advance('galaxy_start')
    except BaseException:  # noqa: BLE001 - probe timer records any failure
        finish(traceback.format_exc())
        return None
    return .05


bpy.app.timers.register(tick, first_interval=2., persistent=True)
