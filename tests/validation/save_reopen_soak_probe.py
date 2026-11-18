"""Installed GUI probe: save/reopen then 60 s live soak, three rounds per mode.

For IMPACT (Moon preset) and then GALAXY (two-galaxy preset), both exactly 100,000
particles: Start -> play -> Pause -> save -> wm.open_mainfile from a timer. After
each reopen the controller must stay STOPPED (no simulation, worker or begin())
for a watch window and the reopened file must still author the exact model. Then
Start and play for --soak seconds: no runtime error, fresh revisions, physical
time advancing and exactly 100,000 active particles throughout.
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
parser.add_argument('--soak', type=float, default=60.)
parser.add_argument('--watch', type=float, default=5.)
parser.add_argument('--modes', default='IMPACT,GALAXY')
parser.add_argument('--timeout', type=float, default=2400.)
args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:])
REPORT = pathlib.Path(args.report)
REPORT.parent.mkdir(parents=True, exist_ok=True)
MODES = [m.strip().upper() for m in args.modes.split(',') if m.strip()]
COUNT = 100000
rt = importlib.import_module(args.module + '.runtime')
state = {'schema': 'acl-save-reopen-soak-probe', 'pass': False, 'checks': {}, 'events': [], 'soaks': [],
         'rounds': args.rounds, 'soak_seconds': args.soak, 'watch_seconds': args.watch, 'modes': MODES,
         'stage': 'preset', 'mode_index': 0, 'round': 0}
started = time.monotonic()
mark = {'t': started}
begins = []
_original_begin = type(rt.controller).begin


def traced_begin(self, context, *, play=True):
    begins.append({'t': round(time.monotonic() - started, 3), 'stage': state['stage'], 'play': play})
    return _original_begin(self, context, play=play)


type(rt.controller).begin = traced_begin


def mode():
    return MODES[state['mode_index']]


def blend():
    return REPORT.with_name(f'soak-{mode().lower()}.blend')


def window():
    return bpy.context.window_manager.windows[0]


def ops(op, **kw):
    win = window()
    with bpy.context.temp_override(window=win, scene=win.scene):
        return op(**kw)


def write():
    REPORT.write_text(json.dumps({**state, 'begins': begins}, indent=2, default=str), encoding='utf-8')


def log(event, **extra):
    c = rt.controller
    state['events'].append({'t': round(time.monotonic() - started, 2), 'event': event, 'mode': mode(),
                            'round': state['round'], 'intent': c.intent, 'playing': c.playing,
                            'revision': c.revision, 'sim': c.simulation is not None, **extra})
    print('SOAK_PROBE', event, mode(), state['round'], flush=True)
    write()


def check(name, value):
    state['checks'][name] = bool(value)
    if not value:
        raise AssertionError(name)


def advance(stage, **extra):
    state['stage'] = stage
    mark['t'] = time.monotonic()
    log(stage, **extra)


def waited(seconds):
    return time.monotonic() - mark['t'] >= seconds


def tag():
    return f'{mode()} round {state["round"] + 1}'


def authored_ok():
    cfg, _bodies, _objects = rt.authored_scene(window().scene)
    return cfg.mode == mode() and cfg.particle_count == COUNT


def open_file():
    bpy.ops.wm.open_mainfile(filepath=str(blend()))
    return None


def finish(error=None):
    state['error'] = error
    state['pass'] = error is None and bool(state['checks']) and all(state['checks'].values())
    state['elapsed_seconds'] = round(time.monotonic() - started, 1)
    try:
        ops(bpy.ops.acl.reset)
    except Exception:  # noqa: BLE001 - best-effort cleanup before writing the report
        pass
    type(rt.controller).begin = _original_begin
    write()
    print('SOAK_PROBE_RESULT', json.dumps({'pass': state['pass'], 'error': error}), flush=True)
    bpy.ops.wm.quit_blender()


def tick():
    try:
        c = rt.controller
        stage = state['stage']
        if time.monotonic() - started > args.timeout:
            raise TimeoutError('probe deadline at stage ' + stage)
        if c.error:
            raise RuntimeError(c.error)
        if stage == 'preset':
            if state['mode_index'] > 0:
                check(f'{mode()} new scene', ops(bpy.ops.scene.new, type='NEW') == {'FINISHED'})
            window().scene.name = f'Soak {mode()}'
            op = bpy.ops.acl.moon_preset if mode() == 'IMPACT' else bpy.ops.acl.galaxy_preset
            check(f'{mode()} preset', ops(op) == {'FINISHED'})
            window().view_layer.update()
            check(f'{mode()} preset authors exactly {COUNT}', authored_ok())
            check(f'{mode()} first start', ops(bpy.ops.acl.start) == {'FINISHED'})
            advance('warm')
        elif stage == 'warm' and c.simulation is not None and c.playing and c.revision > 30:
            ops(bpy.ops.acl.pause)
            advance('paused')
        elif stage == 'paused' and waited(1.):
            check(f'{tag()} paused before save', not c.playing)
            check(f'{tag()} save', bpy.ops.wm.save_as_mainfile(filepath=str(blend())) == {'FINISHED'})
            state['begins_before_open'] = len(begins)
            advance('reopening')
            bpy.app.timers.register(open_file, first_interval=0.)
        elif stage == 'reopening' and bpy.data.filepath and pathlib.Path(bpy.data.filepath) == blend() and waited(.5):
            advance('watch')
        elif stage == 'watch':
            if (c.simulation is not None or c.warming is not None or c.intent != 'STOPPED'
                    or len(begins) > state['begins_before_open']):
                log('UNEXPECTED_AUTOPLAY')
                check(f'{tag()} reopen stays stopped', False)
            if waited(args.watch):
                check(f'{tag()} reopen stays stopped', True)
                check(f'{tag()} reopened file authors exactly {COUNT}', authored_ok())
                check(f'{tag()} start after reopen', ops(bpy.ops.acl.start) == {'FINISHED'})
                advance('soak_start')
        elif stage == 'soak_start' and c.simulation is not None and c.playing and c.revision > 0:
            state['soak'] = {'revision': c.revision, 'physical_time': c.simulation.physical_time,
                             't': time.monotonic(), 'counts': set()}
            advance('soak')
        elif stage == 'soak':
            soak = state['soak']
            soak['counts'].add(c.simulation.count if c.simulation is not None else None)
            check(f'{tag()} keeps playing during soak', c.simulation is not None and c.playing)
            if waited(args.soak):
                seconds = time.monotonic() - soak['t']
                revisions = c.revision - soak['revision']
                progress = c.simulation.physical_time - soak['physical_time']
                row = {'mode': mode(), 'round': state['round'] + 1, 'seconds': round(seconds, 2),
                       'revisions': revisions, 'states_per_second': round(revisions / seconds, 2),
                       'physical_time_advance': progress, 'counts': sorted(soak['counts'], key=str)}
                state['soaks'].append(row)
                check(f'{tag()} fresh revisions during soak', revisions > 0)
                check(f'{tag()} physical time advanced', progress > 0)
                check(f'{tag()} exactly {COUNT} active throughout', soak['counts'] == {COUNT})
                check(f'{tag()} no runtime error', not c.error)
                del state['soak']
                ops(bpy.ops.acl.pause)
                state['round'] += 1
                if state['round'] < args.rounds:
                    advance('paused', soak=row)
                else:
                    ops(bpy.ops.acl.reset)
                    state['round'] = 0
                    state['mode_index'] += 1
                    if state['mode_index'] >= len(MODES):
                        state['mode_index'] -= 1
                        finish()
                        return None
                    advance('reset_wait', soak=row)
        elif stage == 'reset_wait' and waited(1.5):
            advance('preset')
    except BaseException:  # noqa: BLE001 - probe timer records any failure in the report
        finish(traceback.format_exc())
        return None
    return .05


bpy.app.timers.register(tick, first_interval=2., persistent=True)
