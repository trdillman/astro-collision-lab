"""Installed GUI probe: five-period isolation gate (D-010/D-014) before and after an edit.

Galaxy preset (exact 100,000-particle two-galaxy model) -> Validate Galaxy Isolation
-> wait for the background worker's verdict. Then the 2026-11-16 thick-disk defect
edit (first root mass 3, thickness 1.53) -> validate again. Both verdicts must be
"passed for this exact model". Each worker's request/log/result files are copied
next to the report. Thresholds live in the worker; nothing here relaxes them.
"""
import argparse
import importlib
import json
import pathlib
import shutil
import sys
import time
import traceback

import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--module', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--timeout', type=float, default=3300.)
parser.add_argument('--edit-mass', type=float, default=3.)
parser.add_argument('--edit-thickness', type=float, default=1.53)
args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:])
REPORT = pathlib.Path(args.report)
REPORT.parent.mkdir(parents=True, exist_ok=True)
rt = importlib.import_module(args.module + '.runtime')
PASSED = 'Isolation validation passed for this exact model'
state = {'schema': 'acl-isolation-gate-probe', 'pass': False, 'checks': {}, 'events': [],
         'stage': 'setup', 'runs': [], 'edit': {'mass': args.edit_mass, 'thickness': args.edit_thickness}}
started = time.monotonic()
mark = {'t': started}


def window():
    return bpy.context.window_manager.windows[0]


def ops(op, **kw):
    win = window()
    with bpy.context.temp_override(window=win, scene=win.scene):
        return op(**kw)


def settings():
    return window().scene.acl_settings


def write():
    REPORT.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')


def log(event, **extra):
    state['events'].append({'t': round(time.monotonic() - started, 1), 'event': event,
                            'status': settings().authoring_status, **extra})
    print('ISOLATION_PROBE', event, settings().authoring_status, flush=True)
    write()


def check(name, value):
    state['checks'][name] = bool(value)
    if not value:
        raise AssertionError(name)


def advance(stage):
    state['stage'] = stage
    mark['t'] = time.monotonic()
    log(stage)


def waited(seconds):
    return time.monotonic() - mark['t'] >= seconds


def roots():
    return [o for o in settings().source_collection.all_objects if o.acl_body.is_galaxy and o.parent is None]


def launch(label):
    cfg, bodies, _objects = rt.authored_scene(window().scene)
    check(f'{label}: Galaxy model of 100000 particles', cfg.mode == 'GALAXY' and cfg.particle_count == 100000)
    check(f'{label}: validate operator accepted', ops(bpy.ops.acl.validate_isolation) == {'FINISHED'})
    job = rt._VALIDATION_JOB
    check(f'{label}: worker started', job is not None)
    state['runs'].append({'label': label, 'owned': str(job['owned']), 'started_s': round(time.monotonic() - started, 1),
                          'bodies': [{'name': b.name, 'kind': b.kind, 'mass': b.mass, 'thickness': b.thickness,
                                      'particles': b.particle_count} for b in bodies]})
    state['current'] = {'label': label, 'owned': job['owned'], 'marker': str(job['marker'])}


def collect(label):
    run = state['runs'][-1]
    run['finished_s'] = round(time.monotonic() - started, 1)
    run['status'] = settings().authoring_status
    dest = REPORT.parent / ('isolation-' + label)
    owned = pathlib.Path(state['current']['owned'])
    dest.mkdir(parents=True, exist_ok=True)
    if owned.exists():
        shutil.copytree(owned, dest, dirs_exist_ok=True)
        run['evidence'] = str(dest)
    # The worker writes its gate report beside the cache marker; failures add failure.json.
    for path in (pathlib.Path(state['current']['marker'] + '.isolation.json'), dest / 'failure.json'):
        if path.exists():
            shutil.copy2(path, dest / path.name)
            run.setdefault('worker_reports', {})[path.name] = json.loads(path.read_text(encoding='utf-8'))
    check(f'{label}: isolation passed for this exact model', run['status'] == PASSED)


def finished():
    return rt._VALIDATION_JOB is None and 'running' not in settings().authoring_status


def finish(error=None):
    state['error'] = error
    state['pass'] = error is None and bool(state['checks']) and all(state['checks'].values())
    state['elapsed_seconds'] = round(time.monotonic() - started, 1)
    write()
    print('ISOLATION_PROBE_RESULT', json.dumps({'pass': state['pass'], 'error': error}), flush=True)
    bpy.ops.wm.quit_blender()


def tick():
    try:
        stage = state['stage']
        if time.monotonic() - started > args.timeout:
            raise TimeoutError('probe deadline at stage ' + stage)
        if stage == 'setup':
            check('Galaxy preset', ops(bpy.ops.acl.galaxy_preset) == {'FINISHED'})
            window().view_layer.update()
            advance('settle_preset')
        elif stage == 'settle_preset' and waited(2.):
            launch('preset')
            advance('validating_preset')
        elif stage == 'validating_preset' and finished():
            collect('preset')
            root = roots()[0]
            state['edit']['root'] = root.name
            root.acl_body.mass = args.edit_mass
            root.acl_body.thickness = args.edit_thickness
            window().view_layer.update()
            advance('settle_edit')
        elif stage == 'settle_edit' and waited(2.):
            body = next(b for b in rt.authored_scene(window().scene)[1]
                        if b.kind == 'DISK' and b.galaxy_id and abs(b.thickness - args.edit_thickness) < 1e-6)
            check('edit reached the authored disk', body is not None)
            check('edit marks the model not validated', settings().authoring_status != PASSED)
            launch('thick-disk-edit')
            advance('validating_edit')
        elif stage == 'validating_edit' and finished():
            collect('thick-disk-edit')
            finish()
            return None
    except BaseException:  # noqa: BLE001 - probe timer records any failure in the report
        finish(traceback.format_exc())
        return None
    return .5


bpy.app.timers.register(tick, first_interval=2., persistent=True)
