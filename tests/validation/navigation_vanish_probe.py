"""Installed GUI probe for "particles vanish during viewport navigation with overlays off".

Needs Blender started with --enable-event-simulate (real input is ignored).
Plays a Galaxy scene and drives real modal navigation through simulated
mouse events: MMB-drag orbit, Shift+MMB pan, Ctrl+MMB zoom and wheel zoom,
in Solid and Material Preview shading, with overlays off (and on as control).
For every viewport redraw it records whether the add-on's particle draw ran
and how many framebuffer pixels that draw changed; mid-drag and post-release
area screenshots are saved as visual evidence. Any redraw during a live run
whose particle draw is missing or contributes (almost) nothing is a vanish.
"""
import argparse
import importlib
import json
import pathlib
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Quaternion

parser = argparse.ArgumentParser()
parser.add_argument('--module', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--counts', default='2048,100000')
parser.add_argument('--timeout', type=float, default=1500.)
args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:])
REPORT = pathlib.Path(args.report)
SHOTS = REPORT.parent / 'shots'
SHOTS.mkdir(parents=True, exist_ok=True)
rt = importlib.import_module(args.module + '.runtime')
cuda_gl = importlib.import_module(args.module + '.cuda_gl')
COUNTS = [int(c) for c in args.counts.split(',')]
NAVS = ('orbit', 'pan', 'zoom', 'wheel')
CASES = [(n, shading, overlays, nav) for n in COUNTS for shading in ('SOLID', 'MATERIAL')
         for overlays in (False, True) for nav in NAVS]
MOVES = 24
state = {'schema': 'acl-navigation-vanish-probe', 'pass': False, 'checks': {}, 'cases': [],
         'counts': COUNTS, 'stage': 'setup', 'index': 0}
started = time.monotonic()
mark = {'t': started}
cur = {'redraws': 0, 'draws': [], 'phase': None, 'step': 0, 'live_redraws': 0}
_original_draw = cuda_gl._GL.draw


def view3d():
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return win, area, region, area.spaces.active


def traced_draw(self, *a, **kw):
    if cur['phase'] is None:
        return _original_draw(self, *a, **kw)
    import gpu
    fb = gpu.state.active_framebuffer_get()
    x, y, w, h = gpu.state.viewport_get()
    before = np.asarray(fb.read_color(x, y, w, h, 4, 0, 'FLOAT'))
    before = before.copy()
    result = _original_draw(self, *a, **kw)
    after = np.asarray(fb.read_color(x, y, w, h, 4, 0, 'FLOAT'))
    delta = np.max(np.abs(after - before), axis=-1)
    cur['draws'].append({'phase': cur['phase'], 'step': cur['step'],
                         'pixels': int(np.count_nonzero(delta > 1e-5)), 'region': [w, h],
                         'distance': float(bpy.context.region_data.view_distance)})
    return result


def post_pixel():
    # Runs on every 3D-view WINDOW redraw, independent of the particle draw path.
    if cur['phase'] is not None:
        c = rt.controller
        live = c.simulation is not None and bpy.context.scene == c.scene
        cur['redraws'] += 1
        cur['live_redraws'] += int(live)


def check(name, value):
    state['checks'][name] = bool(value)
    if not value:
        raise AssertionError(name)


def ops(op, **kw):
    win, area, region, _ = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region, scene=win.scene):
        return op(**kw)


def advance(stage):
    state['stage'] = stage
    mark['t'] = time.monotonic()
    REPORT.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')


def waited(seconds):
    return time.monotonic() - mark['t'] >= seconds


def screenshot(tag):
    path = SHOTS / f"{state['index']:02d}-{tag}.png"
    win, area, region, _ = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=str(path))
    img = bpy.data.images.load(str(path), check_existing=False)
    px = np.asarray(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], img.channels)[..., :3]
    bpy.data.images.remove(img)
    q = np.round(px * 255).astype(np.int32)
    keys = q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]
    values, counts = np.unique(keys, return_counts=True)
    bg = values[np.argmax(counts)]
    bg_rgb = np.array([bg // 65536, (bg // 256) % 256, bg % 256])
    return {'file': path.name, 'non_background_px': int(np.count_nonzero(np.abs(q - bg_rgb).max(-1) > 12))}


def reset_view(space, overlays, shading):
    space.overlay.show_overlays = overlays
    space.shading.type = shading
    rv = space.region_3d
    rv.view_perspective = 'PERSP'
    rv.view_location = (0, 0, 0)
    rv.view_distance = 25
    rv.view_rotation = Quaternion((1, 0, 0), .35)


def view_key(space):
    rv = space.region_3d
    return (tuple(round(v, 5) for v in rv.view_rotation), tuple(round(v, 4) for v in rv.view_location),
            round(rv.view_distance, 4))


def emit(win, kind, value, x, y, **mods):
    win.event_simulate(type=kind, value=value, x=int(x), y=int(y), **mods)


def finish(error=None):
    cur['phase'] = None
    cuda_gl._GL.draw = _original_draw
    state['error'] = error
    state['pass'] = error is None and bool(state['checks']) and all(state['checks'].values())
    state['elapsed_seconds'] = round(time.monotonic() - started, 1)
    try:
        ops(bpy.ops.acl.reset)
    except Exception:  # noqa: BLE001 - best-effort cleanup before writing the report
        pass
    REPORT.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
    bpy.ops.wm.quit_blender()


def close_case(case):
    n, shading, overlays, nav = case
    draws = cur['draws']
    before = [d['pixels'] for d in draws if d['phase'] == 'before']
    ref = float(np.median(before)) if before else 0.
    ref_distance = float(np.median([d['distance'] for d in draws if d['phase'] == 'before'])) if before else 1.
    drag = [d['pixels'] for d in draws if d['phase'] == 'drag']
    after = [d['pixels'] for d in draws if d['phase'] == 'after']
    # Projected coverage shrinks with the square of view distance when zooming out;
    # zooming in never lowers the floor below the framed reference.
    weak = [d for d in draws if d['phase'] != 'before'
            and d['pixels'] < .25 * ref * min(1., (ref_distance / d['distance']) ** 2)]
    record = {'count': n, 'shading': shading, 'overlays': overlays, 'nav': nav,
              'reference_px': ref, 'reference_distance': ref_distance, 'drag_draws': len(drag), 'after_draws': len(after),
              'drag_px_min': min(drag) if drag else None, 'drag_px_median': float(np.median(drag)) if drag else None,
              'after_px_min': min(after) if after else None, 'weak_draws': len(weak), 'weak_samples': weak[:8],
              'redraws': cur['redraws'], 'live_redraws': cur['live_redraws'], 'particle_draws': len(draws),
              'view_moved': state.pop('_view_start') != view_key(view3d()[3]),
              'shots': state.pop('_shots'), 'frame': rt.controller.status().get('frame'),
              'error': rt.controller.error}
    record['vanish'] = bool(weak) or record['particle_draws'] < record['live_redraws'] or bool(record['error'])
    state['cases'].append(record)
    tag = f'{n}/{shading}/overlays={overlays}/{nav}'
    check(f'navigated {tag}', record['view_moved'] and record['drag_draws'] >= 4)
    check(f'particles drawn every redraw {tag}', record['particle_draws'] >= record['live_redraws'])
    check(f'particles visible through navigation {tag}', ref > 0 and not weak)
    check(f'screenshots show particles {tag}', all(s['non_background_px'] > 0 for s in record['shots']))
    cur.update(redraws=0, live_redraws=0, draws=[], phase=None, step=0)


def tick():
    try:
        c = rt.controller
        if time.monotonic() - started > args.timeout:
            raise TimeoutError('probe deadline at stage ' + state['stage'])
        if c.error:
            raise RuntimeError(c.error)
        stage = state['stage']
        i = state['index']
        if stage == 'setup':
            if i >= len(CASES):
                finish()
                return None
            n = CASES[i][0]
            if c.simulation is None or state.get('live_count') != n:
                ops(bpy.ops.acl.reset)
                scene = bpy.context.window_manager.windows[0].scene
                if not scene.acl_settings.source_collection or not any(
                        o.acl_body.is_galaxy for o in scene.acl_settings.source_collection.objects):
                    check('Galaxy preset', ops(bpy.ops.acl.galaxy_preset) == {'FINISHED'})
                roots = [o for o in scene.acl_settings.source_collection.objects if o.acl_body.is_galaxy]
                for k, o in enumerate(roots):
                    o.acl_body.particle_count = n // len(roots) + (1 if k < n % len(roots) else 0)
                bpy.context.view_layer.update()
                check(f'Galaxy start {n}', ops(bpy.ops.acl.start) == {'FINISHED'})
                state['live_count'] = n
                advance('wait_live')
            else:
                advance('arm')
        elif stage == 'wait_live' and c.simulation is not None and c.playing and c.revision > 20:
            check(f'live count {state["live_count"]}', c.simulation.count == state['live_count'])
            advance('arm')
        elif stage == 'arm':
            _, shading, overlays, _ = CASES[i]
            win, area, region, space = view3d()
            reset_view(space, overlays, shading)
            area.tag_redraw()
            state['_view_start'] = None
            state['_shots'] = []
            advance('settle')
        elif stage == 'settle' and waited(1.5 if CASES[i][1] == 'MATERIAL' else .4):
            cur['phase'] = 'before'
            advance('before')
        elif stage == 'before' and waited(.5):
            win, area, region, space = view3d()
            state['_view_start'] = view_key(space)
            cx, cy = region.x + region.width // 2, region.y + region.height // 2
            state['_center'] = (cx, cy)
            nav = CASES[i][3]
            emit(win, 'MOUSEMOVE', 'NOTHING', cx, cy)
            if nav != 'wheel':
                mods = {'pan': {'shift': True}, 'zoom': {'ctrl': True}}.get(nav, {})
                emit(win, 'MIDDLEMOUSE', 'PRESS', cx, cy, **mods)
            cur['phase'] = 'drag'
            advance('drag')
        elif stage == 'drag' and waited(.04):
            win, area, region, space = view3d()
            cx, cy = state['_center']
            cur['step'] += 1
            s = cur['step']
            nav = CASES[i][3]
            if nav == 'wheel':
                # Four notches out over the first half, two back in over the second.
                if s % 3 == 0 and s <= MOVES // 2:
                    emit(win, 'WHEELDOWNMOUSE', 'PRESS', cx, cy)
                elif s % 6 == 0:
                    emit(win, 'WHEELUPMOUSE', 'PRESS', cx, cy)
                else:
                    emit(win, 'MOUSEMOVE', 'NOTHING', cx + (s % 2), cy)
            else:
                dx = 6 * s if nav != 'zoom' else 0
                dy = (3 * s if nav != 'zoom' else 4 * ((s % 12) - 6))
                emit(win, 'MOUSEMOVE', 'NOTHING', cx + dx, cy + dy)
            if s == MOVES // 2:
                state['_shots'].append(dict(screenshot('mid-drag'), phase='drag'))
            mark['t'] = time.monotonic()
            if s >= MOVES:
                if nav != 'wheel':
                    emit(win, 'MIDDLEMOUSE', 'RELEASE', cx + 6 * s, cy + 3 * s)
                cur['phase'] = 'after'
                advance('after')
        elif stage == 'after' and waited(.8):
            state['_shots'].append(dict(screenshot('after'), phase='after'))
            close_case(CASES[i])
            state['index'] += 1
            advance('setup')
    except BaseException:  # noqa: BLE001 - probe timer records any failure
        finish(traceback.format_exc())
        return None
    return .02


cuda_gl._GL.draw = traced_draw
bpy.types.SpaceView3D.draw_handler_add(post_pixel, (), 'WINDOW', 'POST_PIXEL')
bpy.app.timers.register(tick, first_interval=2.)
