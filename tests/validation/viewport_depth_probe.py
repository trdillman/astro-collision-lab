"""Diagnose which GL state hides the particle draw in a given shading/overlay mode.

For SOLID, MATERIAL and RENDERED shading with overlays off and on, capture
several particle draws normally and several with the depth test forced off,
recording pixels changed, depth-buffer statistics and the GL state Blender
leaves bound. Verification only: framebuffer readback, no product change.
"""
import argparse
import ctypes as C
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
parser.add_argument('--count', type=int, default=2048)
parser.add_argument('--occluder', action='store_true')
args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:])
REPORT = pathlib.Path(args.report)
rt = importlib.import_module(args.module + '.runtime')
cuda_gl = importlib.import_module(args.module + '.cuda_gl')
CASES = [(s, o) for s in ('SOLID', 'MATERIAL', 'RENDERED') for o in (False, True)]
PER = 3
state = {'schema': 'acl-viewport-depth-probe', 'cases': [], 'stage': 'setup', 'index': 0, 'error': None}
cur = {'variant': None, 'samples': []}
started = time.monotonic()
mark = {'t': started}
_draw = cuda_gl._GL.draw
_enable = cuda_gl._GL._enable


def gl_state(gl):
    ints = (C.c_int * 4)()
    gl._call('glGetIntegerv', None, (C.c_uint, C.POINTER(C.c_int)), 0x0C23, ints)
    writemask = list(ints)
    floats = (C.c_float * 2)()
    gl._call('glGetFloatv', None, (C.c_uint, C.POINTER(C.c_float)), 0x0B70, floats)
    fbo = gl._integer(0x8CA6)
    kind = C.c_int(-1)
    if fbo:
        gl._call('glGetFramebufferAttachmentParameteriv', None,
                 (C.c_uint, C.c_uint, C.c_uint, C.POINTER(C.c_int)), 0x8CA9, 0x8D00, 0x8CD0, C.byref(kind))
    return {'draw_fbo': fbo, 'depth_attachment_type': kind.value, 'color_writemask': writemask,
            'depth_range': list(floats), 'depth_func': gl._integer(0x0B74),
            'depth_test': gl._enabled(0x0B71), 'clip_origin': gl._integer(0x935C),
            'clip_depth': gl._integer(0x935D), 'depth_mask': gl._integer(0x0B72),
            'cull': gl._enabled(0x0B44), 'scissor': gl._enabled(0x0C11)}


def traced(self, *a, **kw):
    variant = cur['variant']
    if variant is None or len(cur['samples']) >= 2 * PER:
        return _draw(self, *a, **kw)
    import gpu
    fb = gpu.state.active_framebuffer_get()
    x, y, w, h = gpu.state.viewport_get()
    before = np.asarray(fb.read_color(x, y, w, h, 4, 0, 'FLOAT')).copy()
    depth = np.asarray(fb.read_depth(x, y, w, h)).copy()
    gl_before = gl_state(self)
    mode = 'nodepth' if len(cur['samples']) % 2 else 'normal'
    if mode == 'nodepth':
        def forced(gl, enum, enabled):
            return _enable(gl, enum, False if enum == 0x0B71 else enabled)
        cuda_gl._GL._enable = forced
    try:
        result = _draw(self, *a, **kw)
    finally:
        cuda_gl._GL._enable = _enable
    after = np.asarray(fb.read_color(x, y, w, h, 4, 0, 'FLOAT'))
    delta = np.max(np.abs(after - before), axis=-1)
    d = depth.ravel()
    cur['samples'].append({'mode': mode, 'pixels': int(np.count_nonzero(delta > 1e-5)),
                           'depth_min': float(d.min()), 'depth_max': float(d.max()),
                           'depth_median': float(np.median(d)), 'depth_zero_frac': float(np.mean(d == 0.)),
                           'depth_one_frac': float(np.mean(d == 1.)), 'depth_mid_frac': float(np.mean((d > 0.) & (d < 1.))), 'region': [w, h],
                           'gl_inherited': gl_before})
    return result


def ops(op, **kw):
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    with bpy.context.temp_override(window=win, area=area, region=region, scene=win.scene):
        return op(**kw)


def space():
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    return area, area.spaces.active


def save():
    REPORT.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')


def finish(error=None):
    cur['variant'] = None
    cuda_gl._GL.draw = _draw
    state['error'] = error
    state['elapsed_seconds'] = round(time.monotonic() - started, 1)
    try:
        ops(bpy.ops.acl.reset)
    except Exception:  # noqa: BLE001 - best-effort cleanup before saving the report
        pass
    save()
    bpy.ops.wm.quit_blender()


def advance(stage):
    state['stage'] = stage
    mark['t'] = time.monotonic()
    save()


def tick():
    try:
        c = rt.controller
        if time.monotonic() - started > 600:
            raise TimeoutError(state['stage'])
        if c.error:
            raise RuntimeError(c.error)
        stage = state['stage']
        if stage == 'setup':
            scene = bpy.context.window_manager.windows[0].scene
            assert ops(bpy.ops.acl.galaxy_preset) == {'FINISHED'}
            roots = [o for o in scene.acl_settings.source_collection.objects if o.acl_body.is_galaxy]
            for k, o in enumerate(roots):
                o.acl_body.particle_count = args.count // len(roots) + (1 if k < args.count % len(roots) else 0)
            if args.occluder:
                import bmesh
                mesh = bpy.data.meshes.new('ProbeOccluder')
                bm = bmesh.new()
                bmesh.ops.create_cube(bm, size=4.)
                bm.to_mesh(mesh)
                bm.free()
                cube = bpy.data.objects.new('ProbeOccluder', mesh)
                scene.collection.objects.link(cube)
            bpy.context.view_layer.update()
            assert ops(bpy.ops.acl.start) == {'FINISHED'}
            advance('wait_live')
        elif stage == 'wait_live' and c.simulation is not None and c.playing and c.revision > 20:
            advance('arm')
        elif stage == 'arm':
            if state['index'] >= len(CASES):
                finish()
                return None
            shading, overlays = CASES[state['index']]
            area, sp = space()
            sp.shading.type = shading
            sp.overlay.show_overlays = overlays
            rv = sp.region_3d
            rv.view_location = (0, 0, 0)
            rv.view_distance = 25
            rv.view_rotation = Quaternion((1, 0, 0), .35)
            area.tag_redraw()
            advance('settle')
        elif stage == 'settle' and time.monotonic() - mark['t'] > 2.:
            cur.update(variant=CASES[state['index']], samples=[])
            advance('capture')
        elif stage == 'capture' and (len(cur['samples']) >= 2 * PER or time.monotonic() - mark['t'] > 10):
            shading, overlays = CASES[state['index']]
            s = cur['samples']
            state['cases'].append({'shading': shading, 'overlays': overlays, 'samples': s,
                                   'normal_px': [x['pixels'] for x in s if x['mode'] == 'normal'],
                                   'nodepth_px': [x['pixels'] for x in s if x['mode'] == 'nodepth']})
            cur['variant'] = None
            state['index'] += 1
            advance('arm')
    except BaseException:  # noqa: BLE001 - probe timer records any failure
        finish(traceback.format_exc())
        return None
    return .05


cuda_gl._GL.draw = traced
bpy.app.timers.register(tick, first_interval=2.)
