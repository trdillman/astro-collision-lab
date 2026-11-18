"""Real owned GL shader versus independent double-precision ray/quadric reference.

Execute in an isolated Blender MCP session, with OUTPUT_DIR set by the caller.
Synthetic host uploads and image readbacks exist only in this validation probe.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

import bpy
import gpu
import numpy as np
from astro_collision_lab import cuda_gl


def projection(ortho=False):
    near, far = .1, 50.
    if ortho:
        return np.array([[.4, 0, 0, 0], [0, .4, 0, 0],
                         [0, 0, -2/(far-near), -(far+near)/(far-near)], [0, 0, 0, 1.]])
    f = 1/np.tan(np.pi/6)
    return np.array([[f, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)], [0, 0, -1., 0]])


def reference(matrix, center, axes, radius, size, clear_depth):
    """Intersect rays with a transformed unit quadric, using double precision."""
    y, x = np.mgrid[:size, :size]
    xy = np.stack((2*(x+.5)/size-1, 2*(y+.5)/size-1), axis=-1)
    inv = np.linalg.inv(matrix)
    ends = []
    for z in [-1., 0.]:
        clip = np.concatenate((xy, np.full((*x.shape, 1), z), np.ones((*x.shape, 1))), axis=-1)
        world = clip @ inv.T
        ends.append(world[..., :3]/world[..., 3:])
    transform = np.linalg.inv(axes*radius)
    origin = (ends[0]-center) @ transform.T
    direction = (ends[1]-ends[0]) @ transform.T
    a = np.sum(direction**2, axis=-1)
    b = np.sum(origin*direction, axis=-1)
    c = np.sum(origin**2, axis=-1)-1
    disc = b*b-a*c
    root = np.sqrt(np.maximum(disc, 0))
    t = (-b-root)/a
    t = np.where(t >= 0, t, (-b+root)/a)
    world = ends[0]+t[..., None]*(ends[1]-ends[0])
    clip = np.concatenate((world, np.ones((*x.shape, 1))), axis=-1) @ matrix.T
    depth = .5*(clip[..., 2]/clip[..., 3])+ .5
    mask = (disc >= 0) & (t >= 0) & (clip[..., 3] > 0) & (depth >= 0) & (depth <= clear_depth)
    return mask, depth


def run():
    output = Path(OUTPUT_DIR)  # noqa: F821 - OUTPUT_DIR is injected into the exec namespace by the caller (see docstring)
    output.mkdir(parents=True, exist_ok=True)
    size = 256
    results = []
    cases = [
        ('perspective-centered', (0, 0, -3), 1., False, 'VIEW', 1., 1.),
        ('perspective-off-axis', (1.4, .5, -3), .8, False, 'VIEW', 1., 1.),
        ('large-close-sphere', (.5, 0, -1.5), .8, False, 'VIEW', 1., 1.),
        ('orthographic', (.7, -.3, -3), .9, True, 'VIEW', 1., 1.),
        ('stretched', (.8, .4, -4), .9, False, 'VIEW', 1.7, 1.),
        ('velocity-aligned', (.8, .4, -4), .9, False, 'VELOCITY', 1.7, 1.),
        ('fixed-aligned', (.8, .4, -4), .9, False, 'FIXED', 1.7, 1.),
        ('axis-aligned', (.8, .4, -4), .9, False, 'AXIS', 1.7, 1.),
        ('near-plane-intersection', (0, 0, -.14), .08, False, 'VIEW', 1., 1.),
        ('behind-eye', (0, 0, .2), .05, False, 'VIEW', 1., 1.),
        ('occluding-plane', (.5, 0, -3), 1., False, 'VIEW', 1., .965),
        ('offset-viewport', (1.4, .5, -3), .8, False, 'VIEW', 1., 1.),
        ('rotated-camera', (1.4, .5, -3), .8, False, 'VIEW', 1., 1.),
        ('eye-inside', (0, 0, 0), .5, False, 'VIEW', 1., 1.),
        ('past-far-plane', (0, 0, -60), 1., False, 'VIEW', 1., 1.),
    ]
    gl = cuda_gl._GL()
    offscreen = gpu.types.GPUOffScreen(size+32, size+32, format='RGBA32F')
    try:
        with offscreen.bind():
            vao, vbo, program = gl.create(1)
            try:
                for name, center, radius, ortho, alignment, stretch, clear_depth in cases:
                    view = np.eye(4)
                    if name == 'rotated-camera':
                        c, s = np.cos(.6), np.sin(.6)
                        view[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                        view[:3, 3] = [.4, -.2, .7]
                        center = (np.linalg.inv(view) @ [*center, 1])[:3]
                    matrix = projection(ortho) @ view
                    velocity = np.array([1., 2., .5])
                    normal = np.array([.5, .3, .8]); normal /= np.linalg.norm(normal)
                    axis = np.array([.2, 1., .4]); axis /= np.linalg.norm(axis)
                    right, up, facing = np.linalg.inv(view)[:3, :3].T
                    if alignment == 'VELOCITY':
                        right = np.array([1., 2., 0]); right /= np.linalg.norm(right)
                        up = np.cross(facing, right)
                    elif alignment == 'FIXED':
                        facing = normal
                        right = np.cross([0., 0., 1.], facing); right /= np.linalg.norm(right)
                        up = np.cross(facing, right)
                    elif alignment == 'AXIS':
                        up = axis
                        right = np.cross(up, facing); right /= np.linalg.norm(right)
                        facing = np.cross(right, up); facing /= np.linalg.norm(facing)
                    axes = np.column_stack((right*stretch, up, facing))
                    packed = (C.c_float*12)(*center, radius, 1, .4, .2, 1, *velocity, 0)
                    with gl._state():
                        gl._call('glBindBuffer', None, (C.c_uint, C.c_uint), 0x8892, vbo)
                        gl._call('glBufferSubData', None, (C.c_uint, C.c_ssize_t, C.c_ssize_t, C.c_void_p),
                                 0x8892, 0, C.sizeof(packed), packed)
                    fb = gpu.state.active_framebuffer_get()
                    ox, oy = (13, 17) if name == 'offset-viewport' else (0, 0)
                    gpu.state.viewport_set(ox, oy, size, size)
                    gpu.state.depth_mask_set(True)
                    fb.clear(color=(0, 0, 0, 0), depth=clear_depth)
                    gl.draw(vao, program, 1, view_projection=tuple(matrix.T.flat),
                            view_matrix=tuple(view.T.flat), viewport_size=(size, size),
                            min_diameter_px=.01, shaded=True, stretch=stretch,
                            alignment={'VIEW':0,'VELOCITY':1,'FIXED':2,'AXIS':3}[alignment],
                            fixed_normal=tuple(normal), lock_axis=tuple(axis), blend_mode='OPAQUE')
                    depth = np.array(fb.read_depth(ox, oy, size, size)).reshape(size, size)
                    color = np.array(fb.read_color(ox, oy, size, size, 4, 0, 'FLOAT')).reshape(size, size, 4)
                    actual = color[..., 3] > .5
                    expected, exact = reference(matrix, np.array(center), axes, radius, size, clear_depth)
                    missing = int(np.count_nonzero(expected & ~actual))
                    extra = int(np.count_nonzero(actual & ~expected))
                    common = expected & actual
                    error = float(np.max(np.abs(depth[common]-exact[common]))) if common.any() else 0.
                    result = {'name': name, 'expected_pixels': int(expected.sum()), 'actual_pixels': int(actual.sum()),
                                  'missing': missing, 'extra': extra, 'max_depth_error': error}
                    # At most a one-pixel float32 tangent classification difference.
                    result['passed'] = missing+extra <= 1 and error < 2e-5
                    results.append(result)
                    np.savez_compressed(output/(name+'.npz'), actual_depth=depth, actual_mask=actual,
                                        reference_depth=exact, reference_mask=expected)
            finally:
                gl.delete(vao, vbo, program)
    finally:
        offscreen.free()
    receipt = {'passed': all(r['passed'] for r in results), 'cases': results, 'pid': os.getpid(),
                   'blender': bpy.app.version_string, 'module': cuda_gl.__file__,
                   'shader_strings_sha256': hashlib.sha256((cuda_gl._VERTEX+cuda_gl._FRAGMENT).encode()).hexdigest(),
                   'shader_source_sha256': hashlib.sha256(Path(cuda_gl.__file__).read_bytes()).hexdigest()}
    (output/'sphere-depth-result.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
    assert receipt['passed'], 'Shader differs from analytic reference; see receipt'


run()
