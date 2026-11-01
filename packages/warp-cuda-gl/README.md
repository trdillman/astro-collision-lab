# warp-cuda-gl (`warp_cuda_gl`)

Draw millions of simulation particles without copying them off the GPU. NVIDIA Warp kernels
write position, size and colour straight into an OpenGL vertex buffer that CUDA has
registered, then OpenGL draws it as instanced, shaded billboards. Taken from the Astro
Collision Lab Blender add-on, where it is the live viewport path.

Two parts:

- **`CudaGLBatch`** (`cuda_gl.py`): owns one OpenGL buffer, its CUDA registration and a
  small shader. It talks to the CUDA driver directly through `ctypes`, checks every driver
  status, and never loses track of who owns the buffer, even when a step fails.
- **`DeviceAppearanceWriter`** (`device_appearance.py`): GPU kernels that fill that buffer
  from Warp arrays. Colour by flat colour, per-particle colour, speed, density, damage,
  energy, distance from a point, material, or source object; size uniform or physical.

## Requirements

- Python 3.11+, NumPy, and NVIDIA Warp (the GPU path is proven with `warp-lang==1.15.0`; the CPU tests also pass on 1.18.0, the astro-unified-sdk pin).
- For drawing: Windows, an NVIDIA GPU, and a current OpenGL 4.5 context (or 4.0+ with
  `GL_ARB_clip_control`) on the calling thread, on the same GPU as the Warp device. (Blender's viewport draw callback is one.)
- Importing the package is safe anywhere. The appearance writer can run on Warp's CPU device
  with `allow_cpu=True`, which is how the tests run without a GPU.

## Install

```
pip install "warp-cuda-gl[warp] @ git+https://github.com/trdillman/astro-collision-lab#subdirectory=packages/warp-cuda-gl"
```

## Use

Call everything from the thread that owns the OpenGL context, with that context current.

```python
import types
import warp_cuda_gl as wgl

batch = wgl.CudaGLBatch("cuda:0", capacity=n)        # once per GL context
writer = wgl.DeviceAppearanceWriter("cuda:0")
look = wgl.Appearance(color_source="SPEED", ramp_low=0.0, ramp_high=2.0,
                      size_source="PHYSICAL", radius_multiplier=1.0)

# Each frame: point at your Warp arrays (positions vec3, optional velocities, sizes, ...).
frame = types.SimpleNamespace(channels={"positions": pos, "velocities": vel, "sizes": radius})
writer.validate(frame, look)                          # fails early, before mapping GL memory
batch.update(lambda mapped: writer.write(frame, mapped, look))
batch.draw(n, view_projection=vp16, view_matrix=view16, viewport_size=(width, height),
           blend_mode="OPAQUE", alignment="VIEW", min_diameter_px=2.0)

batch.close()                                         # with the same context current
```

Matrices are 16 floats in column-major order. `blend_mode` is `"OPAQUE"`, `"ALPHA"` or
`"ADDITIVE"`; `alignment` is `"VIEW"`, `"VELOCITY"`, `"FIXED"` or `"AXIS"`. Each point
uses 12 floats (`wgl.STRIDE`): x, y, z, radius, r, g, b, a and four extra slots.

For colour by source object, upload per-object colours once with
`writer.set_metadata(stable_ids=..., colors=..., revision=...)` and give the frame a
`stable_ids` channel.

## Rules that keep it safe

- Never call `update`, `draw` or `close` without the owning context current on the
  original thread; the batch refuses.
- If a driver call fails, the batch marks itself faulted and keeps every handle instead of
  freeing memory it may not own. Close it later from the right context.
- `write` never reads particles back to the CPU or synchronizes the GPU.

## Test

```
pip install -e "packages/warp-cuda-gl[test]"
pytest packages/warp-cuda-gl/tests
```

The deeper contract tests (fake CUDA and OpenGL drivers) live in the add-on's
`tests/test_cuda_gl.py` and `tests/test_device_appearance.py`, which test the same files.
Real GPU drawing is checked with `validation/live_gui_cuda_gl_probe.py` in Blender on a
Windows CUDA machine. `cuda_gl.py` and `device_appearance.py` are kept byte-identical to the
add-on's copies (`python tools/sync_packages.py --check`).
