# Reusable packages

Small, standalone Python packages taken from the best parts of Astro Collision Lab, so
future projects (and Claude or Codex) can install them instead of rebuilding them. None of
them needs Blender. None depends on the add-on or on its vendored solver.

| Package | Import | What it gives you | Needs |
|---|---|---|---|
| [astro-initial-conditions](astro-initial-conditions/) | `astro_ics` | Galaxy setups built to start in equilibrium (Plummer spheres, exponential disks, halos; disks are candidates to confirm with `assess`), the Moon-impact and two-galaxy presets, and stability checks, as NumPy arrays | NumPy 2 |
| [warp-cuda-gl](warp-cuda-gl/) | `warp_cuda_gl` | Zero-copy particle drawing: Warp writes straight into an OpenGL buffer, colour/size by speed, density, material and more | Warp; Windows + CUDA + OpenGL to draw |
| [pinned-wheel-installer](pinned-wheel-installer/) | `pinned_wheel` | Install one exact, hash-checked wheel into a private folder without pip, including from a Blender add-on | Standard library |

Install any of them straight from GitHub, for example:

```
pip install "astro-initial-conditions @ git+https://github.com/trdillman/astro-collision-lab#subdirectory=packages/astro-initial-conditions"
```

## How they fit with astro-unified-sdk

[astro-unified-sdk](https://github.com/trdillman/astro-unified-sdk) is the physics engine
(gravity tree, contacts, unified matter) as a plain-Python SDK. These packages do the jobs
around an engine: build the starting particles (`astro_ics`), draw them live
(`warp_cuda_gl`), and ship the Warp dependency safely inside an app (`pinned_wheel`).
They do not import the SDK, so they work with any solver. When you do use the SDK, import
it only from `astro_warp.sdk`, which is the part with a compatibility promise.

The add-on's vendored `_vendor/astro_warp` is an older copy of the same engine the SDK ships,
so it is deliberately not repackaged here.

## Rules for changing them

- The package folder is the source of truth. Some modules are also used by the add-on and
  are kept byte-identical: edit the package copy, then run `python tools/sync_packages.py`.
  `python tools/sync_packages.py --check` runs in the tests and CI and fails on drift.
- Byte-identical copies mean the add-on's source fingerprint and its Blender/GPU
  qualification are unaffected by packaging. Changing a shared module does change the
  add-on, and needs the add-on's normal native checks on a Windows CUDA machine.
- Each package has its own tests (`pytest packages/<name>/tests`), README and version.
