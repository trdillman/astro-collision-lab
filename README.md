# Astro Collision Lab 0.3.0

Windows x64 Blender 5.1+ add-on for authoring and live-simulating galaxy collisions and planetary (Moon-style) impacts on an NVIDIA GPU, using CUDA and a strict zero-copy CUDA/OpenGL display. It runs 100,000 simulated and displayed particles in real time, and two-galaxy scenes run at up to 800,000 particles with Compact tree storage. This is an experimental simulation tool. The bounded qualification below is not a scientific validation of Moon formation or of arbitrary galaxy models.

## Build and install

1. Run `python tools/build.py` to build the slim extension ZIP under `artifacts/`.
2. Install it with Blender Preferences > Get Extensions > Install from Disk, then enable Astro Collision Lab.
3. In the add-on preferences, press **Install certified Warp 1.18.0**. Warp is a separately managed, hash-pinned dependency, and an interrupted download resumes when you press Install again. The ZIP bundles no runtime wheel or binary.

Use the AstroCollide sidebar in the 3D View: choose **Galaxy Collision** or **Moon Collision**, or add sources with Shift+A > Astro Collision Lab. Move, rotate and scale sources with G / R / S, then press **Start**. Pause, Single Step and Reset work at any time, including while kernels compile.

- Appearance edits apply instantly and never restart physics.
- Physical edits rebuild from the edited starting state and resume.
- **Validate Galaxy Isolation** runs the five-orbit stability gate on the exact model.
- **Tree storage > Compact** enables the large-scene Galaxy solver.

## What's new in 0.3.0

- **800K Galaxy real time.** Compact tree storage uses a stable radix sort and a warp-converged tree walk, both exact against the vendored solver. An 800K step dropped from 68.5 ms to 21.7 ms.
- **Thick-disk isolation.** Thick-disk physical edits (for example mass 3 with thickness 1.53) now pass the five-orbit isolation gate. Disk velocities are relaxed in the model's own field, and the gate uses an exact softened potential energy.
- **No vanishing particles.** Particles no longer vanish during viewport navigation with overlays off.
- **Correct sphere depth.** Shaded-sphere depth is perspective-correct.
- **Runtime.** Warp 1.18.0 is the certified runtime; Blender 5.2 LTS is qualified.

## Verification

See [qualification](docs/QUALIFICATION.md) and [publication provenance](docs/PUBLICATION.md). The CPU suite is `pip install -e ".[dev]"` and then `python -m pytest -q`; the complete suite also needs a separately installed Warp. Native Blender/GPU probes are in `tests/validation/` and require Blender and suitable hardware. Machine-specific launchers and private historical reports are deliberately excluded from this publication.

## Reusable packages

Three parts of the add-on are also published as standalone Python packages under
[`packages/`](packages/): galaxy and impact starting setups (`astro_ics`), zero-copy
CUDA/OpenGL particle drawing (`warp_cuda_gl`), and a hash-pinned dependency installer
(`pinned_wheel`). See [packages/README.md](packages/README.md).

## Licensing

Since 0.3.0 the add-on is licensed GPL-3.0-or-later (manifest `SPDX:GPL-3.0-or-later`; see `LICENSE`). The packages under `packages/` keep their own licensing terms. Existing source notices and vendored solver provenance are retained; see `extension/astro_collision_lab/_vendor/VENDOR.md` and `docs/PROVENANCE.json`.
