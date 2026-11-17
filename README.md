# Astro Collision Lab 0.2.2 candidate

Windows x64 Blender 5.1 add-on for primitive-based planetary and galaxy collision previews using CUDA and OpenGL. This is an experimental simulation tool; the bounded qualification below is not scientific validation of Moon formation or arbitrary galaxy models.

## Build and install

Run `python tools/build.py` to build the slim extension ZIP under `artifacts/`. Install it using Blender Preferences > Get Extensions > Install from Disk, then enable Astro Collision Lab. Warp 1.18.0 is a separately managed dependency; the add-on provides an explicit dependency install action. No runtime wheel or binary is bundled. Historical installation checks reused an existing certified Warp installation; they did not test a fresh dependency download or mouse-driven file picker.

Use the Astro Collision controls in the 3D View sidebar to author sources, preview, Play, Pause, Single Step and Reset. Physical edits rebuild the simulation; appearance edits preserve its state.

## Verification

See [historical qualification](docs/QUALIFICATION.md) and [publication provenance](docs/PUBLICATION.md). CPU tests: `python -m pytest -q`; the complete suite requires NumPy, pytest and separately installed Warp 1.18.0. Native scripts under `validation/` and `tests/validation/` require Blender and appropriate hardware. Machine-specific launchers and private historical reports are deliberately excluded from this publication.

For reproducible main/candidate CUDA solver comparisons, see the [paired benchmark instructions](validation/PAIRED_SOLVER_BENCHMARK.md). Solver timing and live viewport cadence are separate measurements.

This candidate adds editable Galaxy tree depth, opt-in compact node storage, corrected shaded-sphere depth, and clearer isolation status. Controls are in **AstroCollide > Simulation**; physical edits rebuild from authored initial conditions. See the [user guide](docs/USER_GUIDE.md) and [simulation API](docs/SIMULATION_API.md). Existing standalone packages are retained, including the shared renderer correction. These changes do not promote the historical qualification to a new release or establish the 24 updates/s performance target.

## Reusable packages

Three parts of the add-on are also published as standalone Python packages under
[`packages/`](packages/): galaxy and impact starting setups (`astro_ics`), zero-copy
CUDA/OpenGL particle drawing (`warp_cuda_gl`), and a hash-pinned dependency installer
(`pinned_wheel`). See [packages/README.md](packages/README.md).

## Licensing

The existing manifest declares `SPDX:LicenseRef-Private-Internal`. Public repository visibility does not grant an open-source license or additional reuse rights. Existing source notices and vendored solver provenance are retained. See `extension/astro_collision_lab/_vendor/VENDOR.md` and `docs/PROVENANCE.json`.
