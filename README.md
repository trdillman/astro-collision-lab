# Astro Collision Lab 0.2.1

Windows x64 Blender 5.1 add-on for primitive-based planetary and galaxy collision previews using CUDA and OpenGL. This is an experimental simulation tool; the bounded qualification below is not scientific validation of Moon formation or arbitrary galaxy models.

## Build and install

Run `python tools/build.py` to build the slim extension ZIP under `artifacts/`. Install it using Blender Preferences > Get Extensions > Install from Disk, then enable Astro Collision Lab. Warp 1.15.0 is a separately managed dependency; the add-on provides an explicit dependency install action. No runtime wheel or binary is bundled. The verified installation test reused an existing certified Warp installation; it did not test a fresh dependency download or mouse-driven file picker.

Use the Astro Collision controls in the 3D View sidebar to author sources, preview, Play, Pause, Single Step and Reset. Physical edits rebuild the simulation; appearance edits preserve its state.

## Verification

See [qualification](docs/QUALIFICATION.md) and [publication provenance](docs/PUBLICATION.md). CPU tests: `python -m pytest -q`; the complete suite requires NumPy, pytest and separately installed Warp 1.15.0. Native scripts under `validation/` require Blender and appropriate hardware. Machine-specific launchers and private historical reports are deliberately excluded from this publication.

## Licensing

The existing manifest declares `SPDX:LicenseRef-Private-Internal`. Public repository visibility does not grant an open-source license or additional reuse rights. Existing source notices and vendored solver provenance are retained. See `extension/astro_collision_lab/_vendor/VENDOR.md` and `docs/PROVENANCE.json`.
