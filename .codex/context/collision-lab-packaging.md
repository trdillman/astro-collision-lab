# Checkpoint: reusable packages from Astro Collision Lab

Started 2026-11-01 on branch claude/project-thread-sg4m79 (base main a2929ee).

## Loop contract
- Outcome: three standalone, tested, documented packages under `packages/`, made from the
  add-on's own reusable code; add-on keeps working byte-for-byte the same; CI added and green;
  PR open.
- Out of scope: the vendored `_vendor/astro_warp` solver (older copy of what
  trdillman/astro-unified-sdk already ships); `equilibrium.py`, `warm_worker.py`,
  `interaction.py` (tied to the add-on's solver/runtime).
- Owner gates: licence choice for the packages (Tyler); publishing/tagging (Tyler);
  Blender/GPU checks (Tyler's PC). The add-on's files are unchanged, so no new Blender proof
  is needed for this PR.

## Approach
- `packages/<name>/src/...` is the source of truth. Files shared with the add-on are kept
  byte-identical by `tools/sync_packages.py` (`--check` in tests and CI), so the add-on's
  source fingerprint and qualification receipts do not change.

## Progress (2026-11-01)
- [x] astro-initial-conditions (`astro_ics`): galaxy_models + preset_specs (synced) + typed api.py; 15 tests
- [x] warp-cuda-gl (`warp_cuda_gl`): cuda_gl + device_appearance (synced); 7 tests; CPU paths
      also pass on Warp 1.18.0
- [x] pinned-wheel-installer (`pinned_wheel`): generalized rewrite of the Warp installer; 15 tests.
      The add-on's own install_dependency.py is unchanged (it runs as a standalone script in
      background Blender; switching it needs a Blender check on Tyler's PC).
- [x] tools/sync_packages.py + tests/test_package_sync.py
- [x] .github/workflows/ci.yml (Warp 1.15 full suite, Warp 1.18 packages, wheel build)
- [x] Local proof: 244 passed + 41 subtests (CI-like venv, Warp 1.15.0); 69 passed (Warp 1.18.0)
- [ ] PR open + CI green

## Open owner questions
- Licence for the packages: currently LicenseRef-Private-Internal (same as the add-on);
  asked Tyler whether to make them GPL like astro-unified-sdk.

## Resume
Check PR status/CI; if licence answered, update `license` in each packages/*/pyproject.toml.
