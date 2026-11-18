# Publication provenance and development workflow

## 0.3.0 (18 November 2026)

This update derives from canonical local tag `v0.3.0`, commit `4e89620115b335895900ab25c21da6fcaea32898`. It was exported through the same reviewed allowlist as 0.2.1 and applied as one normal commit. The add-on, tests, native probes (now under `tests/validation/`), build and session tools, `LICENSE` and `pyproject.toml` are included. The committed add-on files are identical to canonical v0.3.0. The three package modules shared with the add-on were refreshed from the qualified add-on bytes, so `tools/sync_packages.py --check` passes; `astro-initial-conditions` and `warp-cuda-gl` are now 0.2.0.

Source fingerprint: `3041e9d0bad0d46b491d69bea378923ac5218ac68558a5a90d0c24e954a036b0`.
Qualified slim ZIP SHA256: `bf25a9879d36c2c8000bc304cc41de4094280aa4931e87741fd39c77a1b21ed2` (750715 bytes).
Certified Warp 1.18.0 wheel SHA256: `1769e55746739f2e244a2b412123003c462ec3d977dfb5f420f924a8553d30aa`.

Reproducing from git: a fresh clone of this repository (or of canonical v0.3.0) builds with `python tools/build.py` to ZIP SHA256 `b4a86a0d10b0ec5db92e60dc5ae4504e238a048ef06454870ff15d5e8c82865f`, source fingerprint `d5bcd13d93f8dfe18d0e3d6f19675be9ba16d59d3f741746eace6868771fae1c`. It differs from the qualified ZIP only in line endings. When the qualified ZIP was built, the private working tree had `compact_tree.py` and `galaxy_models.py` checked out with CRLF endings, while git stores them as LF (`.gitattributes`: `eol=lf`). The 99 members, their names and all other bytes match, and the Python code is identical apart from line endings.

## 0.2.1 (initial publication)

This sanitized initial publication derives from canonical local commit `a3a6734889e48f71518cee06d5e90e8245b56d46`. Its runtime Python and manifest bytes are unchanged from the qualified build. Private conversation/tracker snapshots, host execution metadata, original history, machine-specific scripts and dependency binaries are excluded; they remain preserved locally.

The existing canonical repository remains the single development source. This checkout is a publication artifact, not a second development lane. Future changes are implemented and qualified in canonical development, then exported through the same reviewed allowlist into the public repository and normal-pushed as new commits. Never force-push or replace canonical history to synchronize publication. Record canonical-to-public commit mappings locally after each publication.

Runtime fingerprint: `ca532342bdd50920f2ad11d7a7d87194590ebe685520271c4993d36cc1aa6553`.
Qualified slim ZIP SHA256: `756e2608da4670ea0d811272c5c34354ac5ae04b9849403f4fd6201a3dc00daf` (708078 bytes).
Original 0.2.1 source ZIP SHA256: `a751f1bb8534ba9f300eab443d777688f8179d769eb25219761e45a4edf33f32`.
Original 0.2.0 source ZIP SHA256: `871902523ddaefdd39c7b414599dd24fb11af061e918ccc7cbfe8778c3e9411e`.
