# Bounded Windows qualification — 24 October 2026

Blender 5.1.0, Warp 1.15.0, NVIDIA RTX 4080 SUPER. Canonical CPU suite: 205 tests and 41 subtests passed.

| 100K active/displayed workload | Fresh states/s | p95 display ms | p95 completion ms |
|---|---:|---:|---:|
| IMPACT | 41.958 | 25.175 | 15.937 |
| GALAXY after physical edit | 59.929 | 19.571 | 13.704 |

Each bounded live run measured 120 distinct completed states. Both passed strict CUDA/OpenGL interop without host particle copies. Both galaxies passed five-orbit isolation before/after a timestep edit. Appearance preservation, physical rebuild/resume and interrupted Pause/Single Step/Reset passed. Native tests passed 17 checks including Undo/Redo, duplicate independence, deletion budget, 100K preview and positive/negative pixel controls.

The final slim ZIP was installed/enabled through Blender's supported extension installer in a fresh isolated profile. Every archive member matched the installed file. All 17 native checks passed on the installed package. Existing certified Warp was reused without download; production preferences were untouched. The source fingerprint and archive hash are recorded in PUBLICATION.md.

Two earlier Windows attempts failed on generated CUDA filename length; the existing short cache path setting resolved those failures. Raw evidence is retained privately with its original metadata. This public summary does not redistribute private execution logs. Extended soak, manual file-picker clicks, fresh dependency download and arbitrary custom-model scientific convergence remain unrun/unqualified. No CI success is implied by these local tests.
