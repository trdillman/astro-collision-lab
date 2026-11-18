# Qualification

## 0.3.0: Windows release qualification, 18 November 2026

Blender 5.2 LTS, Warp 1.18.0, NVIDIA RTX 4080 SUPER, Windows 11. The candidate was the exact ZIP recorded in PUBLICATION.md. It was installed through Blender's extension installer into a fresh isolated profile, and every one of its 99 archive members was byte-identical to the installed file, with no extra files. Background GPU use was at most 10% before every GPU run. Timing runs held an exclusive GPU lease, and only runs with a clean lease verdict count.

CPU suite: 412 tests and 41 subtests passed; ruff is clean.

Every live run below passed every check. The bar for each is at least 24 fresh states/s, with p95 at or under 41.67 ms for both display interval and step-to-GL-finish:

| Live workload (all particles active and displayed) | Runs | Fresh states/s | p95 display ms | p95 step-to-GL-finish ms |
|---|---:|---:|---:|---:|
| IMPACT, 100,000 | 3/3 | 46.3-47.7 | <=23.1 | <=15.3 |
| GALAXY, 100,000 | 5/5 | 52.5-53.8 | <=21.3 | <=12.7 |
| GALAXY, 800,000, Compact, depth 16 | 3/3 | 29.2-29.6 | <=36.9 | <=28.3 |

Other native gates on the installed package:
- **Edit workflow:** live runs with edits passed in both modes (IMPACT 46.7, GALAXY 53.3 states/s).
- **Native checks:** 17 native GUI checks passed (Undo/Redo, duplicate independence, deletion budget, positive/negative pixel controls), and 25 RNA authoring checks passed.
- **Interruptions:** interrupted Pause / Single Step / Reset passed 6/6 from a cold kernel cache, including a reset during a rebuild.
- **Isolation:** five-orbit isolation passed on the two-galaxy preset and after a thick-disk edit (mass 3, thickness 1.53). Maximum radius drift was 0.020 and 0.028 against a 0.05 limit.
- **Save/reopen:** save, reopen, then 60 s of play, three rounds per mode, passed 67/67 checks. Reopened files always stayed stopped until Start. IMPACT ran 45.0-45.4 and GALAXY 47.8-49.0 states/s, with exactly 100,000 active particles throughout. A separate reopen-autoplay probe also passed.

Earlier failed or inadmissible runs are kept privately as history. They were traced to the test probes, not the add-on, and the probes were fixed without relaxing any threshold. Raw evidence is retained privately; this public summary does not redistribute private execution logs. Not run: Linux GPU/GUI, manual file-picker clicks, soaks longer than 60 s per round, and scientific convergence of arbitrary custom models.

## 0.2.1: Bounded Windows qualification, 24 October 2026

Blender 5.1.0, Warp 1.15.0, NVIDIA RTX 4080 SUPER. Canonical CPU suite: 205 tests and 41 subtests passed.

| 100K active/displayed workload | Fresh states/s | p95 display ms | p95 completion ms |
|---|---:|---:|---:|
| IMPACT | 41.958 | 25.175 | 15.937 |
| GALAXY after physical edit | 59.929 | 19.571 | 13.704 |

Each bounded live run measured 120 distinct completed states, and both passed strict CUDA/OpenGL interop without host particle copies. Both galaxies passed five-orbit isolation before and after a timestep edit. Appearance preservation, physical rebuild/resume and interrupted Pause / Single Step / Reset passed. The native tests passed 17 checks. The installed slim ZIP matched every archive member. Two earlier attempts failed on generated CUDA filename length; a short cache path resolved them.
