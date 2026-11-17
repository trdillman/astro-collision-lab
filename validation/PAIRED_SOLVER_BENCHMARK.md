# Paired CUDA solver benchmark

`paired_solver_probe.py` extends `solver_probe.py` and measures completed
`AstroCollisionSimulation.step()` calls through CUDA device synchronization.
Initialization/JIT, readbacks for validation, and viewport drawing are excluded.
This benchmark cannot establish displayed viewport FPS.

Create immutable source snapshots with `git archive` of the full baseline and
candidate revisions. Pass their `extension/astro_collision_lab` directories to
`prepare`. Both solvers must support the same Warp runtime; the original public
main solver supports Warp 1.18, although its GUI requires Warp 1.15.

```powershell
python validation/paired_solver_probe.py prepare --baseline BASELINE_ADDON_ROOT --candidate CANDIDATE_ADDON_ROOT --baseline-revision FULL_BASELINE_SHA --candidate-revision FULL_CANDIDATE_SHA --dependency WARP_118_IMPORT_ROOT --cache SHORT_OWNED_CACHE --output NEW_STUDY_DIRECTORY --storage COMPACT
python validation/paired_solver_probe.py run --output NEW_STUDY_DIRECTORY
python validation/paired_solver_probe.py analyze --output NEW_STUDY_DIRECTORY
```

`prepare` needs NumPy on the host. Trial dependencies are imported inside Blender
from the supplied runtime root. Run orchestration from a terminal where `bgate`
resolves to the installed Blender gate CLI. No regular Blender profile is used.
Do not modify frozen source snapshots or the verifier during a study.

Defaults are 100,000 and 400,000 particles, depth 16, leaf capacity 8, FP32,
theta 0.6, softening 0.03, dt 0.02, 20 warmup frames and 120 measured frames.
Eight fresh-process A/A and eight counterbalanced A/B pairs are required per
particle count. Pair seeds, initial GPU arrays, active counts and physical time
must match. Full-channel finite checks, sampled paired trajectory comparison,
an unequal-mass direct force reference and actual GPU-output corruption protect
the quality boundary. The trajectory tolerance is an engineering regression
oracle, not scientific validation.

Each trial uses its own exclusive gate lease. The verifier waits outside timing
for actual monitor samples before and after the work. Only clean final receipts
are accepted. Failed or contaminated attempts remain on disk; choose a new
study directory after an invalid attempt. `--resume` continues an interrupted
valid prefix; `--limit N` is a development pilot and cannot yield a full report.

The report uses run-level median ratios and a conservative order-statistic
interval. With eight pairs the achieved confidence is 99.21875%. A/A bounds must
lie within the predeclared 5% engineering repeatability envelope. A speedup
requires the full A/B interval above one; the full interval below one is a
regression. Otherwise the result is inconclusive. Each workload is reported
separately. Allocated solver array bytes are component memory, not peak VRAM.

Use `--storage STRIDED` in a separate study to compare the unchanged default
storage path. Above main's depth-16 admission limit, report capacity separately;
there is no valid baseline speedup ratio when the baseline cannot start.

Keep `contract.json`, `controls.json`, `records.json`, `report.json` and every
trial's `spec.json`, `result.json`, `lease.json` and `process.log`. Analysis checks
executed-file hashes and rejects missing trials, stale source, mismatched work,
CPU fallback, unclean leases and altered aggregate evidence.
