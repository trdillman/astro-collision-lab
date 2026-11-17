# Independent simulation adapter

```python
from astro_collision_lab.simulation import BodySpec, SimulationConfig, AstroCollisionSimulation

config = SimulationConfig(mode='IMPACT', particle_count=100000,
    gravitational_constant=1.0, dt=0.005, softening=0.03, theta=0.6)
bodies = [
    BodySpec('Target', particle_count=80000, radius=1.0, mass=1.0, position=(-0.5,0,0)),
    BodySpec('Impactor', particle_count=20000, radius=0.6, mass=0.15,
             position=(2.2,0.4,0), velocity=(-1.0,-0.1,0)),
]
sim = AstroCollisionSimulation(config)
sim.initialize(bodies)             # CPU primitive sampling, one upload, CUDA initialization
metadata = sim.authoring_metadata() # cache once: stable_ids int64, source_index int32, source_names
gpu_channels = sim.step()           # GPU handles, no CPU particle snapshots/history
gpu_channels = sim.display_channels()
print(sim.count, sim.physical_time, sim.stats)
sim.close()
```

`BodySpec` also accepts `kind` SPHERE/DISK/HALO, a row-major proper 3x3 `rotation`, disk
`thickness`, `spin` in inverse code-time, and `material` ROCK/IRON/FLUID/TILLOTSON. Original
synthetic material spellings are accepted. Rotation maps local offsets to world coordinates
as `rotation @ local`; mass, radius and initial velocities are explicit normalized values.

GALAXY uses the pinned collisionless Barnes-Hut core. Disk velocities are initialized using
an approximate analytic enclosed-mass model; halo dispersion uses a uniform sphere estimate.
These are initial-condition approximations, not a validated equilibrium galaxy library.
All authored sources sum to exactly the configured count. No sampling cap or idle mirror is
used. The usual scene is 35k disk + 15k halo for each of two galaxies.

Galaxy configs accept `tree_max_depth` (integer 1–16, default 16) and `tree_storage`
(`STRIDED` default, or `COMPACT`). Both are part of physical and isolation identity;
changing either requires initialization from authored inputs. The adapter passes the
chosen depth and matching `1 + 7 * depth` stack capacity to the actual tree configuration.
`galaxy_tree_config(config)` is shared by preflight and initialization. Reserved storage
rejects its worst-case node limit before CPU sampling or GPU allocation.

Compact storage is implemented in the Lab-owned `compact_tree.py`, without modifying
the vendored solver. It replaces topology storage/addressing and retains the vendor
force equations. Its storage counters distinguish occupied nodes and allocated capacity.
The compact path uses eager GPU submission so allocation checks and growth occur outside
capture. This behavior needs its own performance qualification; memory reduction alone
does not establish the project's real-time gate.

IMPACT uses the SpaceSim-inspired IISPH pressure and bond-fracture core. Above 32768 live
particles the pinned core's AUTO gravity chooses Barnes-Hut. The adapter builds an exact
identity source map itself, rather than using the vendor sampler that can leave idle slots.
The thermal channel is a shock/fracture heat proxy, not thermodynamic temperature.

COMPRESSIBLE uses the recovered explicit SPH/CFL/energy core with tree self-gravity. Use
material TILLOTSON for the normalized energy-dependent EOS; ROCK/IRON/FLUID select cold
Murnaghan. Nothing silently replaces an authored material. All constants are synthetic,
and no result should be claimed as validated Moon formation. No moon guide is instantiated.

The live API never calls snapshot(), diagnostics(), fragment_report(), or numpy() on a
particle device array. Runtime small nonfinite, adaptive-time and overflow counters remain.
Initialization retains some host source arrays required by vendor ownership, plus static
source-id metadata; it creates no per-frame host cache. Channels stay in the original source
identity order even where an engine internally uses Morton order.

Device channels: positions/velocities vec3 float32 or float64; stable_ids int64 and active
int32; masses; radii (all modes); IMPACT/COMPRESSIBLE densities, internal_energies and
material_ids int64, with damage when meaningful. COMPRESSIBLE physical radii are updated on the GPU from mass and density into one persistent
float32 buffer at initialization and each step, with no particle readback. GALAXY has no physical material, density or thermal state.

`step(dt)` accepts only the initialized dt. Changing it requires reinitialization to avoid
replaying stale captured graphs. Authored dt is never shortened automatically. IMPACT adapts
its own inner steps; a config `substeps` other than one is refused for that mode. `stats`
reports actual softening, theta and gravity backend, physical time and synchronized solver
wall time. These numbers are not complete viewport FPS; the installed draw harness must
measure solver + packing + actual viewport drawing at exactly 100k active particles.


## Solver diagnostic

Run `python tests/validation/solver_probe.py --mode IMPACT --count 512 --frames 3` for a
small CUDA smoke run, then `--count 100000 --frames 120` for the required full count.
GALAXY and COMPRESSIBLE use the same CLI. The probe refuses CPU fallback, uses the same
primitive presets with exact largest-remainder count scaling, and defaults to five warm-up
frames. COMPRESSIBLE explicitly selects normalized TILLOTSON; the UI's ROCK remains Murnaghan.
Use `--dt` to explicitly change the requested physical interval; defaults are .005 impact/
compressible and .02 galaxy. The report retains its chosen configuration and full source specs.

Reports default to `.local/evidence/solver/solver_<mode>_<count>.json`; `--output PATH` overrides.
Initialization includes sampling, upload, settling and first device radius update. Timed
frames include solver work plus the device radius update and synchronize completion before
the stop timestamp. Motion validation copies positions before/after the measured sequence;
finite-state checks and exact active count readback occur afterward. Small substep diagnostic
reads occur between timed frames and their excluded cost is recorded separately. Frame
physical advances and actual inner-step counts are reported. SpaceSim adaptive per-step dt
is summarized by average integrated dt and its planned dt, not falsely reported as fixed.

This is a solver diagnostic, not live viewport acceptance: it does not pack OpenGL buffers,
submit a Blender draw, or establish equivalent SpaceSim benchmark conditions.
