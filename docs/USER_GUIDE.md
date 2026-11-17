# User guide (0.2.2)

Native Blender 5.1+ authoring and live simulation for Windows x64, using the separately installed, hash-pinned Warp 1.18.0 runtime and the CUDA/OpenGL particle display. An NVIDIA GPU is required for live simulation.

## Install
1. Build with `python tools/build.py` (writes `artifacts/astro_collision_lab-<version>-windows-x64.zip`) or take the released ZIP. Install it from Blender: Preferences > Get Extensions > Install from Disk. Enable Astro Collision Lab.
2. In the add-on preferences press **Install certified Warp 1.18.0** if Warp is missing. The ZIP deliberately contains no wheels or binaries. If the connection drops, reopen Preferences and press Install again: the partial download resumes (at most four transfer attempts per press; log at `install.log` in the extension-owned dependency directory). Wheel hash is verified before extraction. A different already-loaded Warp needs a Blender restart.
3. Open the AstroCollide sidebar, pick **Galaxy Collision** or **Moon Collision**, edit, press **Start**. First-use kernel compilation can take time.

## What you can do
- **Add objects:** Shift+A > Astro Collision Lab > Galaxy, Stellar Disk, Stellar Bulge, Dark Matter Halo, Planet / Impactor. Settings live in Object Properties and the sidebar.
- **Move, rotate, scale** with G / R / S. Preview samples are reused for move/rotate and regenerated for shape, size, count or seed changes; full authored counts are kept.
- **Appearance** (color, point size, blending, diagnostic colors) redraws previews and live particles immediately and never restarts physics.
- **Physical edits** replace the evolved display with an initial-state preview; after a 350 ms quiet period the run rebuilds from the latest inputs and follows the latest Play/Pause/Step choice. This changes starting conditions; it does not push an evolved galaxy or apply a live force.
- **Rename** freely: realization and trajectory are unchanged. Replacing a source with a different object rebuilds ownership.
- **Undo/Redo** restores authoring choices; runtime resources are released and reacquired. Opening another file stops the run.
- **Invalid inputs** stop preparation with an explanation; correcting them resumes the retained playback choice.

| Control | Behaviour |
|---|---|
| Start | Prepare current settings if needed, then advance continuously; an unchanged paused run resumes. |
| Pause | Stop updates and block automatic playback; preparation may finish but stays paused. |
| Single Step | Use the latest settings; show the initial state, advance exactly once, stay paused. |
| Reset | Stop, cancel pending restart, return to authoring. |

Controls are independent of the Blender timeline. "Simulation time per step" is model time per update, not screen refresh speed. Repeated presses during preparation replace the pending choice; there is no hidden queue.

### Galaxy tree controls

In **AstroCollide > Simulation**, **Maximum tree depth** selects 1–16 levels (default 16).
It controls the smallest cells the collisionless gravity tree may create. A shallow tree
can leave many particles in a terminal cell, requiring expensive direct interactions.
A deeper tree can split dense cores inside an extended scene, but requires more tree
construction and storage. Depth is not an accuracy slider: opening angle, softening,
particle distribution and the integration time step also affect the result.

Changing depth is a physical edit: it rebuilds from the authored starting conditions,
then follows the latest Start/Pause/Step choice. It does not retessellate an evolved
state in place. Previously saved scenes retain depth 16 unless explicitly edited.

**Tree storage** offers **Reserved** (the original layout and default) and **Compact**.
Reserved allocates `1 + total_particles * depth` node slots, with a nine-million-node
ceiling. Counts sum across every enabled source, including both galaxies' components.
Capacity errors report that total and the selected depth before sampling begins.

Compact uses a Lab-owned tree adapter, preserving the pinned vendor files and force
equations. Its occupied, reachable nodes use compact addresses; terminal cells stop at
the configured leaf capacity or maximum depth. The sidebar shows actual used nodes
and allocated capacity. Reusable buffers may retain capacity after a larger tree has
been built. Compact submission checks allocation outside GPU capture, so reduced
memory use is not a promise of faster playback. Both modes retain overflow checks;
available VRAM and other per-particle buffers still bound feasible scene size.

## Galaxies
Families: disk / spiral progenitor, lenticular, elliptical / spheroidal, dwarf. Structure is authored on the parent Galaxy; children inherit mass, particle allocation, motion, seed and structure (colors stay editable). Deleting a managed child recreates it; changing a mass fraction changes the recipe; duplicate tags, orphaned children or split group membership produce an explicit error. Default Galaxy = 50,000 particles; the two-galaxy and Moon demos use exactly 100,000. Components are fixed exponential / sech-squared disks and spherical Plummer spheroids.

Initial conditions are approximate. Press **Validate Galaxy Isolation** to run the five-period gate on the exact model in a background Blender process; results are tied to the model, solver settings, code, runtime and device, and go stale when anything changes.

## Known limits
Spherical proxies use the largest scale as radius. No galaxy gas, star formation, barred equilibria or arbitrary profiles; materials are normalized synthetic models, not a validated Moon-formation model. The unguided compressible mode has no published giant-impact validation. Known open defects: thick-disk physical edits can fail isolation (e.g. mass 3 with thickness 1.53), particles can vanish during viewport navigation with overlays disabled, and shaded-sphere silhouette depth is not perspective-correct. None is claimed fixed.
