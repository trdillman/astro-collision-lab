# Astro Collision Lab solver dependency

The dependency under `astro_warp/` is pinned to the user's private repository
`trdillman/astro-warp-blender` commit `e5ebdc970a3633bc61cfe8909e28923f97ca2cfc`.
Copy `src/astro_warp/` from that exact commit into this directory, preserving all package
relative imports. Do not copy the original Blender extension, installed settings, cache,
or user working tree. No absolute `astro_warp` imports were found in the product closure.

The independent add-on imports it only as `astro_collision_lab._vendor.astro_warp`.
No old add-on UI, object roles, registration identity, saved graphs, guided Moon runtime,
product checkpoints or CPU history transport is used by `simulation.py`.

Apply `standalone_core.patch` from the new project root. It adds:

- `product/sph.py`, `hydro.py`, `selfgravity.py`, `solids.py`, `sph_runtime.py`, recovered
  from the pre-removal commit `4b1f2b4100abd39f16ec42ec95eceb7da82f8495`.
- `product/planetary_sampling.py`, the historical sphere lattice sampler isolated from
  legacy stage handoffs.
- `product/planetary_sph_runtime.py`, an independent array-driven publication bridge.
- A narrow change to `spacesim_impact_runtime.py`: optional tree angle and world-unit
  gravitational softening pass into the existing engine settings. `None` preserves the
  pinned dependency defaults. The Lab passes its explicit normalized controls.

The recovered SPH adapter also keeps authored zero specific energy for condensed bodies
instead of deriving gas virial energy, and uses a distinct checkpoint backend identity.
No original product registry/model/factory/scheduler/Blender files are changed.

Attribution: the underlying solver implementation is the user's Astro Warp project.
This record identifies source provenance; it does not assert a third-party public license.
Existing source notices are retained. The SpaceSim-inspired IISPH/bond equations and their
limitations remain documented in the vendored source. FLUIDS source was not copied.

The material constants created by the Lab adapter are explicitly normalized synthetic
fixtures. They are not a sourced Earth/iron/ANEOS material library. The unguided
compressible option has no published giant-impact validation or measured 100k viewport
performance yet. The default collisionless and IISPH modes also require the new add-on's
own 100k active-particle live-viewport acceptance measurements.
