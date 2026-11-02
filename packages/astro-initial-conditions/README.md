# astro-initial-conditions (`astro_ics`)

Stable starting setups for galaxy and planetary-impact simulations, as plain NumPy arrays.
No Blender, no GPU, no solver needed. Taken from the Astro Collision Lab add-on, where it
builds every galaxy before the GPU solver runs.

What it makes:

- **Plummer spheres** (stars, bulges, dark halos). Velocities are drawn from the exact
  distribution function, so an isolated sphere starts in equilibrium.
- **Exponential disks** with a sech² vertical profile. Rotation comes from the real
  rotation curve of the whole galaxy, with Toomre-Q radial dispersion, asymmetric drift and
  vertical Jeans support.
- **Disk galaxies**: a disk and halo that feel each other's gravity (shared `galaxy_id`).
- **Presets**: the add-on's two-galaxy collision and Moon-forming impact scenes.
- **Stability checks**: Lagrangian radii, energy, angular momentum, bound fraction, and a
  pass/fail drift test you can run on your own simulation's snapshots.

All values are in normalized simulation units (G = 1 by default). Same seed, same particles.

## Install

```
pip install "astro-initial-conditions @ git+https://github.com/trdillman/astro-collision-lab#subdirectory=packages/astro-initial-conditions"
```

or from a checkout: `pip install -e packages/astro-initial-conditions`. Needs Python 3.11+ and
NumPy 2.

## Use

```python
import astro_ics as ics

# One galaxy: 20% disk, 80% halo, 50k particles.
galaxy = ics.disk_galaxy("milky", particle_count=50_000)
s = ics.sample(galaxy)
s.positions, s.velocities, s.masses, s.radii   # (N,3), (N,3), (N,), (N,)
s.component                                    # which component each particle came from

# Two galaxies on a collision course, the second one tilted.
a = ics.disk_galaxy("a", position=(-8, 0, 0), velocity=(0.12, 0, 0), seed=0)
b = ics.disk_galaxy("b", position=(8, 0, 0), velocity=(-0.12, 0, 0), seed=1,
                    rotation=ics.euler_xyz_matrix((0.6, 0, 0)))
s = ics.sample(a + b, ics.Config(softening=0.03))

# Build components by hand.
bulge = ics.plummer_sphere("bulge", kind="BULGE", particle_count=5_000, mass=0.1,
                           radius=1.0, galaxy_id="milky")

# Ready-made scenes.
components, config = ics.preset("galaxy_collision")   # or "moon_collision"

# Check stability of your own run: compare snapshots to the start.
base = ics.diagnostics(p0, v0, m)
later = [ics.diagnostics(p, v, m) for p, v in snapshots]
print(ics.assess(base, later, periods=6))   # {'pass': True, 'max_energy_drift': ..., ...}
```

`ics.preview(components, limit=2000)` gives fast positions-only points for drawing a setup
before you commit to the full sample.

## Notes and limits

- Velocities for Plummer spheres that share a group with differently-sized spheres or a disk
  come from an Eddington inversion of the combined potential. If the mix has no positive
  distribution function, `sample` raises `ValueError` instead of returning a bad setup.
- Disk setups are good candidates for equilibrium, not proofs. Run `assess` on your own
  solver's output to confirm.
- `galaxy_models.py` and `preset_specs.py` are kept byte-identical to the add-on's copies
  (`python tools/sync_packages.py --check` from the repo root). Edit them here, then run
  `python tools/sync_packages.py` to update the add-on.

## Test

```
pip install -e "packages/astro-initial-conditions[test]"
pytest packages/astro-initial-conditions/tests
```
