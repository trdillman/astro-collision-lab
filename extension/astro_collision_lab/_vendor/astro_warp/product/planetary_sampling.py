"""Historical lattice sphere sampler, isolated from legacy handoffs (4b1f2b41)."""
from __future__ import annotations
import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any
_LATTICE_JITTER=0.05
def _numpy():
    return __import__('numpy')

@dataclass(frozen=True)
class ShellSpec:
    """One concentric layer of an authored body."""

    outer_radius: float
    material_id: int
    density: float

    def validate(self) -> None:
        if self.outer_radius <= 0.0:
            raise ValueError("a shell needs a positive outer radius")
        if self.density <= 0.0:
            raise ValueError("a shell needs a positive density")


@dataclass(frozen=True)
class LayeredBody:
    """A sampled differentiated body: positions, per-shell masses and per-particle materials.

    ``particle_mass`` is the mean particle mass, ``total_mass / len(self)``. Since ADR 0166
    a shell's particles carry that shell's density times the body's volume per particle, so
    a core particle is heavier than a mantle one by the density ratio.
    """

    positions: Any
    velocities: Any
    masses: Any
    material_ids: Any
    radii: Any
    shell_index: Any
    shell_counts: tuple[int, ...]
    shell_masses: tuple[float, ...]
    analytic_shell_masses: tuple[float, ...]
    particle_mass: float
    total_mass: float

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def _shell_lattice(np: Any, generator: Any, inner: float, outer: float, count: int) -> Any:
    """Exactly ``count`` sites of a body-centred cubic lattice filling ``inner <= r < outer``.

    The cell edge is the largest at which the shell holds ``count`` sites, found by
    bisection on sorted site radii (the lattice is built once, in cell units, so each probe
    is a binary search rather than a rebuild). The origin is shifted by a seeded fraction of
    a cell, which breaks the lattice's symmetry so the site count steps nearly one at a
    time; whatever surplus the last step leaves is dropped by the seeded generator.
    """

    volume = (4.0 / 3.0) * math.pi * (outer**3 - inner**3)
    # Two sites per cubic cell, so the edge that gives `count` sites is cbrt(2 V / count).
    nominal = (2.0 * volume / count) ** (1.0 / 3.0)
    origin = generator.random(3)

    def sites(smallest_edge: float) -> tuple[Any, Any]:
        reach = int(math.ceil(outer / smallest_edge)) + 2
        axis = np.arange(-reach, reach + 1, dtype=np.float64)
        cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        cells = np.concatenate((cube + origin, cube + origin + 0.5))
        radii = np.linalg.norm(cells, axis=1)
        order = np.argsort(radii, kind="stable")
        return cells[order], radii[order]

    def held(edge: float, radii: Any) -> int:
        return int(
            np.searchsorted(radii, outer / edge, side="left")
            - np.searchsorted(radii, inner / edge, side="left")
        )

    low = 0.8 * nominal
    cells, radii = sites(low)
    while held(low, radii) < count:
        low *= 0.8
        cells, radii = sites(low)
    high = nominal
    while held(high, radii) >= count:
        high *= 1.25
    for _ in range(60):
        middle = 0.5 * (low + high)
        if held(middle, radii) >= count:
            low = middle
        else:
            high = middle
    edge = low
    start = int(np.searchsorted(radii, inner / edge, side="left"))
    stop = int(np.searchsorted(radii, outer / edge, side="left"))
    chosen = cells[start:stop] * edge
    if chosen.shape[0] > count:
        keep = np.sort(generator.choice(chosen.shape[0], size=count, replace=False))
        chosen = chosen[keep]
    jittered = chosen + generator.normal(scale=_LATTICE_JITTER * edge, size=chosen.shape)
    # A jittered site that left its shell keeps its lattice position, so shell membership -
    # and with it each shell's mass - is unchanged by the jitter.
    distance = np.linalg.norm(jittered, axis=1)
    inside = (distance >= inner) & (distance < outer)
    return np.where(inside[:, None], jittered, chosen)


def _random_rotation(np: Any, generator: Any) -> Any:
    """A rotation drawn uniformly from SO(3).

    A unit quaternion normalised from a 4-D Gaussian is uniform on the 3-sphere, which is
    exactly Haar measure on SO(3) - so no axis is preferred and the deterministic point set
    below picks up no orientation bias from being rotated.
    """

    quaternion = generator.standard_normal(4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def layered_body_mass(shells: tuple[ShellSpec, ...]) -> float:
    """The exact mass of concentric shells, from their radii and densities alone.

    Same volume-times-density sum `sample_layered_body` draws its particle counts from,
    exposed standalone so a caller can price a shell stack's mass without paying for (or
    needing) a particle draw. An authoring tool that hands a differentiated body a mass
    disagreeing with its own material densities by two or three orders of magnitude hands
    the EOS a pressure mismatch violent enough to blow the settling transient past the
    Courant substep budget before it converges - this is that number, so a default never
    starts a body that far from its own equilibrium.
    """

    if not shells:
        raise ValueError("a layered body needs at least one shell")
    previous = 0.0
    total = 0.0
    for shell in shells:
        shell.validate()
        if shell.outer_radius <= previous:
            raise ValueError("shell outer radii must strictly increase from the core out")
        total += shell.density * (4.0 / 3.0) * math.pi * (shell.outer_radius**3 - previous**3)
        previous = shell.outer_radius
    return total


def sample_layered_body(
    shells: tuple[ShellSpec, ...],
    particle_count: int,
    *,
    seed: int | Sequence[int],
    center: Any = (0.0, 0.0, 0.0),
    velocity: Any = (0.0, 0.0, 0.0),
) -> LayeredBody:
    """Sample a differentiated body as one even lattice whose shells differ only in mass.

    Equal spacing, not equal mass (ADR 0166, superseding ADR 0027 section 5). Equal-mass
    shells sit at spacings that differ by the cube root of the density ratio, and a
    mass-weighted kernel sum straddling the interface then reads the heavy side's mass at
    the light side's spacing: on the shipped Asteroid the ice mantle next to the iron core
    started at up to 5.7 times its own density, with a sound speed near 72 km/s, and blew
    outward before anything touched it. So the whole body is one lattice at one spacing,
    each site takes the material of the shell its radius falls in, and each shell's
    particles share that shell's analytic mass equally - which is its density times the
    body's volume per site, to within the lattice's surface noise. The solver reads such a
    body by *number* density (``HydroConfig.number_density``), which has no interface to
    smear across.

    Shell counts are whatever the lattice puts in each shell, so they sum to
    ``particle_count`` exactly, and each shell's sampled mass equals its analytic mass to
    round-off. That is what the gate measures.

    Placement is a lattice, not a random or low-discrepancy draw (ADR 0163). The solver's
    density is the kernel sum over the ~50 neighbours its smoothing length solves for, the
    particle's own mass included. A point set with no minimum spacing - random, or the
    Hammersley set used before - lets neighbours sit arbitrarily close, and under that sum
    it reads about 1 / (pi * 1.49), roughly 21 %, denser than its shells: measured on the
    GPU, a 50 km Asteroid target started 23 % over its reference density, at a pressure
    about 700 times what its own gravity could hold, and blew itself apart before the
    impactor arrived. A lattice has a minimum spacing, which is the condition the kernel
    normalisation assumes, and the same body sampled on one starts within 1 % of its
    reference density and keeps its size. ``seed`` still decorrelates two bodies: it draws
    the lattice origin, the jitter, and the rotation the body is placed under.
    """

    np = _numpy()
    if not shells:
        raise ValueError("a layered body needs at least one shell")
    if particle_count < len(shells):
        raise ValueError("a layered body needs at least one particle per shell")
    previous = 0.0
    for shell in shells:
        shell.validate()
        if shell.outer_radius <= previous:
            raise ValueError("shell outer radii must strictly increase from the core out")
        previous = shell.outer_radius

    inner = np.array([0.0] + [shell.outer_radius for shell in shells[:-1]], dtype=np.float64)
    outer = np.array([shell.outer_radius for shell in shells], dtype=np.float64)
    densities = np.array([shell.density for shell in shells], dtype=np.float64)
    volumes = (4.0 / 3.0) * np.pi * (outer**3 - inner**3)
    analytic = densities * volumes
    total_mass = float(analytic.sum())

    # Handed straight to numpy rather than narrowed to an int: `sampling.py` seeds every
    # other body with `[authored seed, stable id]` so two bindings sharing an authored seed
    # still draw different clouds, and a differentiated body has to seed the same way or it
    # is the one representation where that decorrelation silently stops happening.
    generator = np.random.default_rng(seed)
    body_radius = float(outer[-1])
    # One lattice for the whole body, so the spacing does not jump at an interface. The
    # jitter keeps every site inside the ball; a site's shell is then read off its radius.
    positions = _shell_lattice(np, generator, 0.0, body_radius, particle_count)
    positions = positions @ _random_rotation(np, generator)
    distance = np.linalg.norm(positions, axis=1)
    shell_index = np.searchsorted(outer, distance, side="right").astype(np.int32)
    shell_index = np.minimum(shell_index, len(shells) - 1)
    counts = np.bincount(shell_index, minlength=len(shells)).astype(np.int64)
    # A shell thinner than the spacing can catch no site. It takes the sites nearest its
    # middle radius from a neighbour that can spare them, so its mass is never dropped.
    for row in np.flatnonzero(counts == 0):
        middle = 0.5 * (inner[row] + outer[row])
        donors = np.flatnonzero(counts[shell_index] > 1)
        nearest = int(donors[np.argmin(np.abs(distance[donors] - middle))])
        counts[shell_index[nearest]] -= 1
        shell_index[nearest] = row
        counts[row] = 1

    masses = (analytic / counts)[shell_index]
    material_ids = np.array([shell.material_id for shell in shells], dtype=np.int64)[
        shell_index
    ]
    # One spacing for every particle, which is what the neighbour stage turns into a
    # smoothing length.
    radii = np.full(particle_count, float(np.cbrt(volumes.sum() / particle_count)))

    positions = positions + np.asarray(center, dtype=np.float64)
    velocities = np.tile(np.asarray(velocity, dtype=np.float64), (particle_count, 1))
    return LayeredBody(
        positions=positions,
        velocities=velocities,
        masses=masses,
        material_ids=material_ids,
        radii=radii,
        shell_index=shell_index,
        shell_counts=tuple(int(value) for value in counts),
        shell_masses=tuple(
            float(masses[shell_index == row].sum()) for row in range(len(shells))
        ),
        analytic_shell_masses=tuple(float(value) for value in analytic),
        particle_mass=total_mass / particle_count,
        total_mass=total_mass,
    )


