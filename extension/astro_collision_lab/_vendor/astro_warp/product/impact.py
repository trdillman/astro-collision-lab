"""Impact setup, shell layouts and fragment diagnostics.

Three host-side pieces of the impact story, and they are host-side on purpose. The
encounter prediction runs *before* a simulation exists - that is the whole point of it, per
`USER_STORY_AND_WORKFLOWS.md`: the user chooses a setup by seeing the encounter it produces
rather than by paying for a bake and looking. The shell layout prices a layered body's mass
once, at authoring time. The fragment clustering runs once, after a run. None of the three
is per-step work, so none of them belongs in a kernel.

The compressible impact engine that used to sample layered bodies and settle fragments in
this module was deleted on the owner's order of 2026-10-13; the realtime rock and giant
engines read the shell layout and the fragment clustering from here.

What is *not* host-scale-naive is the clustering: a friends-of-friends pass written as a
Python loop over particles would be the one stage in this file that stops working when
the impact gets the resolution it needs. It is written as a bounded-memory grid sweep
with vectorised connected components instead, so its cost tracks the particle count
rather than the square of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Pairs are enumerated in chunks so the peak allocation does not track the particle
# count. Four million candidate pairs is about 130 MiB across the index arrays the
# expansion needs at once, which is small next to the state a run of that size already
# holds and large enough that the chunk loop runs a handful of times, not thousands.
_PAIR_CHUNK = 4_000_000

# Width of the parabolic band around e == 1. The elliptic and hyperbolic mean-anomaly
# forms both lose their leading digits as `1 - e` cancels; at 1e-8 the surviving relative
# error in either is around 1e-8, which is where Barker's equation - exact at e == 1 -
# is already the more accurate of the two.
_PARABOLIC_BAND = 1e-8

# Half the 27 cell offsets plus the cell itself. The other thirteen are the same pairs
# read from the other end, and `union` does not care which end it is given.
_CELL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) >= (0, 0, 0)
)


def _numpy() -> Any:
    return __import__("numpy")


DEFAULT_IMPACT_SHELL_LAYOUT: tuple[tuple[str, int, float], ...] = (
    ("synthetic-iron", 2, 0.5),
    ("synthetic-rock", 1, 1.0),
)
"""The authoritative default material/shell layout for Quick Effects Impact.

Relocated unchanged from the Blender-side ``IMPACT_SETUP_LAYERS`` (decision D7):
synthetic iron core (material id 2) to outer-radius fraction 0.5, then the
user-selected outer material (default synthetic rock, material id 1) to fraction 1.0.
"""


# -- impact setup ------------------------------------------------------------------------


@dataclass(frozen=True)
class EncounterPrediction:
    """What the two-body problem says the encounter will be, before anything is simulated.

    Every field is derived from the conic through the current relative state, so the
    prediction costs no timesteps and is exact for the point-mass problem. What it does
    not model is the finite-size correction - tides, the pressure gradient once the
    bodies touch, and self-gravity within each body - which is why `contact_time` is a
    prediction the simulation is gated *against* rather than a value it is given.
    """

    contact: bool
    reason: str
    contact_separation: float
    contact_time: float = 0.0
    contact_speed: float = 0.0
    impact_angle_degrees: float = 0.0
    impact_energy: float = 0.0
    eccentricity: float = 0.0
    periapsis: float = 0.0
    specific_energy: float = 0.0
    regime: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact": self.contact,
            "reason": self.reason,
            "contact_separation": self.contact_separation,
            "contact_time": self.contact_time,
            "contact_speed": self.contact_speed,
            "impact_angle_degrees": self.impact_angle_degrees,
            "impact_energy": self.impact_energy,
            "eccentricity": self.eccentricity,
            "periapsis": self.periapsis,
            "specific_energy": self.specific_energy,
            "regime": self.regime,
        }


def _mean_anomaly(eccentricity: float, true_anomaly: float) -> float:
    """Mean anomaly at ``true_anomaly``, on whichever conic the eccentricity names.

    The three branches are the three conics rather than a numerical convenience: the
    eccentric-anomaly substitution that linearises the ellipse is degenerate at ``e == 1``
    and imaginary beyond it, so an impact setup that happens to be near-parabolic - which
    a marginally bound encounter is - needs Barker's equation, not a tolerance on the
    elliptic form.
    """

    half = 0.5 * true_anomaly
    if eccentricity < 1.0 - _PARABOLIC_BAND:
        anomaly = 2.0 * math.atan2(
            math.sqrt(1.0 - eccentricity) * math.sin(half),
            math.sqrt(1.0 + eccentricity) * math.cos(half),
        )
        return anomaly - eccentricity * math.sin(anomaly)
    if eccentricity > 1.0 + _PARABOLIC_BAND:
        tangent = math.sqrt((eccentricity - 1.0) / (eccentricity + 1.0)) * math.tan(half)
        anomaly = 2.0 * math.atanh(tangent)
        return eccentricity * math.sinh(anomaly) - anomaly
    barker = math.tan(half)
    return barker + barker * barker * barker / 3.0


# Width of the band around the rectilinear parabola, in ``|energy * radius / mu|``. The
# elliptic and hyperbolic radial closed forms lose their leading digits there the way the
# mean-anomaly forms do near e == 1 - `angle - sin(angle)` and `sinh(angle) - angle` both
# cancel down to `angle**3 / 6` - so a head-on pair meeting at its escape speed, whose
# energy is round-off either side of zero, would fall no distance in no time. At 1e-2 the
# closed forms still keep about twelve digits, and the series inside converges to double
# precision in under ten terms.
_RADIAL_SERIES_BAND = 1e-2


def _radial_series_time(mu: float, energy: float, radius: float) -> float:
    """``_radial_flight_time`` as its series about the rectilinear parabola.

    ``t(r) = integral from 0 to r of sqrt(s / (2 mu)) (1 + k s)**-0.5 ds`` with
    ``k = energy / mu``; expanding the binomial term by term gives
    ``r sqrt(r / (2 mu)) * sum_n binom(-1/2, n) (k r)**n / (n + 3/2)``, whose first term is
    the parabolic closed form. Summed until a term no longer changes the total.
    """

    ratio = energy * radius / mu
    term = 1.0
    total = term / 1.5
    for n in range(1, 64):
        term *= -(n - 0.5) / n * ratio
        updated = total + term / (n + 1.5)
        if updated == total:
            break
        total = updated
    return radius * math.sqrt(radius / (2.0 * mu)) * total


def _radial_flight_time(mu: float, energy: float, radius: float) -> float:
    """Time for a purely radial trajectory to fall from ``radius`` to the centre.

    The rectilinear conic is the one case the true-anomaly machinery cannot express: with
    no angular momentum the semi-latus rectum is zero, so the orbit has no shape to
    parametrise by an angle. It has to be parametrised by radius instead, which is what
    these three closed forms are. This is not a corner case to tolerate - a head-on impact
    released from rest is the most obvious setup a user will author, and it lands here
    exactly. Near the parabola the closed forms cancel, so there the series stands in for
    all three (ADR 0094).
    """

    if abs(energy * radius / mu) < _RADIAL_SERIES_BAND:
        return _radial_series_time(mu, energy, radius)
    if energy < 0.0:
        axis = -mu / (2.0 * energy)
        angle = math.acos(max(-1.0, min(1.0, 1.0 - radius / axis)))
        return math.sqrt(axis**3 / mu) * (angle - math.sin(angle))
    if energy > 0.0:
        axis = mu / (2.0 * energy)
        angle = math.acosh(max(1.0, 1.0 + radius / axis))
        return math.sqrt(axis**3 / mu) * (math.sinh(angle) - angle)
    return math.sqrt(2.0 * radius**3 / (9.0 * mu))


def _time_scale(mu: float, semi_latus_rectum: float, eccentricity: float) -> float:
    """Seconds per unit of mean anomaly on the conic."""

    if eccentricity < 1.0 - _PARABOLIC_BAND:
        axis = semi_latus_rectum / (1.0 - eccentricity * eccentricity)
        return math.sqrt(axis**3 / mu)
    if eccentricity > 1.0 + _PARABOLIC_BAND:
        axis = semi_latus_rectum / (eccentricity * eccentricity - 1.0)
        return math.sqrt(axis**3 / mu)
    periapsis = 0.5 * semi_latus_rectum
    return math.sqrt(2.0 * periapsis**3 / mu)


def predict_encounter(
    *,
    position_a: Any,
    velocity_a: Any,
    mass_a: float,
    radius_a: float,
    position_b: Any,
    velocity_b: Any,
    mass_b: float,
    radius_b: float,
    gravitational_constant: float,
) -> EncounterPrediction:
    """Predict the encounter of two authored bodies without simulating it.

    The relative motion of two point masses is a conic, so contact - the instant the
    separation reaches the sum of the radii - is a true anomaly, and the time to it is a
    difference of mean anomalies. Nothing here integrates.

    A separating or a missing pair is reported as such rather than as a contact at some
    large time: an impact setup that does not produce an impact is the answer the user
    needs before the bake, which is the requirement this serves.
    """

    np = _numpy()
    if not (mass_a > 0.0 and mass_b > 0.0):
        raise ValueError("both masses must be positive")
    if not (radius_a > 0.0 and radius_b > 0.0):
        raise ValueError("both radii must be positive")
    if gravitational_constant <= 0.0:
        raise ValueError("the gravitational constant must be positive")

    separation = np.asarray(position_b, dtype=np.float64) - np.asarray(
        position_a, dtype=np.float64
    )
    relative = np.asarray(velocity_b, dtype=np.float64) - np.asarray(
        velocity_a, dtype=np.float64
    )
    if separation.shape != (3,) or relative.shape != (3,):
        raise ValueError("positions and velocities must be three-vectors")

    contact_separation = float(radius_a + radius_b)
    distance = float(np.linalg.norm(separation))
    if distance <= 0.0:
        # Coincident centres are not a contact already achieved; they are a state with no
        # conic through it at all - the specific energy is singular and every parameter
        # derived from it is an infinity. Refused here rather than guarded downstream so
        # that the one caller that hands over authored bindings gets a sentence it can
        # record instead of a manifest full of `inf`.
        raise ValueError("the two bodies are coincident, so the encounter has no conic")
    mu = gravitational_constant * (mass_a + mass_b)
    reduced = mass_a * mass_b / (mass_a + mass_b)
    speed = float(np.linalg.norm(relative))
    energy = 0.5 * speed * speed - mu / distance
    momentum = float(np.linalg.norm(np.cross(separation, relative)))
    radial = float(np.dot(separation, relative)) / distance
    eccentricity = math.sqrt(max(0.0, 1.0 + 2.0 * energy * momentum * momentum / (mu * mu)))
    semi_latus_rectum = momentum * momentum / mu
    periapsis = (
        semi_latus_rectum / (1.0 + eccentricity) if eccentricity > 0.0 else distance
    )
    # Below this the cross product is the round-off of two parallel vectors rather than a
    # resolved angular momentum, and every conic parameter derived from it - the
    # semi-latus rectum above all - is that noise cubed. The comparison is `<=` so that a
    # body released from rest, whose angular momentum is exactly zero, lands here: a
    # head-on drop is the most obvious setup a user will author and it is precisely the
    # one the true-anomaly parametrisation cannot express.
    rectilinear = momentum <= 1e-12 * distance * speed
    regime = (
        "radial"
        if rectilinear
        else "elliptic"
        if eccentricity < 1.0 - _PARABOLIC_BAND
        else "hyperbolic"
        if eccentricity > 1.0 + _PARABOLIC_BAND
        else "parabolic"
    )
    base = EncounterPrediction(
        contact=False,
        reason="",
        contact_separation=contact_separation,
        eccentricity=eccentricity,
        periapsis=periapsis,
        specific_energy=energy,
        regime=regime,
    )

    def contacting(time: float) -> EncounterPrediction:
        # Vis-viva at contact rather than the initial speed: the encounter is faster than
        # it starts by exactly the potential the bodies fall through, and for a setup
        # released from rest that difference is the entire impact velocity.
        contact_speed = math.sqrt(max(0.0, 2.0 * (energy + mu / contact_separation)))
        tangential = momentum / contact_separation
        sine = 0.0 if contact_speed <= 0.0 else min(1.0, tangential / contact_speed)
        return EncounterPrediction(
            contact=True,
            reason="",
            contact_separation=contact_separation,
            contact_time=time,
            contact_speed=contact_speed,
            # Measured from the surface normal, so zero is head-on and ninety is grazing.
            impact_angle_degrees=math.degrees(math.asin(sine)),
            impact_energy=0.5 * reduced * contact_speed * contact_speed,
            eccentricity=eccentricity,
            periapsis=periapsis,
            specific_energy=energy,
            regime=regime,
        )

    if distance <= contact_separation:
        return contacting(0.0)
    # Strictly positive, because a zero radial rate is an apsis rather than a departure:
    # released from rest the bodies sit at apoapsis and fall together from there, which is
    # the setup the whole prediction exists to serve.
    if radial > 0.0:
        return _refused(base, "the bodies are separating")
    if periapsis > contact_separation:
        return _refused(base, "the trajectory misses: periapsis exceeds the contact radius")

    if rectilinear:
        return contacting(
            _radial_flight_time(mu, energy, distance)
            - _radial_flight_time(mu, energy, contact_separation)
        )

    # Inbound, so both anomalies are on the approaching branch and negative. The angle at
    # contact is the larger of the two, which is what makes the mean-anomaly difference
    # the positive time still to run.
    now = -math.acos(max(-1.0, min(1.0, (semi_latus_rectum / distance - 1.0) / eccentricity)))
    then = -math.acos(
        max(-1.0, min(1.0, (semi_latus_rectum / contact_separation - 1.0) / eccentricity))
    )
    scale = _time_scale(mu, semi_latus_rectum, eccentricity)
    swept = _mean_anomaly(eccentricity, then) - _mean_anomaly(eccentricity, now)
    return contacting(scale * swept)


def _refused(base: EncounterPrediction, reason: str) -> EncounterPrediction:
    return EncounterPrediction(
        contact=False,
        reason=reason,
        contact_separation=base.contact_separation,
        eccentricity=base.eccentricity,
        periapsis=base.periapsis,
        specific_energy=base.specific_energy,
        regime=base.regime,
    )


# -- layered bodies ----------------------------------------------------------------------


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


def layered_body_mass(shells: tuple[ShellSpec, ...]) -> float:
    """The exact mass of concentric shells, from their radii and densities alone.

    A volume-times-density sum, so a caller can price a shell stack's mass without paying
    for (or needing) a particle draw. An authoring tool that hands a differentiated body a
    mass disagreeing with its own material densities by two or three orders of magnitude
    starts the body far from its own equilibrium - this is the number a default uses
    instead.
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


# -- fragment diagnostics ----------------------------------------------------------------


@dataclass(frozen=True)
class FragmentReport:
    """The post-impact fragment population."""

    labels: Any
    fragment_count: int
    fragment_masses: Any
    fragment_sizes: Any
    largest_remnant_mass: float
    largest_remnant_fraction: float
    total_mass: float
    linking_length: float
    bound_mass_fraction: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_count": self.fragment_count,
            "fragment_masses": [float(value) for value in self.fragment_masses],
            "fragment_sizes": [int(value) for value in self.fragment_sizes],
            "largest_remnant_mass": self.largest_remnant_mass,
            "largest_remnant_fraction": self.largest_remnant_fraction,
            "total_mass": self.total_mass,
            "linking_length": self.linking_length,
            **self.extra,
        }


def _connected_components(np: Any, count: int, left: Any, right: Any) -> Any:
    """Label the connected components of an undirected edge list, without a Python loop.

    Hooking to the smaller label and then jumping pointers to the root, repeated until
    nothing moves. Both halves are whole-array operations, so the cost is the edge count
    times the number of rounds, and the round count is logarithmic in the longest chain
    rather than linear in the particle count - which is what a naive union-find written
    in Python would have been.
    """

    labels = np.arange(count, dtype=np.int64)
    while True:
        hooked = labels.copy()
        if left.size:
            np.minimum.at(hooked, left, labels[right])
            np.minimum.at(hooked, right, labels[left])
        # Pointer jumping to a fixed point: after this, every entry names a root.
        while True:
            jumped = hooked[hooked]
            if np.array_equal(jumped, hooked):
                break
            hooked = jumped
        if np.array_equal(hooked, labels):
            return labels
        labels = hooked


def _candidate_pairs(np: Any, positions: Any, linking_length: float) -> Any:
    """Yield chunks of ``(left, right)`` index pairs closer than ``linking_length``.

    Cells are one linking length across, so every pair within the linking length lies in
    the same cell or in one of its twenty-six neighbours - and thirteen of those are the
    other end of a pair already enumerated. Cell blocks are expanded in bounded chunks so
    the peak allocation is the chunk, not the pair count.
    """

    count = positions.shape[0]
    cells = np.floor(positions / linking_length).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(count)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    cell_count = unique_cells.shape[0]
    starts = np.searchsorted(sorted_inverse, np.arange(cell_count))
    sizes = np.searchsorted(sorted_inverse, np.arange(cell_count), side="right") - starts

    for offset in _CELL_OFFSETS:
        shifted = unique_cells + np.asarray(offset, dtype=np.int64)
        # Locate each shifted cell among the occupied ones by giving both to a single
        # `unique`, so cells that coincide get the same id. Deliberately not a packed
        # integer key: packing needs the grid extent to fit an int64, and impact debris
        # travelling far from a fine linking length is exactly the case where it would
        # not. Deliberately not a raw byte view either - that compares little-endian
        # bytes, which is neither the lexicographic order `unique` produces nor correct
        # across the sign bit.
        _, identifiers = np.unique(
            np.concatenate([unique_cells, shifted]), axis=0, return_inverse=True
        )
        identifiers = identifiers.reshape(2 * cell_count)
        lookup = np.full(int(identifiers.max()) + 1, -1, dtype=np.int64)
        lookup[identifiers[:cell_count]] = np.arange(cell_count, dtype=np.int64)
        located = lookup[identifiers[cell_count:]]
        source = np.nonzero(located >= 0)[0]
        if source.size == 0:
            continue
        target = located[source]
        block = sizes[source].astype(np.int64) * sizes[target].astype(np.int64)
        keep = block > 0
        source, target, block = source[keep], target[keep], block[keep]
        if source.size == 0:
            continue
        bounds = np.cumsum(block)
        begin = 0
        while begin < source.size:
            # Take as many cell blocks as fit in the pair budget, and always at least one
            # so a single block wider than the budget still makes progress.
            consumed = int(bounds[begin - 1]) if begin else 0
            end = int(np.searchsorted(bounds, consumed + _PAIR_CHUNK, side="right"))
            end = min(max(end, begin + 1), source.size)
            span = slice(begin, end)
            total = int(block[span].sum())
            offsets = np.repeat(np.cumsum(block[span]) - block[span], block[span])
            within = np.arange(total, dtype=np.int64) - offsets
            widths = np.repeat(sizes[target[span]].astype(np.int64), block[span])
            left = order[np.repeat(starts[source[span]], block[span]) + within // widths]
            right = order[np.repeat(starts[target[span]], block[span]) + within % widths]
            keep_pair = left != right
            left, right = left[keep_pair], right[keep_pair]
            if left.size:
                delta = positions[left] - positions[right]
                close = np.einsum("ij,ij->i", delta, delta) < linking_length * linking_length
                if close.any():
                    yield left[close], right[close]
            begin = end


def cluster_fragments(
    positions: Any,
    masses: Any,
    linking_length: float,
    *,
    velocities: Any = None,
    gravitational_constant: float = 0.0,
) -> FragmentReport:
    """Group particles into fragments by friends-of-friends and report the population.

    Friends-of-friends rather than a threshold on a density field: the question a fragment
    diagnostic answers is which particles are still *attached*, which is a connectivity
    question, and connectivity has an exact answer for a given linking length. There is no
    fitted constant here - the linking length is an input, and the gate measures that the
    labelling is invariant across every linking length between the widest intra-fragment
    spacing and the narrowest gap, which is the range over which the answer is meaningful
    at all.

    Mass is summed per fragment from the same values, so the fragment masses add back to
    the total to summation round-off and nothing else.
    """

    np = _numpy()
    positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have one entry per particle")
    if linking_length <= 0.0:
        raise ValueError("the linking length must be positive")
    count = positions.shape[0]
    if count == 0:
        raise ValueError("fragment diagnostics need at least one particle")

    lefts: list[Any] = []
    rights: list[Any] = []
    for left, right in _candidate_pairs(np, positions, linking_length):
        lefts.append(left)
        rights.append(right)
    left = np.concatenate(lefts) if lefts else np.zeros(0, dtype=np.int64)
    right = np.concatenate(rights) if rights else np.zeros(0, dtype=np.int64)

    roots = _connected_components(np, count, left, right)
    _, labels, sizes = np.unique(roots, return_inverse=True, return_counts=True)
    labels = labels.reshape(count).astype(np.int64)
    fragment_masses = np.zeros(sizes.size, dtype=np.float64)
    np.add.at(fragment_masses, labels, masses)
    total_mass = float(masses.sum())
    ranked = np.argsort(-fragment_masses, kind="stable")
    largest = float(fragment_masses[ranked[0]])

    bound_fraction = 0.0
    if velocities is not None and gravitational_constant > 0.0:
        bound_fraction = _bound_fraction(
            np,
            positions,
            np.asarray(velocities, dtype=np.float64),
            masses,
            labels,
            int(ranked[0]),
            gravitational_constant,
        )

    return FragmentReport(
        labels=labels,
        fragment_count=int(sizes.size),
        fragment_masses=fragment_masses[ranked],
        fragment_sizes=sizes[ranked],
        largest_remnant_mass=largest,
        largest_remnant_fraction=largest / total_mass,
        total_mass=total_mass,
        linking_length=float(linking_length),
        bound_mass_fraction=bound_fraction,
    )


def _bound_fraction(
    np: Any,
    positions: Any,
    velocities: Any,
    masses: Any,
    labels: Any,
    remnant: int,
    gravitational_constant: float,
) -> float:
    """Fraction of the total mass gravitationally bound to the largest remnant.

    The remnant's own monopole, evaluated at every particle: a fragment escapes when its
    kinetic energy in the remnant's frame exceeds the remnant's potential there. The
    monopole and not a full pair sum because the escaping material is by then far enough
    out that the multipoles it would resolve are smaller than the fragmentation itself.
    """

    inside = labels == remnant
    remnant_mass = float(masses[inside].sum())
    if remnant_mass <= 0.0:
        return 0.0
    center = (positions[inside] * masses[inside, None]).sum(axis=0) / remnant_mass
    drift = (velocities[inside] * masses[inside, None]).sum(axis=0) / remnant_mass
    offset = positions - center
    distance = np.sqrt(np.einsum("ij,ij->i", offset, offset))
    distance = np.maximum(distance, np.finfo(np.float64).tiny)
    relative = velocities - drift
    energy = 0.5 * np.einsum("ij,ij->i", relative, relative) - (
        gravitational_constant * remnant_mass / distance
    )
    bound = energy < 0.0
    return float(masses[bound].sum() / masses.sum())
