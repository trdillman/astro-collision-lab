"""Independent references for gravity, orbit, and neighbour accuracy gates.

These oracles exist so a measured tolerance can be attributed to the solver rather
than to the harness. Every reference here differs from the product runtime in
algorithm, arithmetic precision, and device:

- ``kepler_two_body_state`` is a closed-form conic solution. It performs no time
  integration at all, so it shares no truncation error with the leapfrog solver.
- ``converged_rk4_nbody`` integrates the same equations of motion with a different
  scheme (classical fourth-order Runge-Kutta, not kick-drift-kick) in float64 on the
  CPU, and refines the step count until the result stops moving. It reports the
  Richardson estimate of its own remaining error, so a comparison against it can be
  rejected when the reference is not converged well below the quantity being measured.
- ``direct_accelerations`` is the exact float64 monopole sum, which is the reference an
  approximate tree traversal is defined against.
- ``brute_force_neighbors`` is the exhaustive float64 pair scan a spatial search is
  defined against. It uses no acceleration structure at all, and it reports how close
  the sample came to the support boundary, because that gap is what decides whether the
  device's float32 answer is even allowed to be identical.
- ``radial_kernel_integral`` integrates a published smoothing kernel over its own
  support, which is the independent check that a normalisation constant and a support
  convention were paired correctly.

Only NumPy is required, so the oracles are importable inside Blender's interpreter as
well as on the host. ``tests/unit/test_oracles.py`` runs pure-NumPy self-checks, on the
host, for the gravity, Kepler, neighbour, and Riemann oracles above; ``radial_kernel_integral``
has no host self-check and is exercised only inside the cuda:0 accuracy-gate suites.
SciPy is used, when present, purely to cross-check the oracles against each other in
that same file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Convergence is measured on the Newton increment rather than on the residual: near
# periapsis of an eccentric orbit the residual is damped by (1 - e cos E), so a small
# residual can still hide an eccentric anomaly that is wrong in its last few digits,
# and the resulting velocity error is amplified by the same factor.
KEPLER_TOLERANCE = 1.0e-15
KEPLER_MAX_ITERATIONS = 200


class OracleUnavailableError(RuntimeError):
    """Raised when a reference cannot be produced to the requested quality.

    A silent fallback to a worse reference would let an accuracy gate pass against a
    tolerance the oracle cannot resolve, so every failure to converge is raised.
    """


@dataclass(frozen=True)
class TwoBodyElements:
    """Orbital elements of a bound two-body relative orbit, in the input's units."""

    gravitational_parameter: float
    semi_major_axis: float
    eccentricity: float
    period: float
    mean_anomaly_at_epoch: float
    periapsis_direction: tuple[float, float, float]
    in_plane_direction: tuple[float, float, float]
    normal_direction: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gravitational_parameter": self.gravitational_parameter,
            "semi_major_axis": self.semi_major_axis,
            "eccentricity": self.eccentricity,
            "period": self.period,
            "mean_anomaly_at_epoch": self.mean_anomaly_at_epoch,
            "periapsis_direction": list(self.periapsis_direction),
            "in_plane_direction": list(self.in_plane_direction),
            "normal_direction": list(self.normal_direction),
        }


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Sampled reference states plus the provenance needed to defend a tolerance."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.shape[0] != self.times.shape[0]:
            raise ValueError("sample count must match the number of times")

    @property
    def body_count(self) -> int:
        return int(self.positions.shape[1])


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise OracleUnavailableError("cannot normalize a zero-length vector")
    return vector / norm


def direct_accelerations(
    masses: np.ndarray,
    positions: np.ndarray,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
    *,
    block_size: int | None = None,
) -> np.ndarray:
    """Exact float64 monopole acceleration sum.

    This is the definition an approximate multipole traversal is measured against, so
    it is deliberately the naive O(N^2) pairwise sum with no acceptance criterion.

    ``block_size`` bounds host memory to O(block_size * N) by evaluating the target rows
    in slices. The default computes every row at once and is bit-identical to the
    unblocked formulation; a slice only changes which rows are materialised together,
    never the order in which the N contributions to a row are summed.
    """

    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have shape (N,)")
    count = int(positions.shape[0])
    span = count if block_size is None else max(1, int(block_size))
    accelerations = np.empty_like(positions)
    for start in range(0, count, span):
        stop = min(start + span, count)
        offsets = positions[None, :, :] - positions[start:stop, None, :]
        distance_squared = np.einsum("ijk,ijk->ij", offsets, offsets) + softening * softening
        # A body exerts no force on itself; an infinite separation zeroes the term without
        # branching, and works for a row slice where the diagonal is offset.
        distance_squared[np.arange(stop - start), np.arange(start, stop)] = np.inf
        inverse_cube = distance_squared ** -1.5
        weights = masses[None, :] * inverse_cube
        accelerations[start:stop] = gravitational_constant * np.einsum(
            "ij,ijk->ik", weights, offsets
        )
    return accelerations


@dataclass(frozen=True)
class NeighborReference:
    """Exhaustive neighbour sets plus the pairs no finite precision can decide."""

    counts: np.ndarray
    sets: tuple[frozenset[int], ...]
    ambiguous: tuple[frozenset[int], ...]
    minimum_relative_gap: float
    provenance: dict[str, Any]

    @property
    def particle_count(self) -> int:
        return int(self.counts.shape[0])

    @property
    def ambiguous_pair_count(self) -> int:
        return sum(len(entry) for entry in self.ambiguous)

    def disagreements(self, candidate: Sequence[frozenset[int]]) -> tuple[list[int], list[int]]:
        """Split a candidate's disagreements into the decidable and the undecidable.

        The first list is the particles where the candidate differs on a pair that sits
        outside the margin: those are search errors and nothing excuses them. The second
        is where it differs only on pairs inside the margin, which is the arithmetic's own
        resolution of the support boundary rather than a property of the search.
        """

        if len(candidate) != self.particle_count:
            raise ValueError("candidate must cover every particle")
        resolved: list[int] = []
        undecidable: list[int] = []
        for index, found in enumerate(candidate):
            difference = found ^ self.sets[index]
            if not difference:
                continue
            if difference <= self.ambiguous[index]:
                undecidable.append(index)
            else:
                resolved.append(index)
        return resolved, undecidable


def brute_force_neighbors(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    block_size: int = 512,
    boundary_margin: float = 0.0,
) -> NeighborReference:
    """Exhaustive float64 neighbour sets within a per-particle radius.

    No acceleration structure is involved, so the result depends on nothing the spatial
    search under test also depends on. Self-pairs are excluded and membership is closed
    (``distance <= radius``), matching the runtime's own predicate.

    ``boundary_margin`` is the relative width, in units of the radius, inside which the
    membership test cannot be settled by a lower-precision lane: a pair with
    ``|distance - radius| <= boundary_margin * radius`` is recorded in ``ambiguous`` and a
    candidate is allowed to disagree about it. This is not leniency. Whether such a pair is
    inside the support is a question float32 arithmetic does not have the resolution to
    answer, so requiring agreement there would be requiring luck, and a run that happened to
    pass would be reporting the sample rather than the search. Every other pair is decidable
    and demanded exactly. A margin of zero makes every pair decidable.

    ``minimum_relative_gap`` is the smallest ``|distance - radius| / radius`` over every
    ordered pair, reported so the caller can see how close the sample came to the boundary
    independently of the margin it chose.

    ``block_size`` bounds host memory to ``O(block_size * N)``. It changes only how many
    rows are materialised at once, never which pairs are found.
    """

    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if radii.shape != (positions.shape[0],):
        raise ValueError("radii must have shape (N,)")
    if np.any(radii <= 0.0):
        raise ValueError("radii must be positive")
    if boundary_margin < 0.0:
        raise ValueError("boundary_margin must be non-negative")
    count = int(positions.shape[0])
    span = max(1, int(block_size))
    counts = np.empty(count, dtype=np.int64)
    sets: list[frozenset[int]] = []
    ambiguous: list[frozenset[int]] = []
    minimum_gap = math.inf
    for start in range(0, count, span):
        stop = min(start + span, count)
        offsets = positions[None, :, :] - positions[start:stop, None, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", offsets, offsets))
        rows = np.arange(stop - start)
        distances[rows, np.arange(start, stop)] = math.inf
        block_radii = radii[start:stop, None]
        inside = distances <= block_radii
        counts[start:stop] = inside.sum(axis=1)
        gaps = np.abs(distances - block_radii) / block_radii
        undecidable = gaps <= boundary_margin
        for row in range(stop - start):
            sets.append(frozenset(int(index) for index in np.nonzero(inside[row])[0]))
            ambiguous.append(frozenset(int(index) for index in np.nonzero(undecidable[row])[0]))
        minimum_gap = min(minimum_gap, float(gaps.min()))
    return NeighborReference(
        counts=counts,
        sets=tuple(sets),
        ambiguous=tuple(ambiguous),
        minimum_relative_gap=minimum_gap,
        provenance={
            "method": "exhaustive pairwise scan",
            "precision": "float64",
            "device": "cpu",
            "membership": "distance <= radius, self excluded",
            "particle_count": count,
            "block_size": span,
            "boundary_margin": float(boundary_margin),
        },
    )


def radial_kernel_integral(
    value: Any,
    support_radius: float,
    *,
    breakpoints: tuple[float, ...] = (),
    samples: int = 2001,
) -> float:
    """``4*pi * integral of value(r) * r**2 dr`` over ``[0, support_radius]``.

    A normalised smoothing kernel integrates to one over its own support, so this is the
    check that a published normalisation constant was paired with the support convention
    it was published under. Getting that pairing wrong rescales every density in the
    simulation by a constant nobody would see in a plot of the kernel.

    Composite Simpson is applied between ``breakpoints`` rather than across them, because
    a piecewise kernel is only smooth inside a piece and Simpson's error bound needs a
    fourth derivative that exists. ``samples`` is the odd sample count per piece.
    """

    if support_radius <= 0.0:
        raise ValueError("support_radius must be positive")
    if samples < 3 or samples % 2 == 0:
        raise ValueError("samples must be an odd integer of at least 3")
    edges = sorted({0.0, float(support_radius), *(float(point) for point in breakpoints)})
    if edges[0] < 0.0 or edges[-1] > support_radius:
        raise ValueError("breakpoints must lie inside [0, support_radius]")
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        grid = np.linspace(lower, upper, samples, dtype=np.float64)
        integrand = np.array([float(value(float(r))) * r * r for r in grid], dtype=np.float64)
        weights = np.ones(samples, dtype=np.float64)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        total += float((upper - lower) / (3.0 * (samples - 1)) * (weights @ integrand))
    return 4.0 * math.pi * total


@dataclass(frozen=True)
class GasState:
    """A uniform state of an ideal gas on one side of a discontinuity."""

    density: float
    velocity: float
    pressure: float

    def sound_speed(self, gamma: float) -> float:
        return math.sqrt(gamma * self.pressure / self.density)

    def validate(self) -> None:
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.pressure <= 0.0:
            raise ValueError("pressure must be positive")


@dataclass(frozen=True)
class RiemannSolution:
    """The exact solution of the ideal-gas Riemann problem, sampled in ``x/t``.

    The solution is self-similar in ``xi = x / t``, so a single star-region solve serves
    every output time. It is an algebraic reference rather than a discretised one: no cell
    size, no timestep, and no artificial viscosity enter it, which is what makes a
    comparison against it attributable to the solver being gated.
    """

    left: GasState
    right: GasState
    gamma: float
    pressure_star: float
    velocity_star: float
    iterations: int

    def sample(self, xi: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(density, velocity, pressure)`` at the similarity coordinates ``xi``."""

        coordinates = np.atleast_1d(np.asarray(xi, dtype=np.float64))
        density = np.empty_like(coordinates)
        velocity = np.empty_like(coordinates)
        pressure = np.empty_like(coordinates)
        for index, value in enumerate(coordinates):
            density[index], velocity[index], pressure[index] = self._sample_one(float(value))
        return density, velocity, pressure

    def _sample_one(self, xi: float) -> tuple[float, float, float]:
        gamma = self.gamma
        if xi <= self.velocity_star:
            return self._sample_side(xi, self.left, gamma, -1.0)
        return self._sample_side(xi, self.right, gamma, 1.0)

    def _sample_side(
        self, xi: float, state: GasState, gamma: float, sign: float
    ) -> tuple[float, float, float]:
        # ``sign`` is +1 on the right of the contact and -1 on the left. Every wave speed
        # below is the left-facing expression with the frame mirrored through it, which is
        # why one branch covers both sides instead of two transcriptions that can disagree.
        sound = state.sound_speed(gamma)
        ratio = self.pressure_star / state.pressure
        if ratio > 1.0:
            shock_speed = state.velocity + sign * sound * math.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)
            )
            if sign * (xi - shock_speed) >= 0.0:
                return state.density, state.velocity, state.pressure
            jump = (gamma - 1.0) / (gamma + 1.0)
            density = state.density * (ratio + jump) / (jump * ratio + 1.0)
            return density, self.velocity_star, self.pressure_star
        head = state.velocity + sign * sound
        star_sound = sound * ratio ** ((gamma - 1.0) / (2.0 * gamma))
        tail = self.velocity_star + sign * star_sound
        if sign * (xi - head) >= 0.0:
            return state.density, state.velocity, state.pressure
        if sign * (xi - tail) <= 0.0:
            return state.density * ratio ** (1.0 / gamma), self.velocity_star, self.pressure_star
        fan = 2.0 / (gamma + 1.0)
        velocity = fan * (-sign * sound + (gamma - 1.0) / 2.0 * state.velocity + xi)
        local_sound = fan * (sound - sign * (gamma - 1.0) / 2.0 * (state.velocity - xi))
        density = state.density * (local_sound / sound) ** (2.0 / (gamma - 1.0))
        pressure = state.pressure * (local_sound / sound) ** (2.0 * gamma / (gamma - 1.0))
        return density, velocity, pressure


def exact_riemann(
    left: GasState,
    right: GasState,
    gamma: float = 1.4,
    *,
    tolerance: float = 1.0e-14,
    max_iterations: int = 200,
) -> RiemannSolution:
    """Solve the ideal-gas Riemann problem exactly (Toro, *Riemann Solvers*, chapter 4).

    The star-region pressure is the root of ``f_left(p) + f_right(p) + (u_R - u_L) = 0``,
    where each ``f`` is the shock or rarefaction relation for its side. The root is found
    by Newton iteration from the two-rarefaction guess; convergence is measured on the
    relative pressure increment, and a failure to converge raises rather than returning a
    partially solved state, because a wrong star pressure would silently move every wave.
    """

    left.validate()
    right.validate()
    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    sound_left = left.sound_speed(gamma)
    sound_right = right.sound_speed(gamma)
    if 2.0 / (gamma - 1.0) * (sound_left + sound_right) <= right.velocity - left.velocity:
        raise OracleUnavailableError(
            "the initial states generate a vacuum; the star region does not exist"
        )

    def wave(pressure: float, state: GasState, sound: float) -> tuple[float, float]:
        if pressure > state.pressure:
            coefficient = 2.0 / ((gamma + 1.0) * state.density)
            offset = (gamma - 1.0) / (gamma + 1.0) * state.pressure
            root = math.sqrt(coefficient / (offset + pressure))
            value = (pressure - state.pressure) * root
            slope = root * (1.0 - 0.5 * (pressure - state.pressure) / (offset + pressure))
            return value, slope
        ratio = pressure / state.pressure
        exponent = (gamma - 1.0) / (2.0 * gamma)
        value = 2.0 * sound / (gamma - 1.0) * (ratio**exponent - 1.0)
        slope = ratio ** (-(gamma + 1.0) / (2.0 * gamma)) / (state.density * sound)
        return value, slope

    # Two-rarefaction guess: exact whenever both waves are rarefactions, and positive by
    # construction, which a linearised guess is not for strong shocks.
    exponent = (gamma - 1.0) / (2.0 * gamma)
    numerator = sound_left + sound_right - 0.5 * (gamma - 1.0) * (
        right.velocity - left.velocity
    )
    denominator = sound_left / left.pressure**exponent + sound_right / right.pressure**exponent
    pressure = max((numerator / denominator) ** (1.0 / exponent), tolerance)
    for iteration in range(1, max_iterations + 1):
        value_left, slope_left = wave(pressure, left, sound_left)
        value_right, slope_right = wave(pressure, right, sound_right)
        residual = value_left + value_right + right.velocity - left.velocity
        step = residual / (slope_left + slope_right)
        updated = pressure - step
        if updated <= 0.0:
            updated = 0.5 * pressure
        change = abs(updated - pressure) / (0.5 * (updated + pressure))
        pressure = updated
        if change < tolerance:
            value_left, _ = wave(pressure, left, sound_left)
            value_right, _ = wave(pressure, right, sound_right)
            velocity = 0.5 * (left.velocity + right.velocity) + 0.5 * (value_right - value_left)
            return RiemannSolution(
                left=left,
                right=right,
                gamma=float(gamma),
                pressure_star=float(pressure),
                velocity_star=float(velocity),
                iterations=iteration,
            )
    raise OracleUnavailableError(
        f"the exact Riemann solver did not converge in {max_iterations} iterations"
    )


def strong_shock_density_ratio(gamma: float) -> float:
    """The Rankine-Hugoniot compression limit ``(gamma + 1) / (gamma - 1)``.

    A blast wave driven into a cold ambient medium approaches this ratio from below as the
    upstream Mach number grows, so it is the closed-form ceiling a Sedov density peak is
    checked against without needing the full self-similar solution.
    """

    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    return (gamma + 1.0) / (gamma - 1.0)


def elements_from_state(
    total_mass: float,
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    gravitational_constant: float = 1.0,
) -> TwoBodyElements:
    """Classical elements of a bound relative orbit, taken from one state vector."""

    position = np.asarray(relative_position, dtype=np.float64)
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    mu = float(gravitational_constant) * float(total_mass)
    if mu <= 0.0:
        raise ValueError("gravitational parameter must be positive")
    radius = float(np.linalg.norm(position))
    speed_squared = float(velocity @ velocity)
    specific_energy = 0.5 * speed_squared - mu / radius
    if specific_energy >= 0.0:
        raise OracleUnavailableError(
            "closed-form reference covers bound orbits only; "
            f"specific energy is {specific_energy!r}"
        )
    semi_major_axis = -mu / (2.0 * specific_energy)
    angular_momentum = np.cross(position, velocity)
    eccentricity_vector = (
        (speed_squared - mu / radius) * position - float(position @ velocity) * velocity
    ) / mu
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    if eccentricity >= 1.0:
        raise OracleUnavailableError(f"orbit is not bound: eccentricity {eccentricity!r}")
    normal = _unit(angular_momentum)
    # A circular orbit has no periapsis to point at, so the epoch state defines the
    # in-plane basis instead. Both choices describe the same orbit.
    periapsis = _unit(eccentricity_vector) if eccentricity > 0.0 else _unit(position)
    in_plane = np.cross(normal, periapsis)
    true_anomaly = math.atan2(float(position @ in_plane), float(position @ periapsis))
    cosine_true = math.cos(true_anomaly)
    denominator = 1.0 + eccentricity * cosine_true
    eccentric_anomaly = math.atan2(
        math.sqrt(1.0 - eccentricity * eccentricity) * math.sin(true_anomaly) / denominator,
        (cosine_true + eccentricity) / denominator,
    )
    return TwoBodyElements(
        gravitational_parameter=mu,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        period=2.0 * math.pi * math.sqrt(semi_major_axis**3 / mu),
        mean_anomaly_at_epoch=eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly),
        periapsis_direction=tuple(periapsis.tolist()),
        in_plane_direction=tuple(in_plane.tolist()),
        normal_direction=tuple(normal.tolist()),
    )


def _solve_eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    """Newton solve of Kepler's equation, with the iteration count made visible."""

    wrapped = math.remainder(mean_anomaly, 2.0 * math.pi)
    anomaly = wrapped + eccentricity * math.sin(wrapped)
    for _ in range(KEPLER_MAX_ITERATIONS):
        derivative = 1.0 - eccentricity * math.cos(anomaly)
        if derivative == 0.0:
            raise OracleUnavailableError("Kepler iteration reached a zero derivative")
        increment = (anomaly - eccentricity * math.sin(anomaly) - wrapped) / derivative
        anomaly -= increment
        if abs(increment) <= KEPLER_TOLERANCE:
            return anomaly
    raise OracleUnavailableError(
        f"Kepler's equation did not converge to {KEPLER_TOLERANCE!r} "
        f"for eccentricity {eccentricity!r}"
    )


def kepler_two_body_state(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    gravitational_constant: float = 1.0,
) -> ReferenceTrajectory:
    """Closed-form two-body states at ``times``, with no time integration.

    The barycentre is propagated as the straight line it provably is, and the relative
    orbit is evaluated from its conic elements, so the reference carries no accumulated
    step error for the solver's error to hide inside.
    """

    masses = np.asarray(masses, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if masses.shape != (2,):
        raise ValueError("the closed-form reference covers exactly two bodies")
    if positions.shape != (2, 3) or velocities.shape != (2, 3):
        raise ValueError("positions and velocities must have shape (2, 3)")
    total_mass = float(masses.sum())
    elements = elements_from_state(
        total_mass,
        positions[1] - positions[0],
        velocities[1] - velocities[0],
        gravitational_constant,
    )
    barycentre = (masses[:, None] * positions).sum(axis=0) / total_mass
    barycentre_velocity = (masses[:, None] * velocities).sum(axis=0) / total_mass
    periapsis = np.asarray(elements.periapsis_direction, dtype=np.float64)
    in_plane = np.asarray(elements.in_plane_direction, dtype=np.float64)
    axis = elements.semi_major_axis
    eccentricity = elements.eccentricity
    minor_factor = math.sqrt(1.0 - eccentricity * eccentricity)
    mean_motion = 2.0 * math.pi / elements.period
    speed_scale = math.sqrt(elements.gravitational_parameter * axis)
    shares = np.array([-masses[1] / total_mass, masses[0] / total_mass], dtype=np.float64)

    sampled_positions = np.empty((times.shape[0], 2, 3), dtype=np.float64)
    sampled_velocities = np.empty_like(sampled_positions)
    for index, time in enumerate(times):
        anomaly = _solve_eccentric_anomaly(
            elements.mean_anomaly_at_epoch + mean_motion * float(time), eccentricity
        )
        cosine = math.cos(anomaly)
        sine = math.sin(anomaly)
        radius = axis * (1.0 - eccentricity * cosine)
        relative_position = axis * (cosine - eccentricity) * periapsis + (
            axis * minor_factor * sine
        ) * in_plane
        relative_velocity = (speed_scale / radius) * (
            -sine * periapsis + minor_factor * cosine * in_plane
        )
        centre = barycentre + barycentre_velocity * float(time)
        sampled_positions[index] = centre[None, :] + shares[:, None] * relative_position[None, :]
        sampled_velocities[index] = (
            barycentre_velocity[None, :] + shares[:, None] * relative_velocity[None, :]
        )
    return ReferenceTrajectory(
        times=times,
        positions=sampled_positions,
        velocities=sampled_velocities,
        provenance={
            "method": "closed-form Kepler conic",
            "integration_steps": 0,
            "precision": "float64",
            "kepler_increment_tolerance": KEPLER_TOLERANCE,
            "elements": elements.to_dict(),
        },
    )


def _rk4_segment(
    masses: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    duration: float,
    steps: int,
    gravitational_constant: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    step = duration / float(steps)
    for _ in range(steps):
        a1 = direct_accelerations(masses, position, gravitational_constant, softening)
        p2 = position + 0.5 * step * velocity
        v2 = velocity + 0.5 * step * a1
        a2 = direct_accelerations(masses, p2, gravitational_constant, softening)
        p3 = position + 0.5 * step * v2
        v3 = velocity + 0.5 * step * a2
        a3 = direct_accelerations(masses, p3, gravitational_constant, softening)
        p4 = position + step * v3
        v4 = velocity + step * a3
        a4 = direct_accelerations(masses, p4, gravitational_constant, softening)
        position = position + (step / 6.0) * (velocity + 2.0 * v2 + 2.0 * v3 + v4)
        velocity = velocity + (step / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return position, velocity


def converged_rk4_nbody(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
    relative_tolerance: float = 1.0e-12,
    initial_steps_per_unit_time: int = 512,
    max_refinements: int = 12,
) -> ReferenceTrajectory:
    """Step-refined float64 RK4 reference for an arbitrary number of bodies.

    The step count is doubled until the sampled trajectory stops changing, and the
    Richardson estimate of the remaining error is reported. A reference that cannot be
    driven below ``relative_tolerance`` raises instead of returning, because a gate
    derived from an unconverged reference measures the reference.
    """

    masses = np.asarray(masses, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if np.any(np.diff(times) < 0.0) or float(times[0]) < 0.0:
        raise ValueError("times must be non-negative and non-decreasing")
    span = max(float(times[-1]), 1.0e-30)
    scale = max(float(np.abs(positions).max()), 1.0)

    def sample(multiplier: int) -> tuple[np.ndarray, np.ndarray]:
        sampled_positions = np.empty((times.shape[0], positions.shape[0], 3), dtype=np.float64)
        sampled_velocities = np.empty_like(sampled_positions)
        position = positions.copy()
        velocity = velocities.copy()
        previous_time = 0.0
        for index, time in enumerate(times):
            duration = float(time) - previous_time
            if duration > 0.0:
                steps = max(1, int(math.ceil(duration * initial_steps_per_unit_time * multiplier)))
                position, velocity = _rk4_segment(
                    masses,
                    position,
                    velocity,
                    duration,
                    steps,
                    gravitational_constant,
                    softening,
                )
            sampled_positions[index] = position
            sampled_velocities[index] = velocity
            previous_time = float(time)
        return sampled_positions, sampled_velocities

    if max_refinements < 1:
        raise ValueError("max_refinements must be positive")
    multiplier = 1
    coarse_positions, _ = sample(multiplier)
    estimated_error = math.inf
    for refinement in range(1, max_refinements + 1):
        multiplier *= 2
        fine_positions, fine_velocities = sample(multiplier)
        # Richardson: for a fourth-order scheme the halved-step difference is 15/16 of
        # the coarse error, so the fine solution's error is that difference over 15.
        difference = float(np.abs(fine_positions - coarse_positions).max())
        estimated_error = difference / 15.0 / scale
        coarse_positions = fine_positions
        if estimated_error <= relative_tolerance:
            return ReferenceTrajectory(
                times=times,
                positions=fine_positions,
                velocities=fine_velocities,
                provenance={
                    "method": "step-refined classical RK4",
                    "precision": "float64",
                    "steps_per_unit_time": initial_steps_per_unit_time * multiplier,
                    "total_steps": int(
                        math.ceil(span * initial_steps_per_unit_time * multiplier)
                    ),
                    "refinements": refinement,
                    "requested_relative_tolerance": relative_tolerance,
                    "estimated_relative_error": estimated_error,
                    "softening": softening,
                },
            )
    raise OracleUnavailableError(
        f"RK4 reference did not reach {relative_tolerance!r} after {max_refinements} "
        f"refinements; last estimate was {estimated_error!r}"
    )
