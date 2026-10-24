"""Pure initial-state math for Cloud mass jitter and velocity models (ADR 0040).

Everything here is a pure NumPy function over explicit inputs. No ``bpy``, no Warp, no
global RNG, no per-step work: these run **once**, during expansion, before the single
host-to-device upload. That placement is the whole point -- these are initial-state inputs,
not force terms, so they belong at the sampler seam rather than in the solver.

Three things are worth understanding before changing anything here.

**The streams are domain-separated on purpose.** Position sampling keeps the exact legacy
seed material ``[sampling_seed, runtime_stable_id(binding_id)]`` with *no* appended domain
integer, which is what preserves the existing position stream byte for byte. Each new
stochastic concern appends its own pinned integer instead. The property this buys is that
turning on mass jitter cannot move a single particle, and cannot shift the virial or
eccentricity draws either. The integers are serialized-test fixtures: changing one changes
every existing scene's draws.

**``VIRIALIZE`` is not equilibrium.** It normalizes the global scalar virial ratio against
a uniform-sphere reference potential. It does not solve the collisionless Boltzmann
equation, produce a local Jeans dispersion, or guarantee a stationary profile. ADR 0040
rejects the equilibrium framing explicitly, because claiming it would falsely close part of
``GALAXY-001``. Keep the naming honest in anything added here.

**The float64 bounds are the ADR's, not the implementation's.** Section 4 fixes
``eps64 = 2^-52`` and ``gamma_N = (N * eps64) / (1 - N * eps64)`` and states each tolerance
in terms of them, adding: "an implementation may be more accurate but may not widen the
bounds." If a change here makes a bound fail, the change is wrong -- or the ADR needs
amending first.
"""

from __future__ import annotations

import math
from typing import Any

# Pinned stream-domain integers (ADR 0040 section 2). The legacy position stream appends
# none of these, which is what keeps it byte-identical to the pre-ADR expansion.
MASS_JITTER_DOMAIN = 1
VIRIAL_VELOCITY_DOMAIN = 2
KEPLERIAN_ECCENTRICITY_DOMAIN = 3
ROTATION_CURVE_ECCENTRICITY_DOMAIN = 4

#: Azimuths sampled per radial bin when measuring the disk's own radial force.
#:
#: The quantity wanted is the *azimuthally averaged* inward acceleration at a radius. One
#: sample point would measure the acceleration at one angle, which for a finite particle
#: cloud is dominated by whichever particles happen to lie near it -- a curve that changes
#: when the seed changes is not a rotation curve. Sixteen is the smallest power of two at
#: which the residual anisotropy of a Poisson-sampled annulus falls below the bin-to-bin
#: scatter the binning itself leaves, so raising it buys precision the bins immediately
#: throw away. It is a quadrature order, not a tuned constant: the answer it converges to
#: is the same one, and `test_the_curve_does_not_depend_on_the_seed` is what holds it.
ROTATION_CURVE_AZIMUTHS = 16

#: Field particles swept per block when measuring the curve (ADR 0133). A block holds
#: ``ROTATION_CURVE_AZIMUTHS`` rows of this many float64 values in each of four work arrays,
#: 8 MiB apiece at 65,536, so a million-particle field costs no more memory than a small one.
#: A throughput knob only: every particle is still summed at every probe.
ROTATION_CURVE_FIELD_CHUNK = 65_536


def _numpy() -> Any:
    import numpy as np

    return np


def domain_rng(sampling_seed: int, binding_id: str, domain: int) -> Any:
    """The generator for one stochastic concern of one binding.

    Entropy is exactly ``[sampling_seed, runtime_stable_id(binding_id), domain]``. Seeding
    from the binding ID as well as the authored seed is what makes a cloud sample the same
    way regardless of what else is in the scene or what order bindings compile in.
    """

    from .model import runtime_stable_id

    np = _numpy()
    return np.random.default_rng(
        [int(sampling_seed), runtime_stable_id(binding_id), int(domain)]
    )


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _require_finite_vector(name: str, value: Any) -> Any:
    np = _numpy()
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite, got {value!r}")
    return vector


def jittered_masses(
    *, total_mass: float, count: int, jitter: float, rng: Any
) -> Any:
    """Per-particle masses whose sum is the authored total (ADR 0040 section 3).

    ``jitter = 0`` returns the exact equal split and **does not consume the stream**, so a
    cloud with jitter switched off draws identically to one authored before jitter existed.

    Otherwise weights are drawn uniformly from ``[1-j, 1+j]`` and normalized so the total
    is preserved. The residual left by float64 summation is applied to a single particle:
    the lowest index whose pre-correction mass equals the maximum (the first maximum in
    binding order), and only when the correction cannot drive that mass non-positive.
    Choosing the largest mass is what bounds the relative disturbance; choosing it by first
    index is what removes any dependence on set or dictionary ordering. There is no random
    tie break, so the result is reproducible for fixed inputs.
    """

    np = _numpy()
    total = _require_finite("total_mass", total_mass)
    amount = _require_finite("jitter", jitter)
    if total <= 0.0:
        raise ValueError("total_mass must be positive")
    if int(count) < 1:
        raise ValueError("count must be at least 1")
    if not 0.0 <= amount < 1.0:
        raise ValueError("jitter must satisfy 0 <= jitter < 1")

    n = int(count)
    if amount == 0.0:
        return np.full(n, total / n, dtype=np.float64)

    weights = rng.uniform(1.0 - amount, 1.0 + amount, size=n)
    masses = weights * (total / float(weights.sum()))

    # Fixed-order residual correction. `argmax` returns the lowest index among ties, which
    # is exactly the "first maximum in binding order" the ADR specifies.
    residual = total - float(masses.sum())
    target = int(np.argmax(masses))
    if masses[target] + residual > 0.0:
        masses[target] += residual
    return masses


def virialized_velocities(
    *,
    masses: Any,
    total_mass: float,
    radius: float,
    gravitational_constant: float,
    support_fraction: float,
    bulk_velocity: Any,
    rng: Any,
) -> Any:
    """Global virial-ratio initial velocities for a uniform Ball (ADR 0040 section 4).

    **Not an equilibrium model.** This constrains one global scalar -- total internal
    kinetic energy against the uniform-sphere reference potential
    ``W_ref = -3 G M^2 / (5 R)`` -- so that ``Q_ref = 2 K / |W_ref| = support_fraction^2``.
    It says nothing about local dispersion or stationarity.

    ``support_fraction = 0`` returns the authored bulk velocity tiled exactly, which is the
    pre-ADR cold-cloud field, bit for bit.
    """

    np = _numpy()
    total = _require_finite("total_mass", total_mass)
    outer = _require_finite("radius", radius)
    g = _require_finite("gravitational_constant", gravitational_constant)
    support = _require_finite("support_fraction", support_fraction)
    bulk = _require_finite_vector("bulk_velocity", bulk_velocity)

    if total <= 0.0:
        raise ValueError("total_mass must be positive")
    if outer <= 0.0:
        raise ValueError("radius must be positive")
    if g <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if support < 0.0:
        raise ValueError("support_fraction must be non-negative")

    weights = np.asarray(masses, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("masses must be a 1-D array")
    if weights.size < 2:
        raise ValueError(
            "VIRIALIZE needs at least two particles: a single particle carries no "
            "internal motion once its own centre of mass is removed"
        )
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("masses must be finite and strictly positive")

    count = int(weights.size)
    if support == 0.0:
        return np.tile(bulk, (count, 1))

    internal = rng.standard_normal((count, 3))
    # Exact mass-weighted mean removal: the cloud must not acquire bulk drift from the draw.
    internal -= (weights[:, None] * internal).sum(axis=0) / float(weights.sum())

    kinetic = 0.5 * float((weights * (internal**2).sum(axis=1)).sum())
    if not math.isfinite(kinetic) or kinetic <= 0.0:
        raise ValueError(
            "post-COM kinetic norm is zero or non-finite; cannot rescale to a nonzero "
            "support fraction"
        )
    w_ref = -3.0 * g * total**2 / (5.0 * outer)
    target_kinetic = 0.5 * (support**2) * abs(w_ref)
    internal *= math.sqrt(target_kinetic / kinetic)
    return internal + bulk


def keplerian_velocities(
    *,
    positions: Any,
    central_position: Any,
    central_mass: float,
    central_velocity: Any,
    gravitational_constant: float,
    normal: Any,
    counter_clockwise: bool,
    eccentricity_jitter: float,
    bulk_velocity: Any,
    rng: Any,
    softening: float = 0.0,
) -> Any:
    """Circular-orbit initial velocities about a central Body (ADR 0040 section 6).

    Each particle gets the circular speed of the Plummer-softened point mass it is
    integrated against, ``sqrt(mu * r**2 / (r**2 + eps**2)**1.5)`` at its cylindrical radius
    ``r`` (ADR 0188, the ADR 0081 form ``rotation_curve_velocities`` already uses), directed
    along ``normal x radial_unit`` (negated for clockwise). ``softening = 0`` is exactly the
    unsoftened ``sqrt(mu/r)``. ``eccentricity_jitter`` draws a signed apsis eccentricity
    uniformly in ``[-e_max, e_max]`` and multiplies the tangential speed by
    ``sqrt(1 + e_signed)``.

    The resulting angular momentum is intentional, so -- unlike ``VIRIALIZE`` -- no COM
    subtraction or kinetic rescale is applied. This model does not recoil the central Body,
    solve a barycentric multi-primary disk, add radial epicycles, or promise disk
    self-gravity equilibrium.
    """

    np = _numpy()
    mass = _require_finite("central_mass", central_mass)
    g = _require_finite("gravitational_constant", gravitational_constant)
    jitter = _require_finite("eccentricity_jitter", eccentricity_jitter)
    eps = _require_finite("softening", softening)
    centre = _require_finite_vector("central_position", central_position)
    centre_velocity = _require_finite_vector("central_velocity", central_velocity)
    bulk = _require_finite_vector("bulk_velocity", bulk_velocity)
    axis = _require_finite_vector("normal", normal)

    if mass <= 0.0:
        raise ValueError("central_mass must be positive")
    if g <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if not 0.0 <= jitter < 1.0:
        raise ValueError(
            "eccentricity_jitter must satisfy 0 <= eccentricity_jitter < 1"
        )
    if eps < 0.0:
        raise ValueError("softening must be non-negative")
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 0.0:
        raise ValueError("normal must be a nonzero vector")
    axis = axis / axis_norm

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("positions must be finite")

    offsets = points - centre
    # Project out of the disk plane: the orbit is set by the in-plane radius, so a particle
    # displaced along the normal still orbits at its cylindrical radius.
    axial = (offsets @ axis)[:, None] * axis
    radial = offsets - axial
    radii = np.linalg.norm(radial, axis=1)
    if not np.isfinite(radii).all() or (radii <= 0.0).any():
        raise ValueError(
            "every particle needs a positive, finite cylindrical radius; a particle on "
            "the rotation axis has no defined tangent direction"
        )

    radial_unit = radial / radii[:, None]
    tangent = np.cross(axis, radial_unit)
    if not counter_clockwise:
        tangent = -tangent

    if eps > 0.0:
        squared = radii * radii
        speeds = np.sqrt(g * mass * squared / (squared + eps * eps) ** 1.5)
    else:
        speeds = np.sqrt(g * mass / radii)
    if jitter > 0.0:
        signed = rng.uniform(-jitter, jitter, size=int(points.shape[0]))
        speeds = speeds * np.sqrt(1.0 + signed)

    return tangent * speeds[:, None] + centre_velocity + bulk


def escape_speed(
    *, total_mass: float, radius: float, gravitational_constant: float
) -> float:
    """``sqrt(2 G M / R)``: the speed that just unbinds a particle at ``radius`` from the mass.

    The default an Expansion cloud takes when no speed was set. It is the escape speed of the
    cloud's own mass from its own edge -- exact for a thin shell, and for a uniform ball at its
    surface -- so a cloud switched to Expansion expands at the one speed that is neither a
    fall-back nor a violent blast, whatever its mass, size or unit system.
    """

    mass = _require_finite("total_mass", total_mass)
    outer = _require_finite("radius", radius)
    g = _require_finite("gravitational_constant", gravitational_constant)
    if mass <= 0.0:
        raise ValueError("total_mass must be positive")
    if outer <= 0.0:
        raise ValueError("radius must be positive")
    if g <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    return math.sqrt(2.0 * g * mass / outer)


def expansion_velocities(
    *,
    positions: Any,
    centre: Any,
    speed: float,
    proportional: bool,
    reference_radius: float,
    total_mass: float,
    gravitational_constant: float,
    bulk_velocity: Any,
) -> Any:
    """Radial outflow from ``centre`` (ADR 0202): every particle moves straight outward.

    With ``proportional`` False each particle moves at ``speed`` along the unit vector from
    ``centre`` to itself, so a thin shell stays a shell and a ball keeps its shape only in the
    sense that every layer moves at the same rate. With ``proportional`` True the speed grows
    linearly with distance and is ``speed`` at ``reference_radius`` -- a Hubble flow,
    ``v = (speed / reference_radius) * (x - centre)``, in which every pair of particles
    separates at a rate proportional to their separation and the cloud keeps its shape while it
    grows.

    ``speed == 0`` is not "no outflow": it is the cloud's own escape speed from
    ``reference_radius`` (:func:`escape_speed`). A cloud switched to this model by hand has no
    drawn speed to set, and a model that silently did nothing would look like a broken one.
    A particle exactly at the centre has no outward direction and gets only the bulk velocity.

    Pure, deterministic and free of any random draw, so it consumes no stream and cannot move
    one. The bulk velocity is added on top, as for every other model: it is how a moving
    star carries its nebula.

    **What this is not.** It is an initial velocity field, not an equation of motion. Nothing
    keeps the flow going or holds it back afterwards: the particles then move under the
    System's gravity like any others, which for a shell that is far faster than its own
    escape speed is a coast.
    """

    np = _numpy()
    speed_value = _require_finite("speed", speed)
    outer = _require_finite("reference_radius", reference_radius)
    origin = _require_finite_vector("centre", centre)
    bulk = _require_finite_vector("bulk_velocity", bulk_velocity)
    if speed_value < 0.0:
        raise ValueError("speed must be non-negative")

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("positions must be finite")

    if speed_value == 0.0:
        speed_value = escape_speed(
            total_mass=total_mass, radius=outer, gravitational_constant=gravitational_constant
        )
    elif proportional and outer <= 0.0:
        raise ValueError("reference_radius must be positive for a proportional flow")

    offsets = points - origin
    if proportional:
        return offsets * (speed_value / outer) + bulk

    distances = np.linalg.norm(offsets, axis=1)
    # A particle on the centre has no direction; dividing by one leaves its zero offset zero.
    safe = np.where(distances > 0.0, distances, 1.0)
    return offsets * (speed_value / safe)[:, None] + bulk


def rotation_curve_bin_count(count: int) -> int:
    """Radial bins for a cloud of ``count`` particles.

    Derived, not chosen. Two errors trade off against each other and ``sqrt(N)`` is where
    they cross: too few bins and the curve cannot follow a real density profile; too many
    and each bin holds so few particles that its measured force is mostly shot noise. The
    floor of four keeps a curve rather than a single number for a cloud too small to
    profile, and the ceiling of sixty-four bounds the cost, which is linear in bins.
    """

    n = int(count)
    if n < 1:
        raise ValueError("count must be at least 1")
    return max(4, min(64, math.isqrt(n)))


def _plane_basis(np: Any, axis: Any) -> tuple[Any, Any]:
    """Two orthonormal in-plane directions for a unit ``axis``.

    The reference vector is chosen by the axis's *smallest* component, which is the only
    choice that cannot be near-parallel to the axis: a fixed reference would degenerate for
    a disk whose normal happens to point along it, and a random one would make the sampled
    azimuths depend on a stream this function does not own.
    """

    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(axis)))] = 1.0
    first = np.cross(axis, reference)
    first = first / float(np.linalg.norm(first))
    return first, np.cross(axis, first)


def rotation_curve_velocities(
    *,
    positions: Any,
    masses: Any,
    centre: Any,
    normal: Any,
    counter_clockwise: bool,
    gravitational_constant: float,
    softening: float,
    eccentricity_jitter: float,
    bulk_velocity: Any,
    rng: Any,
    field_positions: Any = None,
    field_masses: Any = None,
) -> Any:
    """Rotational support measured from the softened field the disk actually sits in.

    Every other model here balances motion against a potential stated in closed form:
    ``VIRIALIZE`` against a uniform sphere, ``KEPLERIAN`` against a central point mass.
    Neither describes a self-gravitating disk, and ADR 0081 is the record of what happens
    when a cloud is handed speeds for a potential the solver does not integrate -- the disk
    either flies apart or collapses, and the demo depicts neither its name nor anything
    else.

    So this model does not assume a potential. It **measures** one. The disk is divided
    into radial quantile bins; at each bin's radius the inward radial acceleration produced
    by every particle in the cloud is evaluated at ``ROTATION_CURVE_AZIMUTHS`` angles and
    averaged; the circular speed is ``sqrt(a_R * R)``; and each particle's speed is read off
    that curve by linear interpolation at its own cylindrical radius.

    Three properties follow from measuring rather than assuming, and each is the reason for
    a specific line below:

    **It uses the softened force, not the bare one.** ``softening`` enters the pair term as
    the Plummer denominator ``(d^2 + eps^2)^{3/2}``, which is exactly the force the solver
    integrates. This is the ADR 0081 lesson applied at the only place it can be applied: a
    disk supported against the *unsoftened* force is over-supported by precisely the amount
    the softening removes, and expands.

    **It needs no assumption about the disk's shape.** A spherical ``G M(<R) / R^2`` would
    understate a flattened disk's in-plane force by up to about half again, so a disk given
    those speeds contracts. Nothing here is told the profile, the flattening or the
    thickness; the sampled particles are the distribution.

    **It reports no support where none exists.** A bin whose mean radial acceleration comes
    out non-inward supports no circular orbit, and its speed is zero rather than the square
    root of a negative number. That is a real answer about a real distribution -- an outer
    bin can be dominated by the mass outside it -- and it is left visible rather than
    smoothed.

    **It measures the field, not the member (ADR 0087).** ``positions``/``masses`` are the
    particles being *given* speeds -- they set the radial bins and receive the answer.
    ``field_positions``/``field_masses`` are the particles doing the *pulling*. They
    default to the same arrays, which is the pre-0087 behaviour byte for byte, and a
    caller that knows about other mass in the System passes the whole System instead. That
    distinction is what lets a light disk orbit inside a heavy halo: measuring only its own
    20 percent of the mass would hand every particle a speed for a fifth of the gravity it
    is actually going to feel, and the disk would fall straight in.

    The resulting angular momentum is intentional, so -- as with ``KEPLERIAN`` and unlike
    ``VIRIALIZE`` -- no centre-of-mass subtraction or kinetic rescale is applied.

    **What this is not.** It is not an equilibrium solution. There is no vertical structure,
    no velocity dispersion, no epicyclic or asymmetric drift correction, and no Toomre-Q
    stabilisation, so a cold disk sampled this way is free to form its own spiral structure
    and clumps -- which for a galaxy is the point, not a defect, but it means the disk is
    stationary in the azimuthal mean only. Naming it ``Rotation Curve`` rather than
    anything with "equilibrium" in it is deliberate, in the same spirit as the note above
    about ``VIRIALIZE``.
    """

    np = _numpy()
    g = _require_finite("gravitational_constant", gravitational_constant)
    eps = _require_finite("softening", softening)
    jitter = _require_finite("eccentricity_jitter", eccentricity_jitter)
    origin = _require_finite_vector("centre", centre)
    bulk = _require_finite_vector("bulk_velocity", bulk_velocity)
    axis = _require_finite_vector("normal", normal)

    if g <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if eps < 0.0:
        raise ValueError("softening must be non-negative")
    if not 0.0 <= jitter < 1.0:
        raise ValueError("eccentricity_jitter must satisfy 0 <= eccentricity_jitter < 1")

    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 0.0:
        raise ValueError("normal must be a nonzero vector")
    axis = axis / axis_norm

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("positions must be finite")

    weights = np.asarray(masses, dtype=np.float64)
    if weights.ndim != 1 or weights.size != points.shape[0]:
        raise ValueError("masses must be a 1-D array with one entry per position")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("masses must be finite and strictly positive")

    count = int(points.shape[0])
    if count < 2:
        raise ValueError(
            "a rotation curve needs at least two particles: a single particle has no "
            "mass but its own to be supported against"
        )

    # ADR 0087. Omitting both is the pre-0087 call and reproduces it exactly; supplying
    # one without the other is a caller bug rather than a shorthand, because the two
    # arrays have to describe the same population to mean anything.
    if (field_positions is None) != (field_masses is None):
        raise ValueError(
            "field_positions and field_masses are given together or not at all: a "
            "population's coordinates and its masses cannot come from different sets"
        )
    if field_positions is None:
        field_points, field_weights = points, weights
    else:
        field_points = np.asarray(field_positions, dtype=np.float64)
        if field_points.ndim != 2 or field_points.shape[1] != 3:
            raise ValueError("field_positions must have shape (M, 3)")
        if not np.isfinite(field_points).all():
            raise ValueError("field_positions must be finite")
        field_weights = np.asarray(field_masses, dtype=np.float64)
        if field_weights.ndim != 1 or field_weights.size != field_points.shape[0]:
            raise ValueError(
                "field_masses must be a 1-D array with one entry per field position"
            )
        if not np.isfinite(field_weights).all() or (field_weights <= 0.0).any():
            raise ValueError("field_masses must be finite and strictly positive")

    offsets = points - origin
    axial = (offsets @ axis)[:, None] * axis
    radial = offsets - axial
    radii = np.linalg.norm(radial, axis=1)
    if not np.isfinite(radii).all() or (radii <= 0.0).any():
        raise ValueError(
            "every particle needs a positive, finite cylindrical radius; a particle on "
            "the rotation axis has no defined tangent direction"
        )

    first, second = _plane_basis(np, axis)
    angles = np.arange(ROTATION_CURVE_AZIMUTHS, dtype=np.float64) * (
        2.0 * math.pi / ROTATION_CURVE_AZIMUTHS
    )
    # Half-open quantiles: bin centres rather than edges, so every sample radius is one the
    # disk actually has particles at, and the outermost is inside the cloud rather than on
    # its ragged rim where the measured force is a property of the last few particles.
    bins = rotation_curve_bin_count(count)
    quantiles = (np.arange(bins, dtype=np.float64) + 0.5) / float(bins)
    sample_radii = np.quantile(radii, quantiles)
    # Ties collapse when many particles share a radius (a ring, or a degenerate cloud).
    # `np.interp` requires strictly increasing sample points, so keep the first of each.
    keep = np.concatenate(([True], np.diff(sample_radii) > 0.0))
    sample_radii = sample_radii[keep]

    curve = _measured_curve(
        np,
        field_points=field_points,
        field_weights=field_weights,
        origin=origin,
        first=first,
        second=second,
        angles=angles,
        sample_radii=sample_radii,
        gravitational_constant=g,
        softening=eps,
    )

    speeds = np.interp(radii, sample_radii, curve)
    if jitter > 0.0:
        signed = rng.uniform(-jitter, jitter, size=count)
        speeds = speeds * np.sqrt(1.0 + signed)

    radial_unit = radial / radii[:, None]
    tangent = np.cross(axis, radial_unit)
    if not counter_clockwise:
        tangent = -tangent

    return tangent * speeds[:, None] + bulk


def _measured_curve(
    np,
    *,
    field_points,
    field_weights,
    origin,
    first,
    second,
    angles,
    sample_radii,
    gravitational_constant: float,
    softening: float,
):
    """The circular speed at each sample radius, from the field's softened pull on a ring.

    ADR 0133. Every probe is ``origin + R (cos t first + sin t second)``, so a field
    particle's separation from it has in-plane components ``(a - R cos t, b - R sin t)`` and
    an out-of-plane length ``h`` that no probe changes, where ``a`` and ``b`` are the
    particle's own in-plane coordinates. Projecting the field once and sweeping every probe
    as array arithmetic is the same sum over the same particles at the same probes as the
    probe-at-a-time loop it replaces, which cost 22 s per disk at 100,000 bodies and ran four
    times per Play. The squared distance stays a sum of squares, so it is never negative and
    a particle beside a probe does not cancel to garbage. The field is taken in blocks of
    ``ROTATION_CURVE_FIELD_CHUNK`` so memory stays bounded at any count.
    """

    eps_squared = softening * softening
    cosines = np.cos(angles)[:, None]
    sines = np.sin(angles)[:, None]
    inward = np.zeros(sample_radii.size, dtype=np.float64)
    field_count = int(field_points.shape[0])
    for start in range(0, field_count, ROTATION_CURVE_FIELD_CHUNK):
        stop = min(start + ROTATION_CURVE_FIELD_CHUNK, field_count)
        relative = field_points[start:stop] - origin
        along_first = relative @ first
        along_second = relative @ second
        out_of_plane = relative - along_first[:, None] * first - along_second[:, None] * second
        floor = np.einsum("ij,ij->i", out_of_plane, out_of_plane) + eps_squared
        chunk_masses = field_weights[start:stop]
        shape = (int(angles.size), stop - start)
        dx = np.empty(shape, dtype=np.float64)
        dy = np.empty(shape, dtype=np.float64)
        distance_squared = np.empty(shape, dtype=np.float64)
        pull = np.empty(shape, dtype=np.float64)
        for index in range(int(sample_radii.size)):
            radius = float(sample_radii[index])
            np.subtract(along_first, radius * cosines, out=dx)
            np.subtract(along_second, radius * sines, out=dy)
            np.multiply(dx, dx, out=distance_squared)
            distance_squared += dy * dy
            distance_squared += floor
            # The outward component of the separation, ``(x - probe) . u``.
            dx *= cosines
            dy *= sines
            dx += dy
            # A probe sitting exactly on a particle is only reachable at eps == 0, and the
            # pair term there is a true singularity rather than a large number. Dropping
            # that one particle is the same choice a solver makes when it softens.
            usable = distance_squared > 0.0
            np.sqrt(distance_squared, out=pull)
            pull *= distance_squared
            np.divide(dx, pull, out=pull, where=usable)
            pull[~usable] = 0.0
            inward[index] -= float((pull @ chunk_masses).sum())
    curve = np.empty(sample_radii.size, dtype=np.float64)
    for index in range(int(sample_radii.size)):
        radius = float(sample_radii[index])
        mean_inward = gravitational_constant * float(inward[index]) / float(ROTATION_CURVE_AZIMUTHS)
        curve[index] = math.sqrt(mean_inward * radius) if mean_inward > 0.0 else 0.0
    return curve
