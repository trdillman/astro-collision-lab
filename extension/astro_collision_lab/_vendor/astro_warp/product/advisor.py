"""Pure derived-parameter advisor (design D-B).

Every "suggested" number in the authoring surface comes from this module. It is
side-effect-free and never imports bpy; panels and operators call it. Fixtures in
tests/unit/test_advisor.py are the galaxy-merger demo's hand-derived constants.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class PairOrbit:
    semi_major_axis: float
    period: float
    # Apocentre speed magnitude for (body_a, body_b), opposed directions.
    apocentre_speeds: tuple[float, float]


def pair_orbit(
    mass_a: float,
    mass_b: float,
    separation: float,
    pericentre_ratio: float,
    gravitational_constant: float,
) -> PairOrbit:
    """Derive the two-body Keplerian orbit for a pair released at apoapsis.

    ``separation`` is the pair's apocentre (initial, maximum) separation: the
    derivation assumes the two bodies start at apoapsis, which is how the
    galaxy-merger demo authors the encounter. Passing an arbitrary mid-orbit
    separation makes every returned value wrong.

    ``pericentre_ratio`` is pericentre separation divided by apocentre
    separation, so it must lie in the half-open interval (0, 1]. A ratio of
    exactly ``1.0`` means a circular orbit: pericentre equals apocentre, so
    ``semi_major_axis`` reduces to ``separation`` itself, and the returned
    speeds reduce to the ordinary circular-velocity split of
    ``sqrt(gravitational_constant * (mass_a + mass_b) / separation)`` by mass
    ratio. Nothing in the derivation below is singular at ratio 1.0.

    The returned ``apocentre_speeds`` are speed magnitudes, at apoapsis, in
    the pair's centre-of-mass frame for (body_a, body_b); the two bodies move
    in opposite directions.

    Units are caller-defined and only need to be mutually consistent with
    ``gravitational_constant``.
    """

    if mass_a <= 0.0:
        raise ValueError("mass_a must be positive")
    if mass_b <= 0.0:
        raise ValueError("mass_b must be positive")
    if separation <= 0.0:
        raise ValueError("separation must be positive")
    if pericentre_ratio <= 0.0:
        raise ValueError("pericentre_ratio must be positive")
    if pericentre_ratio > 1.0:
        raise ValueError("pericentre_ratio must not exceed 1 (pericentre inside apocentre)")
    if gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    total_mass = mass_a + mass_b
    pericentre = separation * pericentre_ratio
    semi_major_axis = 0.5 * (separation + pericentre)
    mu = gravitational_constant * total_mass
    period = 2.0 * math.pi * math.sqrt(semi_major_axis**3 / mu)
    relative_speed = math.sqrt(mu * (2.0 / separation - 1.0 / semi_major_axis))
    return PairOrbit(
        semi_major_axis=semi_major_axis,
        period=period,
        apocentre_speeds=(
            relative_speed * mass_b / total_mass,
            relative_speed * mass_a / total_mass,
        ),
    )


@dataclass(frozen=True)
class TimelineFit:
    frame_start: int
    frame_end: int
    seconds_per_frame: float


def fit_timeline(period: float, orbits: int, frame_start: int, frame_count: int) -> TimelineFit:
    """Fit ``orbits`` orbital periods onto ``frame_count`` frame intervals.

    ``frame_count`` counts frame *intervals*, not frames: the returned
    ``frame_end`` is ``frame_start + frame_count``, so 250 intervals starting
    at frame 1 spans frames 1..251 inclusive. Increasing ``orbits`` stretches
    the simulated time represented by each frame and leaves the frame range
    itself unchanged. ``seconds_per_frame`` is in whatever time unit ``period``
    is expressed in -- there is no unit conversion here, only the caller's
    convention.

    ``frame_start`` is deliberately unconstrained: Blender frame numbers may
    legitimately be zero or negative, so any integer is accepted.

    Writes nothing: callers (the Fit Timeline operator) apply frame range and
    time scale together, per the signed-off popover.
    """
    if period <= 0.0:
        raise ValueError("period must be positive")
    if orbits <= 0:
        raise ValueError("orbits must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return TimelineFit(
        frame_start=frame_start,
        frame_end=frame_start + frame_count,
        seconds_per_frame=orbits * period / frame_count,
    )


def circumbinary_critical_radius_ratio(mass_ratio: float, eccentricity: float) -> float:
    """The smallest stable circumbinary orbit, as a multiple of the binary's separation.

    Holman & Wiegert (1999), the P-type ("planet orbits both stars") fit to their integrations:

        a_c/a = 1.60 + 5.10 e - 2.22 e^2 + 4.12 mu - 4.27 e mu - 5.09 mu^2 + 4.61 e^2 mu^2

    with ``mu = m2 / (m1 + m2)``. Inside ``a_c`` the binary's time-varying quadrupole drives
    orbits chaotic and clears the region -- which is why a real circumbinary disc has an inner
    hole several times the separation, and why a demo that draws one without the hole is drawing
    something that would not survive.

    This is a *radius*, not a velocity: a Keplerian initial condition about the pair's combined
    mass is perfectly well defined at any radius, and this says where it also stays put. It is
    therefore used to *derive* the disc geometry the template ships, rather than to refuse a
    geometry someone authored -- deriving it means no eccentricity or mass ratio the operator
    admits can put the disc inside the hole, so there is no refusal to write.

    The fit's own domain is ``0 <= e <= 0.7`` and ``0.1 <= mu <= 0.9``; outside that it is an
    extrapolation, which callers are told rather than protected from, because the alternative is
    a template that refuses a legal binary.
    """

    ratio = float(mass_ratio)
    e = float(eccentricity)
    if not math.isfinite(ratio) or not math.isfinite(e):
        raise ValueError("mass_ratio and eccentricity must be finite")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mass_ratio must satisfy 0 < mu < 1, got {ratio!r}")
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity must satisfy 0 <= e < 1, got {e!r}")
    return (
        1.60
        + 5.10 * e
        - 2.22 * e**2
        + 4.12 * ratio
        - 4.27 * e * ratio
        - 5.09 * ratio**2
        + 4.61 * e**2 * ratio**2
    )


def suggest_softening(cloud_radii: list[float]) -> float:
    """Suggest a gravitational softening length from the bodies' cloud radii.

    Ported from the galaxy-merger demo: the suggestion is simply the *largest*
    cloud radius among ``cloud_radii``. This is an engineering rule of thumb
    inherited from one verified scene, not a derived optimum. Its rationale in
    that scene: a cold, self-gravitating cloud with no internal velocity
    dispersion collapses and would otherwise bounce through a singular core
    and disperse; softening at the cloud radius makes the internal field
    nearly harmonic (``omega**2 ~ G M / s**3``), so the cloud breathes
    coherently instead, while the inter-body orbit -- at separations several
    times the radius -- stays close enough to Newtonian that the encounter
    itself is not distorted. Using the largest of several radii keeps every
    body's internal field softened -- and, where the radii differ, *over*-softens
    the smaller ones by exactly their ratio, which is unavoidable in a single
    scene-wide softening and is why this is a rule of thumb rather than a rule.
    It is only safe while that ratio is near one, as it is in the galaxy pair
    this was ported from. The shipped Asteroid Impact pair is 10:1, so the
    suggestion softens its 5 km impactor at ``10 * r`` -- the exact signature
    :data:`ORBITAL_SOFTENING_TOLERANCE` below names as a misapplication. (An
    earlier version of this line claimed the largest radius avoided
    over-softening the smaller bodies, which ``max`` cannot do. ADR 0121
    measured the asteroid consequence: at that separation self-gravity is some
    two to three orders of magnitude weaker than the bodies' own motion, so
    there it is a wrong claim rather than a wrong scene. ADR 0079's bounded
    helpers are the remedy where it does bite.)

    This is a suggestion the "Suggest" button offers, not an override: the
    user may always type a different value.
    """
    if not cloud_radii:
        raise ValueError("cloud_radii must not be empty")
    for radius in cloud_radii:
        if radius <= 0.0:
            raise ValueError("cloud_radii must be positive")
    return max(cloud_radii)


ORBITAL_SOFTENING_TOLERANCE = 0.01
"""Default fractional force error a resolved orbit may lose to softening (ADR 0079).

One percent puts the softening below the error the integrator itself makes over a frame at
the substep counts the templates ship, so softening stops being the dominant error, without
being so small that close particle pairs come to dominate the step. The coefficient it
produces is not sensitive to the choice: 0.001 gives ``0.0258 * r`` and 0.05 gives
``0.1865 * r``, a factor of seven across two decades. What matters is that the ceiling is
``O(0.1) * r`` rather than the ``10 * r`` a misapplied `suggest_softening` produced.
"""


def suggest_orbital_softening(
    orbit_radius: float, *, tolerance: float = ORBITAL_SOFTENING_TOLERANCE
) -> float:
    """The largest softening that leaves the orbit at ``orbit_radius`` essentially Newtonian.

    The companion to `suggest_softening`, answering the opposite question. That one asks how
    *much* softening a cold self-gravitating cloud needs so it breathes instead of bouncing
    through a singular core; this one asks how *little* a scene can get away with before the
    orbits it authored stop existing. A scene with both a cloud and an orbit is bounded from
    below by the first and from above by this, and ships a value satisfying both.

    Plummer softening -- ``r**2 + eps**2`` in every force kernel in the tree -- scales the
    acceleration at radius ``r`` by ``(1 + (eps/r)**2) ** -1.5``. Requiring that to stay at or
    above ``1 - tolerance`` inverts exactly, with no fitting and no measured constant::

        eps_max = r * sqrt((1 - tolerance) ** (-2/3) - 1)

    This matters because a scene whose softening is comparable to its orbital radii no longer
    has Keplerian orbits at all. At ``eps == r`` the pull is 35 percent of Newtonian. Before
    ADR 0188 `velocity_models.keplerian_velocities` handed out the unsoftened ``sqrt(G*M/r)``,
    1.68 times what would keep that orbit, so the particles left; it now hands out the
    softened circular speed, so they stay, but on orbits whose period and shear are those of
    the softening core rather than of the body the scene depicts. Either way the bound is
    what keeps an authored orbit Keplerian.

    ``orbit_radius`` is the *smallest* radius the scene must resolve, since that is where the
    ratio ``eps/r`` is worst -- the inner edge of a disc, the pericentre of an encounter.
    """
    if not math.isfinite(orbit_radius) or orbit_radius <= 0.0:
        raise ValueError("orbit_radius must be positive and finite")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between 0 and 1")
    return orbit_radius * math.sqrt((1.0 - tolerance) ** (-2.0 / 3.0) - 1.0)


CLOUD_BINDING_TOLERANCE = 0.05
"""Default fraction of its own binding energy a virialized cloud may lose to softening (ADR 0081).

A cloud handed ``Q_ref = 2K/|W_0| = 1`` and then integrated in a field whose true binding
energy is ``|W_eps| = (1 - t) |W_0|`` is really starting at ``Q = 1/(1 - t)``, and re-virializes
at ``R_final/R_initial = 1/(1 - t)``. So the tolerance *is* the fractional breathing the cloud
will do from the softening alone: five percent buys a cloud that settles 5.3 percent larger,
which is under the shot noise in its own radius at the particle counts the templates ship.

Five rather than the one percent `ORBITAL_SOFTENING_TOLERANCE` spends, because this bound runs
the opposite way from that one. An orbit only ever wants *less* softening; a cloud wants as much
as it can afford, since softening is what suppresses the two-body scattering a few thousand
particles standing in for 1e11 stars would otherwise produce. One percent would put the ceiling
at ``0.0568 * R``, below the ``0.0585 * R`` `suggest_discreteness_softening` asks for at the
star cluster's shipped 5,000 bodies -- so a one percent ceiling would bind a cloud that is
nowhere near needing to be bound, and hold it under the length softening exists to cover. Five
percent puts the ceiling at ``0.162 * R``, clear of both shipped templates, which is what leaves
the spacing free to be the number that ships. The ceiling is not sensitive to the
choice at the scale that matters: 0.01 gives ``0.0568 * R`` and 0.10 gives ``0.267 * R``, a
factor of five across a decade, against the ``1.0 * R`` a misapplied `suggest_softening` produced.
"""


def uniform_ball_binding_fraction(softening_ratio: float) -> float:
    """The fraction of a uniform ball's binding energy that survives Plummer softening.

    ``softening_ratio`` is ``eps / R``; the result is ``|W(eps)| / |W(0)|``, which is 1 at zero
    softening and falls monotonically toward 0. Closed form, no fitted constant and no
    quadrature.

    `velocity_models.virialized_velocities` scales a cloud's internal dispersion against the
    *unsoftened* uniform-sphere reference ``W_0 = -3 G M**2 / (5 R)``, but the runtime evolves
    that cloud in a Plummer-softened field (``r**2 + eps**2`` in every force kernel in the
    tree), whose binding energy is strictly smaller. This function is the ratio between the two,
    and therefore the factor by which a cloud's realized virial ratio exceeds the one it was
    authored with.

    The derivation is one integral. The pair-separation density inside a ball of radius ``R``,
    in units ``u = s / R``, is ``p(u) = 3 u**2 (1 - 3u/4 + u**3/16)`` on ``[0, 2]``, so::

        |W(eps)| = (G M**2 / 2R) * Integral p(u) / sqrt(u**2 + a**2) du,   a = eps / R

    and the integral is exactly 1.2 at ``a = 0``, reproducing ``3 G M**2 / (5 R)``. Each of the
    three power terms integrates elementarily against ``1/sqrt(u**2 + a**2)``, so the whole
    thing is arithmetic. The ``u**2`` term's ``a**2 asinh(u/a)`` is written as
    ``a**2 (log(u + s) - log(a))``, which is the same value and stays finite as ``a`` approaches
    the smallest representable positive double instead of producing ``0 * inf``.
    """

    a = float(softening_ratio)
    if not math.isfinite(a) or a < 0.0:
        raise ValueError("softening_ratio must be finite and non-negative")
    if a == 0.0:
        return 1.0

    def terms(u: float) -> float:
        s = math.hypot(u, a)
        # Integral u**2 / s du, Integral u**3 / s du, Integral u**5 / s du, each at u.
        quadratic = 0.5 * u * s - 0.5 * a * a * (math.log(u + s) - math.log(a))
        cubic = s**3 / 3.0 - a * a * s
        quintic = s**5 / 5.0 - (2.0 / 3.0) * a * a * s**3 + a**4 * s
        return 3.0 * quadratic - 2.25 * cubic + 0.1875 * quintic

    return (terms(2.0) - terms(0.0)) / 1.2


def suggest_cloud_softening(
    radius: float, *, tolerance: float = CLOUD_BINDING_TOLERANCE
) -> float:
    """The largest softening a virialized cloud of radius ``radius`` still holds itself with.

    The third member of the softening family, and the one that reaches the scenes the other
    two do not. `suggest_softening` is a *lower* bound: how much softening a cold cloud needs
    so it settles instead of bouncing through a singular core. `suggest_orbital_softening` is
    an upper bound for scenes built out of *orbits*. Neither one sees a cloud that is held up
    by its own internal motion rather than by an orbit, and that is the case this answers:
    softening does not only smooth the centre, it weakens the cloud's grip on itself
    everywhere, and a cloud handed the kinetic energy for a grip it is never given expands.

    Requiring `uniform_ball_binding_fraction` to stay at or above ``1 - tolerance`` inverts by
    bisection on a function that is smooth and strictly decreasing, so the result is the exact
    ceiling to the last bit of a double rather than a fitted coefficient. Bisection rather than
    a closed form because the forward direction is the one with the closed form; the inverse of
    that particular combination of radicals and a logarithm has none.

    This is an upper bound on a suggestion, not an override: the user may always type a
    different value, and a scene that also authors an orbit ships the smallest bound of all
    that apply to it.
    """

    outer = float(radius)
    if not math.isfinite(outer) or outer <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between 0 and 1")

    floor = 1.0 - tolerance
    low, high = 0.0, 1.0
    while uniform_ball_binding_fraction(high) >= floor:
        high *= 2.0
        if high > 1.0e12:  # pragma: no cover - unreachable: the fraction decays like 1/a
            raise ValueError("tolerance is too close to 1 to bracket a ceiling")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if uniform_ball_binding_fraction(middle) >= floor:
            low = middle
        else:
            high = middle
    return outer * low


def suggest_discreteness_softening(radius: float, count: int) -> float:
    """The softening a cloud of ``count`` bodies in radius ``radius`` wants: its own spacing.

    The other half of the answer `suggest_cloud_softening` bounds. That function says how much
    softening a virialized cloud can *afford*; this one says how much it actually *wants*, and
    the two are different questions with different answers.

    Softening exists because a few thousand bodies are standing in for something like 1e11
    stars, and a pair of those stand-ins that happens to pass close scatters off a partner a
    hundred million times too massive. The length that suppresses that, and the standard choice
    in the N-body literature, is the spacing between neighbours: below it the smoothing is not
    yet reaching the particles it exists to smooth over, and above it real structure is being
    erased along with the noise. For ``count`` bodies filling a ball of radius ``radius`` that
    spacing is ``radius * count ** (-1/3)`` — a volume per body of ``R**3 / N``, one side of it.

    Independently measured on real particles. A peer session opened a window on the shipped
    demos and arrived at the same rule from the other end, by sweeping softening against the
    virial balance a cloud actually settles at: a seventeenth of the star cluster's radius at
    5,000 bodies, which is ``5000 ** (-1/3) = 0.0585``, to three digits. Its measured balance
    at that softening was 1.012 against the 1.011 this ratio predicts through
    `uniform_ball_binding_fraction`.

    The count is what makes this a suggestion and not a constant: the same cloud with ten times
    the bodies wants a softening a bit over twice as small, and a cloud sampled thinly enough
    wants one so wide it can no longer hold itself — which is the case `suggest_cloud_softening`
    exists to stop, and the reason a caller takes the smaller of the two rather than this alone.
    """

    outer = float(radius)
    if not math.isfinite(outer) or outer <= 0.0:
        raise ValueError("radius must be positive and finite")
    bodies = int(count)
    if bodies < 1:
        raise ValueError("count must be at least one body")
    return outer * bodies ** (-1.0 / 3.0)


#: Mean nearest-neighbour distance of a Poisson point process, in units of ``n ** (-1/3)``.
#:
#: ``Gamma(4/3) * (3 / (4 * pi)) ** (1/3)``. Derived, not measured: for points scattered at
#: number density ``n``, the distance to the nearest other point has the distribution whose
#: mean is this multiple of the side of the cube one point has to itself. It is evaluated
#: here rather than typed out because that is the difference between a derivation and a fit:
#: a literal can be nudged a digit at a time to make one demo look better, and this cannot.
POISSON_NEAREST_NEIGHBOUR = math.gamma(4.0 / 3.0) * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
"""Gamma(4/3) * cbrt(3 / (4 pi)) = 0.8929795 * 0.6203505 = 0.5539603."""


def suggest_neighbour_softening(volume: float, count: int) -> float:
    """The spacing between neighbours in ``count`` bodies filling ``volume``.

    The same physical length `suggest_discreteness_softening` is after, stated for a shape
    whose volume is not a ball. That function reads a *radius*, and a radius does not say how
    tightly a cloud's bodies are packed: a disk and a ball of the same outer radius holding
    the same count differ in density by the ratio of their volumes, so the ball rule applied
    to a disk returns a softening several times too wide and erases the structure it was
    supposed to protect. Reading the volume the sampler actually filled removes that failure
    mode rather than correcting for it.

    Why the nearest-neighbour distance and not the cube side. Softening exists to stop a pair
    of bodies from scattering off each other at close range, and the range at which that
    starts to happen is set by how far a body typically is from its *closest* neighbour, not
    by how much room it has on average. For Poisson-scattered points those differ by a fixed
    factor, `POISSON_NEAREST_NEIGHBOUR`, which is where the constant comes from -- a
    derivation, not a fit.

    The two rules agree, which is what makes this one a restatement rather than a second
    policy. For a ball of radius ``R`` this returns ``0.893 * R * count ** (-1/3)`` against
    the ball rule's ``1.0 * R * count ** (-1/3)``: the same length to 12 percent, and the
    ball rule's measured anchor -- a seventeenth of the star cluster's radius at 5,000
    bodies -- survives the substitution.

    Measured on the disk it was written for (ADR 0083). Two galaxies of 10 kpc radius and
    2 kpc thickness were sampled at 750, 1,500 and 3,000 bodies, and the rotation curve
    `rotation_curve_velocities` measures from them came out the same curve at all three --
    0.13 to 0.14 code units at one kpc rising to 0.26 at the rim -- with a seed-to-seed
    spread of at most 10, 19 and 5 percent outside the innermost bin. A softening that had
    to be tuned per count could not have done that.
    """

    span = float(volume)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("volume must be positive and finite")
    bodies = int(count)
    if bodies < 1:
        raise ValueError("count must be at least one body")
    return POISSON_NEAREST_NEIGHBOUR * (span / bodies) ** (1.0 / 3.0)


def _positive_finite(value: float, name: str) -> float:
    """``value`` as a float, or a ``ValueError`` naming ``name`` if it is not a usable number.

    Strict about type as well as range: a bool is an ``int`` and would pass a numeric check
    as 1.0, and a string that happens to parse would hide a caller that read the wrong field.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _non_negative_finite(value: float, name: str) -> float:
    """``_positive_finite`` for a quantity that may legitimately be zero."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return numeric


def binding_free_fall_index(
    clouds: Sequence[tuple[float, float]], gravitational_constant: float
) -> int:
    """Index of the ``(mass, radius)`` cloud with the shortest free-fall time.

    The cloud :func:`suggest_substeps` is limited by, so the panel can say which object set
    the value. Ties resolve to the first.
    """
    if not clouds:
        raise ValueError("clouds must not be empty")
    if gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    best_index = 0
    best_time = math.inf
    for index, (cloud_mass, cloud_radius) in enumerate(clouds):
        if cloud_mass <= 0.0 or cloud_radius <= 0.0:
            raise ValueError(f"clouds[{index}] mass and radius must be positive")
        density = cloud_mass / (4.0 / 3.0 * math.pi * cloud_radius**3)
        free_fall = math.sqrt(3.0 * math.pi / (32.0 * gravitational_constant * density))
        if free_fall < best_time:
            best_index, best_time = index, free_fall
    return best_index


def suggest_substeps(
    seconds_per_frame: float,
    clouds: Sequence[tuple[float, float]],
    gravitational_constant: float,
    steps_per_free_fall: int = 40,
) -> int:
    """Suggest an integrator substep count from a System's clouds' free-fall times.

    Ported from the galaxy-merger demo. The step is sized by the fastest
    timescale the scene can reach: the free-fall time of a uniform sphere,
    ``sqrt(3 pi / (32 G rho))``, where ``rho`` is a cloud's mass divided by
    its volume. A step chosen from the orbit alone can advance a close
    encounter by a large fraction of its own crossing time and disperse it,
    which looks like the merger failing when it is really the integration
    failing.

    A System may hold several clouds -- each ``(mass, radius)`` pair in
    ``clouds`` -- with different densities and therefore different free-fall
    times. The suggestion takes the *minimum* free-fall time across all of
    them: the fastest-collapsing cloud is the one whose internal dynamics the
    step can fail to resolve, so it is the binding constraint on the whole
    System's substep count, and every slower-collapsing cloud is automatically
    over-resolved rather than under-resolved. This is the aggregation
    counterpart to ``suggest_softening`` taking the *maximum* radius: there,
    the largest body sets the binding constraint on softening; here, the
    densest (fastest-collapsing) body sets the binding constraint on step
    size.

    ``steps_per_free_fall`` controls how many substeps are demanded per
    free-fall time -- i.e. how finely the collapse timescale is resolved. It
    is an engineering rule of thumb: 40 is the value the merger demo used and
    validated, not a derived optimum, which is why it is exposed as a keyword
    argument -- callers may raise it for stiffer configurations (denser
    clouds, tighter pericentres) where 40 is not enough. The result is
    rounded up (``math.ceil``) and floored at 1 substep per frame.

    This is a suggestion the "Suggest" button offers, not an override: the
    user may always type a different value.
    """
    if seconds_per_frame <= 0.0:
        raise ValueError("seconds_per_frame must be positive")
    if not clouds:
        raise ValueError("clouds must not be empty")
    for index, (cloud_mass, cloud_radius) in enumerate(clouds):
        if cloud_mass <= 0.0:
            raise ValueError(f"clouds[{index}] mass must be positive")
        if cloud_radius <= 0.0:
            raise ValueError(f"clouds[{index}] radius must be positive")
    if gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if steps_per_free_fall <= 0:
        raise ValueError("steps_per_free_fall must be positive")
    free_fall_times = []
    for cloud_mass, cloud_radius in clouds:
        density = cloud_mass / (4.0 / 3.0 * math.pi * cloud_radius**3)
        free_fall_times.append(math.sqrt(3.0 * math.pi / (32.0 * gravitational_constant * density)))
    binding_free_fall_time = min(free_fall_times)
    return max(1, math.ceil(seconds_per_frame * steps_per_free_fall / binding_free_fall_time))


def suggest_orbit_substeps(
    seconds_per_frame: float,
    pericentre_separation: float,
    total_mass: float,
    gravitational_constant: float,
    steps_per_orbit: int = 60,
) -> int:
    """Suggest an integrator substep count for a point-mass *orbital* System.

    The orbital counterpart to :func:`suggest_substeps`, for Systems whose
    binding timescale is a Keplerian orbit between point masses rather than the
    collapse of a diffuse cloud. A single-particle Body is a point mass: its
    ``radius`` is a rendering/photosphere size, not the extent of a
    self-gravitating particle distribution, so feeding it to
    :func:`suggest_substeps` computes the *star's own* internal free-fall time --
    a stellar dynamical time far shorter than, and physically unrelated to, the
    orbit the System actually integrates -- and demands hundreds of times more
    substeps than the motion needs (a two-Body binary asked for 2266 substeps
    per frame where its orbit is resolved to fewer than two). A point mass has no
    internal collapse timescale to resolve; the fastest timescale the orbit
    reaches is the one to size the step by.

    That timescale is the orbit at *pericentre* -- closest approach, where the
    bodies move fastest and the integrator is most stressed. The suggestion
    resolves one circular orbital period at the pericentre separation,
    ``2 pi sqrt(r_p**3 / (G M))``, into ``steps_per_orbit`` integrator steps.
    This is the same "resolve the fastest dynamical timescale to N steps" rule
    :func:`suggest_substeps` applies to clouds, and it scales correctly with
    eccentricity: a tight pericentre (high ``e``) shortens ``r_p`` and raises the
    count exactly where the encounter is fastest, while a near-circular orbit
    already resolved by many frames per period asks for the floor of one. For a
    circular orbit ``r_p`` equals the semi-major axis and the resolved period is
    the true orbital period.

    ``pericentre_separation`` is the closest-approach separation of the two
    masses (``separation * (1 - e) / (1 + e)`` for a pair released at apoapsis);
    ``total_mass`` is their combined mass. ``steps_per_orbit`` mirrors
    ``steps_per_free_fall``: 60 integrator steps per orbit is the leapfrog
    energy-conservation rule of thumb, exposed so callers may raise it for
    stiffer configurations. The result is rounded up (``math.ceil``) and floored
    at 1 substep per frame.

    This is a suggestion the "Suggest" button offers, not an override: the user
    may always type a different value.
    """
    seconds_per_frame = _positive_finite(seconds_per_frame, "seconds_per_frame")
    pericentre_separation = _positive_finite(
        pericentre_separation, "pericentre_separation"
    )
    total_mass = _positive_finite(total_mass, "total_mass")
    gravitational_constant = _positive_finite(
        gravitational_constant, "gravitational_constant"
    )
    if type(steps_per_orbit) is not int or steps_per_orbit <= 0:
        raise ValueError("steps_per_orbit must be a positive int")

    # Preserve the direct expression's ordinary-scale rounding, but detect every
    # underflow/overflow-prone intermediate and fall back to a log-domain evaluation.
    # The fallback avoids leaking raw ZeroDivisionError/OverflowError for otherwise
    # valid IEEE-754 inputs at the extremes of Blender's unit scales.
    try:
        radius_cubed = pericentre_separation**3
        gravitational_parameter = gravitational_constant * total_mass
        if (
            radius_cubed == 0.0
            or gravitational_parameter == 0.0
            or not math.isfinite(radius_cubed)
            or not math.isfinite(gravitational_parameter)
        ):
            raise ArithmeticError
        pericentre_period = 2.0 * math.pi * math.sqrt(
            radius_cubed / gravitational_parameter
        )
        required = seconds_per_frame * steps_per_orbit / pericentre_period
        if not math.isfinite(required):
            raise ArithmeticError
        return max(1, math.ceil(required))
    except (ArithmeticError, OverflowError, ZeroDivisionError):
        log_required = (
            math.log(seconds_per_frame)
            + math.log(steps_per_orbit)
            - math.log(2.0 * math.pi)
            + 0.5
            * (
                math.log(gravitational_constant)
                + math.log(total_mass)
                - 3.0 * math.log(pericentre_separation)
            )
        )
        if log_required <= 0.0:
            return 1
        if log_required > math.log(sys.float_info.max):
            raise ValueError(
                "derived orbital substep count cannot be represented"
            ) from None
        return max(1, math.ceil(math.exp(log_required)))


ROTATION_CURVE_STEPS_PER_FREE_FALL = 18
"""How finely a rotation-curve disk's free-fall time is resolved (ADR 0196).

The 18 the Galaxy Collision template ships for all four of its clouds, its disks included, not
the 40 `suggest_substeps` asks of a cold cloud. ADR 0147 measured what the lower figure costs
and buys on that scene: three substeps where 40 asked for seven, with the encounter ruler still
passing. A standalone rotation-curve disk inherits the rate rather than re-measuring it. It is
restated here instead of imported because `scenario_builder` imports this module, and
`tests/unit/test_aw005_orbit_substeps.py` fails if the two ever stop being equal.
"""

LIMIT_CLOUD_FREE_FALL = "CLOUD_FREE_FALL"
LIMIT_DISK_ORBIT = "DISK_ORBIT"
LIMIT_DISK_FREE_FALL = "DISK_FREE_FALL"
LIMIT_BODY_ORBIT = "BODY_ORBIT"
"""What set a suggested substep count: the third value `suggest_system_substeps` returns.

A Ball cloud's free fall, a Keplerian disk's orbit at its inner edge, a rotation-curve disk's
free fall, or a Body's orbit about the heaviest Body. The Blender panel words each one
differently, so the kind travels with the label rather than being re-derived from it.
"""

DISK_KEPLERIAN = "KEPLERIAN"
DISK_ROTATION_CURVE = "ROTATION_CURVE"
"""How a disk moves, spelled as the Cloud's Velocity Model identifiers.

A caller reading the enum off a Cloud can pass it straight through. Only these two models
describe a disk whose timescale is knowable from its numbers; the others are the caller's to
leave out, and say so.
"""


@dataclass(frozen=True)
class SubstepCloud:
    """A Ball cloud: the uniform sphere its ``mass`` and ``radius`` make has a free-fall time."""

    label: str
    mass: float
    radius: float


@dataclass(frozen=True)
class SubstepDisk:
    """A Disk Annulus cloud, sized by how it moves.

    A ``DISK_KEPLERIAN`` disk orbits ``central_mass`` -- the central Body's mass, plus its
    companion's for a disk round a pair -- and its fastest orbit is the one at ``inner_radius``,
    so those are the only fields it reads. A ``DISK_ROTATION_CURVE`` disk is held up by its own
    gravity and names no central mass; it is sized by the free fall of the uniform sphere
    ``mass`` and ``outer_radius`` make, the way the Galaxy Collision template sizes its disks.
    """

    label: str
    motion: str
    mass: float
    outer_radius: float
    inner_radius: float = 0.0
    central_mass: float = 0.0


@dataclass(frozen=True)
class SubstepBody:
    """A single-particle Body: a point mass at ``position``, in the System's length unit."""

    label: str
    mass: float
    position: tuple[float, float, float]


def suggest_system_substeps(
    seconds_per_frame: float,
    clouds: Sequence[SubstepCloud],
    disks: Sequence[SubstepDisk],
    bodies: Sequence[SubstepBody],
    gravitational_constant: float,
    softening: float,
) -> tuple[int, str, str]:
    """Suggest a substep count for a whole System, and say which member sets it (ADR 0196).

    `suggest_substeps` reads Ball clouds and nothing else, so a System made of Bodies or of a
    disk had no way to ask: eleven of the nineteen Add-menu objects are one of those. The
    templates already size such Systems by their own fastest timescale -- ADR 0057 for orbits,
    ADR 0147 for the galaxy -- and this is that rule written once, for the System as the
    author left it. The answer is the *largest* count any one member asks for, for the reason
    `suggest_substeps` gives for taking the largest across clouds: the fastest member is the
    one the step can fail to resolve, and every slower one is over-resolved rather than
    under-resolved. Four kinds of member contribute:

    * a Ball cloud, by its free-fall time at the default 40 steps per free fall -- unchanged;
    * a Keplerian disk, by its orbit at the inner edge about its central mass, at the 60
      steps per orbit `suggest_orbit_substeps` asks of a point-mass orbit. The inner radius is
      floored at ``softening``: the solver integrates a Plummer-softened centre, whose pull
      falls away inside the softening length (ADR 0188), so the orbital frequency tops out at
      the unsoftened frequency *at* the softening length and no orbit is faster than that. The
      floor also gives a disk with no hole (inner radius 0) a number, where the template
      leaves it to the disk's self-gravity alone;
    * a rotation-curve disk, by its free-fall time at `ROTATION_CURVE_STEPS_PER_FREE_FALL`;
    * a Body, by its circular orbit about the *heaviest* Body at their current separation,
      the same assumption Fit Timeline makes because no eccentricity is authored. The
      separation is floored at ``softening`` for the same reason as the disk's inner edge. The
      heaviest Body has no orbit of its own to resolve, so a lone Body asks for nothing.

    Not sized, and so never the reason for a count: a Ball cloud's bulk orbit about a Body, a
    Keplerian disk's own self-gravity (which the protoplanetary template adds, and which asks
    for more than the inner orbit only when the disk weighs at least a fourteenth of its star
    and more the wider it is than its hole), a Body's eccentricity, and whether a Body is
    pinned.

    Returns ``(substeps, label, kind)``: the count, the ``label`` of the member that set it,
    and which of the ``LIMIT_*`` kinds that member is. A tie goes to the earlier member,
    clouds before disks before Bodies. The count is rounded up and floored at 1, and with
    only Ball clouds it equals `suggest_substeps` of the same clouds exactly.

    Raises ``ValueError`` for an unusable number, and when no member asks for anything -- a
    System with nothing to size the step from, which is the caller's to report.

    This is a suggestion the "Suggest" button offers, not an override: the user may always
    type a different value.
    """

    seconds_per_frame = _positive_finite(seconds_per_frame, "seconds_per_frame")
    gravitational_constant = _positive_finite(gravitational_constant, "gravitational_constant")
    softening = _non_negative_finite(softening, "softening")

    # Each entry is (substeps, label, kind). Order is the tie-break: `max` keeps the first
    # of equal counts, so clouds win over disks, which win over Bodies.
    candidates: list[tuple[int, str, str]] = []

    for cloud in clouds:
        mass = _positive_finite(cloud.mass, f"cloud '{cloud.label}' mass")
        radius = _positive_finite(cloud.radius, f"cloud '{cloud.label}' radius")
        candidates.append(
            (
                suggest_substeps(
                    seconds_per_frame, [(mass, radius)], gravitational_constant
                ),
                cloud.label,
                LIMIT_CLOUD_FREE_FALL,
            )
        )

    for disk in disks:
        if disk.motion == DISK_KEPLERIAN:
            central_mass = _positive_finite(
                disk.central_mass, f"disk '{disk.label}' central mass"
            )
            inner = _non_negative_finite(
                disk.inner_radius, f"disk '{disk.label}' inner radius"
            )
            # Zero is a real authored value (a disk with no hole) and zero softening is a
            # real setting; both together name an orbit with no length, which is the only
            # case here with no honest number to give.
            orbit_radius = max(inner, softening)
            if orbit_radius <= 0.0:
                raise ValueError(
                    f"disk '{disk.label}' runs in to its central mass (Inner Radius 0) and "
                    "Softening is 0, so its fastest orbit has no length; give the disk an "
                    "Inner Radius or the System a Softening"
                )
            candidates.append(
                (
                    suggest_orbit_substeps(
                        seconds_per_frame, orbit_radius, central_mass, gravitational_constant
                    ),
                    disk.label,
                    LIMIT_DISK_ORBIT,
                )
            )
        elif disk.motion == DISK_ROTATION_CURVE:
            mass = _positive_finite(disk.mass, f"disk '{disk.label}' mass")
            radius = _positive_finite(disk.outer_radius, f"disk '{disk.label}' outer radius")
            candidates.append(
                (
                    suggest_substeps(
                        seconds_per_frame,
                        [(mass, radius)],
                        gravitational_constant,
                        steps_per_free_fall=ROTATION_CURVE_STEPS_PER_FREE_FALL,
                    ),
                    disk.label,
                    LIMIT_DISK_FREE_FALL,
                )
            )
        else:
            raise ValueError(
                f"disk '{disk.label}' moves by {disk.motion!r}; only "
                f"{DISK_KEPLERIAN!r} and {DISK_ROTATION_CURVE!r} disks can be sized"
            )

    masses = [_positive_finite(body.mass, f"body '{body.label}' mass") for body in bodies]
    positions = [tuple(float(value) for value in body.position) for body in bodies]
    for body, position in zip(bodies, positions, strict=True):
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError(f"body '{body.label}' position must be three finite numbers")
    # Every pair of Bodies, not every Body about the heaviest: the step has to resolve the
    # fastest two-body orbit in the System, and in a star, planet and moon that is the moon
    # about its planet (about 8x more substeps than the moon about the star). A pair is named
    # by its lighter Body, the one doing the orbiting; equal masses name the later one.
    for first in range(len(bodies)):
        for second in range(first + 1, len(bodies)):
            separation = max(math.dist(positions[first], positions[second]), softening)
            if separation <= 0.0:
                raise ValueError(
                    f"body '{bodies[second].label}' sits exactly on '{bodies[first].label}' "
                    "and Softening is 0, so their orbit has no length; move one of them or "
                    "give the System a Softening"
                )
            lighter = first if masses[first] < masses[second] else second
            candidates.append(
                (
                    suggest_orbit_substeps(
                        seconds_per_frame,
                        separation,
                        masses[first] + masses[second],
                        gravitational_constant,
                    ),
                    bodies[lighter].label,
                    LIMIT_BODY_ORBIT,
                )
            )

    if not candidates:
        raise ValueError(
            "nothing here sets a time step: add a Ball cloud, a Keplerian or Rotation Curve "
            "disk, or a second Body for the first to orbit"
        )
    return max(candidates, key=lambda candidate: candidate[0])
