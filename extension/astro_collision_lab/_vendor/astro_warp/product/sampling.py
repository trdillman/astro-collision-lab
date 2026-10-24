"""Expansion of authored bindings into the solver bodies they represent.

One bound Blender object is one *binding*. A binding is not necessarily one body:
``BodyRepresentation.SAMPLED_PARTICLE`` declares a particle cloud of
``particle_count`` bodies drawn from the object's position and radius. Every place
that used to say ``len(descriptor.bindings)`` when it meant "how many bodies does the
solver hold" routes through :func:`body_count` or :func:`expand_bindings` instead, so
the cardinality rule lives in exactly one file.

Sampling runs once, at load, as vectorized host work followed by a single
host-to-device copy. That is not the per-particle steady-state Python work ``ARCH-005``
and ``AGENTS.md`` forbid: no host code touches a body again after the copy. The ceiling
is host memory for the staging buffer (~56 bytes per body, so ~56 MB at 10^6 bodies);
if authored counts outgrow that, the upgrade path is a device-side sampling kernel in
the ``INITIAL_STATE`` stage writing the same arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .model import (
    BindingDescriptor,
    BindingRole,
    BodyRepresentation,
    CloudVelocityModel,
    SampleShape,
    runtime_stable_id,
)
from .shearing_box import SHEAR_Q
from .velocity_models import (
    KEPLERIAN_ECCENTRICITY_DOMAIN,
    MASS_JITTER_DOMAIN,
    ROTATION_CURVE_ECCENTRICITY_DOMAIN,
    VIRIAL_VELOCITY_DOMAIN,
    domain_rng,
    expansion_velocities,
    jittered_masses,
    keplerian_velocities,
    rotation_curve_velocities,
    virialized_velocities,
)

_FNV_PRIME = 1099511628211
_FNV_MASK = 0x7FFFFFFFFFFFFFFF


def _numpy() -> Any:
    return __import__("numpy")


def binding_body_count(binding: BindingDescriptor) -> int:
    """How many solver bodies one binding expands to.

    A FIELD-role binding is zero (ADR 0045): a Force is authored and bound, but it is a
    thing that acts on bodies, not a body the solver integrates. Zero here is what keeps
    capacities, allocation, and the HUD count all meaning the same population.
    """

    if binding.role == BindingRole.FIELD:
        return 0
    if binding.representation == BodyRepresentation.SAMPLED_PARTICLE:
        return max(1, int(binding.particle_count))
    return 1


def body_count(bindings: tuple[BindingDescriptor, ...]) -> int:
    """Total solver bodies across every binding, active or not."""

    return sum(binding_body_count(binding) for binding in bindings)


def active_body_count(bindings: tuple[BindingDescriptor, ...]) -> int:
    """Total solver bodies contributed by active bindings only."""

    return sum(binding_body_count(binding) for binding in bindings if binding.active)


@dataclass(frozen=True)
class BodySet:
    """Solver-ready state-of-arrays for a whole context, in binding order.

    Arrays are float64 regardless of the runtime's precision lane: sampling in the wide
    type and narrowing at the device copy makes the float32 and float64 lanes start from
    the same numbers, so a lane comparison measures the solver rather than its input.
    """

    positions: Any
    velocities: Any
    masses: Any
    radii: Any
    stable_ids: Any
    active: Any
    # Per-particle material, or None when the body set has no differentiated bodies. Only the
    # Solid engines' own samplers (`sample_giant_impact_bodies`, `sample_spacesim_bodies`) fill
    # it; `expand_bindings` never does.
    material_ids: Any = None
    # Per-particle Body flag (1 = the element is a Body in the ADR 0037 §3.3 sense, the
    # accretion sink of `CollisionMode.ABSORB`; 0 = ordinary particle). Explicit rather
    # than inferred device-side (largest radius, index 0), because an implicit rule goes
    # silently wrong the first time a scene has two Bodies or a particle outgrows the
    # Body's radius. `expand_bindings` is the flag's only origin. None means "no flags
    # sampled", which the runtime treats as all zeros.
    body_flags: Any = None
    # Per-particle Pinned flag (ADR 0158; 1 = the element's position is held fixed and its
    # velocity held at zero for the whole run, 0 = free). Only an analytic Body can carry 1.
    # None means no body is pinned, which is what every scene authored before Pinned was
    # simulated is, so those scenes build exactly the arrays they always did.
    pinned: Any = None

    def __len__(self) -> int:
        return int(self.stable_ids.shape[0])

    def head(self, count: int) -> BodySet:
        """The first ``count`` bodies, for scaling benchmarks over one authored scene."""

        return BodySet(
            positions=self.positions[:count],
            velocities=self.velocities[:count],
            masses=self.masses[:count],
            radii=self.radii[:count],
            stable_ids=self.stable_ids[:count],
            active=self.active[:count],
            material_ids=None if self.material_ids is None else self.material_ids[:count],
            body_flags=None if self.body_flags is None else self.body_flags[:count],
            pinned=None if self.pinned is None else self.pinned[:count],
        )


def particle_stable_ids(binding_id: str, count: int) -> Any:
    """``count`` distinct runtime stable IDs derived from one binding ID.

    Index 0 is exactly ``runtime_stable_id(binding_id)``, so a single-body binding keeps
    the identity it had before particle clouds existed and every bake manifest written
    against the old rule still matches. Later indices continue the same FNV-1a mixing
    over the index bytes.
    """

    np = _numpy()
    base = runtime_stable_id(binding_id)
    ids = np.empty(count, dtype=np.uint64)
    ids[0] = np.uint64(base)
    if count > 1:
        index = np.arange(1, count, dtype=np.uint64)
        value = np.full(count - 1, base, dtype=np.uint64)
        for _ in range(8):
            value ^= index & np.uint64(0xFF)
            value = (value * np.uint64(_FNV_PRIME)) & np.uint64(_FNV_MASK)
            index >>= np.uint64(8)
        ids[1:] = value
    return ids.astype(np.int64)


def _rng(binding: BindingDescriptor, np: Any) -> Any:
    """The cloud's generator, seeded from the authored seed *and* the binding ID.

    Both, so a body samples the same cloud no matter what else is in the scene or what
    order the bindings compile in.
    """

    return np.random.default_rng(
        [int(binding.sampling_seed), runtime_stable_id(binding.binding_id)]
    )


def _sample_ball(binding: BindingDescriptor, ball_radius: float, count: int, np: Any) -> Any:
    """``count`` positions uniform in the ball of ``ball_radius`` about the binding."""

    rng = _rng(binding, np)
    # r ~ R * U^(1/3) is the inverse CDF of a uniform-density ball; sampling r uniformly
    # would pile mass at the centre and change the field the solver is asked to reproduce.
    radius = float(ball_radius) * np.cbrt(rng.random(count))
    direction = rng.standard_normal((count, 3))
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= np.where(norm == 0.0, 1.0, norm)
    return np.asarray(binding.position, dtype=np.float64) + radius[:, None] * direction


def _sample_shell(binding: BindingDescriptor, count: int, np: Any) -> tuple[Any, float]:
    """``count`` positions uniform in volume between the inner radius and ``binding.radius``.

    A thin spherical shell about the binding's position (ADR 0202). The outer radius is the
    cloud's own ``radius`` and the inner one is ``annulus_inner_radius`` -- the same "how much
    of the middle to leave empty" a ring has -- so the shell is the ball with its middle taken
    out. The radial coordinate is the ball's inverse CDF restricted to the shell: with
    ``f = inner / outer``, ``r = outer * cbrt(f**3 + u * (1 - f**3))`` is uniform in volume
    between the two spheres. At ``f == 0`` that is ``outer * cbrt(u)`` evaluated with the very
    same operations, in the very same order, as :func:`_sample_ball`, and the directions come
    from the same draw after it, so a shell with no hole is a ball *bit for bit* and a scene
    that switches Shape from Ball to Shell with Inner Radius zero changes nothing.

    Returns the positions and the volume's characteristic radius -- the radius of the solid
    ball holding the same volume, ``cbrt(outer**3 - inner**3)`` -- so a thin shell's particles
    are sized by the volume each one fills, as a ball's are, and not by the shell's diameter.
    """

    outer = float(binding.radius)
    inner = float(binding.annulus_inner_radius)
    if not (math.isfinite(outer) and outer > 0.0):
        raise ValueError(
            f"a Shell needs a positive radius, but {binding.object_name!r} has {outer!r}; "
            "give the object a size"
        )
    if not (math.isfinite(inner) and 0.0 <= inner < outer):
        raise ValueError(
            f"a Shell's Inner Radius must be at least zero and less than its radius "
            f"({outer:.6g}), but {binding.object_name!r} has {inner!r}; there is no shell "
            "to sample between those two spheres"
        )
    rng = _rng(binding, np)
    hollow = (inner / outer) ** 3
    radius = outer * np.cbrt(hollow + rng.random(count) * (1.0 - hollow))
    direction = rng.standard_normal((count, 3))
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= np.where(norm == 0.0, 1.0, norm)
    positions = np.asarray(binding.position, dtype=np.float64) + radius[:, None] * direction
    # Exactly ``outer`` when there is no hole, not ``cbrt(outer**3)``, which can land an ulp away.
    characteristic = outer if inner == 0.0 else float(np.cbrt(outer**3 - inner**3))
    return positions, characteristic


def _sample_box(
    binding: BindingDescriptor,
    extent: tuple[float, float, float],
    count: int,
    np: Any,
) -> Any:
    """``count`` positions uniform in the axis-aligned box of ``extent``, centred on it.

    Uniform per axis is the whole of it: a box has constant density under a uniform draw,
    unlike a ball, where the radial coordinate needs its inverse CDF.
    """

    rng = _rng(binding, np)
    offsets = (rng.random((count, 3)) - 0.5) * np.asarray(extent, dtype=np.float64)
    return np.asarray(binding.position, dtype=np.float64) + offsets


# ADR 0040 section 5's fixed tolerances. Both are stated in the ADR and may not be widened
# to accommodate a result: they decide whether an authored object is a disk at all.
ANNULUS_BASIS_TOL = 1e-9 + 1e-7


def annulus_length_tol(length: float) -> float:
    """Scale-aware length tolerance in code-length units (ADR 0040 section 5).

    Scale-aware because a 1e6-unit galactic disk and a 1e-3-unit debris ring cannot share
    one absolute epsilon; the 1e-12 floor keeps the tolerance positive at zero length.
    """

    return 1e-9 + 1e-7 * max(abs(float(length)), 1e-12)


def validate_annulus_basis(basis: Any) -> Any:
    """Refuse any basis that is not a right-handed orthonormal frame.

    Every failure here is a *refusal*, never a repair. ADR 0040: "No hidden inner-radius
    fraction, axis-aligned fallback, world-AABB reinterpretation, mirrored-basis repair, or
    post-failure fallback is allowed." Each of those would produce a plausible-looking disk
    from an object the user did not author, which is worse than an error because nothing
    in the result says it happened.

    The determinant test is the one that is easy to leave out. A mirrored basis is still
    perfectly orthonormal, so without it a negative-scaled object samples a disk whose
    particles orbit the wrong way -- visually plausible, physically reversed.
    """

    np = _numpy()
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"sample_basis must be 3x3, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("sample_basis must be finite")

    for index in range(3):
        deviation = abs(float(np.linalg.norm(matrix[index])) - 1.0)
        if deviation > ANNULUS_BASIS_TOL:
            raise ValueError(
                f"sample_basis row {index} is not unit length (off by {deviation:.3e}); "
                "a scaled or sheared transform is not a disk orientation"
            )
    for i in range(3):
        for j in range(i + 1, 3):
            dot = abs(float(matrix[i] @ matrix[j]))
            if dot > ANNULUS_BASIS_TOL:
                raise ValueError(
                    f"sample_basis rows {i} and {j} are not orthogonal (dot {dot:.3e}); "
                    "a sheared transform has no well-defined disk plane"
                )
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.0:
        raise ValueError(
            f"sample_basis is mirrored (determinant {determinant:.3e}); a mirrored "
            "transform would reverse the disk's orbital direction, and repairing it "
            "silently is not allowed"
        )
    return matrix


def annulus_volume(
    *, outer_radius: float, inner_radius: float, half_thickness: float
) -> float:
    """``V = pi * (Rout^2 - Rin^2) * (2 * H)`` (ADR 0040 section 5)."""

    outer = float(outer_radius)
    inner = float(inner_radius)
    half = float(half_thickness)
    for name, value in (
        ("outer_radius", outer),
        ("inner_radius", inner),
        ("half_thickness", half),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if outer <= 0.0:
        raise ValueError("annulus outer radius must be positive")
    if half <= 0.0:
        raise ValueError("annulus thickness must be positive")
    if not 0.0 <= inner < outer:
        raise ValueError(
            f"annulus inner radius must satisfy 0 <= inner < outer, got inner={inner} "
            f"outer={outer}"
        )
    return math.pi * (outer**2 - inner**2) * (2.0 * half)


def _profiled_annulus_radii(
    rng: Any,
    *,
    count: int,
    inner_radius: float,
    outer_radius: float,
    exponent: float,
) -> Any:
    """Radii drawn so the annulus's surface density falls as ``r ** -exponent``.

    Uniform in area is what an annulus sampler does if nobody tells it otherwise, and it is
    the wrong shape for a disc: the number of particles in a ring of width ``dr`` goes as
    ``sigma(r) * 2 * pi * r * dr``, so with ``sigma`` constant the outer rings win on
    circumference alone and the picture reads as a hollow ring. Falling ``sigma`` is what puts
    the light in the middle, which is what makes a disc look like one.

    The draw is exact rather than rejection-sampled, by inverting the cumulative count. With
    ``q = 2 - exponent`` the enclosed count goes as ``r ** q``, so

        r = (inner ** q + u * (outer ** q - inner ** q)) ** (1 / q)

    for a uniform ``u``, and ``exponent = 0`` recovers the uniform-in-area square root exactly.
    At ``exponent == 2`` the count goes as ``log r`` instead and the closed form above divides
    by zero, so that case is drawn from its own expression rather than nudged past the pole --
    a nudge would be a wrong distribution that still returns numbers.

    Two configurations have no answer and are refused rather than clipped. An annulus with no
    hole and ``exponent >= 2`` encloses infinitely many particles near the centre, so there is
    no distribution to draw from; and an exponent extreme enough that ``r ** q`` overflows has
    a distribution in principle but not in float64, where the inversion would return the inner
    edge for every draw and look like a working sampler that had collapsed the disc to a ring.
    """

    q = 2.0 - exponent
    if inner_radius <= 0.0 and q <= 0.0:
        raise ValueError(
            f"a disk annulus with surface-density exponent {exponent} needs a positive inner "
            f"radius: sigma ~ r**-{exponent} puts an unbounded number of particles at the "
            "centre, so there is no distribution to sample. Give the annulus an inner hole, "
            "or use an exponent below 2"
        )
    uniform = rng.uniform(0.0, 1.0, size=count)
    if q == 0.0:
        # sigma ~ r**-2: equal particle counts per decade of radius, the one exponent whose
        # cumulative count is logarithmic rather than a power.
        return inner_radius * (outer_radius / inner_radius) ** uniform
    try:
        inner_q = inner_radius**q if inner_radius > 0.0 else 0.0
        outer_q = outer_radius**q
    except OverflowError:
        # Python's float power raises here rather than returning an infinity, so the guard
        # below would never see the value it is checking for. Both routes end in the same
        # refusal, which is the point: the sampler is out of range either way.
        inner_q = outer_q = math.inf
    if not math.isfinite(inner_q) or not math.isfinite(outer_q):
        raise ValueError(
            f"a disk annulus with surface-density exponent {exponent} cannot be sampled "
            f"between radii {inner_radius} and {outer_radius}: the inverse-transform term "
            "r**(2 - exponent) overflows float64, so every draw would return the inner edge"
        )
    return (inner_q + uniform * (outer_q - inner_q)) ** (1.0 / q)


def sample_disk_annulus(
    binding: BindingDescriptor, count: int, np: Any
) -> tuple[Any, float]:
    """``count`` positions in the binding's oriented annulus.

    Uniform *in volume* by default: azimuth uniform, ``r^2`` uniform between the squared
    radii, local Z uniform across the thickness. Sampling ``r`` uniformly instead would crowd
    the inner edge -- a visibly wrong surface density that no radius bound would catch.

    A binding may instead declare ``annulus_surface_density_exponent``, and then the radii
    come from `_profiled_annulus_radii` so the surface density falls as ``r ** -exponent``.
    Zero is the default and keeps the uniform-in-volume draw itself, not merely a
    distribution equal to it, so every scene authored before ADR 0078 samples exactly what
    it sampled before.

    Returns the world-space offsets from the binding position and the volume's
    characteristic radius ``0.5 * cbrt(V)``, matching what the ball and box lanes return so
    a disk holding the same count at the same volume gives its particles the same size.
    """

    extent = (
        float(binding.sample_extent[0]),
        float(binding.sample_extent[1]),
        float(binding.sample_extent[2]),
    )
    if not all(math.isfinite(value) for value in extent):
        raise ValueError("sample_extent must be finite")
    span = max(extent[0], extent[1])
    if abs(extent[0] - extent[1]) > annulus_length_tol(span):
        raise ValueError(
            f"a Disk Annulus needs circular X and Y extents, got {extent[0]} and "
            f"{extent[1]}; scale the object uniformly in its own plane rather than "
            "letting the sampler pick one of them"
        )

    outer_radius = 0.5 * extent[0]
    half_thickness = 0.5 * extent[2]
    inner_radius = float(binding.annulus_inner_radius)
    volume = annulus_volume(
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        half_thickness=half_thickness,
    )
    matrix = validate_annulus_basis(binding.sample_basis)

    rng = _rng(binding, np)
    azimuth = rng.uniform(0.0, 2.0 * math.pi, size=count)
    exponent = float(getattr(binding, "annulus_surface_density_exponent", 0.0))
    if exponent == 0.0:
        # The uniform-in-area draw, unchanged and deliberately not routed through the general
        # branch below. Zero is every existing descriptor's value, and reproducing the old
        # distribution *analytically* is not the same promise as reproducing the old draw; the
        # evidence on disk was measured from these exact two lines.
        radii = np.sqrt(rng.uniform(inner_radius**2, outer_radius**2, size=count))
    else:
        radii = _profiled_annulus_radii(
            rng,
            count=count,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            exponent=exponent,
        )
    heights = rng.uniform(-half_thickness, half_thickness, size=count)
    local = np.stack([radii * np.cos(azimuth), radii * np.sin(azimuth), heights], axis=1)
    # Rows of `matrix` are the basis vectors, so local coordinates map to world offsets by
    # left-multiplying: offset = sum_i local_i * b_i. The offsets are then placed on the
    # binding's own origin -- ADR 0040 section 5's centre -- exactly as the ball and box
    # lanes do. Returning the raw offsets instead puts every disk at the world origin
    # whatever the author did, and no radius or thickness bound can see that.
    centre = np.asarray(binding.position, dtype=np.float64)
    return centre + local @ matrix, 0.5 * volume ** (1.0 / 3.0)


def _object_extent(binding: BindingDescriptor) -> tuple[float, float, float] | None:
    """The object extent this binding samples from, or ``None`` to use its own radius.

    ``None`` covers both the default ``BALL`` shape and an object shape whose object turned
    out to have no size --- an Empty, or a mesh scaled to nothing. Sampling a zero extent
    would put every particle of the cloud at one point, which reads as the solver having
    lost the body rather than as the authoring mistake it is.
    """

    if binding.sample_shape == SampleShape.BALL:
        return None
    extent = (
        float(binding.sample_extent[0]),
        float(binding.sample_extent[1]),
        float(binding.sample_extent[2]),
    )
    if max(extent) <= 0.0:
        return None
    return extent


def _sample_mesh_cloud(binding: BindingDescriptor, count: int) -> tuple[Any, float]:
    """A cloud drawn from the bound object's actual geometry.

    The mesh arrives already in world space --- the host evaluates it there, because only
    the host knows the object's modifiers and transform --- so unlike every proxy shape,
    this one does *not* offset by ``binding.position``. Adding it would place the cloud at
    twice the object's own offset.
    """

    from . import mesh_store
    from .mesh_sampling import sample_surface, sample_volume

    mesh = mesh_store.resolve(binding.sample_mesh_key)
    seed = [int(binding.sampling_seed), runtime_stable_id(binding.binding_id)]
    # Neither branch guards its measure against zero: each sampler refuses a mesh it cannot
    # draw from -- no triangles, no area, nothing enclosed -- before the measure is read, so
    # a guard here would be a second owner of a rule that already has one.
    if binding.sample_shape == SampleShape.OBJECT_SURFACE:
        points = sample_surface(mesh, count, seed)
        # A surface cloud's spacing falls off as N^-1/2, not as the N^-1/3 the caller
        # divides this length by. Compensating here is what makes `radius / count**(1/3)`
        # come out as half the true mean spacing over the area, which is the same meaning
        # the volume shapes give their own particles. It is why this one length depends on
        # the count while every other characteristic radius does not.
        radius = 0.5 * (mesh.area() / count) ** 0.5 * count ** (1.0 / 3.0)
        return points, float(radius)
    points = sample_volume(mesh, count, seed)
    # Same convention as the box: half the side of the cube of equal volume. A hollow or
    # concave mesh therefore reports less than the box around it, which is the point --- the
    # proxy hands every particle a radius too large by the cube root of the space it invents.
    return points, float(0.5 * mesh.volume() ** (1.0 / 3.0))


def _sample_cloud(binding: BindingDescriptor, count: int, np: Any) -> tuple[Any, float]:
    """A binding's cloud, and the length its per-particle radius is derived from.

    The second return is the volume's characteristic radius: for a ball that is the ball's
    radius, and for a box it is the radius of the cube of equal volume, so a box and a ball
    holding the same count at the same volume give their particles the same size.
    """

    if binding.sample_shape.needs_mesh:
        return _sample_mesh_cloud(binding, count)
    if binding.sample_shape == SampleShape.DISK_ANNULUS:
        # Routed before _object_extent because a disk must never silently degrade to a
        # ball: a zero or non-circular extent is an authoring error the sampler refuses,
        # not a case to fall back from.
        return sample_disk_annulus(binding, count, np)
    if binding.sample_shape == SampleShape.SHELL:
        # Likewise routed before _object_extent: a shell is sized by the cloud's own radius,
        # and the extent-reading branches below would silently turn it into a box.
        return _sample_shell(binding, count, np)
    extent = _object_extent(binding)
    if extent is None:
        radius = float(binding.radius)
        return _sample_ball(binding, radius, count, np), radius
    if binding.sample_shape == SampleShape.OBJECT_BALL:
        radius = 0.5 * max(extent)
        return _sample_ball(binding, radius, count, np), radius
    volume = extent[0] * extent[1] * extent[2]
    # A flat or linear object has no volume, so the equal-volume cube is degenerate and its
    # radius would be zero. Fall back to the longest axis, which is the only length left.
    radius = 0.5 * (volume ** (1.0 / 3.0) if volume > 0.0 else max(extent))
    return _sample_box(binding, extent, count, np), radius


def _keplerian_central_body(
    bindings: tuple[BindingDescriptor, ...],
    binding: BindingDescriptor,
    binding_id: str,
    role: str,
) -> BindingDescriptor:
    """Resolve one half of a Keplerian centre, refusing where the author can act (ADR 0085).

    Shared by the central and the companion so the two cannot drift into different rules
    about what is orbitable: an id that is not in the System, or one that is inactive or
    sampled, has no single mass and no single position to stand at.
    """

    central = next(
        (other for other in bindings if other.binding_id == binding_id),
        None,
    )
    if central is None:
        raise ValueError(
            f"{binding.object_name!r} orbits {role} binding {binding_id!r}, which is not "
            "in this System; bind that body to the same System or pick another one"
        )
    if not central.active or central.representation != BodyRepresentation.ANALYTICAL:
        raise ValueError(
            f"{binding.object_name!r}'s {role} binding {central.object_name!r} must be an "
            "active analytical Body; a sampled or inactive binding has no single mass to "
            "orbit"
        )
    return central


def _keplerian_centre(
    bindings: tuple[BindingDescriptor, ...],
    binding: BindingDescriptor,
    central: BindingDescriptor,
) -> tuple[Any, Any, float, str]:
    """The point a Keplerian cloud orbits: one body, or a pair's barycentre (ADR 0085).

    With no companion this returns the central body's own position, velocity and mass, so
    every scene authored before ADR 0085 gets exactly the initial condition it got then --
    the barycentre of a single mass is that mass.

    With a companion the cloud orbits the pair as one point mass, which is only defensible
    outside the pair. The combined mass is the sum, the position is the mass-weighted mean,
    and the velocity is the mass-weighted mean too -- the momentum of the pair divided by its
    mass, which is the frame the cloud must be given so a drifting binary carries its disc
    with it instead of leaving it behind. Whether the cloud is *far enough* outside is the
    template's judgement and not this function's: the initial condition is well defined at any
    radius, it is only unstable inside the critical one.
    """

    np = _numpy()
    if not binding.keplerian_companion_binding_id:
        return (
            np.asarray(central.position, dtype=np.float64),
            np.asarray(central.velocity, dtype=np.float64),
            float(central.mass),
            repr(central.object_name),
        )
    companion = _keplerian_central_body(
        bindings, binding, binding.keplerian_companion_binding_id, "companion"
    )
    total = float(central.mass) + float(companion.mass)
    if total <= 0.0:
        raise ValueError(
            f"{binding.object_name!r} orbits the pair "
            f"{central.object_name!r} + {companion.object_name!r}, whose combined mass is "
            f"{total:.6g}; there is no orbit about a massless centre"
        )
    weights = np.asarray(
        [float(central.mass), float(companion.mass)], dtype=np.float64
    ).reshape(2, 1)
    positions = np.asarray(
        [central.position, companion.position], dtype=np.float64
    ).reshape(2, 3)
    velocities = np.asarray(
        [central.velocity, companion.velocity], dtype=np.float64
    ).reshape(2, 3)
    return (
        (weights * positions).sum(axis=0) / total,
        (weights * velocities).sum(axis=0) / total,
        total,
        f"the barycentre of {central.object_name!r} and {companion.object_name!r}",
    )


def _cloud_velocities(
    binding: BindingDescriptor,
    positions: Any,
    masses: Any,
    count: int,
    bindings: tuple[BindingDescriptor, ...],
    gravitational_constant: float,
    softening: float,
    np: Any,
    field: tuple[Any, Any] | None = None,
) -> Any:
    """Per-particle initial velocities for one cloud (ADR 0040 sections 4, 6, 7, 8).

    ``UNIFORM`` and ``FROM_OBJECT`` share one arithmetic path -- tile the authored
    ``velocity`` -- because a ``FROM_OBJECT`` descriptor already carries the measured
    translational velocity folded into that tuple. The model identifier survives so
    save/reopen can still tell authored uniform motion from sampled animation; it is
    provenance, not a second code path.
    """

    bulk = np.asarray(binding.velocity, dtype=np.float64)
    model = binding.velocity_model

    if model == CloudVelocityModel.VIRIALIZE:
        # W_ref = -3GM^2/5R is a *uniform ball's* potential. Any other shape would
        # silently inherit a coefficient that does not describe it.
        if binding.sample_shape != SampleShape.BALL:
            raise ValueError(
                f"VIRIALIZE is defined for a Ball cloud only, but "
                f"{binding.object_name!r} samples {binding.sample_shape.value!r}. The "
                "uniform-sphere reference potential it normalizes against does not "
                "describe that volume; choose Ball, or choose another velocity model"
            )
        return virialized_velocities(
            masses=masses,
            total_mass=float(binding.mass),
            radius=float(binding.radius),
            gravitational_constant=gravitational_constant,
            support_fraction=float(binding.support_fraction),
            bulk_velocity=bulk,
            rng=domain_rng(
                binding.sampling_seed, binding.binding_id, VIRIAL_VELOCITY_DOMAIN
            ),
        )

    if model == CloudVelocityModel.EXPANSION:
        # ADR 0202. A velocity *field* about the cloud's centre, defined over every shape: a
        # shell, a ball and a disk all have an outward direction. The centre is the binding's
        # position, which every analytic shape is drawn about; no random stream is consumed.
        return expansion_velocities(
            positions=positions,
            centre=binding.position,
            speed=float(binding.expansion_speed),
            proportional=bool(binding.expansion_proportional),
            reference_radius=float(binding.radius),
            total_mass=float(binding.mass),
            gravitational_constant=gravitational_constant,
            bulk_velocity=bulk,
        )

    if model == CloudVelocityModel.ROTATION_CURVE:
        if binding.sample_shape != SampleShape.DISK_ANNULUS:
            raise ValueError(
                f"ROTATION_CURVE is defined for a Disk Annulus cloud only, but "
                f"{binding.object_name!r} samples {binding.sample_shape.value!r}; the "
                "curve is measured in a plane, and only the annulus carries one"
            )
        # A refusal rather than a fallback, and the reason is arithmetic. The curve is the
        # azimuthal mean of a sum of `m / (d^2 + eps^2)^{3/2}` terms evaluated inside the
        # cloud. At eps = 0 in a disk that sum is dominated by whichever single particle
        # happens to lie nearest the probe, so the measured "rotation curve" is a property
        # of the sampling seed rather than of the disk: measured over three seeds at 20,000
        # particles the inner curve moved by more than its own value, and it got *worse*
        # with count, because more particles means a closer nearest neighbour. Softening is
        # what removes that divergence, in this measurement for exactly the reason it does
        # in `nbody`'s kernel, which uses the same `dot(r, r) + softening * softening`.
        # Accepting eps = 0 here would hand back a disk that is wrong in a way nothing
        # downstream can detect.
        if float(softening) <= 0.0:
            raise ValueError(
                f"{binding.object_name!r} uses ROTATION_CURVE, which measures the disk's "
                f"own gravity, but this System's softening is {float(softening)!r}. "
                "Without softening that measurement is dominated by each sample point's "
                "nearest particle and describes the sampling seed rather than the disk. "
                "Give the System a softening of at least the particle spacing"
            )
        basis = validate_annulus_basis(binding.sample_basis)
        # ADR 0087: the disk is given speeds for the field it will actually be integrated
        # in, which is every particle in the System and not only its own block. `field` is
        # None only for a direct call that has no System to offer, and then this is the
        # pre-0087 self-measurement byte for byte.
        field_positions, field_masses = field if field is not None else (None, None)
        return rotation_curve_velocities(
            positions=positions,
            masses=masses,
            field_positions=field_positions,
            field_masses=field_masses,
            centre=binding.position,
            normal=basis[2],
            counter_clockwise=bool(binding.keplerian_counter_clockwise),
            gravitational_constant=gravitational_constant,
            softening=float(softening),
            eccentricity_jitter=float(binding.eccentricity_jitter),
            bulk_velocity=bulk,
            rng=domain_rng(
                binding.sampling_seed,
                binding.binding_id,
                ROTATION_CURVE_ECCENTRICITY_DOMAIN,
            ),
        )

    if model == CloudVelocityModel.KEPLERIAN:
        if binding.sample_shape != SampleShape.DISK_ANNULUS:
            raise ValueError(
                f"KEPLERIAN is defined for a Disk Annulus cloud only, but "
                f"{binding.object_name!r} samples {binding.sample_shape.value!r}; a "
                "circular orbit needs the disk plane the annulus carries"
            )
        central = _keplerian_central_body(
            bindings, binding, binding.keplerian_central_binding_id, "central"
        )
        # ADR 0085. With no companion this is the pre-0085 single body, byte for byte: the
        # barycentre of one mass is its own position and its own velocity.
        centre_position, centre_velocity, centre_mass, centre_name = _keplerian_centre(
            bindings, binding, central
        )
        # ADR 0040 sections 5 and 6: the disk's origin must coincide with what it orbits,
        # within the scale-aware length tolerance. Without this a displaced disk is accepted
        # and every particle is given the speed for a radius its geometry does not have -- a
        # plausible-looking cloud orbiting a point it is nowhere near. For a central pair the
        # point to be centred on is the barycentre, and it is on *neither* body once the
        # masses differ, which is the whole reason ADR 0085 exists.
        outer_radius = 0.5 * float(binding.sample_extent[0])
        separation = float(
            np.linalg.norm(
                np.asarray(binding.position, dtype=np.float64) - centre_position
            )
        )
        if separation > annulus_length_tol(outer_radius):
            raise ValueError(
                f"{binding.object_name!r} must be centred on what it orbits, but its "
                f"origin is {separation:.6g} from {centre_name} (tolerance "
                f"{annulus_length_tol(outer_radius):.6g} for an outer radius of "
                f"{outer_radius:.6g}). Move the disk onto that centre, or orbit a "
                "body that is at the disk's centre"
            )
        # The disk's own stored basis decides the plane, so a tilted disk orbits tilted.
        basis = validate_annulus_basis(binding.sample_basis)
        return keplerian_velocities(
            positions=positions,
            central_position=tuple(float(value) for value in centre_position),
            central_mass=centre_mass,
            central_velocity=tuple(float(value) for value in centre_velocity),
            gravitational_constant=gravitational_constant,
            normal=basis[2],
            counter_clockwise=bool(binding.keplerian_counter_clockwise),
            eccentricity_jitter=float(binding.eccentricity_jitter),
            bulk_velocity=bulk,
            rng=domain_rng(
                binding.sampling_seed,
                binding.binding_id,
                KEPLERIAN_ECCENTRICITY_DOMAIN,
            ),
            # ADR 0188: the orbit the solver integrates is about a softened point mass.
            softening=float(softening),
        )

    return np.tile(bulk, (count, 1))


def expand_bindings(
    bindings: tuple[BindingDescriptor, ...],
    *,
    gravitational_constant: float,
    softening: float = 0.0,
    positions_only: bool = False,
    shear_flow_omega: float = 0.0,
) -> BodySet:
    """Every binding's bodies, concatenated in binding order.

    ``shear_flow_omega`` (ADR 0202) is a shearing box's angular speed, passed by the runtime that
    plays one and by nobody else. In a box every velocity is measured *relative to the circular
    shear flow* ``(0, -q Omega x, 0)`` (``q = 3/2``), so a body or a cloud authored with no
    velocity is on a circular orbit instead of falling out of equilibrium, and any velocity model
    composes with it. The flow is added to every body after its own velocity is drawn, at the
    body's authored ``x`` (the box is centred on the world origin). Zero, the default, adds
    nothing and leaves every expansion byte-identical.

    ``positions_only`` (ADR 0162) skips the velocity pass and returns zero velocities. The
    draws are the same draws from the same streams, so the positions are bit-identical to a
    full expansion's; what is skipped is the second pass, which can refuse a velocity model
    or measure the whole System's field and is irrelevant to where a particle is drawn. It is
    what the start-frame preview uses to show a scene before any physics exists.

    ``gravitational_constant`` is **required and keyword-only** (ADR 0040). ``VIRIALIZE``
    and ``KEPLERIAN`` consume it once here, and there is no defensible default: SI ``G``
    and a code-unit ``G=1`` differ by twenty orders of magnitude, so a silent fallback
    would not raise -- it would return a cloud moving at entirely the wrong speed, which
    reads as a physics result rather than as a bug. Keyword-only so it cannot be slipped
    in by a positional call.

    ``softening`` is the same scene scalar the solver kernel integrates, and it is here
    for the two models that orbit a field: ``ROTATION_CURVE``, which measures the disk's
    own gravity and so must measure the *softened* force rather than a bare one the solver
    never applies -- ADR 0081 is the record of what a mismatch there costs -- and
    ``KEPLERIAN``, whose circular speed is that of the softened point mass the solver
    integrates (ADR 0188). Unlike ``gravitational_constant`` it does carry a default, and
    the difference is not laxity: ``0.0`` is what ``UnitScale.softening`` itself defaults
    to, so this signature's default is the shipped scene default rather than an invented
    fallback, and at zero every model expands exactly as it did before softening existed.
    The one model that cannot tolerate zero refuses there instead of returning a plausible
    wrong disk; that refusal is in ``_cloud_velocities`` with its arithmetic.

    Expansion runs in **two passes** (ADR 0087): every binding's particles are drawn
    first, and only then is any velocity model evaluated. No model influences where a
    particle is drawn, so the draws and their random streams are untouched by the split.
    What it buys is that a model may be handed the field the *whole System* makes rather
    than only its own block's -- which is what lets a light disk be given the speeds it
    needs to orbit inside a heavy halo it does not itself contain.

    This function stays graph-agnostic: it knows shapes, not runtimes.
    """

    np = _numpy()
    positions = []
    velocities = []
    velocity_jobs: list[tuple[BindingDescriptor, int, Any, Any, bool]] = []

    masses = []
    radii = []
    stable_ids = []
    active = []
    body_flags: list[Any] = []
    pinned_flags: list[Any] = []
    for binding in bindings:
        count = binding_body_count(binding)
        if count == 0:
            # A FIELD-role binding expands to nothing: the runtime reads it from the
            # descriptor's bindings, never from the body arrays.
            continue
        # Set by the sampled-cloud branches, which are the only ones whose velocities go
        # through a model at all. Every other binding tiles its authored velocity.
        sampled = False
        # The Body of ADR 0037 §3.3 is the analytic point mass: role BODY carried by an
        # ANALYTICAL representation. A sampled or imported population is a particle
        # cloud (flag 0) even when its role is BODY, and non-body roles never flag.
        is_body = (
            binding.role == BindingRole.BODY
            and binding.representation == BodyRepresentation.ANALYTICAL
        )
        body_flags.append(np.full(count, 1 if is_body else 0, dtype=np.int32))
        # Pinned is meaningful only for the analytic point mass: a cloud has no single
        # position to hold. `_validate_plan` refuses it on any other backend, so a cloud
        # that reaches here with the flag set is a bug elsewhere, not an authored scene.
        pinned_flags.append(
            np.full(count, 1 if (is_body and binding.pinned) else 0, dtype=np.int32)
        )
        if binding.representation == BodyRepresentation.SAMPLED_PARTICLE and count == 1:
            # One particle is not enough volume to sample, so position, mass and radius
            # stay exactly what the single-body lane always produced. The velocity model
            # still runs: ADR 0040 section 4 names "fewer than two particles" as a
            # degenerate input that must fail visibly, and routing this case around the
            # dispatch made that refusal unreachable rather than satisfied.
            positions.append(np.asarray([binding.position], dtype=np.float64))
            single_mass = np.full(1, float(binding.mass), dtype=np.float64)
            masses.append(single_mass)
            radii.append(np.full(1, float(binding.radius), dtype=np.float64))
            sampled = True
        elif binding.representation == BodyRepresentation.SAMPLED_PARTICLE and count > 1:
            cloud, cloud_radius = _sample_cloud(binding, count, np)
            positions.append(cloud)
            # The cloud carries the authored total mass. At zero jitter this is the exact
            # equal split it always was, drawn from its own stream, so an existing scene
            # expands byte-identically.
            cloud_masses = jittered_masses(
                total_mass=float(binding.mass),
                count=count,
                jitter=float(binding.mass_jitter),
                rng=domain_rng(
                    binding.sampling_seed, binding.binding_id, MASS_JITTER_DOMAIN
                ),
            )
            masses.append(cloud_masses)
            # Per-particle radius is the mean interparticle spacing, which is the only
            # length the cloud's own size and count together determine.
            radii.append(np.full(count, cloud_radius / count ** (1.0 / 3.0)))
            sampled = True
        else:
            positions.append(np.asarray([binding.position], dtype=np.float64))
            masses.append(np.full(count, float(binding.mass), dtype=np.float64))
            radii.append(np.full(count, float(binding.radius), dtype=np.float64))
        # ADR 0087: velocities are computed in a second pass, once every binding's
        # particles exist, because a velocity model may need to measure the field the
        # whole System makes rather than only the block it belongs to. Sampling is
        # unaffected -- no model influences where a particle is drawn -- so this defers
        # the answer without changing any of the draws or their random streams.
        velocity_jobs.append((binding, count, positions[-1], masses[-1], sampled))
        stable_ids.append(particle_stable_ids(binding.binding_id, count))
        # An Emitter's particles are all drawn here, exactly like any other cloud, and all
        # start dead: emission is the activation of a slot the solver already owns, and the
        # release schedule (`astro_warp.product.emitters`) decides which slots are live at
        # which frame. Expansion has no frame to evaluate that schedule at, and inventing
        # one here would put the timeline into a function whose whole contract is that it
        # produces the state *before* step zero. The runtime applies frame zero's release
        # when it loads these bodies.
        emitting = binding.role == BindingRole.EMITTER
        active.append(
            np.full(count, 1 if binding.active and not emitting else 0, dtype=np.int32)
        )
    # Second pass. The field is every particle the System holds, which is the same set the
    # solver's kernel will sum over on step one -- so a model that measures it is measuring
    # the force its particles are about to feel, rather than a fraction of it.
    if positions_only:
        velocities = [np.zeros((count, 3), dtype=np.float64) for _, count, _, _, _ in velocity_jobs]
    elif velocity_jobs:
        field_positions = np.concatenate(
            [block for _, _, block, _, _ in velocity_jobs]
        ).astype(np.float64)
        field_masses = np.concatenate([block for _, _, _, block, _ in velocity_jobs])
        # A massless particle pulls on nothing, so it is not part of the field. Keeping it
        # would be harmless arithmetic and a refusal in the measurement, which validates
        # that its masses are strictly positive.
        pulls = field_masses > 0.0
        field = (field_positions[pulls], field_masses[pulls])
        for binding, count, block_positions, block_masses, sampled in velocity_jobs:
            if sampled:
                velocities.append(
                    _cloud_velocities(
                        binding,
                        block_positions,
                        block_masses,
                        count,
                        bindings,
                        gravitational_constant,
                        softening,
                        np,
                        field,
                    )
                )
            else:
                velocities.append(
                    np.tile(np.asarray(binding.velocity, dtype=np.float64), (count, 1))
                )
            if count > 1 and any(binding.spin):
                velocities[-1] = velocities[-1] + rigid_rotation_velocities(
                    block_positions, block_masses, binding.spin, np
                )

    all_positions = np.concatenate(positions).astype(np.float64)
    all_velocities = np.concatenate(velocities)
    if shear_flow_omega != 0.0 and not positions_only:
        # A shearing box measures velocity from the circular flow v_y = -q Omega x (ADR 0202).
        all_velocities = all_velocities.copy()
        all_velocities[:, 1] -= SHEAR_Q * float(shear_flow_omega) * all_positions[:, 0]
    return BodySet(
        positions=all_positions,
        velocities=all_velocities,
        masses=np.concatenate(masses),
        radii=np.concatenate(radii),
        stable_ids=np.concatenate(stable_ids),
        active=np.concatenate(active),
        body_flags=np.concatenate(body_flags),
        pinned=(
            np.concatenate(pinned_flags) if any(binding.pinned for binding in bindings) else None
        ),
    )


def rigid_rotation_velocities(positions: Any, masses: Any, spin: Any, np: Any) -> Any:
    """The velocity field of a body turning rigidly: ``spin x (position - centre of mass)``.

    ``spin`` is the angular-velocity vector in radians per code time unit (the direction is
    the axis, the length the rate). The pivot is the body's mass-weighted centre, so a
    cloud turns about itself and not about the world origin, and the spin adds no net
    momentum (the mass-weighted offsets about the centre of mass sum to zero). Pure and
    array-only, so the add-on and the tests can both ask it the one question.
    """

    omega = np.asarray(spin, dtype=np.float64)
    weights = np.asarray(masses, dtype=np.float64)
    total = float(weights.sum())
    if not total > 0.0:
        return np.zeros_like(np.asarray(positions, dtype=np.float64))
    block = np.asarray(positions, dtype=np.float64)
    centre = (block * weights[:, None]).sum(axis=0) / total
    return np.cross(omega, block - centre)


def runtime_stable_ids(bindings: tuple[BindingDescriptor, ...]) -> list[int]:
    """Solver stable IDs in body order, for manifest and output identity checks."""

    # One concatenate and one ``tolist`` rather than an ``int()`` per body: at 50,000 bodies
    # the per-element loop was ~9 ms of every committed frame inside Blender (ADR 0094).
    parts = [
        particle_stable_ids(binding.binding_id, binding_body_count(binding))
        for binding in bindings
    ]
    if not parts:
        return []
    return _numpy().concatenate(parts).tolist()
