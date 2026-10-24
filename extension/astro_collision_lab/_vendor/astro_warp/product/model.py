"""Serializable product descriptors shared by Blender UI and runtimes."""

from __future__ import annotations

import json
import math
import struct
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from typing import Any

from .timeline_stages import TimelineStage, TimelineStageKind, validate_stage_list

SCHEMA_VERSION = "astro-warp-product-v1"
CHECKPOINT_SCHEMA_VERSION = "astro-warp-checkpoint-v1"

# GRAV-002's accepted accuracy bound, from adr/0006-accuracy-gate-thresholds.md: the tree's
# relative acceleration error stays below TREE_ACCURACY_BOUND_CONSTANT * theta**2, measured
# across a 64x span of body count and proven by
# status/proofs/astro_warp_accuracy_gate_suite.json. Auto uses it as its default acceptance
# tolerance because it is the only tree error figure with evidence behind it: inside the
# bound the tree is performing as certified, and outside it something has regressed, so
# failing closed to direct is the right answer either way.
TREE_ACCURACY_POLICY = "adr-0006-tree-opening-angle-bound"
TREE_ACCURACY_BOUND_CONSTANT = 0.1
# The bound was measured over opening angles 0.8 down to 0.2. Error falls faster than
# theta**2, so applying it below 0.2 is conservative, but above 0.8 it would be
# extrapolation and no default is offered there.
TREE_ACCURACY_CERTIFIED_MAX_THETA = 0.8
# Blender holds Theta as a float32, so the certified 0.8 arrives as 0.800000011920929; judged
# against the float64 0.8 it lost its bound at exactly the certified angle and Auto played
# direct (ADR 0136). The edge is therefore 0.8 as a float32 holds it, rounded to nearest,
# which is the lenient side; the next float32 up is still refused.
_CERTIFIED_MAX_THETA_AS_STORED = struct.unpack(
    "f", struct.pack("f", TREE_ACCURACY_CERTIFIED_MAX_THETA)
)[0]


#: The Barnes-Hut opening angle the plan layer authors when nothing says otherwise (ADR 0136,
#: ADR 0208). Every graph builder, the scenario builder, the add-on's System default and the
#: factory's ``_tree_config`` fallback read this one value. ``tree.BarnesHutConfig`` keeps its own
#: engine-class default; that is a property of the engine, not of a plan.
DEFAULT_TREE_THETA = 0.6
#: Auto sends a scene of at most this many bodies straight to Direct (ADR 0155). The factory
#: decides with it and the memory pre-flight sizes with it, so it lives where both can import it.
AUTO_SMALL_SCENE_DIRECT_LIMIT = 4096


def tree_accuracy_tolerance(theta: float) -> float | None:
    """The certified GRAV-002 error bound at ``theta``, or None outside the swept range."""

    value = float(theta)
    if not 0.0 < value <= _CERTIFIED_MAX_THETA_AS_STORED:
        return None
    return TREE_ACCURACY_BOUND_CONSTANT * value**2


class ContextState(StrEnum):
    UNLOADED = "UNLOADED"
    READY = "READY"
    PREVIEWING = "PREVIEWING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    DIRTY = "DIRTY"
    BAKING = "BAKING"
    FAULTED = "FAULTED"
    MISSING_CACHE = "MISSING_CACHE"
    CLOSED = "CLOSED"


class BindingRole(StrEnum):
    BODY = "Body"
    EMITTER = "Emitter"
    COLLIDER = "Collider"
    FIELD = "Field"
    REGION_MASK = "Region-Mask"


class BodyRepresentation(StrEnum):
    ANALYTICAL = "Analytical"
    RIGID_EXTENDED = "Rigid Extended"
    SAMPLED_PARTICLE = "Sampled Particle"
    IMPORTED_PARTICLE_SET = "Imported Particle Set"


class SampleShape(StrEnum):
    """The volume a ``SAMPLED_PARTICLE`` binding draws its cloud from.

    ``BALL`` is the original and the default: a ball of the binding's own ``radius`` about
    its position, so a scene authored before this enum existed samples exactly what it
    always did. The two object shapes size the volume from the bound object's evaluated
    dimensions instead, which is what "use this object as a volume of particles" has to
    mean --- a 40-unit mesh sampled at the default 1-unit radius is a bug the user cannot
    see.

    ``OBJECT_BOX`` and ``OBJECT_BALL`` are convex proxies sized from that bounding box,
    and they are kept because every scene authored against them must keep sampling what
    it sampled. ``OBJECT_VOLUME`` and ``OBJECT_SURFACE`` are the shapes that read the
    geometry itself: the first fills the region the surface encloses, the second
    scatters over the surface by area. Those two are the only ones for which "use this
    object as a cloud" means the object rather than the box around it, which is what
    makes a torus, an L-block or a spiral arm authorable at all.

    The mesh itself does not travel in the descriptor --- see
    ``BindingDescriptor.sample_mesh_key`` and ``astro_warp.product.mesh_store``.
    """

    BALL = "Ball"
    OBJECT_BOX = "Object Box"
    OBJECT_BALL = "Object Ball"
    OBJECT_VOLUME = "Object Volume"
    OBJECT_SURFACE = "Object Surface"
    # An oriented annulus in the object's own transformed local XY plane (ADR 0040
    # section 5). Unlike the two object shapes above, it needs an orientation as well as an
    # extent, because a disk the user tilted must sample tilted -- so the compiled binding
    # also carries `sample_basis`. Its inner radius cannot be derived from the object at
    # all, which is why `annulus_inner_radius` is an authored field rather than a fraction
    # of the outer radius.
    DISK_ANNULUS = "Disk Annulus"
    # A thin spherical shell about the binding's position (ADR 0202): uniform in volume
    # between an inner radius and the cloud's own ``radius``. The inner radius is the
    # existing ``annulus_inner_radius`` -- "how much of the middle to leave empty" means the
    # same for a ring and a shell, so the shell reuses the field and its drawn control
    # rather than adding a second one. Zero leaves no hole, and then a shell samples exactly
    # what a ``BALL`` of the same radius samples. Appended last: the enum persists by value,
    # but the add-on's static EnumProperty stores an item's index.
    SHELL = "Shell"

    @property
    def needs_mesh(self) -> bool:
        """Whether this shape reads real geometry rather than a proxy of its size.

        ``DISK_ANNULUS`` is deliberately absent: it is oriented by ``sample_basis`` and
        sized from the object's extent, so it needs the object's transform but never its
        triangles.
        """

        return self in (SampleShape.OBJECT_VOLUME, SampleShape.OBJECT_SURFACE)


class CloudVelocityModel(StrEnum):
    """How a ``SAMPLED_PARTICLE`` cloud's per-particle initial velocities are drawn.

    ``UNIFORM`` is the original and the default: the authored bulk ``velocity`` tiled
    across every particle, so a cloud authored before this enum existed keeps exactly the
    initial condition it had.

    ``VIRIALIZE`` is a **global virial-ratio initializer**, not an equilibrium
    distribution function. It constrains total kinetic energy for a uniformly sampled
    ``BALL``; it does not solve the collisionless Boltzmann equation, produce a local
    Jeans dispersion, or guarantee a stationary profile. ADR 0040 names it this way
    deliberately -- calling it equilibrium would falsely close part of ``GALAXY-001``.

    ``KEPLERIAN`` gives each particle the circular speed about a named central binding.
    ``FROM_OBJECT`` measures the source object's animated translational velocity; the
    descriptor keeps recording ``FROM_OBJECT`` afterwards rather than being rewritten to
    ``UNIFORM``, because that rewrite would erase how the initial condition was obtained
    and leave save/reopen unable to tell authored motion from sampled motion.

    ``ROTATION_CURVE`` is the only member that assumes no potential at all. The other four
    each balance against something stated in closed form -- a uniform sphere, a central
    point mass, or nothing -- and none of those describes a disk held up by its own
    gravity. This one measures the disk's softened radial self-gravity from the sampled
    particles and reads each particle's speed off the resulting curve, which is what makes
    a spiral galaxy a thing a Cloud can be rather than only a thing a Cloud can be named.
    It is a rotation curve and not an equilibrium solution: no dispersion, no vertical
    structure, no Toomre stabilisation, so the disk is free to form its own structure.

    ``EXPANSION`` (ADR 0202) sends every particle straight out from the cloud's centre --
    the binding's position -- at ``expansion_speed``, either the same speed everywhere or,
    with ``expansion_proportional``, growing with distance (a Hubble-like flow, ``H r``). It
    is a velocity *field* like ``KEPLERIAN``, and like it assumes nothing about the cloud's
    own gravity: it is the outflow of a shell, a blast or a wind, not an equilibrium. Defined
    over every shape. A speed of zero means the cloud's own escape speed, so a cloud that
    is switched to Expansion by hand moves at once instead of staying at rest.
    """

    VIRIALIZE = "Virialize"
    KEPLERIAN = "Keplerian"
    ROTATION_CURVE = "Rotation Curve"
    UNIFORM = "Uniform"
    FROM_OBJECT = "From Object"
    EXPANSION = "Expansion"


class ForceKind(StrEnum):
    """What acceleration a FIELD-role binding applies to every solver body (ADR 0045).

    ``UNIFORM`` pushes every body along the bound object's local +Z axis (the row-2
    ``sample_basis`` direction, matching Blender's Wind convention) at ``strength``
    code-units of acceleration. ``POINT_ATTRACTOR`` pulls toward the bound object's
    position with magnitude ``strength / r^2`` -- ``strength`` plays the role of GM, so
    a negative value repels. Sign is meaning for both kinds, not an error.
    """

    UNIFORM = "Uniform"
    POINT_ATTRACTOR = "Point Attractor"


class Precision(StrEnum):
    """Precision of the authoritative solver state.

    Interactive preview runs float32 because the display path only needs positions that
    are correct to viewport resolution. Validation and production bakes run float64
    because the conservation metrics they are judged on are otherwise dominated by
    round-off rather than by integration error, which makes them useless as a signal.
    """

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class StageFamily(StrEnum):
    CONTEXT = "context"
    INITIAL_STATE = "initial_state"
    SPATIAL_SERVICE = "spatial_service"
    FORCE = "force"
    INTEGRATION = "integration"
    EVENTS = "events"
    DIAGNOSTICS = "diagnostics"
    OUTPUT = "output"


STAGE_ORDER = {
    StageFamily.CONTEXT: 0,
    StageFamily.INITIAL_STATE: 1,
    StageFamily.SPATIAL_SERVICE: 2,
    StageFamily.FORCE: 3,
    StageFamily.INTEGRATION: 4,
    StageFamily.EVENTS: 5,
    StageFamily.DIAGNOSTICS: 6,
    StageFamily.OUTPUT: 7,
}


def _require_finite(name: str, value: Any) -> None:
    """Refuse NaN and +/-inf before any sign comparison.

    NaN compares False against every bound, so ``x <= 0`` and ``x < 0`` both let it through;
    the finiteness test has to come first (SR-10).
    """

    try:
        finite = math.isfinite(value)
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_finite_all(name: str, values: Any) -> None:
    for index, value in enumerate(values):
        _require_finite(f"{name}[{index}]", value)


def _from_known_keys(cls: type, name: str, data: dict[str, Any]) -> Any:
    """``cls(**data)`` with a clean validation error for an unknown key (SR-47)."""

    known = {f.name for f in fields(cls)}
    unknown = sorted(str(key) for key in data if key not in known)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    return cls(**data)


@dataclass(frozen=True)
class Units:
    length: str = "m"
    mass: str = "kg"
    time: str = "s"
    gravitational_constant: float = 6.67430e-11
    softening: float = 0.0
    # Compile-time SI bases are persisted because Custom label strings and derived G
    # alone are not injective: (1,1,1) and (2,2,2) have the same G but give every
    # authored SI value a different code-space meaning. Defaults preserve old descriptors.
    length_scale_m: float = 1.0
    mass_scale_kg: float = 1.0
    time_scale_s: float = 1.0

    def validate(self) -> None:
        if not self.length or not self.mass or not self.time:
            raise ValueError("units length, mass, and time are required")
        _require_finite("gravitational_constant", self.gravitational_constant)
        _require_finite("softening", self.softening)
        if self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be positive")
        if self.softening < 0.0:
            raise ValueError("softening must be non-negative")
        for name in ("length_scale_m", "mass_scale_kg", "time_scale_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ReferenceFrame:
    name: str = "BlenderWorld"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axes: str = "BLENDER_Z_UP"
    inertial: bool = True


# The shortest frame the authored `seconds_per_frame` property can hold without being
# altered on the way in: the smallest positive *normal* float32 (2**-126). The property is
# float32, so a bound above this truncates a shorter frame silently -- which is how a
# giant-impact scene calibrated at 3e-12 s was stored as 1e-9 s and then refused at Play
# while compile still said OK (ADR 0108). Below this the stored value is subnormal and no
# longer round-trips, so it is a representation floor rather than a taste one; the comfort
# floor artists drag against is the property's `soft_min`.
MIN_SECONDS_PER_FRAME = 2.0**-126


@dataclass(frozen=True)
class TimelineMapping:
    frame_start: int = 1
    frame_end: int = 250
    seconds_per_frame: float = 1.0
    substeps: int = 1

    def validate(self) -> None:
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        _require_finite("seconds_per_frame", self.seconds_per_frame)
        if self.seconds_per_frame <= 0.0:
            raise ValueError("seconds_per_frame must be positive")
        if self.seconds_per_frame < MIN_SECONDS_PER_FRAME:
            raise ValueError(
                f"seconds_per_frame must be at least {MIN_SECONDS_PER_FRAME!r}, "
                f"got {self.seconds_per_frame!r}"
            )
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1")


# Seam 7 sub-commit 2 collapsed the three-way playback selector (Timeline-Locked /
# Free-Run / Manual) down to the single implied Timeline transport; ``TimelineMapping``
# has carried no ``mode`` field since. Files saved before that collapse can still carry
# one of these three strings in two stranded locations (a raw IDProperty left behind by
# the deleted RNA enum, and a ``timeline.mode`` key inside ``descriptor_json``). This is
# the closed, known set of legacy values a load-time migration is responsible for
# recognizing and dropping; sub-commit 3 adds the versioned normalization/migration-
# notice path referenced in ``from_dict`` below.
#
# The pre-collapse build stored the selector as a Blender ``EnumProperty``. Blender
# serialises an enum's value as the integer INDEX into its ``items`` list, not the
# identifier string, so the raw IDProperty stranded on a real old-build row (left behind
# by the deleted RNA enum) carries an ``int`` (0/1/2), while ``descriptor_json``'s
# timeline mode key -- a plain JSON string -- carries the identifier.
# ``LEGACY_PLAYBACK_MODE_ORDER`` is that
# build's frozen ``items`` order (properties.py at the pre-collapse SLICE4_BASE:
# TIMELINE_LOCKED=0, FREE_RUN=1, MANUAL=2); it is the sole source of truth for both the
# index->identifier mapping and the known-set membership below, so the two can never drift.
LEGACY_PLAYBACK_MODE_ORDER = ("TIMELINE_LOCKED", "FREE_RUN", "MANUAL")
KNOWN_LEGACY_PLAYBACK_MODES = frozenset(LEGACY_PLAYBACK_MODE_ORDER)


class UnknownLegacyPlaybackModeError(ValueError):
    """Raised when a stored legacy playback-mode value is not one of the known three.

    Fail-closed: the caller must refuse to mutate the row that carried this value
    rather than guess at a mapping, per ADR 0041 house style.
    """


def normalize_legacy_playback_mode(value: str) -> None:
    """Classify a stored legacy playback-mode *identifier string*.

    Pure and total over the known set: returns ``None`` (there is nothing to map it
    to -- every known legacy value collapses onto the single implied Timeline
    transport) for any of ``KNOWN_LEGACY_PLAYBACK_MODES``, and raises
    :class:`UnknownLegacyPlaybackModeError` naming the offending value for anything
    else. Callers use the raise to fail closed and leave the offending row untouched.

    This validates the ``descriptor_json`` string path only. The raw IDProperty path
    stores an enum *index* rather than the identifier string; use
    :func:`resolve_legacy_playback_mode`, which accepts either encoding.
    """

    if value not in KNOWN_LEGACY_PLAYBACK_MODES:
        raise UnknownLegacyPlaybackModeError(f"unknown legacy playback mode: {value!r}")


def resolve_legacy_playback_mode(value: object) -> str:
    """Resolve a stored legacy playback-mode value to its canonical identifier string.

    Accepts either encoding a pre-collapse ``.blend`` can carry on a context row:

    * the integer enum index left in the raw playback-mode IDProperty, mapped through
      :data:`LEGACY_PLAYBACK_MODE_ORDER`; or
    * the identifier string written into ``descriptor_json``'s timeline mode key,
      validated against :data:`KNOWN_LEGACY_PLAYBACK_MODES`.

    Returns the canonical legacy identifier (one of the three strings). Fail-closed like
    :func:`normalize_legacy_playback_mode`: an out-of-range index, an unknown string, a
    ``bool`` (an ``int`` subclass, but never a valid stored enum index), or any other
    type raises :class:`UnknownLegacyPlaybackModeError` so the caller leaves the row
    untouched rather than guessing at a mapping.
    """

    if isinstance(value, bool):
        raise UnknownLegacyPlaybackModeError(f"unknown legacy playback mode: {value!r}")
    if isinstance(value, int):
        if 0 <= value < len(LEGACY_PLAYBACK_MODE_ORDER):
            return LEGACY_PLAYBACK_MODE_ORDER[value]
        raise UnknownLegacyPlaybackModeError(
            f"unknown legacy playback mode index: {value!r}"
        )
    if isinstance(value, str) and value in KNOWN_LEGACY_PLAYBACK_MODES:
        return value
    raise UnknownLegacyPlaybackModeError(f"unknown legacy playback mode: {value!r}")


@dataclass(frozen=True)
class BindingDescriptor:
    binding_id: str
    context_id: str
    object_name: str
    object_persistent_id: str
    role: BindingRole = BindingRole.BODY
    representation: BodyRepresentation = BodyRepresentation.ANALYTICAL
    active: bool = True
    mass: float = 1.0
    radius: float = 1.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_object_name: str = ""
    orbital_elements: dict[str, float] = field(default_factory=dict)
    # A SAMPLED_PARTICLE binding expands to this many solver bodies at load: one bound
    # object becomes a particle cloud of `mass` spread over `particle_count` bodies inside
    # `radius`. Every other representation is one body, so a Phase 2 scene keeps its
    # meaning without carrying either field. `astro_warp.product.sampling` owns the
    # expansion; nothing else may assume one binding is one body.
    particle_count: int = 1
    sampling_seed: int = 0
    # Where a sampled cloud is drawn from, and how big that volume is. The extent is the
    # bound object's evaluated dimensions, resolved once at compile time by the host: the
    # descriptor carries three floats whatever the particle count, so a 10^6-body cloud
    # hashes into `graph_hash` no differently from a 10-body one. Sampled points must never
    # be embedded here for that reason.
    sample_shape: SampleShape = SampleShape.BALL
    sample_extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The geometry a mesh-shaped cloud is drawn from, named by its content digest rather
    # than carried. `astro_warp.product.mesh_store` holds the triangles under this key,
    # so the descriptor holds 64 characters whatever the mesh --- for exactly the reason
    # the comment above gives for the extent. Empty for every shape that samples a proxy,
    # which keeps a descriptor authored before the mesh shapes existed byte-identical.
    sample_mesh_key: str = ""
    # Emitter-role surface (ADR 0037 section 3.6). An Emitter releases `rate` particles per
    # frame across the inclusive frame span `active_range`, up to a declared ceiling of
    # `emitter_capacity` particles ever emitted. The compiler enforces
    # `rate * (active_range[1] - active_range[0] + 1) <= emitter_capacity` at compile time so
    # an over-budget scene is rejected before a bake starts rather than faulting at frame N.
    # These fields have no effect on any other role, and their defaults leave every existing
    # (non-Emitter) descriptor unchanged. The Emitter role itself stays "not simulated": this
    # is the descriptor surface and its validator, not runtime emission.
    rate: float = 0.0
    active_range: tuple[int, int] = (0, 0)
    emitter_capacity: int = 0
    # Force-role surface (ADR 0045). A FIELD-role binding carries what it applies and how
    # hard; the defaults are inert (a zero-strength uniform field accelerates nothing), so
    # every existing descriptor decodes unchanged -- the Emitter precedent above. These are
    # binding fields rather than graph parameters on purpose: the external-field graph node
    # is parameter-free, so editing a Force invalidates solved state (every binding field
    # invalidates unless exempted) without changing the plan identity.
    force_kind: ForceKind = ForceKind.UNIFORM
    strength: float = 0.0
    # Cloud initial-velocity surface (ADR 0040 section 1). Every default reproduces a
    # pre-ADR descriptor exactly -- UNIFORM bulk velocity, no jitter, identity basis, no
    # provenance -- so a scene saved before these fields existed decodes to what it was
    # already simulating. Changing a default silently changes old scenes.
    #
    # All of these are retroactive initial-state inputs: they are consumed once during
    # expansion, before the first step, so an edit invalidates solved state from step 0.
    # They are retroactive by construction rather than by allowlist -- `invalidation` keeps
    # every binding field unless it is named in EXCLUDED_BINDING_FIELDS, so a new field is
    # invalidating until someone deliberately exempts it, which is the safe default.
    velocity_model: CloudVelocityModel = CloudVelocityModel.UNIFORM
    # Fraction of the virial kinetic energy VIRIALIZE gives the cloud. 1.0 is full virial
    # support; 0.0 is a cold cloud, which is the pre-ADR behaviour.
    support_fraction: float = 1.0
    keplerian_central_binding_id: str = ""
    # The optional second half of the central mass (ADR 0085). Empty is every scene authored
    # before this field existed and means exactly what it meant then: one central body, at its
    # own position, with its own mass. Set, it names a companion, and the cloud then orbits the
    # *barycentre* of the pair at their combined mass -- which is what a circumbinary disc
    # orbits, and what neither body alone can stand in for once the masses are unequal.
    #
    # Two and not N. The point-mass approximation a Keplerian initial condition rests on is
    # well-posed outside a *binary* by a criterion that exists (Holman & Wiegert 1999); there is
    # no equivalent for an arbitrary central set, so a list here would be generality with no
    # rule to make it safe.
    keplerian_companion_binding_id: str = ""
    keplerian_counter_clockwise: bool = True
    eccentricity_jitter: float = 0.0
    mass_jitter: float = 0.0
    annulus_inner_radius: float = 0.0
    # How steeply the annulus's surface density falls with radius: sigma(r) proportional to
    # r**-annulus_surface_density_exponent. Zero is the pre-ADR-0078 behaviour and is uniform
    # in area, which is why it is the default -- every existing descriptor keeps exactly the
    # distribution it had. A real disc is centrally concentrated, and a uniform-in-area one
    # reads in the viewport as a hollow ring, because a ring at twice the radius is twice the
    # area and gets twice the particles while the eye expects the opposite.
    annulus_surface_density_exponent: float = 0.0
    # The EXPANSION velocity model's outflow (ADR 0202): the speed, in code velocity units,
    # at which a particle moves straight out from the cloud's centre. Zero means the cloud's
    # own escape speed, ``sqrt(2 G M / R)``, so the model does something sensible before
    # anyone has set a number. With ``expansion_proportional`` the speed is the one reached at
    # the cloud's radius and grows linearly from the centre (a Hubble flow). Both read only
    # under EXPANSION; both are solver inputs, and both sit at their defaults on every
    # descriptor that existed before the model did (see invalidation.py).
    expansion_speed: float = 0.0
    expansion_proportional: bool = False
    # Compiled-only observations, not author inputs. The host resolves these at compile
    # time and the authoring-document decoder refuses to read them, so an imported JSON
    # cannot manufacture a provenance the host never measured.
    sample_basis: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    sample_frame_provenance: tuple[int, ...] = ()
    # ADR 0158. Spin is the body's angular velocity in radians per *code* time unit, as an
    # (x, y, z) axis-scaled vector (the direction is the axis, the length is the rate). For
    # a Body it turns the visible object; for a sampled Solid it gives every particle the
    # rigid-rotation starting velocity about the body's centre of mass. Pinned holds a
    # Body's position fixed for the whole run: it still pulls on everything else, it is
    # just never pushed. Zero and False are what every earlier binding already was.
    spin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pinned: bool = False
    # Compiled-only, like the basis above: the (w, x, y, z) rotation the Body's object had at
    # compile time, which a Body Spin turns *from*. Only stamped when spin is non-zero.
    spin_rest: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def validate(self) -> None:
        if not self.binding_id or not self.context_id:
            raise ValueError("binding_id and context_id are required")
        if len(self.spin) != 3 or not all(math.isfinite(value) for value in self.spin):
            raise ValueError(f"spin must be three finite numbers, got {self.spin!r}")
        if len(self.spin_rest) != 4 or not all(math.isfinite(v) for v in self.spin_rest):
            raise ValueError(f"spin_rest must be four finite numbers, got {self.spin_rest!r}")
        _require_finite("mass", self.mass)
        _require_finite("radius", self.radius)
        for name in ("position", "velocity", "sample_extent"):
            vector = getattr(self, name)
            if len(vector) != 3:
                raise ValueError(f"{name} must be three numbers, got {vector!r}")
            _require_finite_all(name, vector)
        _require_finite("rate", self.rate)
        _require_finite_all("active_range", self.active_range)
        # A sampled cloud carries mass whatever its role, so it is held to a sign like a
        # Body; a field or region mask keeps its default and is not (SR-10).
        if (
            self.role == BindingRole.BODY
            or self.representation == BodyRepresentation.SAMPLED_PARTICLE
        ) and self.mass <= 0.0:
            raise ValueError("body mass must be positive")
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative")
        if self.particle_count < 1:
            raise ValueError("particle_count must be at least 1")
        if self.representation == BodyRepresentation.SAMPLED_PARTICLE and self.radius <= 0.0:
            raise ValueError("a sampled particle binding needs a positive radius")
        # Unconditional (SR-48): numpy's default_rng refuses a negative seed whatever the
        # representation, so the descriptor refuses it first.
        if self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative")
        if any(value < 0.0 for value in self.sample_extent):
            raise ValueError("sample_extent must be non-negative on every axis")
        # A mesh shape with no key names no geometry, and the sampler's only alternatives
        # would be to raise there or to fill a proxy. Refusing here puts the error on the
        # descriptor that is wrong rather than on the draw downstream of it.
        if self.sample_shape.needs_mesh and not self.sample_mesh_key:
            raise ValueError(
                f"sample_shape {self.sample_shape.value!r} requires a sample_mesh_key"
            )
        # Cloud initial-velocity guards (ADR 0040). Enforced unconditionally: every
        # default satisfies them, so no pre-ADR descriptor is affected, while a
        # hand-authored or imported one is refused before it reaches the sampler.
        #
        # Non-finite is checked first and separately. NaN compares False against every
        # bound, so a plain range test lets it through and it reaches the sampler as a
        # silently poisoned initial condition rather than an error the author can act on.
        for name in (
            "support_fraction",
            "mass_jitter",
            "eccentricity_jitter",
            "annulus_inner_radius",
            # Finiteness only. Whether a given exponent is *usable* depends on the annulus's
            # inner and outer radii, which this descriptor does not yet know -- they come from
            # the host-measured extent -- so that judgement is made in the sampler, next to the
            # radii it needs, rather than guessed here.
            "annulus_surface_density_exponent",
            "expansion_speed",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.support_fraction < 0.0:
            raise ValueError("support_fraction must be non-negative")
        if self.expansion_speed < 0.0:
            raise ValueError("expansion_speed must be non-negative")
        if not 0.0 <= self.mass_jitter < 1.0:
            raise ValueError("mass_jitter must satisfy 0 <= mass_jitter < 1")
        if not 0.0 <= self.eccentricity_jitter < 1.0:
            raise ValueError(
                "eccentricity_jitter must satisfy 0 <= eccentricity_jitter < 1"
            )
        if self.annulus_inner_radius < 0.0:
            raise ValueError("annulus_inner_radius must be non-negative")
        if (
            self.velocity_model == CloudVelocityModel.KEPLERIAN
            and not self.keplerian_central_binding_id
        ):
            raise ValueError(
                "a Keplerian cloud needs keplerian_central_binding_id naming the central "
                "body it orbits; there is no default central binding to fall back to"
            )
        # ADR 0085. A companion with no central is a half-authored pair, and it would decode
        # to a *single*-central orbit about whichever id happened to be set -- the failure
        # mode this whole field exists to remove. A companion equal to the central would
        # double the mass of one body and put the barycentre on top of it.
        if self.keplerian_companion_binding_id:
            if not self.keplerian_central_binding_id:
                raise ValueError(
                    "keplerian_companion_binding_id names the second half of a central "
                    "pair, so keplerian_central_binding_id must name the first"
                )
            if self.keplerian_companion_binding_id == self.keplerian_central_binding_id:
                raise ValueError(
                    "a Keplerian central pair needs two different bindings, but the "
                    f"central and the companion are both "
                    f"{self.keplerian_central_binding_id!r}"
                )
        # Emitter-field guards (ADR 0037 section 3.6). Enforced unconditionally, mirroring
        # TimelineMapping.validate()'s frame-order rule: the defaults (0.0, (0, 0), 0) that
        # every non-Emitter binding carries satisfy all three, so no existing descriptor is
        # affected, while a hand-authored Emitter with a reversed range or negative field is
        # rejected before it can reach the compiler's budget check.
        if self.active_range[1] < self.active_range[0]:
            raise ValueError("active_range end must be at or after its start")
        if self.rate < 0.0:
            raise ValueError("rate must be non-negative")
        if self.emitter_capacity < 0:
            raise ValueError("emitter_capacity must be non-negative")
        # Force-field guard (ADR 0045). Finite-only, unconditional: the 0.0 default every
        # non-FIELD binding carries passes trivially, while NaN/inf would reach the kernel
        # as a silently poisoned acceleration. No sign or range check -- negative strength
        # is a repulsive attractor or a downward uniform push, which is authored meaning.
        if not math.isfinite(self.strength):
            raise ValueError(f"strength must be finite, got {self.strength!r}")

    def without_presentation(self) -> BindingDescriptor:
        """Copy with output-contract-only fields cleared.

        ``render_object_name`` names the datablock an output build writes into. It
        reaches no solver input, so a runtime holding a compiled plan is still current
        after it changes; anything asking whether a runtime went stale compares this
        instead of the whole binding.
        """

        return replace(self, render_object_name="")


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    family: StageFamily
    module: str
    # Empty means unpinned: the compiler resolves it against MODULE_REGISTRY and stamps the
    # installed version on the compiled node. A non-empty value is a pin, and the compiler
    # rejects it when it disagrees with the installed module, because a graph authored
    # against an older module must be upgraded explicitly (PRODUCT-004 c2).
    module_version: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AstroSystemGraph:
    graph_id: str
    schema_version: str = SCHEMA_VERSION
    nodes: tuple[GraphNode, ...] = ()
    manifests: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeStep:
    index: int
    node_id: str
    family: StageFamily
    module: str
    module_version: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePlan:
    plan_id: str
    graph_hash: str
    steps: tuple[RuntimeStep, ...]
    backend: str = "none"
    capacities: dict[str, int] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviceDescriptor:
    required_warp_version: str = "1.15.0"
    required_device: str = "cuda:0"
    cuda_required: bool = True


@dataclass(frozen=True)
class SimulationDescriptor:
    context_id: str
    name: str
    schema_version: str = SCHEMA_VERSION
    state: ContextState = ContextState.UNLOADED
    units: Units = field(default_factory=Units)
    reference_frame: ReferenceFrame = field(default_factory=ReferenceFrame)
    display_scale: float = 1.0
    precision: Precision = Precision.FLOAT32
    timeline: TimelineMapping = field(default_factory=TimelineMapping)
    cache_root: str = ""
    controller_object_name: str = ""
    bindings: tuple[BindingDescriptor, ...] = ()
    graph: AstroSystemGraph = field(default_factory=lambda: default_graph())
    runtime_plan: RuntimePlan | None = None
    selected_bake_id: str = ""
    fault_text: str = ""
    # System-owned, frame-authored timeline stage list (ENGINE-002). Defaulted empty so every
    # descriptor built before this field existed is unchanged. A stage edit is a solver input:
    # it is not excluded anywhere in invalidation.py, so it flows into both
    # solved_state_inputs and retroactive_state_inputs by default and invalidates from step 0.
    stages: tuple[TimelineStage, ...] = ()

    def validate(self) -> None:
        if not self.context_id:
            raise ValueError("context_id is required")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.display_scale <= 0.0:
            raise ValueError("display_scale must be positive")
        self.units.validate()
        self.timeline.validate()
        binding_ids = set()
        for binding in self.bindings:
            binding.validate()
            if binding.binding_id in binding_ids:
                raise ValueError(f"duplicate binding_id {binding.binding_id}")
            binding_ids.add(binding.binding_id)
        validate_stage_list(
            self.stages,
            frame_start=self.timeline.frame_start,
            frame_end=self.timeline.frame_end,
        )

    def with_state(self, state: ContextState, fault_text: str = "") -> SimulationDescriptor:
        return replace(self, state=state, fault_text=fault_text)

    def to_dict(self) -> dict[str, Any]:
        return _encode(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationDescriptor:
        graph_data = data.get("graph") or {}
        plan_data = data.get("runtime_plan")
        descriptor = cls(
            context_id=str(data["context_id"]),
            name=str(data.get("name") or "Astro Simulation Context"),
            schema_version=_required_schema_version(data, "descriptor"),
            state=ContextState(str(data.get("state") or ContextState.UNLOADED.value)),
            units=_from_known_keys(Units, "units", data.get("units") or {}),
            reference_frame=ReferenceFrame(
                **{
                    **(data.get("reference_frame") or {}),
                    # `origin` is a tuple field that _encode emits as a list; decode it back
                    # to a tuple so a descriptor round-trips to an equal value.
                    "origin": tuple(
                        float(v)
                        for v in (data.get("reference_frame") or {}).get(
                            "origin", (0.0, 0.0, 0.0)
                        )
                    ),
                }
            ),
            display_scale=float(data.get("display_scale", 1.0)),
            precision=Precision(str(data.get("precision") or Precision.FLOAT32.value)),
            timeline=_from_known_keys(
                TimelineMapping,
                "timeline",
                {
                    # A file saved before this field was removed may still carry a stored
                    # `mode` key; it is simply ignored here rather than erroring. The
                    # versioned normalization/migration-notice/fail-closed path for that
                    # stored value lives at load time in the extension's session-lifecycle
                    # load_post handler (`normalize_legacy_playback_mode`, this module),
                    # not here -- `from_dict` stays a passive, side-effect-free decoder.
                    key: value
                    for key, value in (data.get("timeline") or {}).items()
                    if key != "mode"
                },
            ),
            cache_root=str(data.get("cache_root") or ""),
            controller_object_name=str(data.get("controller_object_name") or ""),
            bindings=tuple(_binding_from_dict(item) for item in data.get("bindings", ())),
            graph=_graph_from_dict(graph_data, fallback_seed=str(data["context_id"])),
            runtime_plan=_plan_from_dict(plan_data) if plan_data else None,
            selected_bake_id=str(data.get("selected_bake_id") or ""),
            fault_text=str(data.get("fault_text") or ""),
            stages=tuple(
                _stage_from_dict(item, index=index)
                for index, item in enumerate(data.get("stages", ()))
            ),
        )
        descriptor.validate()
        return descriptor

    @classmethod
    def from_json(cls, text: str) -> SimulationDescriptor:
        return cls.from_dict(json.loads(text))


def new_context_id() -> str:
    return str(uuid.uuid4())


def new_binding_id() -> str:
    return str(uuid.uuid4())


def runtime_stable_id(binding_id: str) -> int:
    value = 1469598103934665603
    for byte in binding_id.encode("utf-8"):
        value ^= byte
        value *= 1099511628211
        value &= 0x7FFFFFFFFFFFFFFF
    return value


def default_graph(graph_id: str | None = None) -> AstroSystemGraph:
    gid = graph_id or str(uuid.uuid4())
    return AstroSystemGraph(
        graph_id=gid,
        nodes=(
            GraphNode(
                node_id="context.units",
                family=StageFamily.CONTEXT,
                module="astro.context.units",
                outputs=("units", "reference_frame", "timeline"),
            ),
            GraphNode(
                node_id="inputs.bindings",
                family=StageFamily.INITIAL_STATE,
                module="astro.inputs.bindings",
                inputs=("units", "reference_frame"),
                outputs=("body_bindings",),
            ),
            GraphNode(
                node_id="outputs.contract",
                family=StageFamily.OUTPUT,
                module="astro.outputs.contract",
                inputs=("body_bindings",),
                outputs=("output_contracts",),
            ),
        ),
    )


def direct_orbital_graph(
    graph_id: str | None = None,
    *,
    collision_mode: str = "ignore",
    external_field: bool = False,
    emitter: bool = False,
    shearing_box: dict[str, Any] | None = None,
) -> AstroSystemGraph:
    gid = graph_id or str(uuid.uuid4())
    nodes: tuple[GraphNode, ...] = (
            GraphNode(
                node_id="context.units",
                family=StageFamily.CONTEXT,
                module="astro.context.units",
                outputs=("units", "reference_frame", "timeline"),
            ),
            GraphNode(
                node_id="inputs.analytical_bodies",
                family=StageFamily.INITIAL_STATE,
                module="astro.inputs.analytical_bodies",
                inputs=("units", "reference_frame"),
                outputs=("body_soa",),
            ),
            GraphNode(
                node_id="spatial.direct_all_pairs",
                family=StageFamily.SPATIAL_SERVICE,
                module="astro.gravity.direct_all_pairs",
                inputs=("body_soa",),
                outputs=("pair_iterator",),
                parameters={"backend": "direct"},
            ),
            GraphNode(
                node_id="force.newtonian_gravity",
                family=StageFamily.FORCE,
                module="astro.gravity.newtonian",
                inputs=("body_soa", "pair_iterator"),
                outputs=("accelerations",),
            ),
            GraphNode(
                node_id="integration.kdk",
                family=StageFamily.INTEGRATION,
                module="astro.integrators.kdk_leapfrog",
                inputs=("body_soa", "accelerations"),
                outputs=("body_soa", "checkpoints"),
            ),
            GraphNode(
                node_id="events.point_collisions",
                family=StageFamily.EVENTS,
                module="astro.collision.point",
                inputs=("body_soa",),
                outputs=("collision_events",),
                parameters={"mode": collision_mode},
            ),
            GraphNode(
                node_id="diagnostics.orbital",
                family=StageFamily.DIAGNOSTICS,
                module="astro.diagnostics.orbital_scalars",
                inputs=("body_soa", "accelerations", "collision_events"),
                outputs=("diagnostics",),
            ),
            GraphNode(
                node_id="outputs.bake",
                family=StageFamily.OUTPUT,
                module="astro.outputs.immutable_bake",
                inputs=("body_soa", "diagnostics", "checkpoints"),
                outputs=("bake_revision", "render_contract"),
            ),
    )
    if external_field:
        # The external-field node is parameter-free and spliced in only when asked
        # (ADR 0045): the field physics lives on FIELD-role bindings, so a force-free
        # scene builds the byte-identical node tuple this factory emitted before the flag
        # existed, and adding or editing a Force never changes the graph hash -- the same
        # reason body masses are not graph parameters. It sits directly after the gravity
        # node because the compiler requires authored nodes in staged order; FORCE-family
        # steps are commutative accumulations into the one `accelerations` array, so
        # within-family order carries no physics.
        gravity_index = next(
            index
            for index, node in enumerate(nodes)
            if node.node_id == "force.newtonian_gravity"
        )
        nodes = (
            nodes[: gravity_index + 1]
            + (
                GraphNode(
                    node_id="force.external_field",
                    family=StageFamily.FORCE,
                    module="astro.force.external_field",
                    inputs=("body_soa",),
                    outputs=("accelerations",),
                ),
            )
            + nodes[gravity_index + 1 :]
        )
    if shearing_box is not None:
        # The local shearing box (ADR 0202): a frame option of this Direct pipeline, spliced in
        # only when the System asks for it, so a System that does not builds the byte-identical
        # node tuple this factory emitted before the option existed. Unlike the external-field
        # node it carries parameters -- the box (omega, box_x, box_y, images) is a solver input
        # with no binding to live on -- so editing the box moves `graph_hash`, which is right: a
        # different box is a different simulation. It sits directly after the gravity node, where
        # the (family, node_id) plan sort puts it. Only this factory takes the option: the tree
        # and Auto factories build their own gravity nodes and cannot sum images.
        gravity_index = next(
            index
            for index, node in enumerate(nodes)
            if node.node_id == "force.newtonian_gravity"
        )
        nodes = (
            nodes[: gravity_index + 1]
            + (
                GraphNode(
                    node_id="force.shearing_box",
                    family=StageFamily.FORCE,
                    module="astro.frame.shearing_box",
                    inputs=("body_soa",),
                    outputs=("accelerations",),
                    parameters=dict(shearing_box),
                ),
            )
            + nodes[gravity_index + 1 :]
        )
    if emitter:
        # The emitter-activation node, spliced in only when an EMITTER binding is present,
        # and parameter-free for the same reason the external field is: the release schedule
        # (`rate`, `active_range`, `emitter_capacity`) lives on the binding, so retiming an
        # Emitter never moves the graph hash and never retires a bake that ran the identical
        # kernel. It is an EVENTS node and the sibling of `events.point_collisions`: both are
        # population changes applied after integration, one raising the live watermark and one
        # lowering it.
        #
        # `GraphCompiler.compile` sorts within a family by node_id, so the compiled plan lists
        # this node before the collision node while the runtime applies it once per frame
        # *after* the substep loop. That is not a contradiction: the two write disjoint slots.
        # Activation only ever raises slots above the previous frame's watermark, which have
        # never been live and so were never candidates for a merge; collisions only ever
        # deactivate slots that were already live. Within-family order therefore carries no
        # physics here, exactly as it carries none among the FORCE accumulations above.
        collision_index = next(
            index for index, node in enumerate(nodes) if node.node_id == "events.point_collisions"
        )
        nodes = (
            nodes[: collision_index + 1]
            + (
                GraphNode(
                    node_id="events.emitter_activate",
                    family=StageFamily.EVENTS,
                    module="astro.emitter.activate",
                    inputs=("body_soa", "timeline"),
                    outputs=("body_soa",),
                ),
            )
            + nodes[collision_index + 1 :]
        )
    return AstroSystemGraph(graph_id=gid, nodes=nodes)


# The tree's shipped depth and walk-stack defaults, shared by every graph that builds a
# tree and by the runtimes' own configs (ADR 0134). The depth is the deepest supported
# (`tree.MAX_TREE_DEPTH`): a galaxy's tidal tails stretch the root cube about 130 times
# past its cores, and at the former depth 9 the cores' finest cells held about 250 bodies
# each, so the walk summed them pair by pair. The stack is the least that depth needs.
DEFAULT_TREE_MAX_DEPTH = 16
DEFAULT_TREE_STACK_CAPACITY = 1 + 7 * DEFAULT_TREE_MAX_DEPTH


def barnes_hut_graph(
    graph_id: str | None = None,
    *,
    theta: float = DEFAULT_TREE_THETA,
    max_depth: int = DEFAULT_TREE_MAX_DEPTH,
    leaf_capacity: int = 8,
    stack_capacity: int = DEFAULT_TREE_STACK_CAPACITY,
    node_capacity: int = 0,
    quadrupoles: bool = True,
    collision_mode: str = "ignore",
    external_field: bool = False,
    emitter: bool = False,
) -> AstroSystemGraph:
    graph = direct_orbital_graph(
        graph_id=graph_id,
        collision_mode=collision_mode,
        external_field=external_field,
        emitter=emitter,
    )
    nodes = []
    for node in graph.nodes:
        if node.node_id == "spatial.direct_all_pairs":
            nodes.append(
                GraphNode(
                    node_id="spatial.barnes_hut_tree",
                    family=StageFamily.SPATIAL_SERVICE,
                    module="astro.gravity.barnes_hut_tree",
                    inputs=("body_soa",),
                    outputs=("linear_octree", "tree_traversal"),
                    parameters={
                        "backend": "tree",
                        "theta": theta,
                        "max_depth": max_depth,
                        "leaf_capacity": leaf_capacity,
                        "stack_capacity": stack_capacity,
                        "node_capacity": node_capacity,
                        "quadrupoles": quadrupoles,
                    },
                )
            )
        elif node.node_id == "force.newtonian_gravity":
            nodes.append(
                GraphNode(
                    node_id="force.tree_newtonian_gravity",
                    family=StageFamily.FORCE,
                    module="astro.gravity.tree_newtonian",
                    inputs=("body_soa", "linear_octree", "tree_traversal"),
                    outputs=("accelerations",),
                )
            )
        else:
            nodes.append(node)
    return AstroSystemGraph(graph_id=graph.graph_id, nodes=tuple(nodes), manifests=graph.manifests)


def auto_orbital_graph(
    graph_id: str | None = None,
    *,
    accuracy_tolerance: float | None = None,
    accuracy_policy: str = "",
    theta: float = DEFAULT_TREE_THETA,
    max_depth: int = DEFAULT_TREE_MAX_DEPTH,
    leaf_capacity: int = 8,
    stack_capacity: int = DEFAULT_TREE_STACK_CAPACITY,
    node_capacity: int = 0,
    quadrupoles: bool = True,
    collision_mode: str = "ignore",
    external_field: bool = False,
    emitter: bool = False,
) -> AstroSystemGraph:
    if not accuracy_policy and accuracy_tolerance is None:
        # Neither was chosen, so fall back to the certified bound rather than to nothing:
        # an unset policy makes Auto refuse the tree unconditionally.
        derived = tree_accuracy_tolerance(theta)
        if derived is not None:
            accuracy_policy = TREE_ACCURACY_POLICY
            accuracy_tolerance = derived
    graph = direct_orbital_graph(
        graph_id=graph_id,
        collision_mode=collision_mode,
        external_field=external_field,
        emitter=emitter,
    )
    nodes = []
    for node in graph.nodes:
        if node.node_id == "spatial.direct_all_pairs":
            nodes.append(
                GraphNode(
                    node_id="spatial.auto_gravity",
                    family=StageFamily.SPATIAL_SERVICE,
                    module="astro.gravity.auto_select",
                    inputs=("body_soa",),
                    outputs=("gravity_backend",),
                    parameters={
                        "backend": "auto",
                        "accuracy_tolerance": accuracy_tolerance,
                        "accuracy_policy": accuracy_policy,
                        "theta": theta,
                        "max_depth": max_depth,
                        "leaf_capacity": leaf_capacity,
                        "stack_capacity": stack_capacity,
                        "node_capacity": node_capacity,
                        "quadrupoles": quadrupoles,
                    },
                )
            )
        elif node.node_id == "force.newtonian_gravity":
            nodes.append(
                GraphNode(
                    node_id="force.auto_newtonian_gravity",
                    family=StageFamily.FORCE,
                    module="astro.gravity.auto_newtonian",
                    inputs=("body_soa", "gravity_backend"),
                    outputs=("accelerations",),
                )
            )
        else:
            nodes.append(node)
    return AstroSystemGraph(graph_id=graph.graph_id, nodes=tuple(nodes), manifests=graph.manifests)


def giant_impact_graph(
    graph_id: str | None = None,
    *,
    linking_length_factor: float = 0.75,
    moon_start: float = 20.0,
    moon_end: float = 0.0,
    materials: dict[str, Any] | None = None,
    shells: dict[str, Any] | None = None,
) -> AstroSystemGraph:
    """The giant-impact pipeline (ADR 0094): two planet-sized Solids meet and a moon forms.

    What decides a giant impact is gravity, so this graph is built around it: a
    position-based pressure constraint keeps each body a body at any step length, direct tiled
    gravity holds it together, and a moon guide gathers the orbiting debris into one moon
    inside a shot's length. It is a VFX method, chosen to run in the viewport, not to be
    scientifically accurate.

    The FORCE node ids are numbered because the plan publishes steps in node-id order, and
    that order is the order a frame runs them - the pressure substeps, then gravity once
    around them, then the guide that reads the debris gravity has just moved.
    """

    gid = graph_id or str(uuid.uuid4())
    return AstroSystemGraph(
        graph_id=gid,
        nodes=(
            GraphNode(
                node_id="context.units",
                family=StageFamily.CONTEXT,
                module="astro.context.units",
                outputs=("units", "reference_frame", "timeline"),
            ),
            GraphNode(
                node_id="inputs.giant_impact_bodies",
                family=StageFamily.INITIAL_STATE,
                module="astro.inputs.giant_impact_bodies",
                inputs=("units", "reference_frame"),
                outputs=("particle_soa", "material_ids"),
                # Plan parameters, not runtime configuration: a moved core boundary is a
                # different authored input, so it moves `graph_hash`.
                parameters={"materials": materials or {}, "shells": shells or {}},
            ),
            GraphNode(
                node_id="spatial.giant_impact_grid",
                family=StageFamily.SPATIAL_SERVICE,
                module="astro.giant_impact.neighbor_grid",
                inputs=("particle_soa",),
                outputs=("neighbor_iterator",),
                parameters={"backend": "giant_impact"},
            ),
            GraphNode(
                node_id="force.1_pressure",
                family=StageFamily.FORCE,
                module="astro.giant_impact.pbf_pressure",
                inputs=("particle_soa", "neighbor_iterator", "material_ids"),
                outputs=("particle_soa",),
            ),
            GraphNode(
                node_id="force.2_gravity",
                family=StageFamily.FORCE,
                module="astro.giant_impact.tiled_gravity",
                inputs=("particle_soa",),
                outputs=("accelerations",),
            ),
            GraphNode(
                # In dynamical times of the most massive body, so a scene retimed or rescaled
                # forms its moon at the same point in its own story.
                node_id="force.3_moon_guide",
                family=StageFamily.FORCE,
                module="astro.giant_impact.moon_guide",
                inputs=("particle_soa", "accelerations"),
                outputs=("accelerations",),
                parameters={"start": moon_start, "end": moon_end},
            ),
            GraphNode(
                node_id="integration.giant_impact_frame",
                family=StageFamily.INTEGRATION,
                module="astro.integrators.giant_impact_frame",
                inputs=("particle_soa", "accelerations"),
                outputs=("particle_soa", "checkpoints"),
            ),
            GraphNode(
                node_id="diagnostics.1_giant_impact",
                family=StageFamily.DIAGNOSTICS,
                module="astro.diagnostics.giant_impact_scalars",
                inputs=("particle_soa",),
                outputs=("diagnostics",),
            ),
            GraphNode(
                node_id="diagnostics.2_fragments",
                family=StageFamily.DIAGNOSTICS,
                module="astro.diagnostics.fragments",
                inputs=("particle_soa",),
                outputs=("fragments",),
                parameters={"linking_length_factor": linking_length_factor},
            ),
            GraphNode(
                node_id="outputs.bake",
                family=StageFamily.OUTPUT,
                module="astro.outputs.immutable_bake",
                inputs=("particle_soa", "diagnostics", "fragments", "checkpoints"),
                outputs=("bake_revision", "render_contract"),
            ),
        ),
    )


def spacesim_impact_graph(
    graph_id: str | None = None,
    *,
    linking_length_factor: float = 0.75,
    materials: dict[str, Any] | None = None,
    shells: dict[str, Any] | None = None,
    fault_strain: float = 0.003,
    fault_cell_particles: int = 400,
) -> AstroSystemGraph:
    """The realtime rock-impact pipeline (ADR 0193): bodies that crack, at a viewport's pace.

    Not `giant_impact_graph`, which keeps every body a fluid, so debris never reads as rock.
    This one keeps SpaceSim's implicit pressure and its geometric fracture, holds each body
    together with reference bonds that break past a strain taken from the material, and runs
    two clocks: outer steps rebuild the neighbours and solve pressure once, inner steps touch
    only the bonds. It is a VFX method, chosen to run in the viewport, not to be
    scientifically accurate.

    The FORCE node ids are numbered because the plan publishes steps in node-id order, and
    that order is the order a frame runs them - pressure, then the bonds that hold the bodies
    together, then gravity once around them.
    """

    gid = graph_id or str(uuid.uuid4())
    return AstroSystemGraph(
        graph_id=gid,
        nodes=(
            GraphNode(
                node_id="context.units",
                family=StageFamily.CONTEXT,
                module="astro.context.units",
                outputs=("units", "reference_frame", "timeline"),
            ),
            GraphNode(
                node_id="inputs.spacesim_bodies",
                family=StageFamily.INITIAL_STATE,
                module="astro.inputs.spacesim_bodies",
                inputs=("units", "reference_frame"),
                outputs=("particle_soa", "material_ids"),
                # Plan parameters, not runtime configuration: a moved yield stress or fault
                # strain is a different authored input, so it moves `graph_hash`.
                parameters={
                    "materials": materials or {},
                    "shells": shells or {},
                    "fault_strain": fault_strain,
                    "fault_cell_particles": fault_cell_particles,
                },
            ),
            GraphNode(
                node_id="spatial.spacesim_grid",
                family=StageFamily.SPATIAL_SERVICE,
                module="astro.spacesim.neighbor_grid",
                inputs=("particle_soa",),
                outputs=("neighbor_iterator",),
                parameters={"backend": "spacesim_impact"},
            ),
            GraphNode(
                node_id="force.1_pressure",
                family=StageFamily.FORCE,
                module="astro.spacesim.iisph_pressure",
                inputs=("particle_soa", "neighbor_iterator", "material_ids"),
                outputs=("particle_soa",),
            ),
            GraphNode(
                node_id="force.2_bond_strength",
                family=StageFamily.FORCE,
                module="astro.spacesim.bond_strength",
                inputs=("particle_soa", "material_ids"),
                outputs=("particle_soa", "damage"),
            ),
            GraphNode(
                node_id="force.3_gravity",
                family=StageFamily.FORCE,
                module="astro.spacesim.tiled_gravity",
                inputs=("particle_soa",),
                outputs=("accelerations",),
            ),
            GraphNode(
                node_id="integration.spacesim_two_rate",
                family=StageFamily.INTEGRATION,
                module="astro.integrators.spacesim_two_rate",
                inputs=("particle_soa", "accelerations"),
                outputs=("particle_soa", "checkpoints"),
            ),
            GraphNode(
                node_id="diagnostics.1_spacesim",
                family=StageFamily.DIAGNOSTICS,
                module="astro.diagnostics.spacesim_scalars",
                inputs=("particle_soa",),
                outputs=("diagnostics",),
            ),
            GraphNode(
                node_id="diagnostics.2_fragments",
                family=StageFamily.DIAGNOSTICS,
                module="astro.diagnostics.fragments",
                inputs=("particle_soa",),
                outputs=("fragments",),
                parameters={"linking_length_factor": linking_length_factor},
            ),
            GraphNode(
                node_id="outputs.bake",
                family=StageFamily.OUTPUT,
                module="astro.outputs.immutable_bake",
                inputs=("particle_soa", "diagnostics", "fragments", "checkpoints"),
                outputs=("bake_revision", "render_contract"),
            ),
        ),
    )


def _binding_from_dict(data: dict[str, Any]) -> BindingDescriptor:
    return BindingDescriptor(
        binding_id=str(data["binding_id"]),
        context_id=str(data["context_id"]),
        object_name=str(data.get("object_name") or ""),
        object_persistent_id=str(data.get("object_persistent_id") or ""),
        role=BindingRole(str(data.get("role", BindingRole.BODY.value))),
        representation=BodyRepresentation(
            str(data.get("representation", BodyRepresentation.ANALYTICAL.value))
        ),
        active=bool(data.get("active", True)),
        mass=float(data.get("mass", 1.0)),
        radius=float(data.get("radius", 1.0)),
        position=tuple(float(v) for v in data.get("position", (0.0, 0.0, 0.0))),
        velocity=tuple(float(v) for v in data.get("velocity", (0.0, 0.0, 0.0))),
        render_object_name=str(data.get("render_object_name") or ""),
        orbital_elements={
            str(k): float(v) for k, v in (data.get("orbital_elements") or {}).items()
        },
        # A binding persisted before particle clouds existed carries neither field and
        # reads back as the single body it always was.
        particle_count=int(data.get("particle_count", 1)),
        sampling_seed=int(data.get("sampling_seed", 0)),
        # Likewise for a binding persisted before sample shapes existed: BALL and a zero
        # extent are what it was already sampling.
        sample_shape=SampleShape(str(data.get("sample_shape", SampleShape.BALL.value))),
        sample_mesh_key=str(data.get("sample_mesh_key", "")),
        sample_extent=tuple(
            float(v) for v in data.get("sample_extent", (0.0, 0.0, 0.0))
        ),
        # A binding persisted before the Emitter surface existed carries none of these and
        # reads back at zero rate/range/capacity — which the compiler's budget check passes
        # trivially (and only Emitter-role bindings are checked at all), so an old scene keeps
        # its meaning. to_dict serializes these via asdict(self), so no separate encode path is
        # needed; `active_range` round-trips through the list _encode emits.
        rate=float(data.get("rate", 0.0)),
        active_range=tuple(int(v) for v in data.get("active_range", (0, 0))),
        emitter_capacity=int(data.get("emitter_capacity", 0)),
        # A binding persisted before the Force surface existed carries neither field and
        # reads back as the zero-strength uniform field default, which applies nothing.
        force_kind=ForceKind(str(data.get("force_kind", ForceKind.UNIFORM.value))),
        strength=float(data.get("strength", 0.0)),
        # A binding persisted before the Cloud velocity surface existed carries none of
        # these, and reads back as the UNIFORM, unjittered, identity-basis cloud it was.
        velocity_model=CloudVelocityModel(
            str(data.get("velocity_model", CloudVelocityModel.UNIFORM.value))
        ),
        support_fraction=float(data.get("support_fraction", 1.0)),
        keplerian_central_binding_id=str(data.get("keplerian_central_binding_id") or ""),
        keplerian_companion_binding_id=str(
            data.get("keplerian_companion_binding_id") or ""
        ),
        keplerian_counter_clockwise=bool(data.get("keplerian_counter_clockwise", True)),
        eccentricity_jitter=float(data.get("eccentricity_jitter", 0.0)),
        mass_jitter=float(data.get("mass_jitter", 0.0)),
        annulus_inner_radius=float(data.get("annulus_inner_radius", 0.0)),
        annulus_surface_density_exponent=float(
            data.get("annulus_surface_density_exponent", 0.0)
        ),
        # A binding persisted before the Expansion model existed carries neither, and reads
        # back at the escape-speed default that nothing but an Expansion cloud reads.
        expansion_speed=float(data.get("expansion_speed", 0.0)),
        expansion_proportional=bool(data.get("expansion_proportional", False)),
        sample_basis=tuple(
            tuple(float(component) for component in row)
            for row in data.get(
                "sample_basis",
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            )
        ),
        sample_frame_provenance=tuple(
            int(v) for v in data.get("sample_frame_provenance", ())
        ),
        # A binding persisted before Spin and Pinned were simulated (ADR 0158) carries
        # neither, and reads back as the non-spinning, free body it always ran as.
        spin=tuple(float(v) for v in data.get("spin", (0.0, 0.0, 0.0))),
        pinned=bool(data.get("pinned", False)),
        spin_rest=tuple(float(v) for v in data.get("spin_rest", (1.0, 0.0, 0.0, 0.0))),
    )


def _stage_from_dict(data: dict[str, Any], *, index: int) -> TimelineStage:
    # A corrupted save can carry any JSON value in a stages slot; refuse it with a
    # field-specific message (the same shape validate_stage_list uses) instead of an
    # AttributeError from reading .get() off a non-object.
    if not isinstance(data, dict):
        raise ValueError(f"stages[{index}] must be an object")
    if "kind" not in data:
        raise ValueError(f"stages[{index}].kind must be present")
    boundary = data.get("boundary_frame")
    if boundary is not None:
        # ``int()`` alone would silently coerce 42.9 -> 42 and True -> 1, re-authoring
        # a corrupted timeline instead of refusing it; a JSON number carrying an
        # integral value (42.0) still decodes as the integer 42.
        integral_type = isinstance(boundary, int) and not isinstance(boundary, bool)
        integral_float = (
            isinstance(boundary, float) and boundary.is_integer() and abs(boundary) < 2**53
        )
        if not (integral_type or integral_float):
            raise ValueError(f"stages[{index}].boundary_frame must be an integer")
    return TimelineStage(
        kind=TimelineStageKind(str(data["kind"])),
        boundary_frame=None if boundary is None else int(boundary),
    )


def _required_schema_version(data: dict[str, Any], what: str) -> str:
    """A persisted record must name its schema; absence is not "current" (SR-47)."""

    value = data.get("schema_version")
    if not value:
        raise ValueError(f"{what} schema_version is missing or empty")
    return str(value)


def _graph_from_dict(data: dict[str, Any], fallback_seed: str = "") -> AstroSystemGraph:
    # A graph persisted without an id must decode to the *same* id every load, because the
    # id is hashed into `graph_hash`: a fresh uuid4 here made an unchanged file look edited
    # on every open (SR-46). Derive it from the owning context and the graph's own content.
    graph_id = data.get("graph_id")
    if not graph_id:
        content = json.dumps(data, sort_keys=True, default=str)
        graph_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"astro-warp-graph:{fallback_seed}:{content}")
    )
    return AstroSystemGraph(
        graph_id=str(graph_id),
        schema_version=_required_schema_version(data, "graph") if data else SCHEMA_VERSION,
        nodes=tuple(
            GraphNode(
                node_id=str(item["node_id"]),
                family=StageFamily(str(item["family"])),
                module=str(item["module"]),
                # A graph persisted before module versioning existed carries none, which
                # reads as unpinned and resolves on the next compile.
                module_version=str(item.get("module_version") or ""),
                inputs=tuple(str(v) for v in item.get("inputs", ())),
                outputs=tuple(str(v) for v in item.get("outputs", ())),
                parameters=dict(item.get("parameters") or {}),
            )
            for item in data.get("nodes", ())
        ),
        manifests=dict(data.get("manifests") or {}),
    )


def _plan_from_dict(data: dict[str, Any]) -> RuntimePlan:
    return RuntimePlan(
        plan_id=str(data["plan_id"]),
        graph_hash=str(data["graph_hash"]),
        backend=str(data.get("backend", "none")),
        steps=tuple(
            RuntimeStep(
                index=int(item["index"]),
                node_id=str(item["node_id"]),
                family=StageFamily(str(item["family"])),
                module=str(item["module"]),
                module_version=str(item.get("module_version") or ""),
                parameters=dict(item.get("parameters") or {}),
            )
            for item in data.get("steps", ())
        ),
        capacities={str(k): int(v) for k, v in (data.get("capacities") or {}).items()},
        diagnostics=tuple(str(v) for v in data.get("diagnostics", ())),
    )


def _encode(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    return value
