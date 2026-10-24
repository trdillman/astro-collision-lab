"""The product runtime for the giant-impact solver: two Solids meet and a moon forms.

`giant_impact.GiantImpactSolver` is the engine (ADR 0094). It is a VFX method, not a
scientific one: position-based-fluid pressure keeps each body a body, direct tiled gravity
holds it together, and a moon guide steers the debris disk into one moon inside a shot's
length. This module is what makes it an addon backend rather than a script: it turns the
bindings a System compiles into the engine's particles, publishes the engine's state on the
same display channels every other backend publishes, and speaks the resumable frame
protocol `ProductRuntime` plays with.

Two shapes differ from the authored descriptor, and both are deliberate:

* **One pressure resolution.** The pressure solve has one rest density and one smoothing
  length, so every body is sampled at one common particle spacing: the coarsest any body's
  authored count implies. A body whose authored count is finer than that uses fewer
  particles, and the rest of its slots stay idle (``active`` 0, mass 0) for the life of the
  run. The descriptor's capacity, its stable ids, and the display's row count are unchanged,
  so nothing downstream has a second population to reconcile; the display compacts idle
  slots away exactly as it compacts merged bodies (ADR 0046).
* **One primitive per frame.** Gravity is evaluated once per displayed frame around the
  pressure substeps, so the smallest unit of work that leaves the state consistent is a whole
  frame. ``config.substeps`` is therefore 1: one resumable primitive per authored frame, one
  captured CUDA graph replay inside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .impact_frame import ImpactFrameRuntime
from .model import (
    CHECKPOINT_SCHEMA_VERSION,
    BindingDescriptor,
    BindingRole,
    BodyRepresentation,
    CloudVelocityModel,
    Precision,
    SampleShape,
    runtime_stable_id,
)
from .runtime_foundation import DeviceContext
from .sampling import (
    BodySet,
    binding_body_count,
    particle_stable_ids,
    rigid_rotation_velocities,
)

BACKEND = "giant_impact"

# A frame is this fraction of the most massive body's dynamical time. The moon guide is timed
# in dynamical times (it starts at 20), so at this fraction the moon has formed about 400
# frames in: long enough to watch the impact, the disk and the moon, short enough to be a
# shot, and few enough that every frame fits the scrub cache. A longer frame would reach the
# moon in fewer frames, but the pressure substeps grow with it and so does the cost of each
# frame. It is the frame length the verified reference runs used (ADR 0094).
FRAME_DYNAMICAL_FRACTION = 0.1

# Pressure substeps one frame may take before the run is refused. The substep count follows
# the resolution (it grows as N^(1/3)), and past this the frame no longer fits an interactive
# budget at any count the engine is built for.
DEFAULT_MAX_SUBSTEPS_PER_FRAME = 64

# The engine's smoothing length, in particle spacings. Mirrors
# `GiantImpactSettings.smoothing`; kept here so the sampler can refuse an unresolved body
# without importing Warp.
_SMOOTHING = 2.0

# The other two numbers the substep rule reads, mirrored for the same reason: `giant_impact`
# imports Warp at module scope, so compile cannot ask the engine how many substeps a frame
# needs before a device exists. `test_giant_impact_admission` pins all three against the
# engine's own values and pins `pressure_substeps_for` against `giant_impact.substeps_for`,
# so the mirror cannot drift out from under the compile-time check (ADR 0108).
_CFL = 0.6
_EJECTA_SPEED_MARGIN = 1.5


class GiantImpactTimestepError(RuntimeError):
    """Raised when a frame would need more pressure substeps than the ceiling allows."""


def minimum_particles_per_body(smoothing: float = _SMOOTHING) -> int:
    """The fewest particles a body can have and still be a body to the pressure solve.

    One smoothing kernel's worth: a body with fewer particles than one kernel volume holds
    has its whole mass inside a single neighbourhood, so it has no interior for the pressure
    constraint to hold up and it dissolves on the first frame.
    """

    return math.ceil(4.0 / 3.0 * math.pi * smoothing**3)


def body_radius(binding: BindingDescriptor) -> float:
    """The radius a Solid is sampled at: the ball `sampling` would draw it from."""

    if binding.sample_shape == SampleShape.OBJECT_BALL and max(binding.sample_extent) > 0.0:
        return 0.5 * max(float(value) for value in binding.sample_extent)
    return float(binding.radius)


def dynamical_frame_seconds(
    total_mass: float, radius: float, gravitational_constant: float
) -> float:
    """``FRAME_DYNAMICAL_FRACTION`` of the dynamical time ``sqrt(R^3 / (G M_total))``.

    The one formula for a giant-impact frame. Compile calibration and the Moon Birth
    template's plan both call it, so the two cannot disagree about how long a frame is.
    """

    if gravitational_constant <= 0.0 or not math.isfinite(gravitational_constant):
        raise ValueError("a giant-impact scene needs a positive gravitational constant")
    if not (total_mass > 0.0 and radius > 0.0):
        raise ValueError("a giant-impact body needs a positive mass and radius")
    return FRAME_DYNAMICAL_FRACTION * math.sqrt(
        radius**3 / (gravitational_constant * total_mass)
    )


def giant_impact_frame_seconds(
    bindings: tuple[BindingDescriptor, ...], gravitational_constant: float
) -> float:
    """The frame length a giant-impact scene plays at.

    :func:`dynamical_frame_seconds` of the most massive body's radius over the total mass.
    A pure function of the authored bodies, so calibrating a scene costs nothing and a
    changed body re-times it the same way every time.
    """

    bodies = [
        binding
        for binding in bindings
        if binding.active and binding.role == BindingRole.BODY
    ]
    if not bodies:
        raise ValueError("a giant-impact scene needs at least one active body")
    total = sum(float(binding.mass) for binding in bodies)
    largest = max(bodies, key=lambda binding: float(binding.mass))
    return dynamical_frame_seconds(total, body_radius(largest), gravitational_constant)


# The fastest a pair of Solids may meet, as a multiple of their mutual escape speed, and still
# be a giant impact. Moon-forming impacts meet at about 1 to 2.5 times it and hit-and-run
# grazes reach about 3; past that the impactor is a projectile, the target barely notices it
# gravitationally, and the scene is a cratering shot the `impact` backend's material model is
# built for. A heuristic regime boundary, not a physical constant (ADR 0094).
GIANT_IMPACT_SPEED_RATIO = 3.0


def contact_speed(
    gm_total: float, contact_distance: float, distance: float, relative_speed: float
) -> float:
    """Speed at which two bodies ``distance`` apart, closing at ``relative_speed``, meet at
    ``contact_distance`` under their mutual gravity (energy conservation)."""

    v2 = relative_speed**2 + 2.0 * gm_total * (1.0 / contact_distance - 1.0 / distance)
    return math.sqrt(max(v2, 0.0))


def giant_impact_regime(
    bindings: tuple[BindingDescriptor, ...], gravitational_constant: float
) -> bool:
    """Whether a set of Solids is a giant impact: every pair of active bodies meets no faster
    than ``GIANT_IMPACT_SPEED_RATIO`` times its mutual escape speed.

    A lone body is always one: it has no projectile to be a crater for, only its own gravity to
    hold it together. A scene this cannot judge (no gravity, a body without mass or size) is
    not one either, so it keeps the `impact` backend, whose own validation says what is wrong.
    Pure and Warp-free, so the add-on can route a System before any device exists.
    """

    bodies = [
        binding
        for binding in bindings
        if binding.active and binding.role == BindingRole.BODY
    ]
    if not bodies:
        return False
    g = float(gravitational_constant)
    if not (g > 0.0 and math.isfinite(g)):
        return False
    masses = [float(binding.mass) for binding in bodies]
    radii = [body_radius(binding) for binding in bodies]
    if any(not (value > 0.0 and math.isfinite(value)) for value in (*masses, *radii)):
        return False
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            reach = radii[i] + radii[j]
            gm = g * (masses[i] + masses[j])
            distance = max(float(math.dist(bodies[i].position, bodies[j].position)), reach)
            relative = float(math.dist(bodies[i].velocity, bodies[j].velocity))
            escape = math.sqrt(2.0 * gm / reach)
            if contact_speed(gm, reach, distance, relative) > GIANT_IMPACT_SPEED_RATIO * escape:
                return False
    return True


@dataclass(frozen=True)
class GiantImpactConfig:
    """Everything the engine needs that is not derived from the bodies themselves."""

    count: int
    gravitational_constant: float
    seconds_per_frame: float
    # The authored substep count, read as a floor on the pressure substeps. The engine sizes
    # its own from the resolution and the impact speed; an author asking for more gets more.
    minimum_pressure_substeps: int = 1
    precision: Precision = Precision.FLOAT32
    max_substeps_per_frame: int = DEFAULT_MAX_SUBSTEPS_PER_FRAME
    materials: Any = None
    shells: tuple[tuple[str, tuple[tuple[float, int, float], ...]], ...] = ()
    # Fragment linking length, in smoothing lengths, as `astro.diagnostics.fragments` means
    # it everywhere. 0.75 is one and a half particle spacings: neighbours in a settled body
    # are one spacing apart, so this links a body and not the gap between two.
    linking_length_factor: float = 0.75
    # The moon guide. Always on: a giant impact that leaves a ring and no moon is not the
    # shot this backend exists for. Timing overrides are for tests and the reference runs.
    # Mirrors `MoonSettings.start`, which this module cannot import without Warp.
    moon_start: float = 20.0
    moon_end: float = 0.0
    use_graph: bool = True

    @property
    def substeps(self) -> int:
        """Resumable primitives per authored frame: one, a whole engine frame."""

        return 1

    @property
    def shell_map(self) -> dict[str, tuple[tuple[float, int, float], ...]]:
        return {binding_id: layers for binding_id, layers in self.shells}

    def validate(self) -> None:
        if self.count < 1:
            raise ValueError("count must be positive")
        if not math.isfinite(self.gravitational_constant) or self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be positive")
        if not math.isfinite(self.seconds_per_frame) or self.seconds_per_frame <= 0.0:
            raise ValueError("seconds_per_frame must be positive")
        if self.minimum_pressure_substeps < 1:
            raise ValueError("minimum_pressure_substeps must be positive")
        if self.max_substeps_per_frame < 1:
            raise ValueError("max_substeps_per_frame must be positive")
        if self.precision != Precision.FLOAT32:
            # The engine's state is float32 end to end. Accepting float64 would publish a
            # precision label the arrays do not have.
            raise ValueError("the giant-impact backend runs float32 state only")
        if (self.materials is None) != (not self.shells):
            raise ValueError("materials and shells must be configured together")
        if self.linking_length_factor <= 0.0:
            raise ValueError("linking_length_factor must be positive")
        if self.moon_start <= 0.0:
            raise ValueError("moon_start must be positive")


@dataclass(frozen=True)
class GiantImpactBodies:
    """The sampled scene: the descriptor-shaped body set and the engine's view of it.

    ``bodies`` is capacity-long and in binding order, exactly the population the descriptor
    declares. The engine holds only the active particles; ``solver_slots`` says which slot
    each engine particle is, and ``source`` which engine particle each slot shows (an idle
    slot mirrors one of its body's live particles, so it never samples a stale point).
    """

    bodies: BodySet
    solver_slots: Any
    source: Any
    body_ids: Any
    centres: Any
    spacing: float
    characteristic_speed: float
    body_names: tuple[str, ...] = field(default=())

    @property
    def active_count(self) -> int:
        return int(self.solver_slots.shape[0])


def _refuse(binding: BindingDescriptor) -> None:
    name = binding.object_name or binding.binding_id
    if not binding.active:
        raise ValueError(f"{name}: a giant-impact scene cannot hold an inactive body")
    if binding.role != BindingRole.BODY:
        raise ValueError(
            f"{name}: the giant-impact backend simulates Solids only, not a {binding.role.value}"
        )
    if binding.representation != BodyRepresentation.SAMPLED_PARTICLE:
        raise ValueError(f"{name}: a giant-impact body must be a sampled particle body")
    if binding.sample_shape not in (SampleShape.BALL, SampleShape.OBJECT_BALL):
        raise ValueError(
            f"{name}: a giant-impact body is sampled as a ball, not a "
            f"{binding.sample_shape.value}"
        )
    if binding.velocity_model not in (CloudVelocityModel.UNIFORM, CloudVelocityModel.FROM_OBJECT):
        raise ValueError(
            f"{name}: a giant-impact body moves as one piece; "
            f"{binding.velocity_model.value} velocities are not supported"
        )
    if binding.mass_jitter != 0.0:
        raise ValueError(f"{name}: a giant-impact body takes its masses from its layers")
    if body_radius(binding) <= 0.0 or binding.mass <= 0.0:
        raise ValueError(f"{name}: a giant-impact body needs a positive mass and radius")


def _characteristic_speed(
    masses: list[float],
    radii: list[float],
    centres: Any,
    velocities: Any,
    g: float,
    rims: list[float] | None = None,
) -> float:
    """The speed the impact is sized from: the fastest speed any pair of bodies meets at
    under their mutual gravity. The substep rule's ejecta margin covers what the collision
    throws faster than that, so this is not padded again here. A lone body has no pair; its
    own escape speed stands in, the speed its surface material would move at.

    ``rims`` is each body's surface speed from its own spin (ADR 0158). A pair can meet at
    that on top of its bulk approach, because a spinning body's limb may move against the
    other body, so it is added to the pair's relative speed. A lone body is sized by
    whichever is larger, its escape speed or its rim speed."""

    rims = rims or [0.0] * len(masses)
    if len(masses) == 1:
        return max(math.sqrt(2.0 * g * masses[0] / radii[0]), rims[0])
    speed = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            distance = float(math.dist(centres[i], centres[j]))
            relative = float(math.dist(velocities[i], velocities[j])) + rims[i] + rims[j]
            speed = max(
                speed,
                contact_speed(
                    g * (masses[i] + masses[j]),
                    radii[i] + radii[j],
                    max(distance, radii[i] + radii[j]),
                    relative,
                ),
            )
    return speed


def pressure_substeps_for(
    frame_dt: float, smoothing_length: float, characteristic_speed: float, cfl: float = _CFL
) -> int:
    """`giant_impact.substeps_for`, mirrored so compile can size a frame without Warp.

    The engine's own copy stays the one the solver runs on; this is the one the add-on calls
    from a compile that may have no device at all. `test_giant_impact_admission` pins them
    equal, because a compile that sizes a frame differently from Play is the whole defect
    ADR 0108 exists to close.
    """

    v = _EJECTA_SPEED_MARGIN * characteristic_speed
    return max(1, math.ceil(frame_dt * v / (cfl * smoothing_length) - 1.0e-9))


@dataclass(frozen=True)
class GiantImpactScale:
    """The scene's common particle spacing and the speed its impact is sized from.

    The two numbers `sample_giant_impact_bodies` derives before it samples anything, and the
    only two the substep rule reads. Split out so a compile with no device can compute them
    from the authored bindings alone, and so the compile-time check and the load-time one
    cannot disagree about what they are.
    """

    radii: tuple[float, ...]
    volumes: tuple[float, ...]
    spacing: float
    characteristic_speed: float


def giant_impact_layout_scale(
    bindings: tuple[BindingDescriptor, ...], gravitational_constant: float
) -> GiantImpactScale:
    """Derive :class:`GiantImpactScale` from the authored bodies. Pure and Warp-free."""

    if not bindings:
        raise ValueError("a giant-impact scene needs at least one body")
    radii = [body_radius(binding) for binding in bindings]
    counts = [binding_body_count(binding) for binding in bindings]
    masses = [float(binding.mass) for binding in bindings]
    if any(not (value > 0.0 and math.isfinite(value)) for value in (*radii, *masses)):
        raise ValueError("a giant-impact body needs a positive mass and radius")
    if any(count < 1 for count in counts):
        raise ValueError("a giant-impact body needs at least one particle")
    volumes = [4.0 / 3.0 * math.pi * radius**3 for radius in radii]
    spacing = max(
        (volume / count) ** (1.0 / 3.0) for volume, count in zip(volumes, counts, strict=True)
    )
    return GiantImpactScale(
        radii=tuple(radii),
        volumes=tuple(volumes),
        spacing=float(spacing),
        characteristic_speed=_characteristic_speed(
            masses,
            radii,
            [binding.position for binding in bindings],
            [binding.velocity for binding in bindings],
            float(gravitational_constant),
            [
                math.hypot(*binding.spin) * radius
                for binding, radius in zip(bindings, radii, strict=True)
            ],
        ),
    )


def giant_impact_sized_substeps(
    bindings: tuple[BindingDescriptor, ...],
    *,
    seconds_per_frame: float,
    gravitational_constant: float,
) -> int:
    """Pressure substeps a frame of this scene needs, without sampling or a device.

    What `GiantImpactRuntime.load_layout` computes from its sampled layout, computed instead
    from the authored bindings, so compile can ask the question Play asks.
    """

    scale = giant_impact_layout_scale(bindings, gravitational_constant)
    return pressure_substeps_for(
        float(seconds_per_frame), _SMOOTHING * scale.spacing, scale.characteristic_speed
    )


def giant_impact_timestep_refusal(
    *,
    seconds_per_frame: float,
    sized: int,
    minimum_pressure_substeps: int,
    max_substeps_per_frame: int = DEFAULT_MAX_SUBSTEPS_PER_FRAME,
) -> str | None:
    """Why this frame cannot be run, or ``None`` if it can.

    One wording for both the compile-time refusal and the load-time one, so the artist is
    told the same thing wherever the scene is stopped. Which knob is named depends on which
    number is over: an authored Substeps floor above the ceiling is not a resolution problem,
    and telling someone to use fewer particles when the engine sized the frame at two
    substeps sends them to rebuild a scene that was never the fault (ADR 0108).
    """

    sized = int(sized)
    floor = int(minimum_pressure_substeps)
    ceiling = int(max_substeps_per_frame)
    if max(sized, floor) <= ceiling:
        return None
    if floor > ceiling:
        return (
            f"Substeps is set to {floor}, above the ceiling of {ceiling} pressure substeps a "
            f"frame; this resolution and impact speed need only {sized}. Lower Substeps"
        )
    return (
        f"a frame of {seconds_per_frame:g} s needs {sized} pressure substeps at this "
        f"resolution and impact speed, above the ceiling of {ceiling}; shorten Seconds per "
        f"Frame or use fewer particles"
    )


def sample_giant_impact_bodies(
    bindings: tuple[BindingDescriptor, ...],
    shells: dict[str, tuple[tuple[float, int, float], ...]] | None = None,
    *,
    gravitational_constant: float,
) -> GiantImpactBodies:
    """Sample every body at one common spacing, and lay the particles out by binding.

    Sampling is vectorised host work done once at load, followed by one copy to the device:
    the same boundary `sampling` keeps (ARCH-005).
    """

    np = __import__("numpy")
    from .giant_impact import lattice_ball

    shells = shells or {}
    if not bindings:
        raise ValueError("a giant-impact scene needs at least one body")
    for binding in bindings:
        _refuse(binding)
    counts = [binding_body_count(binding) for binding in bindings]
    # The same derivation compile's admission check runs, so the spacing and speed this scene
    # is sampled at are the ones the frame was admitted at (ADR 0108).
    scale = giant_impact_layout_scale(bindings, gravitational_constant)
    radii = list(scale.radii)
    volumes = list(scale.volumes)
    spacing = scale.spacing
    floor = minimum_particles_per_body()
    has_shells = bool(shells)

    positions, velocities, masses, material_ids = [], [], [], []
    slots, source, body_ids, stable_ids, active = [], [], [], [], []
    slot = 0
    live = 0
    for ordinal, (binding, radius, count) in enumerate(
        zip(bindings, radii, counts, strict=True)
    ):
        name = binding.object_name or binding.binding_id
        rng = np.random.default_rng(
            [int(binding.sampling_seed), runtime_stable_id(binding.binding_id)]
        )
        lattice = lattice_ball(radius, spacing, rng)
        order = np.argsort(np.linalg.norm(lattice, axis=1), kind="stable")
        used = min(count, int(round(volumes[ordinal] / spacing**3)), int(lattice.shape[0]))
        if used < floor:
            raise ValueError(
                f"{name}: only {used} particles fit it at the scene's common spacing, and a "
                f"body needs at least {floor}; give the most coarsely sampled body more "
                f"particles"
            )
        local = lattice[order[:used]]
        radial = np.linalg.norm(local, axis=1) / radius
        layers = shells.get(binding.binding_id)
        if layers:
            outer = np.asarray([float(layer[0]) for layer in layers])
            index = np.minimum(np.searchsorted(outer, radial, side="left"), len(layers) - 1)
            density = np.asarray([float(layer[2]) for layer in layers])[index]
            ids = np.asarray([int(layer[1]) for layer in layers], dtype=np.int64)[index]
        else:
            density = np.ones(used)
            ids = np.full(used, -1, dtype=np.int64)
        weights = density / density.sum() * float(binding.mass)

        full = np.zeros((count, 3))
        full[:used] = local
        # An idle slot sits on the live particle it mirrors, so a consumer that ignores
        # `active` draws a doubled point rather than one at the origin.
        mirror = np.arange(count) % used
        full[used:] = local[mirror[used:]]
        positions.append(full + np.asarray(binding.position, dtype=np.float64))
        bulk = np.broadcast_to(np.asarray(binding.velocity, dtype=np.float64), (count, 3))
        if any(binding.spin):
            # ADR 0158: the rigid turn of the body about its own centre of mass, taken over
            # the live particles and copied onto the idle slots that mirror them.
            turn = rigid_rotation_velocities(local, weights, binding.spin, np)
            bulk = bulk + turn[mirror]
        velocities.append(bulk)
        mass_row = np.zeros(count)
        mass_row[:used] = weights
        masses.append(mass_row)
        material_ids.append(ids[mirror])
        flags = np.zeros(count, dtype=np.int32)
        flags[:used] = 1
        active.append(flags)
        stable_ids.append(particle_stable_ids(binding.binding_id, count))
        slots.append(slot + np.arange(used, dtype=np.int64))
        source.append(live + mirror.astype(np.int32))
        body_ids.append(np.full(used, ordinal, dtype=np.int32))
        slot += count
        live += used

    stacked_positions = np.concatenate(positions)
    particle_radius = spacing * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    bodies = BodySet(
        positions=stacked_positions,
        velocities=np.ascontiguousarray(np.concatenate(velocities)),
        masses=np.concatenate(masses),
        radii=np.full(slot, particle_radius),
        stable_ids=np.concatenate(stable_ids),
        active=np.concatenate(active),
        material_ids=np.concatenate(material_ids) if has_shells else None,
    )
    centres = np.asarray([binding.position for binding in bindings], dtype=np.float64)
    return GiantImpactBodies(
        bodies=bodies,
        solver_slots=np.concatenate(slots),
        source=np.concatenate(source).astype(np.int32),
        body_ids=np.concatenate(body_ids),
        centres=centres,
        spacing=float(spacing),
        characteristic_speed=scale.characteristic_speed,
        body_names=tuple(binding.object_name or binding.binding_id for binding in bindings),
    )


@dataclass(frozen=True)
class GiantImpactDiagnostics:
    """The engine's scalars in one flat record (serialised with `dataclasses.asdict`)."""

    active_count: int
    idle_count: int
    particle_spacing: float
    smoothing_length: float
    body_radius: float
    dynamical_time: float
    timestep: float
    pressure_substeps: int
    frames_advanced: int
    simulation_time: float
    step_ms: float
    moon_guide_started: bool
    moon_orbit_radius: float
    submission: str


def _kernels(wp: Any) -> Any:
    @wp.kernel
    def publish(
        source: wp.array(dtype=wp.int32),
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        heat: wp.array(dtype=float),
        mass: wp.array(dtype=float),
        number_density: wp.array(dtype=float),
        out_x: wp.array(dtype=wp.vec3),
        out_v: wp.array(dtype=wp.vec3),
        out_e: wp.array(dtype=float),
        out_rho: wp.array(dtype=float),
    ):
        i = wp.tid()
        j = source[i]
        out_x[i] = x[j]
        out_v[i] = v[j]
        out_e[i] = heat[j]
        # The engine's ``dens`` is a poly6 sum with no mass in it (a number density, one over
        # length cubed). Times the particle's own mass it is the mass density of the material
        # the particle is made of, so a heavy core particle reads denser than a mantle one, and
        # it means the same thing the rock engine's density does.
        out_rho[i] = mass[j] * number_density[j]

    return publish


class GiantImpactRuntime(ImpactFrameRuntime):
    """One giant-impact scene on the device, played one whole frame at a time."""

    _ENGINE_LABEL = "GiantImpactRuntime"
    # ``internal_energies`` is the engine's heat: how hard a particle has been shocked, cooling
    # over time. It is not a thermodynamic energy, but it is what the Energy colour mode is for,
    # and it is what makes the impact glow. ``densities`` is the mass density, ``m * dens`` (see
    # ``publish``), from the pressure solve's last density pass, so it needs no pass of its own.
    _DISPLAY_CHANNELS = (
        "positions",
        "velocities",
        "masses",
        "radii",
        "active",
        "stable_ids",
        "internal_energies",
        "densities",
        "material_ids",
    )
    _LIVE_STATE_READBACKS = 3

    def __init__(self, device_context: DeviceContext, config: GiantImpactConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.frames_advanced = 0
        self.last_step_ms = 0.0
        self.last_solver_primitive_ms = 0.0
        self.explicit_snapshot_readbacks = 0
        self.full_state_playback_readbacks = 0
        self._layout: GiantImpactBodies | None = None
        self._solver: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._publish: Any | None = None
        self._frame_started_at: float | None = None

    # -- loading ------------------------------------------------------------------------

    def load_bindings(self, bindings: tuple[BindingDescriptor, ...]) -> GiantImpactBodies:
        layout = sample_giant_impact_bodies(
            bindings,
            self.config.shell_map or None,
            gravitational_constant=self.config.gravitational_constant,
        )
        self.load_layout(layout)
        return layout

    def load_layout(self, layout: GiantImpactBodies) -> None:
        if len(layout.bodies) != self.config.count:
            raise ValueError("body count must match GiantImpactConfig.count")
        from . import giant_impact as engine

        self.device_context.initialize()
        wp = self.device_context.wp
        np = __import__("numpy")
        device = self.device_context.device
        bodies = layout.bodies
        live = layout.solver_slots
        settings = replace(
            engine.GiantImpactSettings(),
            frame_dt=float(self.config.seconds_per_frame),
            smoothing=_SMOOTHING,
            moon=engine.MoonSettings(start=self.config.moon_start, end=self.config.moon_end),
        )
        # `pressure_substeps_for` rather than `engine.substeps_for`: compile runs the same
        # call on the same numbers to admit the scene before it reports READY, and one
        # function is the only way the two cannot drift apart (ADR 0108).
        sized = pressure_substeps_for(
            settings.frame_dt,
            settings.smoothing * layout.spacing,
            layout.characteristic_speed,
            settings.cfl,
        )
        refusal = giant_impact_timestep_refusal(
            seconds_per_frame=self.config.seconds_per_frame,
            sized=sized,
            minimum_pressure_substeps=self.config.minimum_pressure_substeps,
            max_substeps_per_frame=self.config.max_substeps_per_frame,
        )
        if refusal is not None:
            raise GiantImpactTimestepError(refusal)
        substeps = max(sized, int(self.config.minimum_pressure_substeps))
        self.close()
        self._solver = engine.GiantImpactSolver(
            bodies.positions[live],
            bodies.velocities[live],
            bodies.masses[live],
            layout.body_ids,
            spacing=layout.spacing,
            characteristic_speed=layout.characteristic_speed,
            gravitational_constant=self.config.gravitational_constant,
            settings=replace(settings, substeps=substeps),
            device=device,
            use_graph=self.config.use_graph,
        )
        self._solver.settle(centres=layout.centres)
        self._layout = layout
        self._publish = _kernels(wp)
        count = self.config.count
        self._arrays = {
            "positions": wp.zeros(count, dtype=wp.vec3, device=device),
            "velocities": wp.zeros(count, dtype=wp.vec3, device=device),
            "internal_energies": wp.zeros(count, dtype=float, device=device),
            "densities": wp.zeros(count, dtype=float, device=device),
            "masses": wp.array(bodies.masses.astype(np.float32), dtype=float, device=device),
            "radii": wp.array(bodies.radii.astype(np.float32), dtype=float, device=device),
            "active": wp.array(bodies.active.astype(np.int32), dtype=wp.int32, device=device),
            "stable_ids": wp.array(bodies.stable_ids, dtype=wp.int64, device=device),
            "source": wp.array(layout.source, dtype=wp.int32, device=device),
        }
        if bodies.material_ids is not None:
            self._arrays["material_ids"] = wp.array(
                np.asarray(bodies.material_ids, dtype=np.int64), dtype=wp.int64, device=device
            )
        self._publish_state()
        wp.synchronize_device(device)
        self.frames_advanced = 0
        self._frame_started_at = None

    def _publish_state(self) -> None:
        solver = self._require_solver()
        wp = self.device_context.wp
        a = self._arrays
        wp.launch(
            self._publish,
            dim=self.config.count,
            inputs=[a["source"], solver.x, solver.v, solver.heat, solver.m, solver.dens],
            outputs=[a["positions"], a["velocities"], a["internal_energies"], a["densities"]],
            device=self.device_context.device,
        )

    def snapshot(self) -> dict[str, Any]:
        np = __import__("numpy")
        solver = self._require_solver()
        layout = self._layout
        assert layout is not None
        bodies = layout.bodies
        state = solver.state(include_acceleration=True)
        self.explicit_snapshot_readbacks += 4
        source = np.asarray(layout.source, dtype=np.int64)
        positions = state["positions"][source]
        velocities = state["velocities"][source]
        heat = state["heat"][source]
        accelerations = state["accelerations"][source]
        records = []
        for index in range(self.config.count):
            record = {
                "index": index,
                "stable_id": int(bodies.stable_ids[index]),
                "active": bool(bodies.active[index]),
                "mass": float(bodies.masses[index]),
                "radius": float(bodies.radii[index]),
                "position": [float(value) for value in positions[index]],
                "velocity": [float(value) for value in velocities[index]],
                # The acceleration the next frame opens with. Stored, not rebuilt: the moon
                # guide's share of it depends on a velocity the snapshot no longer holds.
                "acceleration": [float(value) for value in accelerations[index]],
                "internal_energy": float(heat[index]),
            }
            if bodies.material_ids is not None:
                record["material_id"] = int(bodies.material_ids[index])
            records.append(record)
        return {
            "schema": CHECKPOINT_SCHEMA_VERSION,
            "backend": BACKEND,
            "capacity": self.config.count,
            "count": self.config.count,
            "collision_mode": "ignore",
            "step": self.frames_advanced,
            "simulation_time": self.frames_advanced * self.config.seconds_per_frame,
            # The engine's state beyond the particles: its frame clock, the barycentre it
            # steers about, and the moon's orbit once the guide has read it off the debris.
            "engine": solver.checkpoint(),
            "recomputed_on_restore": ["neighbour_grid"],
            "bodies": records,
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        np = __import__("numpy")
        solver = self._require_solver()
        layout = self._layout
        assert layout is not None
        if snapshot.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint snapshot schema")
        if snapshot.get("backend") != BACKEND:
            raise ValueError("checkpoint backend does not match the giant-impact runtime")
        records: list[dict[str, Any]] = list(snapshot.get("bodies") or ())
        if int(snapshot.get("count", -1)) != self.config.count or len(records) != self.config.count:
            raise ValueError("snapshot body count does not match runtime")
        if [int(record.get("index", -1)) for record in records] != list(range(self.config.count)):
            raise ValueError("snapshot body ordering is invalid")
        if [int(record["stable_id"]) for record in records] != [
            int(value) for value in layout.bodies.stable_ids
        ]:
            raise ValueError("snapshot stable IDs do not match runtime ordering")
        if [bool(record["active"]) for record in records] != [
            bool(value) for value in layout.bodies.active
        ]:
            raise ValueError("snapshot idle slots do not match runtime layout")
        engine = snapshot.get("engine")
        if not isinstance(engine, dict):
            raise ValueError("snapshot carries no engine checkpoint")
        live = layout.solver_slots
        if any("acceleration" not in records[i] for i in live):
            raise ValueError("snapshot carries no opening accelerations")
        solver.restore(
            np.asarray([records[i]["position"] for i in live], dtype=np.float64),
            np.asarray([records[i]["velocity"] for i in live], dtype=np.float64),
            np.asarray([records[i]["internal_energy"] for i in live], dtype=np.float64),
            engine,
            accelerations=np.asarray([records[i]["acceleration"] for i in live], dtype=np.float64),
        )
        self._publish_state()
        self.device_context.wp.synchronize_device(self.device_context.device)
        self.frames_advanced = int(snapshot.get("step", 0))
        self._frame_started_at = None
        self.last_step_ms = 0.0

    def diagnostics(self) -> GiantImpactDiagnostics:
        solver = self._require_solver()
        layout = self._layout
        assert layout is not None
        orbit = solver.orbit or {}
        return GiantImpactDiagnostics(
            active_count=layout.active_count,
            idle_count=self.config.count - layout.active_count,
            particle_spacing=float(layout.spacing),
            smoothing_length=float(solver.h),
            body_radius=float(solver.body_radius),
            dynamical_time=float(solver.dynamical_time),
            timestep=float(solver.dt),
            pressure_substeps=int(solver.substeps),
            frames_advanced=self.frames_advanced,
            simulation_time=self.frames_advanced * self.config.seconds_per_frame,
            step_ms=self.last_step_ms,
            moon_guide_started=solver.orbit is not None,
            moon_orbit_radius=float(orbit.get("radius", 0.0)),
            submission="cuda_graph" if solver._use_graph else "eager",
        )

    def close(self) -> None:
        if self._solver is not None and self.device_context.state.name == "READY":
            self.device_context.wp.synchronize_device(self.device_context.device)
        self._solver = None
        self._layout = None
        self._arrays = {}
