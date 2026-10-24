"""Pure scenario-authoring calculations shared by Blender-facing operators."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .advisor import (
    TimelineFit,
    circumbinary_critical_radius_ratio,
    fit_timeline,
    pair_orbit,
    suggest_cloud_softening,
    suggest_discreteness_softening,
    suggest_neighbour_softening,
    suggest_orbit_substeps,
    suggest_orbital_softening,
    suggest_softening,
    suggest_substeps,
)
from .giant_impact_runtime import dynamical_frame_seconds
from .impact import (
    DEFAULT_IMPACT_SHELL_LAYOUT,
    ShellSpec,
    layered_body_mass,
    predict_encounter,
)
from .materials import PRESETS_BY_KEY
from .model import DEFAULT_TREE_THETA, TREE_ACCURACY_CERTIFIED_MAX_THETA
from .sampling import annulus_volume
from .timeline_stages import TimelineStage, TimelineStageKind, validate_stage_list
from .unit_presets import UnitScale

Vector3 = tuple[float, float, float]

AuthoringValue = str | int | float | bool | tuple[float, float, float]


@dataclass(frozen=True)
class EncounterInitialState:
    """A two-body state whose point-mass conic reaches requested contact conditions."""

    position_a: Vector3
    position_b: Vector3
    velocity_a: Vector3
    velocity_b: Vector3
    relative_position: Vector3
    relative_velocity: Vector3
    contact_time: float
    contact_radius: float
    contact_speed: float
    contact_angle_radians: float


@dataclass(frozen=True)
class CloudGravityInput:
    """Validated-by-consumer gravity facts for one authored cloud."""

    persistent_id: str
    mass: float
    radius: float


@dataclass(frozen=True)
class DistributionTiming:
    """Distribution frame timing and the cloud that controls it."""

    seconds_per_frame: float
    controlling_persistent_id: str


@dataclass(frozen=True)
class MemberFacts:
    """Everything authoring reads about one selected object, read once.

    This is the pure input record the Blender-facing candidate resolver gathers
    from evaluated object state without mutating the object.  ``role`` is the
    object's existing role string (``'NONE'`` when unassigned; the canonical
    sentinel from ``authoring.ROLES`` -- any other assigned role outside
    Body/Cloud fails closed in the builders), ``object_type`` is
    ``'MESH'`` or ``'EMPTY'``, and the physical values are already in code units
    except the SI-authoring fields ``particle_delta_m``, ``spin_rad_s`` (SI
    rad/s), and ``temperature_k``.
    """

    candidate_key: str
    persistent_id: str
    existing_persistent_id: str | None
    selection_ordinal: int
    name: str
    role: str
    object_type: str
    position: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    mass: float
    radius: float
    material_key: str = "synthetic-rock"
    resolution_mode: str = "TARGET_COUNT"
    particle_delta_m: float = 1000.0
    spin_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temperature_k: float = 0.0


@dataclass(frozen=True)
class MemberPlan:
    """One member's authored plan: solver inputs plus persisted settings keys.

    ``shell_layout`` entries are ``(material_key, material_id, outer_radius_fraction)``.
    ``settings`` carries the persisted solver inputs the application seam writes onto
    the role; its keys mirror the role's persisted Solid/Cloud field names, so nothing
    here invents a solver field the role surface cannot store.
    """

    persistent_id: str
    role: str
    mass: float
    radius: float
    velocity: tuple[float, float, float]
    particle_count: int
    sample_shape: str
    material_key: str
    shell_layout: tuple[tuple[str, int, float], ...]
    settings: tuple[tuple[str, AuthoringValue], ...]


OUTER_MATERIAL_IDS: dict[str, int] = {
    "synthetic-rock": 1,
    "synthetic-ice": 3,
    "synthetic-iron": 4,
    "synthetic-vaporizing-rock": 5,
}
"""System-wide outer-shell material ids; the core Synthetic Iron is always id 2."""


#: Every value `color_source` accepts on a context row (`properties.py:321-332`). Restated
#: here rather than imported because this module is deliberately `bpy`-free; the two are held
#: together by `tests/unit/test_scenario_look.py`, which reads the real annotation.
COLOR_SOURCE_KEYS = (
    "FLAT",
    "MATERIAL",
    "SPEED",
    "DENSITY",
    "SOURCE_OBJECT",
    "OBJECT_COLOR",
    "DISTANCE",
    "DAMAGE",
    "ENERGY",
)


@dataclass(frozen=True)
class LookPlan:
    """How a scenario is drawn *while it plays* --- a different question from ``color_by``.

    ``color_by`` is the baked-output path's authority: `_baked_color_values` resolves it out
    of a completed bake into the ``astro_color_value`` attribute a display group's material
    reads. Nothing on that path draws anything during playback. What the GPU overlay draws is
    `draw.RenderSpec`, built by name off the per-context row, and until ADR 0080 no template
    authored a single field of it --- so every demo played at the row's defaults, flat blue at
    four screen pixels, whatever it was a demo *of*.

    This is the template's half of that row. Deliberately only the colour: `particle_size`,
    `particle_size_source`, `render_mode` and `blend_mode` stay unseeded because ADR 0078
    decision 7 reserves the size to its own record and decision 8 reserves every measured
    number to a windowed sweep, and a size chosen in an argument is not evidence.
    """

    color_source: str = "FLAT"
    #: Equal bounds mean auto-range (`properties.py:344-351`) --- normalise over whatever the
    #: frame actually holds. That is the only bound that stays correct when a template's count
    #: or extent moves under it, which is why it is the shipped pair rather than two numbers.
    ramp_low: float = 0.0
    ramp_high: float = 0.0
    # How big each particle is drawn while the scene plays, in the units the render mode
    # reads: POINT reads ``max(1, size * 10)`` screen pixels.
    #
    # ``0.0`` means the scenario authors no size, and is not a size of zero --- a scenario
    # that names none draws at whatever the artist's Viewport Display tier says, which is
    # the right answer for a scenario whose readability does not depend on the number.
    # ADR 0078 decision 7 reserved this field to a record of its own and decision 8 reserved
    # the number in it to a windowed sweep. Both are now paid: ADR 0091 swept 0.4/0.8/1.5/3.0
    # across all five shipped templates in twenty windows and authored the three sizes that
    # sweep measured; ADR 0105 re-swept Galaxy Collision on pictures of the moment it is
    # named for and authored the fourth. A value still arrives here only that way --- a size
    # chosen in an argument is not evidence --- and one of the five templates deliberately
    # still carries none, because a picture taken at a frame nobody chose cannot choose a
    # size either.
    particle_size: float = 0.0

    def __post_init__(self) -> None:
        if self.color_source not in COLOR_SOURCE_KEYS:
            raise ValueError(
                f"color_source '{self.color_source}' is not one of "
                f"{sorted(COLOR_SOURCE_KEYS)}"
            )
        for name in ("ramp_low", "ramp_high", "particle_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        # A ramp bound is allowed to be negative --- it is a normalisation window over
        # whatever channel a later source names, and equal bounds already mean auto-range.
        # A size is not: it is written to a `bpy.props.FloatProperty` with ``min=0.0``,
        # which clamps in silence, so a negative authored here would ship as 0.0 --- as
        # *unauthored* --- while the source that produced it still read as authored.
        if self.particle_size < 0.0:
            raise ValueError("particle_size must be zero or positive")


#: The look that changes nothing: exactly the context row's own shipped defaults, so a plan
#: carrying it plays precisely as every plan did before ADR 0080. It is *not* a field default
#: --- ``ScenarioPlan`` authors every field or none, and a caller with no opinion about the
#: live look has to say so by naming this constant.
DEFAULT_LOOK = LookPlan()

#: What a template authors instead. Not a new choice --- every builder has declared
#: ``color_by="SPEED"`` since Slice 3; this is that same declaration reaching the path that
#: actually plays. Servable today on both gravity lanes: `display._RAMP_FIELDS` maps SPEED to
#: the ``velocities`` channel, and `DirectNBodyRuntime` and `BarnesHutRuntime` both offer it.
SPEED_LOOK = LookPlan(color_source="SPEED")


def look_for_color_by(color_by: str, particle_size: float = 0.0) -> LookPlan:
    """The live look serving a plan's declared ``color_by``, where the live path has one.

    The two enums are not the same enum and must not be mapped as if they were.
    ``color_by`` is D13's five modes over baked sources; ``color_source`` is what the GPU
    overlay can resolve from a device channel it is actually offered. Only ``SPEED`` names
    the same physical quantity on both sides *and* is servable on every live lane, so it is
    the only entry here. ``DAMAGE`` and ``ENERGY`` now have live channels (ADR 0236), but only
    the rock-impact (and, for energy, giant-impact) engines publish them, so a plan on any
    other lane would play one flat colour: they are not mapped. ``TEMPERATURE`` has no live
    channel at all; ``MATERIAL`` names a persistent material id on one side and a bound Blender
    material on the other, which is a resemblance rather than a match. Everything but SPEED
    therefore keeps the default look, which is what those plans play as today anyway --- an
    unauthored look is an honest one, a wrong mapping is not.

    ``particle_size`` is why this is not `swept_look` with an extra branch. The four
    templates that call `swept_look` pin ``color_by="SPEED"`` on the plan as well, so a look
    that hardcodes ``color_source="SPEED"`` states what they already are; the two impact
    planners carry the *caller's* ``color_by``, and handing them `swept_look` would silently
    override a caller who asked for another mode. A swept size is measured on one picture,
    and that picture had a colour: it rides the arm it was photographed on and no other. A
    caller on a different arm gets `DEFAULT_LOOK` exactly as before --- unsized, because the
    sweep measured nothing about that picture --- which is also what keeps that constant's
    "the look that changes nothing" contract true. Defaulted to 0.0 so a caller with no
    measured size still gets the two singletons by value.
    """

    if color_by != "SPEED":
        return DEFAULT_LOOK
    return SPEED_LOOK if particle_size == 0.0 else swept_look(particle_size)


def swept_look(particle_size: float) -> LookPlan:
    """The authored look for a template whose size a windowed sweep has actually measured.

    A factory rather than more module-level constants, because the four templates that
    carry a size measured it *separately* and agreed by coincidence rather than by
    construction. A shared `LOOK_0P8` would encode that coincidence as a fact and would
    have to be unpicked the first time one template's next sweep moves. The fourth is the
    case in point: Galaxy Collision was swept two ADRs and six weeks after the other three
    (ADR 0105), against a different scene at a different moment, and arrived at the same
    number by its own measurement --- with a disc share at 0.8 of 24 per cent against their
    45 to 50, which is agreement about the size and not about the picture. `SPEED_LOOK` is
    the cautionary case: one singleton naming what eight builders had in common, which is
    exactly why it cannot carry a per-template number now.

    The size reaches the drawn picture through `product_registry._seed_live_look`, which
    seeds it onto the context row --- so an authored template keeps the look it was
    measured at when the artist moves the scene tier. Withholding a size does *not* hand
    the template back to that tier: the same seed ticks ``use_display_override`` whenever
    the colour source is not ``FLAT``, and every template here authors ``SPEED``, so a
    template-built row overrides the scene whether or not it names a size. What a template
    that authors nothing draws is its context row's own ``particle_size`` default, 0.4.
    Withholding is still a real choice rather than an omission --- it declines to state a
    measured number for a scene whose picture is of the wrong moment (ADR 0091) --- but it
    is not a choice about which tier wins, and the two are easy to confuse because the row
    default and the scene default are both 0.4.
    """

    return LookPlan(color_source="SPEED", particle_size=particle_size)


@dataclass(frozen=True)
class ImpactOptions:
    """The four addendum-visible Impact operator properties."""

    impact_speed_km_s: float = 5.0
    impact_angle_degrees: float = 45.0
    fragments_k: float = 10.0
    color_by: str = "SPEED"


@dataclass(frozen=True)
class OrbitOptions:
    """The one visible Orbit operator property: eccentricity in [0.0, 0.95]."""

    eccentricity: float = 0.0


@dataclass(frozen=True)
class DistributionOptions:
    """The four visible Distribution operator properties."""

    shape: str = "OBJECT_VOLUME"
    count_per_object: int = 4096
    total_mass_per_object: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class ScenarioPlan:
    """One fully-derived, validated, application-ready scenario plan."""

    unit_preset: str
    solver: str
    color_by: str
    seconds_per_frame: float
    softening: float
    substeps: int
    members: tuple[MemberPlan, ...]
    timeline_stages: tuple[TimelineStage, ...]
    display_groups: tuple[tuple[str, str], ...]
    result_frame_end: int
    #: How the scene is drawn while it plays (ADR 0080). ``display_groups`` above is the
    #: other half of the same question and answers it for the *baked* output only; the two
    #: are separate fields because they are consumed by two paths that share no property.
    #: Undefaulted like every other field on this dataclass: a plan authors the whole
    #: scenario or none of it, and "this one plays as it always did" is a decision a builder
    #: states by naming `DEFAULT_LOOK`, not one it gets by omission.
    look: LookPlan
    #: The Barnes-Hut opening angle the System is authored with (ADR 0136). It is read only
    #: when the solver the System plays is the tree, which under ``AUTO`` is decided at Play;
    #: a template that has no reason to differ names `DEFAULT_TREE_THETA`, the System's own
    #: default, so its compiled bytes are the ones it had before the field existed.
    tree_theta: float

    def __post_init__(self) -> None:
        theta = _finite_scalar(self.tree_theta, "tree_theta")
        if not 0.0 < theta <= TREE_ACCURACY_CERTIFIED_MAX_THETA:
            # Above the certified angle the Auto accuracy bound 0.1 * theta**2 is no longer a
            # measured claim (GRAV-002), so a template may not author past it.
            raise ValueError(
                f"tree_theta must be within (0, {TREE_ACCURACY_CERTIFIED_MAX_THETA}]"
            )


# ``DEFAULT_TREE_THETA`` (imported above from ``model``) is the System's own Barnes-Hut opening
# angle, which every template but one authors.


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(value: Any, name: str) -> Vector3:
    try:
        if len(value) != 3:
            raise ValueError(f"{name} must be a finite three-vector")
        result = tuple(_finite_scalar(component, name) for component in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite three-vector") from exc
    return result  # type: ignore[return-value]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Vector3) -> float:
    return math.hypot(*vector)


def _scaled_close(actual: float, expected: float) -> bool:
    tolerance = 2048.0 * sys.float_info.epsilon
    absolute = 2048.0 * math.ulp(abs(expected))
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=absolute)


#: Where continuous first contact lands inside a template's authored frame range.
#:
#: An impact has two natural timescales and they are three orders of magnitude apart. The
#: approach is the one the operator authored -- a 200 km separation closing at 5 km/s takes
#: 31.5 s -- and the settling of the debris the collision throws off is the other, 8.9
#: hours for that same pair. One constant frame duration cannot resolve both, so a template
#: has to choose which of the two its authored range is for, and it chooses the approach,
#: because the approach is the half the operator's own properties actually set. The
#: remaining three quarters carry the collision and its spray, which at 5 km/s crosses nine
#: target radii before the range ends.
#:
#: Framing is where this number comes from and there is no physics that fixes it. A tenth
#: (contact in the first second at 24 fps) since ADR 0152: a quarter spent most of the clip
#: on two bodies converging, and left the settling boundary beyond the range. (The 5 km/s,
#: 31.5 s and 8.9 hour figures above are the pre-ADR 0152 defaults.) It is stated here, once,
#: rather than spread across the planner.
TEMPLATE_CONTACT_RANGE_FRACTION = 0.1


def template_contact_interval(*, frame_start: int, frame_end: int) -> int:
    """Return how many frames a template spends approaching contact within its range.

    The result always leaves room behind it for the two later stage boundaries, so a
    caller that places Impact here can still place Reaccumulate strictly after it and
    still finish strictly inside the authored range.
    """

    _require_frame_range(frame_start, frame_end)
    span = frame_end - frame_start
    if span < 3:
        raise ValueError(
            "an impact needs at least four frames to author its three stages inside the "
            f"range: {frame_start}..{frame_end} is {span + 1} frames"
        )
    return max(1, min(span - 2, math.ceil(TEMPLATE_CONTACT_RANGE_FRACTION * span)))


def template_impact_seconds_per_frame(
    *, contact_time: float, frame_start: int, frame_end: int
) -> float:
    """Return the frame duration that lands first contact inside the authored range.

    This is the impact counterpart of :func:`orbit_timing`, which already fits one derived
    orbital period across the frames the operator authored. Templates and Quick Impact
    both preserve that range; neither uses a fixed physical duration per frame.
    """

    time = _positive_scalar(contact_time, "contact_time")
    interval = template_contact_interval(frame_start=frame_start, frame_end=frame_end)
    step = time / interval
    # ``interval * (time / interval)`` can round one ulp below ``time``, and
    # :func:`contact_boundary_frame` then reads first contact as just after the sampled
    # interval and puts it one frame late (a 14 km impactor on a 50 km target, 200 km apart
    # at 0.2 km/s, landed on frame 27 rather than 26). Nudge up until the product reaches it.
    while interval * step < time:
        step = math.nextafter(step, math.inf)
    return step


def orbit_timing(period: float, frame_start: int, frame_end: int) -> TimelineFit:
    """Fit exactly one caller-derived orbital period over the authored frame intervals."""

    orbital_period = _positive_scalar(period, "period")
    if isinstance(frame_start, bool) or not isinstance(frame_start, int):
        raise ValueError("frame_start must be an integer")
    if isinstance(frame_end, bool) or not isinstance(frame_end, int):
        raise ValueError("frame_end must be an integer")
    if frame_end <= frame_start:
        raise ValueError("frame_end must be after frame_start")
    return fit_timeline(orbital_period, 1, frame_start, frame_end - frame_start)


def reaccumulate_boundary_frame(
    *,
    contact_frame: int,
    contact_speed: float,
    total_mass: float,
    contact_radius: float,
    gravitational_constant: float,
    seconds_per_frame: float,
    maximum_boundary_frame: int,
    clamp_to_maximum: bool = False,
) -> int:
    """Return the conservative authored reaccumulation boundary from D6.

    This is a pure estimate from already-verified contact facts.  It neither advances a
    solver nor evaluates the runtime fragment-settling diagnostic. Authoring may clamp
    the diagnostic checkpoint to its range; calibration keeps the strict forecast.
    """

    if isinstance(contact_frame, bool) or not isinstance(contact_frame, int):
        raise ValueError("contact_frame must be an integer")
    if isinstance(maximum_boundary_frame, bool) or not isinstance(maximum_boundary_frame, int):
        raise ValueError("maximum_boundary_frame must be an integer")
    if maximum_boundary_frame <= contact_frame:
        raise ValueError("maximum_boundary_frame must be after contact_frame")

    speed = _positive_scalar(contact_speed, "contact_speed")
    mass = _positive_scalar(total_mass, "total_mass")
    radius = _positive_scalar(contact_radius, "contact_radius")
    gravity = _positive_scalar(gravitational_constant, "gravitational_constant")
    frame_duration = _positive_scalar(seconds_per_frame, "seconds_per_frame")

    try:
        escape_speed = math.sqrt(2.0 * gravity * mass / radius)
        sigma = min(0.10 * speed, 0.90 * escape_speed)
        semi_major_axis = 1.0 / (2.0 / radius - sigma**2 / (gravity * mass))
        return_time = math.pi * math.sqrt(semi_major_axis**3 / (gravity * mass))
        dynamical_time = math.sqrt(radius**3 / (gravity * mass))
        frame_count = max(1, math.ceil(2.0 * max(return_time, dynamical_time) / frame_duration))
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("reaccumulate boundary is outside float range") from exc
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (escape_speed, sigma, semi_major_axis, return_time, dynamical_time)
    ):
        raise ValueError("reaccumulate boundary is outside float range")

    boundary_frame = contact_frame + frame_count
    if boundary_frame > maximum_boundary_frame:
        if clamp_to_maximum:
            return maximum_boundary_frame
        raise ValueError("the reaccumulate boundary exceeds maximum_boundary_frame")
    return boundary_frame


def _distribution_seconds_per_frame(radius: float, gravity: float, mass: float) -> float:
    """Evaluate the free-fall expression without avoidable intermediate range loss."""

    radius_mantissa, radius_exponent = math.frexp(radius)
    gravity_mantissa, gravity_exponent = math.frexp(gravity)
    mass_mantissa, mass_exponent = math.frexp(mass)
    coefficient = radius_mantissa**3 / (8.0 * gravity_mantissa * mass_mantissa)
    exponent = 3 * radius_exponent - gravity_exponent - mass_exponent
    if exponent % 2:
        coefficient *= 2.0
        exponent -= 1
    scaled_root = (math.pi / 250.0) * math.sqrt(coefficient)
    try:
        result = math.ldexp(scaled_root, exponent // 2)
    except OverflowError as exc:
        raise ValueError("distribution seconds_per_frame is outside float range") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("distribution seconds_per_frame is outside float range")
    return result


def distribution_timing(
    clouds: tuple[CloudGravityInput, ...], gravitational_constant: float
) -> DistributionTiming:
    """Derive frame timing from the shortest cloud free-fall time."""

    if not isinstance(clouds, tuple) or not clouds:
        raise ValueError("clouds must be a nonempty tuple of CloudGravityInput values")
    gravity = _positive_scalar(gravitational_constant, "gravitational_constant")
    seen_ids: set[str] = set()
    candidates: list[tuple[float, str]] = []
    for cloud in clouds:
        if not isinstance(cloud, CloudGravityInput):
            raise ValueError("clouds must contain only CloudGravityInput values")
        if not isinstance(cloud.persistent_id, str):
            raise ValueError("persistent_id must be a string")
        if cloud.persistent_id in seen_ids:
            raise ValueError(f"duplicate persistent_id: {cloud.persistent_id}")
        seen_ids.add(cloud.persistent_id)
        mass = _positive_scalar(cloud.mass, "mass")
        radius = _positive_scalar(cloud.radius, "radius")
        seconds_per_frame = _distribution_seconds_per_frame(radius, gravity, mass)
        candidates.append((seconds_per_frame, cloud.persistent_id))

    seconds_per_frame, controlling_id = min(candidates, key=lambda item: (item[0], item[1]))
    return DistributionTiming(
        seconds_per_frame=seconds_per_frame,
        controlling_persistent_id=controlling_id,
    )


def solve_encounter_initial_state(
    *,
    position_a: Any,
    position_b: Any,
    mass_a: float,
    mass_b: float,
    radius_a: float,
    radius_b: float,
    contact_speed: float,
    contact_angle_radians: float,
    gravitational_constant: float,
) -> EncounterInitialState:
    """Invert requested first-contact conditions into an inbound two-body state.

    The inverse uses specific orbital energy and angular momentum.  The existing conic
    predictor then independently verifies the result and supplies continuous contact time.
    """

    first_position = _finite_vector(position_a, "position_a")
    second_position = _finite_vector(position_b, "position_b")
    first_mass = _positive_scalar(mass_a, "mass_a")
    second_mass = _positive_scalar(mass_b, "mass_b")
    first_radius = _positive_scalar(radius_a, "radius_a")
    second_radius = _positive_scalar(radius_b, "radius_b")
    requested_speed = _positive_scalar(contact_speed, "contact_speed")
    angle = _finite_scalar(contact_angle_radians, "contact_angle_radians")
    gravity = _positive_scalar(gravitational_constant, "gravitational_constant")
    if angle < 0.0 or angle >= math.pi / 2.0:
        raise ValueError("contact_angle_radians must be in [0, pi/2)")

    separation = tuple(second_position[index] - first_position[index] for index in range(3))
    relative_position: Vector3 = separation  # type: ignore[assignment]
    distance = _norm(relative_position)
    if not math.isfinite(distance):
        raise ValueError("the initial separation must be finite")
    if distance == 0.0:
        raise ValueError("the two bodies must not have coincident centres")
    contact_radius = first_radius + second_radius
    if not math.isfinite(contact_radius):
        raise ValueError("the contact radius must be finite")
    if distance <= contact_radius:
        raise ValueError("the initial bodies must be strictly separated before contact")

    total_mass = first_mass + second_mass
    if not math.isfinite(total_mass):
        raise ValueError("the total mass must be finite")
    mu = gravity * total_mass
    if not math.isfinite(mu):
        raise ValueError("gravitational_constant times total mass must be finite")
    specific_energy = 0.5 * requested_speed * requested_speed - mu / contact_radius
    specific_angular_momentum = contact_radius * requested_speed * math.sin(angle)
    speed_at_distance_squared = 2.0 * (specific_energy + mu / distance)
    tangential_speed = specific_angular_momentum / distance
    radial_speed_squared = speed_at_distance_squared - tangential_speed * tangential_speed
    if not all(
        math.isfinite(value)
        for value in (
            specific_energy,
            specific_angular_momentum,
            speed_at_distance_squared,
            tangential_speed,
            radial_speed_squared,
        )
    ):
        raise ValueError("the requested encounter produces nonfinite orbital invariants")
    invariant_scale = max(
        abs(specific_energy),
        abs(mu / distance),
        abs(speed_at_distance_squared),
        tangential_speed * tangential_speed,
    )
    roundoff = 4.0 * sys.float_info.epsilon * invariant_scale
    if radial_speed_squared < -roundoff:
        raise ValueError("the requested contact state cannot pass through the initial separation")
    radial_speed = -math.sqrt(max(0.0, radial_speed_squared))

    radial_axis = tuple(component / distance for component in relative_position)
    radial_axis_vector: Vector3 = radial_axis  # type: ignore[assignment]
    transverse = _cross(radial_axis_vector, (0.0, 0.0, 1.0))
    transverse_norm = _norm(transverse)
    if transverse_norm <= 64.0 * sys.float_info.epsilon:
        transverse = _cross(radial_axis_vector, (0.0, 1.0, 0.0))
        transverse_norm = _norm(transverse)
    transverse_axis = tuple(component / transverse_norm for component in transverse)
    relative_velocity = tuple(
        radial_speed * radial_axis_vector[index] + tangential_speed * transverse_axis[index]
        for index in range(3)
    )
    relative_velocity_vector: Vector3 = relative_velocity  # type: ignore[assignment]

    first_fraction = -second_mass / total_mass
    second_fraction = first_mass / total_mass
    first_velocity = tuple(first_fraction * component for component in relative_velocity_vector)
    second_velocity = tuple(second_fraction * component for component in relative_velocity_vector)
    velocity_a: Vector3 = first_velocity  # type: ignore[assignment]
    velocity_b: Vector3 = second_velocity  # type: ignore[assignment]

    prediction = predict_encounter(
        position_a=first_position,
        velocity_a=velocity_a,
        mass_a=first_mass,
        radius_a=first_radius,
        position_b=second_position,
        velocity_b=velocity_b,
        mass_b=second_mass,
        radius_b=second_radius,
        gravitational_constant=gravity,
    )
    if not prediction.contact:
        raise ValueError(f"the solved encounter failed forward verification: {prediction.reason}")
    expected_angle_degrees = math.degrees(angle)
    verified = (
        _scaled_close(prediction.contact_separation, contact_radius)
        and _scaled_close(prediction.contact_speed, requested_speed)
        and _scaled_close(prediction.impact_angle_degrees, expected_angle_degrees)
        and math.isfinite(prediction.contact_time)
        and prediction.contact_time > 0.0
    )
    if not verified:
        raise ValueError("the solved encounter failed forward verification")

    return EncounterInitialState(
        position_a=first_position,
        position_b=second_position,
        velocity_a=velocity_a,
        velocity_b=velocity_b,
        relative_position=relative_position,
        relative_velocity=relative_velocity_vector,
        contact_time=prediction.contact_time,
        contact_radius=contact_radius,
        contact_speed=requested_speed,
        contact_angle_radians=angle,
    )


def contact_boundary_frame(
    *,
    contact_time: float,
    seconds_per_frame: float,
    frame_start: int,
    maximum_boundary_frame: int,
) -> int:
    """Return the first integer frame at or after continuous first contact."""

    time = _finite_scalar(contact_time, "contact_time")
    if time < 0.0:
        raise ValueError("contact_time must be nonnegative")
    step = _positive_scalar(seconds_per_frame, "seconds_per_frame")
    if isinstance(frame_start, bool) or not isinstance(frame_start, int):
        raise ValueError("frame_start must be an integer")
    if isinstance(maximum_boundary_frame, bool) or not isinstance(maximum_boundary_frame, int):
        raise ValueError("maximum_boundary_frame must be an integer")
    if maximum_boundary_frame <= frame_start:
        raise ValueError("maximum_boundary_frame must be after frame_start")
    maximum_interval = maximum_boundary_frame - frame_start

    def sampled_time(interval: int) -> float:
        try:
            return interval * step
        except OverflowError:
            return math.inf

    if time > sampled_time(maximum_interval):
        raise ValueError("the contact boundary exceeds maximum_boundary_frame")
    quotient = time / step
    if not math.isfinite(quotient):
        raise ValueError("contact_time divided by seconds_per_frame must be finite")
    candidate = min(maximum_interval, max(1, math.ceil(quotient)))
    # Division can round just above an integer even when ``time`` was produced by the
    # exact same sampled multiplication (for example, ``(13 * 0.0001) / 0.0001``).
    # Correct around that estimate with sampled-product comparisons.  Binary search keeps
    # the correction bounded even for extreme finite inputs and proves the result minimal.
    if time <= sampled_time(candidate):
        lower, upper = 1, candidate
    else:
        lower, upper = candidate + 1, maximum_interval
    while lower < upper:
        midpoint = (lower + upper) // 2
        if time <= sampled_time(midpoint):
            upper = midpoint
        else:
            lower = midpoint + 1
    interval = lower
    boundary = frame_start + interval
    return boundary


# -- scenario builders (Seams 3E and 3F) --------------------------------------------------

_ORBIT_ROLE_PRESETS = {"BODY": "STELLAR", "CLOUD": "GALACTIC"}


def _require_member_tuple(
    members: object, *, operator: str
) -> tuple[MemberFacts, ...]:
    if not isinstance(members, tuple) or not members:
        raise ValueError(
            f"{operator} requires a nonempty tuple of MemberFacts values"
        )
    for member in members:
        if not isinstance(member, MemberFacts):
            raise ValueError(f"{operator} requires MemberFacts values")
    return members  # type: ignore[return-value]


def _require_frame_range(frame_start: int, frame_end: int) -> None:
    for name, value in (("frame_start", frame_start), ("frame_end", frame_end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if frame_end <= frame_start:
        raise ValueError("frame_end must be after frame_start")


def _member_physics(member: MemberFacts) -> tuple[float, float, Vector3]:
    return (
        _positive_scalar(member.mass, "mass"),
        _positive_scalar(member.radius, "radius"),
        _finite_vector(member.position, "position"),
    )


def _member_volume(member: MemberFacts) -> float:
    volume = 1.0
    for extent in member.dimensions:
        volume *= _finite_scalar(extent, "dimensions")
    return volume


def _separation(first: MemberFacts, second: MemberFacts) -> float:
    offset = tuple(
        second.position[index] - first.position[index] for index in range(3)
    )
    return _norm(offset)  # type: ignore[arg-type]


def _evolve_stages(frame_start: int, frame_end: int) -> tuple[TimelineStage, ...]:
    stages = (TimelineStage(kind=TimelineStageKind.EVOLVE, boundary_frame=frame_start),)
    validate_stage_list(stages, frame_start=frame_start, frame_end=frame_end)
    return stages


def _softening_bounded_by_orbit(
    cloud_radii: list[float], orbit_radius: float
) -> float:
    """The binding minimum of a scene's two softening bounds (ADR 0079).

    ``suggest_softening`` answers one question: how much softening a cold self-gravitating
    cloud needs so it settles instead of bouncing through a singular core. It is a *lower*
    bound dressed as a suggestion, and it is correct for the clouds it was written for.

    ``suggest_orbital_softening`` answers the other: how much softening an orbit at a given
    radius can absorb before the force that holds it stops being the force its authored
    velocity was computed from. That is an *upper* bound, and nothing in the codebase
    enforced it until this function existed.

    A scene holding both a cloud and an orbit has to satisfy both, so it ships the smaller
    of the two. Pass ``orbit_radius`` as zero (or negative) for a scene with no orbit to
    resolve, and the cloud bound stands alone -- that is not an error, it is the case
    ``suggest_softening`` was ported from.
    """

    softening = suggest_softening(cloud_radii)
    if orbit_radius > 0.0:
        softening = min(softening, suggest_orbital_softening(orbit_radius))
    return softening


def _softening_bounded_by_binding(clouds: list[tuple[float, int]]) -> float:
    """The softening a *virialized* cloud wants, bounded by what it can still hold (ADR 0081).

    The third bound in the family, and the one that catches the scenes ADR 0079's orbital
    ceiling cannot see. ``virialized_velocities`` hands a cloud the kinetic energy that
    balances its *unsoftened* binding energy, so a cloud integrated in a softened field is
    really starting at ``Q = |W_0| / |W_eps|`` and expands until it finds the size that
    ratio makes right. ``suggest_cloud_softening`` is the largest softening that keeps that
    ratio inside `CLOUD_BINDING_TOLERANCE` of one. Several clouds under one scene-wide
    softening must all satisfy it, so the smallest cloud sets the number.

    The ceiling alone would be the wrong number to ship, though, because it is the largest
    value that is merely *tolerable* and nothing wants to sit at its own limit. What a cloud
    wants is `suggest_discreteness_softening`: the spacing between its own bodies, the length
    that suppresses the two-body scattering softening exists for. So this helper takes the
    smaller of the two per cloud, then the smallest over the scene, since one scene-wide
    softening has to satisfy every cloud in it.

    Both bounds are live, which is what makes this a minimum and not a preference. Spacing
    falls off as ``count ** (-1/3)`` while the ceiling is a fixed fraction of the radius, so
    they cross at 237 bodies: above that the cloud is sampled finely enough that its own
    spacing is the tighter number and the ceiling never binds, and below it a thinly sampled
    cloud would ask for a softening so wide it could no longer hold itself, and the ceiling is
    the only thing standing between the user and the defect this ADR was written for. The
    shipped templates are on the fine side -- 5,000 and 1,500 bodies, wanting ``0.059 * R`` and
    ``0.087 * R`` against a ``0.162 * R`` ceiling -- so both ship the spacing.

    Unlike `_softening_bounded_by_orbit` neither term is ``suggest_softening``, and the
    difference is the point rather than an omission. That rule returns the *largest* radius it
    is handed, and both of these are small fractions of the *smallest*, so it wins for no
    possible input -- including it would be decoration dressed as a check. For a cloud whose
    velocities are virialized, ADR 0081 replaces the older rule rather than bounding it, which
    is exactly the "built on an older contract" the templates were suspected of. What is not
    lost is the job the older rule existed for, because the spacing term *is* that job stated
    as a length instead of a rule of thumb.

    Nothing here is an orbit, which is why ADR 0079 left these scenes alone: Star Cluster is
    one cold spheroid and Galaxy Collision is two, and their softening was the cloud radius
    itself -- ``eps/R = 1``, where 41 percent of the binding energy is gone and the cloud is
    handed 70 percent more motion than the gravity it is actually integrated with can hold.
    """

    if not clouds:
        raise ValueError("clouds must not be empty")
    return min(
        min(suggest_discreteness_softening(radius, count), suggest_cloud_softening(radius))
        for radius, count in clouds
    )


def build_orbit(
    members: tuple[MemberFacts, MemberFacts],
    *,
    options: OrbitOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author one homogeneous two-member orbit plan (decisions D1 and D2).

    Exactly two members; both already Body, both already Cloud, or both unassigned
    (which become Body).  Empty objects are accepted only for Body pairs; Cloud pairs
    must be nonzero-volume Meshes; mixed roles refuse.  Member velocities are the
    opposed apoapsis barycentric split from :func:`pair_orbit` with
    ``pericentre_ratio = (1 - e) / (1 + e)``.
    """

    pair = _require_member_tuple(members, operator="Orbit")
    if len(pair) != 2:
        raise ValueError(f"Orbit requires exactly two members, got {len(pair)}")
    _require_frame_range(frame_start, frame_end)

    assigned = tuple(member.role for member in pair if member.role != "NONE")
    if any(role not in _ORBIT_ROLE_PRESETS for role in assigned):
        raise ValueError(
            "Orbit members already assigned to another role refuse: "
            "both members must be Body, both Cloud, or both unassigned"
        )
    if len(set(assigned)) > 1 or (assigned and len(assigned) != len(pair)):
        raise ValueError(
            "Orbit requires both members Body, both Cloud, or both unassigned; "
            "mixed roles refuse"
        )
    role = assigned[0] if assigned else "BODY"
    preset = _ORBIT_ROLE_PRESETS[role]

    for member in pair:
        if role == "CLOUD":
            if member.object_type != "MESH" or _member_volume(member) <= 0.0:
                raise ValueError(
                    "a Cloud Orbit pair requires nonzero-volume Mesh objects"
                )
        elif member.object_type not in ("MESH", "EMPTY"):
            raise ValueError("a Body Orbit pair accepts Mesh or Empty objects only")

    first, second = pair
    for member in pair:
        _member_physics(member)
    separation = _separation(first, second)
    if separation <= first.radius + second.radius:
        raise ValueError("Orbit members must be strictly separated at apoapsis")

    eccentricity = _finite_scalar(options.eccentricity, "eccentricity")
    if not 0.0 <= eccentricity <= 0.95:
        raise ValueError("eccentricity must be within [0.0, 0.95]")

    orbit = pair_orbit(
        first.mass,
        second.mass,
        separation,
        (1.0 - eccentricity) / (1.0 + eccentricity),
        units.gravitational_constant,
    )
    timing = orbit_timing(orbit.period, frame_start, frame_end)

    tangent = _orbit_tangent(first, second)
    member_plans = []
    for member, speed, sign in zip(
        pair, orbit.apocentre_speeds, (1.0, -1.0), strict=True
    ):
        velocity = tuple(sign * speed * component for component in tangent)
        member_plans.append(
            MemberPlan(
                persistent_id=member.persistent_id,
                role=role,
                mass=member.mass,
                radius=member.radius,
                velocity=velocity,  # type: ignore[arg-type]
                particle_count=1,
                sample_shape="BALL",
                material_key=member.material_key,
                shell_layout=(),
                settings=(),
            )
        )

    return ScenarioPlan(
        unit_preset=preset,
        solver="AUTO",
        color_by="SPEED",
        seconds_per_frame=timing.seconds_per_frame,
        # The pericentre bounds the softening for the same reason it sizes the step
        # below, and the comment below is where that reason is written down: a Body
        # pair's radii are rendering sizes, so nothing here is a cloud extent and the
        # orbit is the only bound that means anything; a Cloud pair carries both, and
        # ships whichever is smaller (ADR 0079). Without this, a pair authored a few
        # radii apart -- which the overlap check above allows -- is softened at nearly
        # its own separation and never flies the orbit it was handed.
        softening=_softening_bounded_by_orbit(
            [member.radius for member in pair],
            separation * (1.0 - eccentricity) / (1.0 + eccentricity),
        ),
        # A Body pair is two point masses: their radii are rendering sizes, not
        # self-gravitating extents, so the step is sized by the orbit at
        # pericentre, never a spurious internal free-fall (ADR 0057). A Cloud
        # pair holds genuine diffuse distributions, so its collapse time governs.
        substeps=(
            suggest_orbit_substeps(
                timing.seconds_per_frame,
                separation * (1.0 - eccentricity) / (1.0 + eccentricity),
                first.mass + second.mass,
                units.gravitational_constant,
            )
            if role == "BODY"
            else suggest_substeps(
                timing.seconds_per_frame,
                [(member.mass, member.radius) for member in pair],
                units.gravitational_constant,
            )
        ),
        members=tuple(member_plans),
        timeline_stages=_evolve_stages(frame_start, frame_end),
        display_groups=(),
        result_frame_end=frame_end,
        look=SPEED_LOOK,
        tree_theta=DEFAULT_TREE_THETA,
    )


def _orbit_tangent(first: MemberFacts, second: MemberFacts) -> Vector3:
    """The unit tangent of the pair's separation in the world-Z orbit plane.

    The separation axis cross world Z gives the in-plane tangent; the world-Y fallback
    covers a Z-parallel pair, mirroring the encounter inverse's axis convention.
    """

    offset = tuple(
        second.position[index] - first.position[index] for index in range(3)
    )
    tangent = _cross(offset, (0.0, 0.0, 1.0))  # type: ignore[arg-type]
    norm = _norm(tangent)
    if norm <= 64.0 * sys.float_info.epsilon * _norm(offset):
        tangent = _cross(offset, (0.0, 1.0, 0.0))  # type: ignore[arg-type]
        norm = _norm(tangent)
    return tuple(component / norm for component in tangent)  # type: ignore[return-value]


_DISTRIBUTION_SHAPES = ("OBJECT_VOLUME",)


def build_distribution(
    members: tuple[MemberFacts, ...],
    *,
    options: DistributionOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author a per-object Cloud distribution plan (decisions D1 and D2).

    One or more selected Meshes; every member becomes a Cloud on the Galactic
    preset.  Each object receives its own ``count_per_object`` particles, its own
    ``total_mass_per_object`` mass, and the shared ``seed``; the sampling proxy is
    the compilable convex ``OBJECT_VOLUME`` shape with UNIFORM velocities and zero
    mass jitter.  Timing is the shortest cloud free-fall time via
    :func:`distribution_timing`, and the controlling id is carried in settings.
    """

    selected = _require_member_tuple(members, operator="Distribution")
    _require_frame_range(frame_start, frame_end)

    if options.shape not in _DISTRIBUTION_SHAPES:
        raise ValueError(
            f"unsupported distribution shape {options.shape!r}; "
            "Slice 3 supports OBJECT_VOLUME only"
        )
    if isinstance(options.count_per_object, bool) or not isinstance(
        options.count_per_object, int
    ):
        raise ValueError("count_per_object must be an integer")
    if options.count_per_object <= 0:
        raise ValueError("count_per_object must be positive")
    total_mass = _positive_scalar(
        options.total_mass_per_object, "total_mass_per_object"
    )
    if isinstance(options.seed, bool) or not isinstance(options.seed, int):
        raise ValueError("seed must be an integer")

    clouds = []
    member_plans = []
    for member in selected:
        if member.object_type != "MESH" or _member_volume(member) <= 0.0:
            raise ValueError(
                "Distribution requires nonzero-volume Mesh objects"
            )
        _mass, radius, _position = _member_physics(member)
        clouds.append(
            CloudGravityInput(
                persistent_id=member.persistent_id,
                mass=total_mass,
                radius=radius,
            )
        )

    timing = distribution_timing(tuple(clouds), units.gravitational_constant)
    for member in selected:
        radius = _positive_scalar(member.radius, "radius")
        member_plans.append(
            MemberPlan(
                persistent_id=member.persistent_id,
                role="CLOUD",
                mass=total_mass,
                radius=radius,
                velocity=(0.0, 0.0, 0.0),
                particle_count=options.count_per_object,
                sample_shape=options.shape,
                material_key=member.material_key,
                shell_layout=(),
                settings=(
                    ("velocity_model", "UNIFORM"),
                    ("mass_jitter", 0.0),
                    ("sampling_seed", options.seed),
                    ("count_per_object", options.count_per_object),
                    ("total_mass_per_object", total_mass),
                    (
                        "controlling_persistent_id",
                        timing.controlling_persistent_id,
                    ),
                ),
            )
        )

    return ScenarioPlan(
        unit_preset="GALACTIC",
        solver="AUTO",
        color_by="SPEED",
        seconds_per_frame=timing.seconds_per_frame,
        softening=suggest_softening(
            [member.radius for member in selected]
        ),
        substeps=suggest_substeps(
            timing.seconds_per_frame,
            [(total_mass, member.radius) for member in selected],
            units.gravitational_constant,
        ),
        members=tuple(member_plans),
        timeline_stages=_evolve_stages(frame_start, frame_end),
        display_groups=(),
        result_frame_end=frame_end,
        look=SPEED_LOOK,
        tree_theta=DEFAULT_TREE_THETA,
    )


# -- Template builders (Slice 5 Chunk 4 Task 12, decision D8) ------------------------------

# D8's Cluster override, spelled out because it is a decision and not a default: the
# template *reuses* Astro Distribution's transactional mechanics and free-fall timing rule
# while replacing every scenario field. The two values named here are the ones that must
# not survive from Distribution -- its OBJECT_VOLUME shape and its 4,096 count -- because
# inheriting either would silently give the user a different cloud than the one D8 accepted.
CLUSTER_COUNT = 5_000
"""Chosen for headroom by measurement (ADR 0076), replacing D8's 50,000.

Swept in a real Blender window at 2,000 / 5,000 / 8,000 / 12,000 / 20,000 bodies. 5,000 is the
largest swept count whose whole frame costs at most half of one service budget --- 9.674 ms of
20.833 --- which is the margin ADR 0076 states as *still full speed on a machine half as fast*.
The next count up, 8,000, costs 14.192 ms and is inside one budget but not inside half of one.

This is a default derived from a measurement, not a cap: the property's ``min`` is unchanged and
an artist who types 50,000 is warned about the rate rather than refused (ADR 0067), which is what
keeps ADR 0056's rejection of magic-number count caps intact.
"""

CLUSTER_TOTAL_MASS = 5.0e-5
"""50,000 nominal solar masses, in Galactic mass units."""

CLUSTER_RADIUS = 0.01
"""10 pc as a Galactic-unit (kpc) radius; the proxy's diameter is 0.02 on each axis."""

CLUSTER_SEED = 0


@dataclass(frozen=True)
class ClusterOptions:
    """The Cluster template's authored inputs (D8), each defaulting to its accepted value."""

    count: int = CLUSTER_COUNT
    total_mass: float = CLUSTER_TOTAL_MASS
    radius: float = CLUSTER_RADIUS
    seed: int = CLUSTER_SEED


def build_cluster(
    member: MemberFacts,
    *,
    options: ClusterOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author D8's virialized 50k cluster as one Galactic/Tree Cloud.

    Shares Astro Distribution's timing rule -- :func:`distribution_timing`, the shortest
    cloud free-fall time -- and nothing else. Every scenario field is this template's own:
    ``BALL`` rather than ``OBJECT_VOLUME``, 50,000 rather than 4,096, and ``VIRIALIZE``
    rather than ``UNIFORM``. Timing is evaluated from *these* values, not from
    Distribution's, which is the whole reason the override has to be explicit: a free-fall
    time computed from the discarded mass and radius would be wrong by orders of magnitude
    and would look like a considered choice.

    Solver ``AUTO`` rather than a pinned backend: at 50,000 bodies the direct lane is not
    only a candidate, on current GPUs it is *faster* than the O(N log N) tree, whose deeper
    per-body walk the device-independent realtime floor bounds above the frame slice -- so a
    pinned ``TREE`` here would author both the slower backend and the one realtime admission
    refuses, and the cluster would only ever bake. ``AUTO`` lets the runtime measure the
    compiled graph once at initialize (ADR 0056) and play whichever backend fits the slice,
    which is what lets this flagship template scrub in realtime rather than only bake.

    Nothing here compiles, initializes or advances. The plan is authored intent; Collision
    Ignore, ``FLOAT32`` preview precision and POINT-only display are the shipped defaults
    D8 names, so the template does not restate them -- restating a default is how a
    "template value" and a "shipped value" drift apart with nothing to say which moved.
    """

    _require_frame_range(frame_start, frame_end)
    if isinstance(options.count, bool) or not isinstance(options.count, int):
        raise ValueError("count must be an integer")
    if options.count <= 0:
        raise ValueError("count must be positive")
    if isinstance(options.seed, bool) or not isinstance(options.seed, int):
        raise ValueError("seed must be an integer")
    if options.seed < 0:
        raise ValueError("seed must not be negative")
    total_mass = _positive_scalar(options.total_mass, "total_mass")
    radius = _positive_scalar(options.radius, "radius")

    timing = distribution_timing(
        (
            CloudGravityInput(
                persistent_id=member.persistent_id,
                mass=total_mass,
                radius=radius,
            ),
        ),
        units.gravitational_constant,
    )

    plan = MemberPlan(
        persistent_id=member.persistent_id,
        role="CLOUD",
        mass=total_mass,
        radius=radius,
        velocity=(0.0, 0.0, 0.0),
        particle_count=options.count,
        sample_shape="BALL",
        material_key=member.material_key,
        shell_layout=(),
        settings=(
            ("velocity_model", "VIRIALIZE"),
            ("support_fraction", 1.0),
            ("mass_jitter", 0.0),
            ("eccentricity_jitter", 0.0),
            # Retained at its shipped value and inactive: VIRIALIZE reads neither, and
            # clearing them would make a later switch to KEPLERIAN look like the user had
            # chosen clockwise motion and no central body.
            ("keplerian_counter_clockwise", True),
            ("keplerian_central_persistent_id", ""),
            ("annulus_inner_radius", 0.0),
            ("sampling_seed", options.seed),
            ("count", options.count),
            ("total_mass", total_mass),
            ("controlling_persistent_id", timing.controlling_persistent_id),
        ),
    )

    return ScenarioPlan(
        unit_preset="GALACTIC",
        solver="AUTO",
        color_by="SPEED",
        seconds_per_frame=timing.seconds_per_frame,
        softening=_softening_bounded_by_binding([(radius, options.count)]),
        substeps=suggest_substeps(
            timing.seconds_per_frame,
            [(total_mass, radius)],
            units.gravitational_constant,
        ),
        members=(plan,),
        timeline_stages=_evolve_stages(frame_start, frame_end),
        display_groups=((member.persistent_id, "STAR_POINTS"),),
        result_frame_end=frame_end,
        # 0.8, swept in a window against 0.4/1.5/3.0 (ADR 0091): 9.83 -> 22.76 -> 33.50 ->
        # 45.25 lit pixels per body, so 45% of each disc survives overlap at 0.8 against
        # 19% at 1.5. In the pictures that is the difference between separate stars over a
        # dense core and confetti with the density gradient washed out. 0.4 reads correctly
        # but faint, which is a cluster that is right and hard to see.
        look=swept_look(0.8),
        tree_theta=CLUSTER_TREE_THETA,
    )


# D8's Binary System. Stellar units, so masses are nominal solar masses and lengths are au.
# The template creates its own fixture rather than reading a selection: it is a *template*,
# and reading whatever the user happened to have selected would make its accepted defaults
# advisory.
BINARY_PRIMARY_MASS = 1.0
BINARY_SECONDARY_MASS = 0.6
BINARY_RADIUS_SOLAR = 1.0
"""Each Body's radius in nominal solar radii; the redo property's unit."""

BINARY_SEPARATION = 1.0
BINARY_ECCENTRICITY = 0.0

SOLAR_RADIUS_AU = 695700000.0 / 149597870700.0
"""`Rsun / au`, as the ratio D8 states rather than a pasted rounding of it."""

# The circumbinary disc ADR 0078 decision 3 adds, so that Binary System has something to *see*
# moving. Two point masses on a closed orbit are correct and nearly invisible: nothing changes
# shape, nothing changes colour, and at the shipped 1 au separation the pair reads as two dots
# rotating. A disc around them is what makes the coupling legible -- and it is the multi-role
# case (two BODY members plus one CLOUD) that no other shipped template exercises.
#
# Every length below is a *ratio*, never an au figure, and that is the point. The disc's inner
# edge has to sit outside the binary's chaotic zone or the demo draws a disc that would not
# survive its own first orbits; where that zone ends depends on the mass ratio and the
# eccentricity, both of which are Add Last Operation properties the user can drag. Deriving the
# geometry from the criterion means the invariant holds at every redo, so the template never has
# to refuse, clamp, or silently move a disc the user just asked for.
BINARY_DISK_INNER_MARGIN = 1.25
"""Inner edge as a multiple of the Holman & Wiegert critical radius.

Twenty-five percent outside the last chaotic orbit. The criterion is a fit to integrations, not
a sharp edge, so sitting exactly on it would be authoring the demo onto the boundary of its own
validity -- the same "a demo that's a coin flip isn't a demo" the owner applied to counts.
"""

BINARY_DISK_RADIUS_RATIO = 2.5
"""Outer edge as a multiple of the inner edge.

Wide enough to read as a disc rather than a ring, and narrow enough that the inner and outer
orbital periods differ by 2.5**1.5 = 4.0 -- so the disc visibly *shears* over the shipped frame
range instead of turning as a rigid wheel, which is the one thing a Keplerian disc does that a
texture could not fake.
"""

BINARY_DISK_THICKNESS_RATIO = 0.02
"""Full thickness as a fraction of the outer radius: a thin disc, `H/R = 0.02`."""

BINARY_DISK_MASS = 0.02
"""Total disc mass in solar masses -- about 20 Jupiters, and 1.25 percent of the pair.

Light enough that the disc is a tracer of the binary's field rather than a third body
restructuring it, which is what keeps the picture readable as "these particles orbit that pair".
"""

BINARY_DISK_COUNT = 2000
"""Particles in the circumbinary disc.

An interim conservative number under ADR 0076, not a swept one, and honestly labelled as such:
the shipped Protoplanetary Disk was swept to 2,000 at eight substeps, and this scene runs the
same count at two, so it costs about a quarter of an already-measured frame. The sweep that
would let it rise is the ADR 0084 debt and is blocked on windowed runs.
"""

BINARY_DISK_SEED = 0
"""The sampling seed, fixed so the shipped demo is the same demo on every machine."""


def solar_radii_to_au(value: float) -> float:
    """One redo property's unit conversion, named because it is the only one here.

    D8 puts the Binary and Disk radius properties in nominal solar radii while Stellar
    lengths are au, so this is the single place the two meet. Inlining it at each call site
    is how one of them ends up with a rounded constant and the other with the ratio.
    """

    return _finite_scalar(value, "solar radii") * SOLAR_RADIUS_AU


@dataclass(frozen=True)
class BinaryOptions:
    """The Binary System template's Add Last Operation properties (D8), in D8's units."""

    primary_mass: float = BINARY_PRIMARY_MASS
    secondary_mass: float = BINARY_SECONDARY_MASS
    primary_radius_solar: float = BINARY_RADIUS_SOLAR
    secondary_radius_solar: float = BINARY_RADIUS_SOLAR
    separation: float = BINARY_SEPARATION
    eccentricity: float = BINARY_ECCENTRICITY
    disk_mass: float = BINARY_DISK_MASS
    disk_count: int = BINARY_DISK_COUNT
    seed: int = BINARY_DISK_SEED


def circumbinary_disk_geometry(
    primary_mass: float,
    secondary_mass: float,
    separation: float,
    eccentricity: float,
) -> tuple[float, float, float]:
    """The circumbinary disc's `(inner_radius, outer_radius, full_thickness)` (ADR 0085).

    Derived from the binary the user authored, never from stored au figures, so the disc is
    outside the pair's chaotic zone at *every* setting of the operator's mass and separation
    properties rather than only at the shipped ones. Widen the separation and the disc widens
    with it; make the pair more eccentric and the hole opens further, because that is what the
    Holman & Wiegert criterion says happens.

    Exposed rather than inlined because the operator has to build the annulus proxy mesh at the
    same three numbers the plan is authored at; two copies of this arithmetic is exactly how a
    proxy ends up describing a different disc from the one being solved.
    """

    primary = _positive_scalar(primary_mass, "primary_mass")
    secondary = _positive_scalar(secondary_mass, "secondary_mass")
    distance = _positive_scalar(separation, "separation")
    ratio = circumbinary_critical_radius_ratio(
        secondary / (primary + secondary),
        _finite_scalar(eccentricity, "eccentricity"),
    )
    inner = BINARY_DISK_INNER_MARGIN * ratio * distance
    outer = BINARY_DISK_RADIUS_RATIO * inner
    return (inner, outer, BINARY_DISK_THICKNESS_RATIO * outer)


def binary_positions(
    primary_mass: float,
    secondary_mass: float,
    separation: float,
    cursor: Vector3 = (0.0, 0.0, 0.0),
) -> tuple[Vector3, Vector3]:
    """D8's exact barycentric split along world +X through the 3D cursor.

    `x_primary = -m_secondary * d / (m1 + m2)` and `x_secondary = +m_primary * d / (m1 + m2)`,
    which puts the barycentre on the cursor rather than either body. A template that centred
    the *primary* on the cursor would look identical for equal masses and drift visibly for
    the 1.0/0.6 pair D8 actually specifies -- and the orbit would then precess across the
    viewport instead of staying where the user clicked.

    Returned as world positions rather than folded into the plan because `MemberPlan` carries
    solver inputs, and an object's transform is the operator's to set.
    """

    primary = _positive_scalar(primary_mass, "primary_mass")
    secondary = _positive_scalar(secondary_mass, "secondary_mass")
    distance = _positive_scalar(separation, "separation")
    origin = tuple(_finite_scalar(value, "cursor") for value in cursor)
    if len(origin) != 3:
        raise ValueError("cursor must be a 3-vector")
    total = primary + secondary
    return (
        (origin[0] - secondary * distance / total, origin[1], origin[2]),
        (origin[0] + primary * distance / total, origin[1], origin[2]),
    )


def build_binary(
    primary_persistent_id: str,
    secondary_persistent_id: str,
    *,
    disk_persistent_id: str,
    options: BinaryOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author D8's two-Body Stellar binary, plus ADR 0085's circumbinary disc.

    The orbital arithmetic is Astro Orbit's shared primitive -- :func:`pair_orbit` at
    `pericentre_ratio = (1 - e) / (1 + e)`, released at apoapsis -- so the two operators
    cannot disagree about what a given eccentricity means. What is this template's own is
    the fixture: D8's masses, radii, separation and geometry, none of which come from a
    selection.

    Velocities are perpendicular to the separation axis and opposed, which is what puts the
    pair on a closed orbit about the barycentre the positions place on the cursor. The
    primary takes `-Y` and the secondary `+Y`, so both contribute positive angular momentum
    about `+Z`; giving them the same sign would be a pair translating together, not orbiting.

    No material is assigned: ADR 0037 gives Body no physics Material field, and each output
    instances its generated source mesh plus whatever ordinary Blender material the user
    later puts on it.

    The third member is ADR 0085's circumbinary Cloud, and it is what makes this demo a demo:
    the two Bodies alone are a correct simulation of almost nothing visible, while a disc drawn
    around them shears, and the shear is the binary's gravity made legible. It orbits the pair's
    *barycentre* at their combined mass, which is a centre lying on neither star -- the engine
    seam ADR 0085 opens, because the pre-0085 KEPLERIAN contract could only orbit one body.

    Timing moves with it. The pre-0085 template fitted one *binary* period across the range;
    that step is far finer than the disc needs and the disc would barely turn. The range is now
    one revolution of the disc's inner edge, which at the shipped fixture is 5.29 binary orbits
    -- so the user sees the pair go round five times inside one turn of the disc, which is the
    relationship between the two timescales stated as a picture instead of a number.
    """

    _require_frame_range(frame_start, frame_end)
    identifiers = (primary_persistent_id, secondary_persistent_id, disk_persistent_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the two Bodies and the disk must be three different objects")
    if isinstance(options.disk_count, bool) or not isinstance(options.disk_count, int):
        raise ValueError("disk_count must be an integer")
    if options.disk_count <= 0:
        raise ValueError("disk_count must be positive")
    if isinstance(options.seed, bool) or not isinstance(options.seed, int):
        raise ValueError("seed must be an integer")
    if options.seed < 0:
        raise ValueError("seed must not be negative")
    disk_mass = _positive_scalar(options.disk_mass, "disk_mass")

    primary_mass = _positive_scalar(options.primary_mass, "primary_mass")
    secondary_mass = _positive_scalar(options.secondary_mass, "secondary_mass")
    separation = _positive_scalar(options.separation, "separation")
    primary_radius = solar_radii_to_au(
        _positive_scalar(options.primary_radius_solar, "primary_radius_solar")
    )
    secondary_radius = solar_radii_to_au(
        _positive_scalar(options.secondary_radius_solar, "secondary_radius_solar")
    )
    eccentricity = _finite_scalar(options.eccentricity, "eccentricity")
    if not 0.0 <= eccentricity <= 0.95:
        raise ValueError("eccentricity must be within [0.0, 0.95]")
    if separation <= primary_radius + secondary_radius:
        raise ValueError(
            "the two Bodies overlap at apoapsis: separation must exceed the sum of the "
            f"radii ({primary_radius + secondary_radius:.6g})"
        )

    orbit = pair_orbit(
        primary_mass,
        secondary_mass,
        separation,
        (1.0 - eccentricity) / (1.0 + eccentricity),
        units.gravitational_constant,
    )
    inner_radius, outer_radius, thickness = circumbinary_disk_geometry(
        primary_mass, secondary_mass, separation, eccentricity
    )
    total_mass = primary_mass + secondary_mass
    # One revolution of the disc's *inner* edge, not of the binary and not of the disc's outer
    # edge. The inner edge is the fastest orbit in the disc, so a range fitted to it shows the
    # whole disc turning at least once while the shear against the slower outer edge -- a factor
    # of BINARY_DISK_RADIUS_RATIO**1.5 = 4 in period -- is plainly visible by the last frame.
    inner_period = 2.0 * math.pi * math.sqrt(
        inner_radius**3 / (units.gravitational_constant * total_mass)
    )
    timing = orbit_timing(inner_period, frame_start, frame_end)

    members = []
    for persistent_id, mass, radius, speed, sign in (
        (primary_persistent_id, primary_mass, primary_radius, orbit.apocentre_speeds[0], -1.0),
        (
            secondary_persistent_id,
            secondary_mass,
            secondary_radius,
            orbit.apocentre_speeds[1],
            1.0,
        ),
    ):
        members.append(
            MemberPlan(
                persistent_id=persistent_id,
                role="BODY",
                mass=mass,
                radius=radius,
                velocity=(0.0, sign * speed, 0.0),
                particle_count=1,
                sample_shape="BALL",
                material_key="",
                shell_layout=(),
                settings=(
                    ("mass", mass),
                    ("radius", radius),
                    ("spin", (0.0, 0.0, 0.0)),
                    ("pinned", False),
                ),
            )
        )

    members.append(
        MemberPlan(
            persistent_id=disk_persistent_id,
            role="CLOUD",
            mass=disk_mass,
            radius=outer_radius,
            velocity=(0.0, 0.0, 0.0),
            particle_count=options.disk_count,
            sample_shape="DISK_ANNULUS",
            material_key="",
            shell_layout=(),
            settings=(
                ("velocity_model", "KEPLERIAN"),
                # ADR 0085's seam. Naming both halves is what turns "orbit that star" into
                # "orbit that pair": the sampler then centres the cloud on the barycentre and
                # gives every particle the speed for the combined mass. Naming only the primary
                # would hand out speeds for 1.0 solar masses about a point 0.375 au off the
                # true centre -- a disc that looks right in the first frame and unwinds.
                ("keplerian_central_persistent_id", primary_persistent_id),
                ("keplerian_companion_persistent_id", secondary_persistent_id),
                # Prograde with the pair, and not an operator property. The two Bodies are
                # released with positive angular momentum about +Z, and a retrograde
                # circumbinary disc is not the scene this template exists to show.
                ("keplerian_counter_clockwise", True),
                ("annulus_inner_radius", inner_radius),
                # The same MMSN-motivated profile the Protoplanetary Disk ships (ADR 0084), so
                # both discs in the product concentrate their dust the same way; over this
                # narrower annulus it moves the median radius in by 12.5 percent.
                ("annulus_surface_density_exponent", DISK_SURFACE_DENSITY_EXPONENT),
                ("eccentricity_jitter", 0.0),
                ("mass_jitter", 0.0),
                # Inactive under KEPLERIAN, kept at its shipped value for the same reason the
                # Protoplanetary Disk keeps it: zeroing an unread field reads as a choice.
                ("support_fraction", 1.0),
                ("sampling_seed", options.seed),
                ("count", options.disk_count),
                ("total_mass", disk_mass),
                ("proxy_extents", disk_proxy_extents(outer_radius, thickness)),
            ),
        )
    )

    return ScenarioPlan(
        unit_preset="STELLAR",
        solver="AUTO",
        color_by="SPEED",
        seconds_per_frame=timing.seconds_per_frame,
        # Bounded by the pericentre as well (ADR 0079). At the shipped defaults this
        # changes nothing -- a solar photosphere is 0.00465 au against a 1 au separation,
        # a 17.6-fold margin inside the ceiling -- but `separation`, `eccentricity` and both
        # radii are Add Last Operation properties, and a user who tightens the orbit or
        # fattens the stars would otherwise cross the ceiling with no sign that anything
        # had changed except that the pair stopped orbiting.
        # The disc's outer radius joins the cloud-extent list because it is what that list
        # means -- a genuinely diffuse self-gravitating extent, unlike the two photosphere
        # radii beside it. It changes nothing at the shipped fixture and is not decoration:
        # the orbital ceiling is the binding term at every setting the operator admits, and
        # writing the cloud term correctly is what keeps that a *fact* rather than a
        # coincidence of the defaults.
        softening=_softening_bounded_by_orbit(
            [primary_radius, secondary_radius, outer_radius],
            separation * (1.0 - eccentricity) / (1.0 + eccentricity),
        ),
        # Three timescales, and the fastest wins. The two Bodies are point masses whose
        # stellar radii are photosphere sizes, not self-gravitating cloud extents, so the
        # cloud free-fall advisor would demand thousands of substeps for a two-body orbit
        # that needs one or two (ADR 0057) -- they are sized by their orbit at pericentre.
        # The disc's inner edge is sized by its own orbit about the combined mass, and the
        # disc's self-gravity by the cloud rule at its real extent. At the shipped fixture
        # the binary's own orbit binds, at 2.
        substeps=max(
            suggest_orbit_substeps(
                timing.seconds_per_frame,
                separation * (1.0 - eccentricity) / (1.0 + eccentricity),
                total_mass,
                units.gravitational_constant,
            ),
            suggest_orbit_substeps(
                timing.seconds_per_frame,
                inner_radius,
                total_mass,
                units.gravitational_constant,
            ),
            suggest_substeps(
                timing.seconds_per_frame,
                [(disk_mass, outer_radius)],
                units.gravitational_constant,
            ),
        ),
        members=tuple(members),
        timeline_stages=_evolve_stages(frame_start, frame_end),
        display_groups=(
            (primary_persistent_id, "BODIES"),
            (secondary_persistent_id, "BODIES"),
            (disk_persistent_id, "STAR_POINTS"),
        ),
        result_frame_end=frame_end,
        # 0.8, same sweep: 9.66 -> 25.12 -> 38.89 -> 47.98, a 50% disc share at 0.8. This
        # demo is judged on its circumbinary disk, and 0.8 is where the grains stay
        # individual and the plane still reads as a plane. It does *not* settle the open
        # defect that the two stars cannot be told from their dust: one RenderSpec is
        # resolved per System, so telling them apart needs a per-member look tier, which
        # is a record of its own and not a size.
        look=swept_look(0.8),
        tree_theta=DEFAULT_TREE_THETA,
    )


# -- Asteroid Impact template (ADR 0048) ---------------------------------------------------

# D8 originally authored the signed mockup's `Solid impactor + Body target`, accepting that
# the pair had no runtime graph: Body Contact was `planned - WP1`, and the template shipped
# as honest authoring scaffolding with a bespoke refusal. ADR 0048 takes the alternative D8
# itself named ("if the owner instead chooses working Quick-Effect parity (`Solid + Solid`),
# revise this packet and its exact defaults before any template code"), because the alpha's
# binding scope is that every shipped template is one click from a System that compiles and
# plays. Both members are now Solids, so the accepted `impact` backend runs the encounter.
#
# Nothing else about the scenario moves: the radii, separation, contact speed and angle are
# D8's accepted numbers, and the timing still comes from Slice 3's inverse two-body solver
# and its contact/reaccumulate frame rules.
ASTEROID_TARGET_RADIUS_KM = 50.0
# 14 km (ADR 0207): a 20 km ice impactor at 4 km/s carries about a hundred times the target's
# gravitational binding energy, so the rock engine correctly breaks the target in two (half of it
# lost, no crater). At 14 km the target keeps three quarters of itself and a bite is taken out.
ASTEROID_IMPACTOR_RADIUS_KM = 14.0
ASTEROID_SEPARATION_KM = 200.0
# 4 km/s (the owner's choice of 2026-10-14, ADR 0204): at the old 0.2 km/s the rock engine's
# ice impactor bounced off intact once the contact was resolved; from 2 km/s it is destroyed
# and throws material out.
ASTEROID_CONTACT_SPEED_KM_S = 4.0
ASTEROID_CONTACT_ANGLE_DEGREES = 45.0
ASTEROID_FRAGMENTS_TOTAL = 10_000
"""The particle budget for the *pair*, split by volume so the two share a particle *volume*.

Not a particle mass: :func:`_split_particles` weights by volume, which equalises per-particle
volume and leaves per-particle masses in the bodies' bulk-density ratio. ADR 0048 measures
that for the shipped pair and accepts it; this line previously claimed the split equalised
mass, which is the same wrong claim ADR 0048 corrected on :func:`_split_particles` itself and
did not correct here.

Under D8 this was the impactor's own count and the Body target held a single point mass.
With both members sampled, one budget split by the accepted D7 rule is what keeps the
solver's density estimate well conditioned; see ADR 0048 for what that costs the impactor,
and ADR 0121 for what the resulting 9,862/138 split costs the *timestep* -- at this budget
the 128-per-body floor, not the volume weight, is what sets the impactor's count.
"""

ASTEROID_TARGET_MATERIAL = "synthetic-rock"
"""The target's outer shell, over the accepted Synthetic Iron core of the shared layout."""

ASTEROID_IMPACTOR_MATERIAL = "synthetic-ice"
"""The impactor's outer shell. Its core is the accepted Synthetic Iron of the shared layout."""


@dataclass(frozen=True)
class AsteroidImpactOptions:
    """The Add Last Operation properties D8 lists, in D8's units.

    The two masses are deliberately absent: D8 makes them "visibly derived readouts, not
    independent redo inputs". A mass the user could type alongside a radius and a material
    would let the three disagree, and nothing would say which one the solver used.
    """

    target_radius_km: float = ASTEROID_TARGET_RADIUS_KM
    impactor_radius_km: float = ASTEROID_IMPACTOR_RADIUS_KM
    separation_km: float = ASTEROID_SEPARATION_KM
    contact_speed_km_s: float = ASTEROID_CONTACT_SPEED_KM_S
    contact_angle_degrees: float = ASTEROID_CONTACT_ANGLE_DEGREES
    fragments: int = ASTEROID_FRAGMENTS_TOTAL
    target_material: str = ASTEROID_TARGET_MATERIAL
    impactor_material: str = ASTEROID_IMPACTOR_MATERIAL
    color_by: str = "SPEED"


def asteroid_shell_layout(material_key: str) -> tuple[tuple[str, int, float], ...]:
    """A member's accepted Synthetic Iron core under its chosen outer material.

    The same layout Slice 3's Impact operator builds, by the same helper, so the two cannot
    disagree about what "the accepted core" is. Both members use it: ADR 0048 makes the
    target a Solid, and a Solid without shells is not a differentiated impact body.
    """

    if material_key not in OUTER_MATERIAL_IDS:
        raise ValueError(
            f"unknown material {material_key!r}; choose one of "
            f"{', '.join(sorted(OUTER_MATERIAL_IDS))}"
        )
    core_key, core_id, core_fraction = DEFAULT_IMPACT_SHELL_LAYOUT[0]
    return (
        (core_key, core_id, core_fraction),
        (material_key, OUTER_MATERIAL_IDS[material_key], 1.0),
    )


def asteroid_layered_mass(
    radius: float, layout: tuple[tuple[str, int, float], ...], units: UnitScale
) -> float:
    """The accepted Slice 3 layered-body mass over a member's proxy radius."""

    shells = []
    previous = 0.0
    for key, material_id, fraction in layout:
        if fraction <= previous:
            raise ValueError("shell fractions must strictly increase")
        shells.append(
            ShellSpec(
                outer_radius=fraction * radius,
                material_id=material_id,
                density=units.density_to_code(
                    PRESETS_BY_KEY[key].eos.parameters["reference_density"]
                ),
            )
        )
        previous = fraction
    return layered_body_mass(tuple(shells))


def _template_solid(
    persistent_id: str,
    mass: float,
    radius: float,
    velocity: Vector3,
    count: int,
    material_key: str,
    layout: tuple[tuple[str, int, float], ...],
) -> MemberPlan:
    """One layered Solid a template authors on proxy geometry it created itself.

    Shared by Asteroid Impact and Moon Birth so the two templates cannot disagree about what
    a template's Solid member carries.
    """

    return MemberPlan(
        persistent_id=persistent_id,
        role="SOLID",
        mass=mass,
        radius=radius,
        velocity=velocity,
        particle_count=count,
        sample_shape="OBJECT_BALL",
        material_key=material_key,
        shell_layout=layout,
        settings=(
            ("material", material_key),
            ("resolution_mode", "TARGET_COUNT"),
            ("target_count", count),
            ("spin_rad_s", (0.0, 0.0, 0.0)),
            ("temperature_k", 0.0),
        ),
    )


def build_asteroid_impact(
    target_persistent_id: str,
    impactor_persistent_id: str,
    *,
    options: AsteroidImpactOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author the Planetary two-Solid asteroid encounter (D8 as amended by ADR 0048).

    Every physical value comes from the operator's own properties in D8's units -- km, km/s,
    degrees -- converted here through the one unit service. Both masses are derived, not
    authored, from the accepted layered-body layout over each member's radius and outer
    material. Letting a user type one alongside its radius and material would let the three
    disagree, with nothing to say which the solver used.

    Timing reuses Slice 3's accepted encounter solver and its contact/reaccumulate frame
    rules verbatim. Reimplementing them here would give the same scenario two answers.

    Unlike :func:`build_impact`, this planner takes persistent ids rather than
    :class:`MemberFacts`: the template creates its own proxy geometry at the radii the user
    typed, so there is no evaluated object to read extents back off yet.
    """

    _require_frame_range(frame_start, frame_end)
    if target_persistent_id == impactor_persistent_id:
        raise ValueError("the target and the impactor must be different objects")
    if isinstance(options.fragments, bool) or not isinstance(options.fragments, int):
        raise ValueError("fragments must be an integer")
    if options.fragments <= 0:
        raise ValueError("fragments must be positive")

    target_radius = units.length_to_code(
        1000.0 * _positive_scalar(options.target_radius_km, "target_radius_km")
    )
    impactor_radius = units.length_to_code(
        1000.0 * _positive_scalar(options.impactor_radius_km, "impactor_radius_km")
    )
    separation = units.length_to_code(
        1000.0 * _positive_scalar(options.separation_km, "separation_km")
    )
    if separation <= target_radius + impactor_radius:
        raise ValueError(
            "the target and impactor overlap at the authored separation: it must exceed "
            f"the sum of the radii ({options.target_radius_km + options.impactor_radius_km:g} km)"
        )

    contact_speed = units.speed_to_code(
        1000.0 * _positive_scalar(options.contact_speed_km_s, "contact_speed_km_s")
    )
    angle = math.radians(
        _finite_scalar(options.contact_angle_degrees, "contact_angle_degrees")
    )
    if not 0.0 <= angle < math.pi / 2.0:
        raise ValueError("contact_angle_degrees must be within [0, 90)")

    target_layout = asteroid_shell_layout(options.target_material)
    impactor_layout = asteroid_shell_layout(options.impactor_material)
    target_mass = asteroid_layered_mass(target_radius, target_layout, units)
    impactor_mass = asteroid_layered_mass(impactor_radius, impactor_layout, units)

    # D7's rule, by D7's function: one budget split by volume, so the two bodies share a
    # particle mass rather than the target's particles outweighing the impactor's by the
    # cube of the radius ratio. The 128-per-body floor is what keeps a 10:1 pair's smaller
    # member representable at all -- see ADR 0048.
    target_count, impactor_count = _split_particles(
        options.fragments,
        (target_persistent_id, impactor_persistent_id),
        (target_radius, impactor_radius),
    )

    # Along cursor-local +X, with the target on the origin: the operator places the objects,
    # and the solver only needs their separation and the axis.
    initial = solve_encounter_initial_state(
        position_a=(0.0, 0.0, 0.0),
        position_b=(separation, 0.0, 0.0),
        mass_a=target_mass,
        mass_b=impactor_mass,
        radius_a=target_radius,
        radius_b=impactor_radius,
        contact_speed=contact_speed,
        contact_angle_radians=angle,
        gravitational_constant=units.gravitational_constant,
    )

    # ADR 0075. The range the operator authored is the range the template plays in, so the
    # frame duration is solved for it rather than fixed ahead of it. Contact is still
    # placed by Slice 3's accepted function off that duration, not asserted: the two have
    # to agree, and only one of them is allowed to be the answer.
    seconds_per_frame = template_impact_seconds_per_frame(
        contact_time=initial.contact_time, frame_start=frame_start, frame_end=frame_end
    )
    contact_frame = contact_boundary_frame(
        contact_time=initial.contact_time,
        seconds_per_frame=seconds_per_frame,
        frame_start=frame_start,
        maximum_boundary_frame=_IMPACT_MAX_BOUNDARY,
    )
    # Reaccumulate is a diagnostic checkpoint -- the frame at which the runtime asks
    # whether the largest remnant has settled -- so it is worth nothing on a frame that is
    # never played. Where the prediction lands inside the range it is kept; where it lands
    # beyond it, the checkpoint moves to the last frame the range can show it on and the
    # diagnostic answers honestly for that frame. Stretching the range to meet the
    # prediction instead is what authored a 320,590-frame demo out of a 250-frame scene.
    reaccumulate_frame = min(
        reaccumulate_boundary_frame(
            contact_frame=contact_frame,
            contact_speed=initial.contact_speed,
            total_mass=target_mass + impactor_mass,
            contact_radius=initial.contact_radius,
            gravitational_constant=units.gravitational_constant,
            seconds_per_frame=seconds_per_frame,
            maximum_boundary_frame=_IMPACT_MAX_BOUNDARY,
        ),
        frame_end - 1,
    )
    result_frame_end = frame_end

    target_plan = _template_solid(
        target_persistent_id,
        target_mass,
        target_radius,
        initial.velocity_a,
        target_count,
        options.target_material,
        target_layout,
    )
    impactor_plan = _template_solid(
        impactor_persistent_id,
        impactor_mass,
        impactor_radius,
        initial.velocity_b,
        impactor_count,
        options.impactor_material,
        impactor_layout,
    )

    stages = (
        TimelineStage(kind=TimelineStageKind.IMPACT, boundary_frame=contact_frame),
        TimelineStage(
            kind=TimelineStageKind.REACCUMULATE, boundary_frame=reaccumulate_frame
        ),
    )
    validate_stage_list(stages, frame_start=frame_start, frame_end=result_frame_end)

    return ScenarioPlan(
        unit_preset="PLANETARY",
        solver="AUTO",
        color_by=options.color_by,
        seconds_per_frame=seconds_per_frame,
        # The particle spacing, not a body radius (ADR 0169). Both bodies are sampled at one
        # lattice spacing (ADR 0166), so the target's spacing is the impactor's too, and the
        # pressure already keeps particles apart below it. Softening at the target's radius
        # held every particle near its surface at a quarter of its weight, so debris that
        # should fall back drifted off instead.
        softening=suggest_discreteness_softening(target_radius, target_count),
        substeps=suggest_substeps(
            seconds_per_frame,
            [(target_mass, target_radius), (impactor_mass, impactor_radius)],
            units.gravitational_constant,
        ),
        members=(target_plan, impactor_plan),
        timeline_stages=stages,
        display_groups=(
            (target_persistent_id, "SPH_POINTS"),
            (impactor_persistent_id, "SPH_POINTS"),
        ),
        result_frame_end=result_frame_end,
        # 0.8, swept inside one shutter at the moment this demo is named for (ADR 0116).
        # ADR 0091 withheld a size here and ADR 0105 could not lift it: every picture either
        # had was of frame 32-34 of 250, before the two bodies touch. ADR 0115 moved the
        # sweep inside the frozen instant, and the window it bought landed on frame 64
        # exactly -- aim 64, achieved 64, overshoot 0, inside the solved Impact stage of
        # 64-248 -- and drew all four sizes off that one held state, spanning zero frames on
        # the display clock and zero on the timeline. What the four pictures show: at 0.4 the
        # bodies are separated grains and read as dust rather than rock; at 0.8 both clouds
        # read as solid volumes and the two detached fragment clumps become legible; at 1.5
        # the silhouette is visibly inflated; at 3.0 the neck between impactor and target
        # fills in until the pair reads as one peanut rather than two bodies in contact. The
        # rule is the smallest swept size at which the bodies read as bodies -- below it the
        # scene is a nebula, above it the size invents extent the simulation does not have.
        # It rides the SPEED arm only, because that is the arm the picture was taken on.
        look=look_for_color_by(options.color_by, 0.8),
        tree_theta=DEFAULT_TREE_THETA,
    )


# -- Moon Birth template (ADR 0094) --------------------------------------------------------

# The owner's requirement is that the addon can show the Moon being born, live. This template
# authors that scene and nothing else: a proto-Earth and a Mars-sized Theia on a grazing
# collision course, both layered Solids, close enough that the approach is a few seconds of
# the timeline and slow enough that the pair routes to the `giant_impact` backend. That
# backend owns the rest -- its frame length, its substeps, and the moon guide that turns the
# debris disc into one moon a viewer can see.
#
# The numbers are the reference run's, not a claim about the real event: Earth's and Mars's
# radii, a contact at the pair's mutual escape speed, and 46 degrees, the angle at which the
# spike's impact parameter (b = sin 46 = 0.72) put a moon in orbit at every resolution tried.
MOON_BIRTH_TARGET_RADIUS_KM = 6371.0
MOON_BIRTH_IMPACTOR_RADIUS_KM = 3300.0
MOON_BIRTH_CONTACT_SPEED_ESCAPE = 1.0
"""Contact speed as a multiple of the pair's mutual escape speed at contact."""

MOON_BIRTH_CONTACT_SPEED_RANGE = (1.0, 2.5)
"""What the Contact Speed property admits, in the same multiples.

The floor is the slowest contact a pair falling in from rest can make. The ceiling sits under
`GIANT_IMPACT_SPEED_RATIO` (3.0) with room to spare: past it the scene would silently route
to the fragmenting impact backend, which forms no moon, and a template that stopped being
the thing its name says at the top of a slider would be a template that lies.
"""

MOON_BIRTH_CONTACT_ANGLE_DEGREES = 46.0
MOON_BIRTH_COUNT = 50_000
"""Particles for the pair, split by volume, and the Moon Birth operator's default. At 50,000 the
shipped timing formed a moon inside ``MOON_BIRTH_FRAMES`` headless at about 6 ms of engine time
a frame on the reference GPU, and played to a moon in a window at 21.7 of 24 fps (ADR 0094). A
larger count costs scrub-cache room as well as time: the cache holds every played frame, and 400
frames of 50,000 bodies already fill most of it. It is a default, not an ADR 0076 headroom
count: the window measured 16.4 ms of whole-frame work, over the half budget that rule allows,
and ADR 0094 records the exception."""

MOON_BIRTH_SEPARATION_RADII = 3.0
"""Initial centre separation, in proto-Earth radii, along cursor-local +X.

Not a property. It only sets how long the fall-in takes -- about 15 frames at the shipped
defaults -- and a wider one buys nothing but a longer wait for the impact.
"""

MOON_BIRTH_FRAMES = 400
"""Frames from the start of the run the scene needs to show a moon.

The moon guide starts 20 dynamical times into the run, 200 frames at 0.1 of one per frame,
and the reference runs at the default count held one lunar mass on the anchor by frame 300
and a finished moon by frame 400. The range is extended to reach this rather than squeezed
into the one the scene had (ADR 0094 records that exception to ADR 0075): the frame length is
the giant-impact backend's, not the template's to choose.

It is also a ceiling. Every played frame is saved for scrubbing and the scrub cache never
evicts, so a range longer than the cache holds at ``MOON_BIRTH_COUNT`` stops playing partway
through; 400 records at 50,000 bodies is about 940 MB of the 1 GiB.
"""

MOON_BIRTH_MATERIAL = "synthetic-rock"
"""Both bodies' outer shell, over the shared layout's Synthetic Iron core."""


@dataclass(frozen=True)
class MoonBirthOptions:
    """The Moon Birth template's Add Last Operation properties.

    As with Asteroid Impact, the masses are derived from each radius and the shared layout,
    never typed: a mass beside a radius and a material is three answers to one question.
    """

    target_radius_km: float = MOON_BIRTH_TARGET_RADIUS_KM
    impactor_radius_km: float = MOON_BIRTH_IMPACTOR_RADIUS_KM
    contact_speed_escape: float = MOON_BIRTH_CONTACT_SPEED_ESCAPE
    contact_angle_degrees: float = MOON_BIRTH_CONTACT_ANGLE_DEGREES
    count: int = MOON_BIRTH_COUNT
    color_by: str = "SPEED"


def build_moon_birth(
    target_persistent_id: str,
    impactor_persistent_id: str,
    *,
    options: MoonBirthOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author the Planetary giant impact that forms a moon (ADR 0094).

    The frame length is :func:`dynamical_frame_seconds` of the proto-Earth, the same function
    compile calibration re-times a giant-impact scene with, so the plan and the running scene
    agree about how long a frame is. The range extends to :data:`MOON_BIRTH_FRAMES` past the
    start when the scene's own is shorter, because a moon that forms after the last frame is
    a moon nobody sees.
    """

    _require_frame_range(frame_start, frame_end)
    if target_persistent_id == impactor_persistent_id:
        raise ValueError("the proto-Earth and Theia must be different objects")
    if isinstance(options.count, bool) or not isinstance(options.count, int):
        raise ValueError("count must be an integer")

    target_radius = units.length_to_code(
        1000.0 * _positive_scalar(options.target_radius_km, "target_radius_km")
    )
    impactor_radius = units.length_to_code(
        1000.0 * _positive_scalar(options.impactor_radius_km, "impactor_radius_km")
    )
    if impactor_radius > target_radius:
        # The backend times the scene off the heavier body and the guide builds the moon out
        # of what orbits it. Swapping the two would still collide, but not as this template.
        raise ValueError("Theia must not be larger than the proto-Earth")

    low, high = MOON_BIRTH_CONTACT_SPEED_RANGE
    factor = _finite_scalar(options.contact_speed_escape, "contact_speed_escape")
    if not low <= factor <= high:
        raise ValueError(
            f"contact_speed_escape must be within [{low:g}, {high:g}] times the escape speed"
        )
    angle = math.radians(
        _finite_scalar(options.contact_angle_degrees, "contact_angle_degrees")
    )
    if not 0.0 <= angle < math.pi / 2.0:
        raise ValueError("contact_angle_degrees must be within [0, 90)")

    layout = asteroid_shell_layout(MOON_BIRTH_MATERIAL)
    target_mass = asteroid_layered_mass(target_radius, layout, units)
    impactor_mass = asteroid_layered_mass(impactor_radius, layout, units)
    total_mass = target_mass + impactor_mass
    gravitational_constant = units.gravitational_constant

    target_count, impactor_count = _split_particles(
        options.count,
        (target_persistent_id, impactor_persistent_id),
        (target_radius, impactor_radius),
    )

    contact_radius = target_radius + impactor_radius
    escape_speed = math.sqrt(2.0 * gravitational_constant * total_mass / contact_radius)
    initial = solve_encounter_initial_state(
        position_a=(0.0, 0.0, 0.0),
        position_b=(MOON_BIRTH_SEPARATION_RADII * target_radius, 0.0, 0.0),
        mass_a=target_mass,
        mass_b=impactor_mass,
        radius_a=target_radius,
        radius_b=impactor_radius,
        contact_speed=factor * escape_speed,
        contact_angle_radians=angle,
        gravitational_constant=gravitational_constant,
    )

    seconds_per_frame = dynamical_frame_seconds(
        total_mass, target_radius, gravitational_constant
    )
    result_frame_end = max(frame_end, frame_start + MOON_BIRTH_FRAMES)
    if result_frame_end > _IMPACT_FRAME_CEILING:
        raise ValueError(
            "the Moon Birth scene needs frames beyond Blender's timeline ceiling "
            f"{_IMPACT_FRAME_CEILING}; start it earlier"
        )

    return ScenarioPlan(
        unit_preset="PLANETARY",
        solver="AUTO",
        color_by=options.color_by,
        seconds_per_frame=seconds_per_frame,
        softening=suggest_softening([target_radius, impactor_radius]),
        substeps=suggest_substeps(
            seconds_per_frame,
            [(target_mass, target_radius), (impactor_mass, impactor_radius)],
            gravitational_constant,
        ),
        members=(
            _template_solid(
                target_persistent_id,
                target_mass,
                target_radius,
                initial.velocity_a,
                target_count,
                MOON_BIRTH_MATERIAL,
                layout,
            ),
            _template_solid(
                impactor_persistent_id,
                impactor_mass,
                impactor_radius,
                initial.velocity_b,
                impactor_count,
                MOON_BIRTH_MATERIAL,
                layout,
            ),
        ),
        # One stage: the giant-impact backend has no stage machinery of its own, and the
        # moment worth photographing is the end, where the moon is.
        timeline_stages=_evolve_stages(frame_start, result_frame_end),
        display_groups=(
            (target_persistent_id, "SPH_POINTS"),
            (impactor_persistent_id, "SPH_POINTS"),
        ),
        result_frame_end=result_frame_end,
        look=look_for_color_by(options.color_by),
        tree_theta=DEFAULT_TREE_THETA,
    )


# D8's Protoplanetary Disk. Stellar units, so lengths are au and masses are nominal solar
# masses. Two of these are *operator inputs to proxy geometry* rather than role fields:
# outer radius and full thickness resize the annular-prism proxy so its evaluated extents
# are exactly `E0 = E1 = 2 * outer_radius` and `E2 = full_thickness`. Only the inner radius
# persists as a role/descriptor field, because ADR 0040 section 5 makes it the one disk
# quantity no object can imply. Duplicating outer radius or thickness as role fields would
# create two answers to one question the moment a user scaled the proxy.
DISK_CENTRAL_MASS = 1.0
DISK_CENTRAL_RADIUS = SOLAR_RADIUS_AU
"""One nominal solar radius in au. The same ratio the Binary template uses, by reference:
two copies of it would be two constants to keep equal, and nothing would notice if they
stopped being."""

DISK_MASS = 0.01
DISK_COUNT = 2_000
"""Chosen for headroom by measurement (ADR 0076), replacing D8's 32,768.

Swept at 1,000 / 2,000 / 3,000 / 4,500 / 6,000 / 9,000 bodies. 2,000 costs 8.887 ms of the
20.833 ms budget; 3,000 costs 12.079 and is over half. This template supplies the sharpest
evidence for why the margin is generous rather than snug: at 6,000 it is two percent over one
whole budget --- 21.306 ms --- and its achieved rate has already fallen from 0.971 of the rate
it authored to 0.628. Two percent of cost for a third of the frame rate is the owner's *"a demo
that's a coin flip isn't a demo"* in numbers.

The disk authors eight substeps against the cluster's one, which is why it reaches the same
budget at a fifth of the bodies; the substep count follows the orbital timescale and does not
move with the count, which ``tests/unit/test_template_headroom_counts.py`` pins.
"""

DISK_SEED = 0
DISK_INNER_RADIUS = 1.0
DISK_OUTER_RADIUS = 10.0
DISK_FULL_THICKNESS = 0.2

DISK_SURFACE_DENSITY_EXPONENT = 1.5
"""The Minimum Mass Solar Nebula's exponent, cited rather than measured (ADR 0084).

`sigma ~ r**-1.5` is the profile Hayashi (1981) obtained by taking the solar system's planets,
restoring each to solar composition, and smearing the result across the annulus it occupies. It
is the reference profile for the object this template depicts, and it is a *citation*: unlike
Galaxy Collision's exponent, which ADR 0083 fixed by measuring the rotation curve it produces,
this one changes no dynamics at all. Every particle here orbits the central Body, so
`keplerian_velocities` gives it the softened point-mass circular speed wherever it sits
(`sqrt(G M / r)` outside the softening core, ADR 0188); moving the particles inward
moves the picture and nothing else. That is why no measurement in this repository can choose it
and why ADR 0084 records it as an interim under ADR 0078 decision 8 rather than as swept.

What it buys, measured off-device at the shipped constants: the fraction of the disk inside half
the outer radius goes from **0.2424 to 0.5717** (drawn 0.2421 / 0.5731 at 200,000 samples), and
the median particle radius from 7.094 to 4.315 au. ADR 0078 diagnosed the hollow ring from that
first number. The surface density then runs 31.6 times higher at the inner edge than the outer.

Safe at the shipped geometry, checked rather than assumed. `DISK_INNER_RADIUS` is positive, so
the `p >= 2` no-hole refusal in `_profiled_annulus_radii` is not in reach; `q = 0.5` is far from
the float64 overflow bound; and neither substep term moves, because both read radii only ---
measured self-gravity 1, inner-orbit 8, shipped maximum 8, unchanged by the profile.
"""


@dataclass(frozen=True)
class ProtoplanetaryDiskOptions:
    """The Protoplanetary Disk template's authored inputs (D8)."""

    central_mass: float = DISK_CENTRAL_MASS
    central_radius: float = DISK_CENTRAL_RADIUS
    disk_mass: float = DISK_MASS
    count: int = DISK_COUNT
    seed: int = DISK_SEED
    inner_radius: float = DISK_INNER_RADIUS
    outer_radius: float = DISK_OUTER_RADIUS
    full_thickness: float = DISK_FULL_THICKNESS
    surface_density_exponent: float = DISK_SURFACE_DENSITY_EXPONENT
    counter_clockwise: bool = True


def disk_proxy_extents(
    outer_radius: float, full_thickness: float
) -> tuple[float, float, float]:
    """The evaluated extents the proxy must have, given the two operator inputs (D8).

    `E0 = E1 = 2 * outer_radius` and `E2 = full_thickness`. Stated as a function because
    both the operator that builds the proxy and the tests that check it need the same
    equations, and because after the operator finishes ADR 0040 section 5 is authoritative:
    any later transform or modifier changes the extents compilation reads, and this is only
    what the template starts them at.
    """

    outer = _positive_scalar(outer_radius, "outer_radius")
    thickness = _positive_scalar(full_thickness, "full_thickness")
    return (2.0 * outer, 2.0 * outer, thickness)


def build_protoplanetary_disk(
    central: MemberFacts,
    disk: MemberFacts,
    *,
    options: ProtoplanetaryDiskOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author D8's central Body plus oriented Disk Annulus Cloud, linked by KEPLERIAN.

    The central Body supplies the mass the disk orbits, so the two members are linked by
    the Body's persistent id -- never its name, which a rename would break, and never an
    object reference, which does not serialize.

    Timing is one orbital period at the disk's outer radius spread across 250 frames,
    matching the constant Slice 3's distribution rule uses. The outer radius rather than a
    mean: it is the slowest orbit in the disk, so a frame step sized for it resolves every
    faster one, while a step sized for the inner edge would resolve the outer disk far more
    finely than anyone needs and cost 30x the frames to see one revolution.

    The central Body carries no material: ADR 0037 gives Body no physics Material field, and
    inventing one so the template could fill it would put a field on the surface that
    nothing else in the product reads.
    """

    _require_frame_range(frame_start, frame_end)
    if isinstance(options.count, bool) or not isinstance(options.count, int):
        raise ValueError("count must be an integer")
    if options.count <= 0:
        raise ValueError("count must be positive")
    if isinstance(options.seed, bool) or not isinstance(options.seed, int):
        raise ValueError("seed must be an integer")
    if options.seed < 0:
        raise ValueError("seed must not be negative")
    if not isinstance(options.counter_clockwise, bool):
        raise ValueError("counter_clockwise must be a boolean")

    central_mass = _positive_scalar(options.central_mass, "central_mass")
    central_radius = _positive_scalar(options.central_radius, "central_radius")
    disk_mass = _positive_scalar(options.disk_mass, "disk_mass")
    outer_radius = _positive_scalar(options.outer_radius, "outer_radius")
    thickness = _positive_scalar(options.full_thickness, "full_thickness")
    inner_radius = _finite_scalar(options.inner_radius, "inner_radius")
    if inner_radius < 0.0:
        raise ValueError("inner_radius must not be negative")
    exponent = _finite_scalar(options.surface_density_exponent, "surface_density_exponent")
    if exponent < 0.0:
        raise ValueError("surface_density_exponent must not be negative")
    if inner_radius >= outer_radius:
        raise ValueError(
            f"inner_radius must be less than outer_radius, got {inner_radius} and "
            f"{outer_radius}"
        )
    if central.persistent_id == disk.persistent_id:
        raise ValueError("the central Body and the disk must be different objects")

    seconds_per_frame = _disk_seconds_per_frame(
        outer_radius, units.gravitational_constant, central_mass
    )
    extents = disk_proxy_extents(outer_radius, thickness)

    central_plan = MemberPlan(
        persistent_id=central.persistent_id,
        role="BODY",
        mass=central_mass,
        radius=central_radius,
        velocity=(0.0, 0.0, 0.0),
        particle_count=1,
        sample_shape="BALL",
        material_key="",
        shell_layout=(),
        settings=(
            ("mass", central_mass),
            ("radius", central_radius),
            ("spin", (0.0, 0.0, 0.0)),
            ("pinned", False),
        ),
    )

    disk_plan = MemberPlan(
        persistent_id=disk.persistent_id,
        role="CLOUD",
        mass=disk_mass,
        radius=outer_radius,
        velocity=(0.0, 0.0, 0.0),
        particle_count=options.count,
        sample_shape="DISK_ANNULUS",
        material_key=disk.material_key,
        shell_layout=(),
        settings=(
            ("velocity_model", "KEPLERIAN"),
            ("keplerian_central_persistent_id", central.persistent_id),
            ("keplerian_counter_clockwise", options.counter_clockwise),
            ("annulus_inner_radius", inner_radius),
            # ADR 0078 decision 4's disc half, reaching the demo it was built for (ADR 0084).
            # The mechanism landed with ADR 0078; until now no shipped template declared a
            # nonzero exponent, so the sampler drew uniform-in-area and the disc read as the
            # hollow ring ADR 0078 diagnosed. Physics is untouched: KEPLERIAN gives each
            # particle its circular speed wherever it sits (softened, ADR 0188), so this
            # moves the picture only.
            ("annulus_surface_density_exponent", exponent),
            ("eccentricity_jitter", 0.0),
            ("mass_jitter", 0.0),
            # Retained at its shipped value and inactive: KEPLERIAN never reads it, and
            # zeroing it would present as an authored choice of a cold cloud if the user
            # later switched to VIRIALIZE.
            ("support_fraction", 1.0),
            ("sampling_seed", options.seed),
            ("count", options.count),
            ("total_mass", disk_mass),
            # Proxy geometry, not role fields: the operator resizes the mesh to these, and
            # from then on the evaluated extents are D4's authority.
            ("proxy_extents", extents),
        ),
    )

    # Two distinct timescales bind the step here, and the fast one is orbital, not
    # collapse (ADR 0057). The disk Cloud is a genuine diffuse distribution, so its own
    # self-gravity free-fall time is a real constraint -- but the central Body is a point
    # mass whose `central_radius` is a photosphere size, not a self-gravitating extent, so
    # feeding it to the cloud advisor computes the star's stellar dynamical time and
    # demands tens of thousands of substeps the motion never needs. Size the orbital part
    # from the inner-edge Keplerian orbit around the central mass (the fastest orbit in the
    # disk) and take the binding maximum of the two. A full disk (inner_radius == 0) has no
    # inner edge to resolve, so only the self-gravity term applies; the shipped template's
    # default inner_radius is positive and takes the maximum.
    disk_self_gravity_substeps = suggest_substeps(
        seconds_per_frame,
        [(disk_mass, outer_radius)],
        units.gravitational_constant,
    )
    if inner_radius > 0.0:
        substeps = max(
            disk_self_gravity_substeps,
            suggest_orbit_substeps(
                seconds_per_frame,
                inner_radius,
                central_mass,
                units.gravitational_constant,
            ),
        )
    else:
        substeps = disk_self_gravity_substeps

    # The softening splits the same way the substeps just did, and ADR 0057 fixed only one of
    # the two halves (ADR 0079). `suggest_softening` sizes a *cold self-gravitating cloud* so it
    # breathes instead of bouncing through a singular core, and the disk Cloud is one, so its
    # outer radius belongs here. `central_radius` does not: it is a photosphere, not a
    # self-gravitating extent, and since the rule is a maximum, handing it in at 0.00465 au next
    # to a 10 au disk did nothing anyway. What the rule cannot see is that this scene also
    # contains orbits, and softening at the disk's own outer radius destroys every one of them:
    # the pull at the inner edge falls to 0.099 percent of Newtonian. When ADR 0079 was written
    # `keplerian_velocities` handed out unsoftened speeds, so the particles started 31.9 times
    # too fast and left on the first frame; since ADR 0188 the speeds match the softened pull,
    # so the disk would stay, but as a near-solid-body rotator with no Keplerian shear, which
    # is not the object this template depicts. The orbital ceiling is still the binding
    # constraint, so take the minimum. At the shipped value the softened inner-edge speed is
    # 0.50 percent below `sqrt(G M / r)` (ADR 0188).
    softening = _softening_bounded_by_orbit([outer_radius], inner_radius)

    return ScenarioPlan(
        unit_preset="STELLAR",
        solver="AUTO",
        color_by="SPEED",
        seconds_per_frame=seconds_per_frame,
        softening=softening,
        substeps=substeps,
        members=(central_plan, disk_plan),
        timeline_stages=_evolve_stages(frame_start, frame_end),
        display_groups=(
            (central.persistent_id, "BODIES"),
            (disk.persistent_id, "STAR_POINTS"),
        ),
        result_frame_end=frame_end,
        # 0.8, same sweep: 9.37 -> 23.34 -> 38.64 -> 47.74, a 46% disc share at 0.8. The
        # thing this demo has to show is a tilted plane with a hot fast inner edge against
        # a slow outer rim, and at 0.8 both the tilt and that colour gradient survive.
        look=swept_look(0.8),
        tree_theta=DISK_TREE_THETA,
    )


def _disk_seconds_per_frame(
    outer_radius: float, gravity: float, central_mass: float
) -> float:
    """One outer-edge orbital period across 250 frames.

    `T = 2*pi*sqrt(r^3 / (G*M))`. The same 250 the distribution rule uses, so the two
    templates present the same "one characteristic time fills the default range" behaviour
    rather than two unexplained cadences.
    """

    period = 2.0 * math.pi * math.sqrt(outer_radius**3 / (gravity * central_mass))
    result = period / 250.0
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("protoplanetary disk seconds_per_frame is outside float range")
    return result


# -- Galaxy Collision template (WP-B2 sub-commit 3) ----------------------------------------
#
# The fifth Add-menu template, and the one the advisor primitives were literally ported from:
# `pair_orbit`, `suggest_softening` and `fit_timeline` each name "the galaxy-merger demo" as
# their source scene (advisor.py). It is Binary System's two-body encounter with each point
# Body replaced by a virialized Cloud -- two Galactic spheroids released at apocentre on a
# bound eccentric orbit, so they fall together, interpenetrate at pericentre half a period
# later, and the frame range is fit to exactly that one period so the collision lands at
# mid-timeline where it cannot fall off either end.
#
# Fixture numbers, stated for owner review (Galactic units: length kpc, mass 1e9 Msun, time
# Myr). Two equal galaxies -- a 1:1 major merger, the symmetric case:
#   * GALAXY_MASS = 100.0   -> 1e11 Msun each, a Milky-Way-scale spheroid.
#   * GALAXY_RADIUS = 10.0  -> 10 kpc each.
#   * GALAXY_COUNT = 1500 per galaxy -> 3,000 across the pair. Chosen for headroom by
#     measurement (ADR 0076); see the docstring on the constant. This replaces D8's 25,000,
#     whose stated reason was that 50,000 across the pair matched CLUSTER_COUNT so "one click"
#     cost the same. That reason does not survive measurement: a frame's cost is the substep
#     count times the per-body primitive, and Galaxy Collision authors five substeps against
#     the Cluster's one, so an equal body count was never an equal cost. The two counts are
#     now unequal for exactly that reason -- 3,000 here against the Cluster's 5,000 -- and
#     each is the largest swept count that leaves the same margin.
#   * GALAXY_SEPARATION = 60.0 kpc apocentre -> six galaxy radii, well inside the regime
#     `suggest_softening` assumes (the inter-body orbit stays Newtonian at separations several
#     times the radius) yet far enough that the pair starts as two distinct galaxies.
#   * GALAXY_ECCENTRICITY = 0.7 -> pericentre = 60*(1-e)/(1+e) = 10.6 kpc centre separation,
#     about one galaxy radius, so the two deeply interpenetrate at closest approach. Lowering
#     it widens the encounter toward a grazing tidal flyby; raising it deepens the plunge.
GALAXY_MASS = 100.0
GALAXY_RADIUS = 10.0

GALAXY_THICKNESS = 2.0
"""Full disk thickness in kpc, so ``h / R = 0.1`` at the shipped radius (ADR 0083).

A galaxy is a disk, and the thing this demo is named for --- two disks tearing tidal tails out
of each other --- cannot happen to two spheroids, which is what it used to author. The ratio is
the observed one for a spiral's stellar disk to within a factor of about two, and it is what
makes the disk read as a disk from a camera that is not exactly edge-on.
"""

GALAXY_INNER_RADIUS = 0.0
"""No hole: a galaxy disk is filled to its centre, and a hole would strand particles (ADR 0083).

The field exists because a protoplanetary disk needs it, and the temptation was to reuse it here
for a bulge-shaped gap. Measurement refused it. An annulus pulls its own inner-edge particles
*outward*, so at the inner rim the mean radial acceleration is not inward and those particles get
no rotational support at all: a 1 kpc hole leaves 1.3 percent of the disk unsupported and a 2 kpc
hole 6.6 percent. That is physically correct and visibly wrong -- the stranded particles fall
straight through the middle. At zero the unsupported fraction is exactly zero.
"""

GALAXY_SURFACE_DENSITY_EXPONENT = 1.0
"""``Sigma ~ r**-1``, which is the exponent that makes the rotation curve flat (ADR 0083).

Not a tuned number: a surface density falling as ``1/r`` encloses mass in proportion to radius,
``M(<r) ~ r``, and a circular orbit in that mass distribution has ``v**2 = G M(<r) / r =
constant``. A flat rotation curve is the defining observational signature of a spiral galaxy,
so the exponent is chosen by what the demo is depicting and then confirmed by measurement.

Measured at the shipped geometry over three seeds. Uniform surface density (exponent 0) gives an
outer/inner speed ratio of 2.334 --- a steeply rising curve, not a galaxy --- and, worse, a
seed-to-seed spread of 1.517, because a uniform disk puts almost no mass in its inner region and
whichever particles happen to land near the probe then set the answer. At exponent 1.0 the ratio
is 1.151 and the spread falls to 0.123: the same curve every time. Exponent 0.5 sits between at
1.543; 1.5 overshoots into a falling curve at 0.885.
"""

GALAXY_COUNT = 1_500
"""Particles *per galaxy*, so the pair carries twice this. Chosen for headroom (ADR 0076).

Swept in a real window at 1,000 / 2,000 / 3,000 / 4,000 / 6,000 / 9,000 bodies across the pair.
3,000 total costs 9.992 ms of the 20.833 ms service budget; 4,000 costs 12.252 and is over half.

The count that ships is the one place this template's asymmetry has to be remembered twice: the
sweep measured totals and the property authors a half, so a change here moves two particles for
every one it appears to. A test asserts the plan's two clouds sum to twice this constant.

Lowering the count costs this demo less of its picture than the number suggests. Across the same
sweep the lit-pixel count rose only from 106,129 at 1,000 bodies to 191,808 at 9,000 --- a ninefold
body count for 1.8x the visible cloud --- because the bodies overlap on screen. At the shipped
3,000 the demo keeps 76 percent of the brightest measurement at a third of its bodies.
"""

GALAXY_SEPARATION = 60.0
GALAXY_ECCENTRICITY = 0.7
GALAXY_SEED = 0

GALAXY_HALO_MASS_FRACTION = 0.8
"""Share of each galaxy's mass carried by its halo rather than its disk (ADR 0087).

ADR 0083 gave the disk a rotation curve measured in its own gravity and said, in its own
honest limits, that a self-gravitating disk with no velocity dispersion has nothing holding
it up against itself. Integrating one galaxy for the whole shipped run, off device, at the
shipped 1,500 bodies and 5 substeps, showed what that costs: the disk's RMS thickness goes
from 0.576 to 4.297 kpc and its median radius from 5.194 to 14.194 kpc. It does not collide
with anything -- it evaporates on its own.

The remedy is the one both research N-body and VFX practice already use: a galaxy is not a
disk, it is a disk inside a much larger and much heavier halo. Neither industry ever ships a
fully self-gravitating disk at zero dispersion; research codes add a dispersion *and* a
dominant halo, and VFX drops the disk's self-gravity entirely. This template picked the one
configuration both treatments reject.

0.8 measured against 0.0, 0.5, 0.7 and 0.9 at four halo radii. Below 0.7 the disk still grows
unstable modes at its outer radii; at 0.9 the disk is *thicker* at the end of the run than at
0.8, because what is left of the thickening is softened-force heating rather than the disk's
own gravity, and a deeper halo does not touch it. The total stays ``GALAXY_MASS`` either way,
so the encounter orbit, its pericentre and its timing are exactly what ADR 0083 shipped.
"""

GALAXY_HALO_RADIUS_FRACTION = 0.7
"""Halo radius as a multiple of the disk radius, so 7.0 kpc at the shipped 10.0 (ADR 0087).

Extended, not compact, and that is the whole reason this fix is affordable. ``suggest_substeps``
sizes the step from the free-fall time at a cloud's *mean* density, so mass spread over a large
radius is cheap and the same mass in a compact core is not: a bulge concentrated enough to
dominate the disk's inner gravity priced at 24 to 132 substeps against the shipped 5. A halo
at 0.7 of the disk radius prices at 7 -- 1.4x the solve -- and closes every resolvable unstable
mode at every radius.

Measured against 0.5, 1.0 and 1.5 of the disk radius. At 0.5 the halo is dense enough to cost
11 substeps for a disk no better held. At 1.0 the nearest-neighbour softening it forces (1.125
kpc) exceeds half the disk's own thickness, which would smear the disk into its own halo.
"""

GALAXY_HALO_COUNT_DIVISOR = 3
"""One body in three goes to the halo; the rest stay in the disk (ADR 0087).

The body budget is *not* raised. ADR 0076 measured 1,500 per galaxy as the largest count that
plays with real headroom, and that measurement is preserved by splitting the same budget rather
than adding to it. The split is what sets the softening, because ``suggest_neighbour_softening``
answers per population and the plan takes the larger: at 1,500 the halo's spacing (0.788 kpc)
dominates the disk's (0.474), and it stays below half the disk's 2.0 kpc thickness, which a
larger halo radius would not. Swept at 300/500/750/1,000 halo bodies -- the cost is flat across
all four, so the split is chosen by softening, not by speed.
"""

GALAXY_HALO_SUPPORT_FRACTION = 1.09
"""Virial support for the halo, in the field it is actually in -- disk included (ADR 0087).

``virialized_velocities`` normalises against the *unsoftened* uniform-sphere reference
``3 G M^2 / (5 R)``, and ADR 0081 recorded what happens when that reference is not the field
the solver integrates: the old ``VIRIALIZE`` galaxies were super-virial and swelled from 10 to
67 kpc instead of colliding. So this number is not 1.0 and is not a guess. It is measured
directly from the sampled configuration as ``2K = -sum_i m_i r_i . a_i`` over the halo's own
particles, with the acceleration taken from *every* particle in the System -- the disk sitting
inside the halo pulls on it too, which is why the answer exceeds 1.0.

Measured across an eightfold body count at three seeds: 1.068 at 750 bodies per galaxy, 1.085
at 1,500, 1.104 at 3,000, 1.115 at 6,000, with a seed-to-seed spread of at most 0.005 at fixed
count. It is a property of the geometry, not of the count, so a single rounded constant carries
it; the residual drift with count is the softening falling away toward the unsoftened limit.
"""

GALAXY_SUPPORT_FRACTION = 1.0
"""Retained at its shipped value and inert under ``ROTATION_CURVE`` (ADR 0083).

It was this template's virial support while the galaxies were ``VIRIALIZE`` balls, and that is
exactly the contract ADR 0081 found broken here: ``virialized_velocities`` balances against the
*unsoftened* binding energy, so at the softening the plan actually shipped the two clouds were
super-virial and swelled instead of colliding. ``ROTATION_CURVE`` measures its support in the
same softened field the solver integrates, so the mismatch cannot arise and this number stops
being read. It stays at 1.0 for the reason the Keplerian fields stay set on the old plan: zeroing
it would present as an authored choice of a cold cloud if a user switched the model back.
"""

DISK_TREE_THETA = 0.8
"""The Disk opens tree cells at the widest angle GRAV-002 certifies (Disk try 3).

The disk's own pull is about 1 percent of the star's, which is summed exactly every substep
(ADR 0144), so the tree's error is a percent of a percent of the motion. ADR 0147's reach
criterion made the 0.6 walk dearer; the wider angle buys that time back. Auto's Play-time
accuracy check still decides whether the tree is used at all.
"""

GALAXY_TREE_THETA = 0.8
"""The Galaxy opens tree cells at the widest angle GRAV-002 certifies (ADR 0136).

At 100,000 bodies the tree walk was 29.8 of the 43.8 ms a frame's solve cost at the System's
default 0.6; at 0.8 it is 16.0 of 30.4 ms, measured interleaved in one process. The encounter
ruler passes at both angles with the same pericentre, tails and disk shape, and Auto still picks
the tree because its Play-time accuracy check passes (0.052 against the 0.064 bound at 0.8). The
worst single body later drifts past the bound at both angles by the same factor (2.4x at 0.6,
2.2x at 0.8); that is not new with this angle, and the ADR records it rather than hiding it.
"""

CLUSTER_TREE_THETA = 0.8
"""The Star Cluster opens tree cells at the widest angle GRAV-002 certifies (ADR 0230).

At 500,000 bodies the walk was 18.1-18.9 ms at the System's default 0.6 and is 8.2 ms at 0.8
(100,000 bodies: 3.5-3.7 to 1.7-2.0 ms), measured interleaved headless. The worst body's
normalised force error against a direct sum rises from 0.0018 to 0.0093 at 500,000 (0.0021 to
0.0094 at 100,000),
still more than six times inside the 0.064 bound Auto checks at Play. The cluster is one
relaxing ball whose look is set by its whole-ball potential, which the tree's far field carries
the same at either angle.
"""

GALAXY_STEPS_PER_FREE_FALL = 18
"""How finely the Galaxy's step resolves its densest cloud's free-fall time (ADR 0147).

The advisor's default is 40, which ADR 0087 turned into 7 substeps once the halo was added. At
100,000 bodies that is about 30 ms of solve a frame, two timer ticks, and 8.6 fps in a window.
18 gives 3 substeps: the encounter ruler passes (galaxies close to 6.7 against the 10.6 bound,
disks hold at 0.19 against 0.29), and each body's position stays within 0.07 galaxy radii
(median) of a 28-substep run up to the approach. A uniform sphere's free fall is also the
fastest time this scene has; the disks' orbits are several times longer.
"""


@dataclass(frozen=True)
class GalaxyCollisionOptions:
    """The Galaxy Collision template's Add Last Operation properties (WP-B2 sub-commit 3).

    Two equal galaxies, so a single ``galaxy_mass`` / ``galaxy_radius`` / ``count`` drives
    both; the pair is symmetric about the barycentre and the two apocentre speeds match.
    """

    galaxy_mass: float = GALAXY_MASS
    galaxy_radius: float = GALAXY_RADIUS
    count: int = GALAXY_COUNT
    separation: float = GALAXY_SEPARATION
    eccentricity: float = GALAXY_ECCENTRICITY
    seed: int = GALAXY_SEED


def build_galaxy_collision(
    first_persistent_id: str,
    second_persistent_id: str,
    *,
    first_halo_persistent_id: str,
    second_halo_persistent_id: str,
    options: GalaxyCollisionOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author two Galactic rotating disks on a colliding encounter.

    Binary System's two-body encounter with each point Body replaced by a ``ROTATION_CURVE``
    Cloud. The orbital arithmetic is Astro Orbit's shared primitive -- :func:`pair_orbit` at
    ``pericentre_ratio = (1 - e) / (1 + e)``, released at apoapsis -- so this template and
    Binary System cannot disagree about what an eccentricity means; and :func:`orbit_timing`
    fits exactly one period onto the frame range, so the pair starts apart at ``frame_start``,
    interpenetrates at pericentre at mid-timeline, and returns to apocentre at ``frame_end``.

    What each galaxy is, is the part this template got wrong for its whole life and ADR 0083
    fixes. A galaxy is a rotating disk, and the tidal tails this demo is named for are drawn
    out of a disk's ordered rotation by the other galaxy's pull; two pressure-supported balls
    with random internal motions have no ordered rotation to draw a tail out of, and so the
    demo could not depict its own name however well it was tuned. Each galaxy is now a
    ``DISK_ANNULUS`` with no hole, a thickness a tenth of its radius, a surface density falling
    as ``1/r``, and a ``ROTATION_CURVE`` velocity field.

    **A galaxy is a disk inside a halo, not a disk on its own (ADR 0087).** ADR 0083's own
    honest limits conceded that a self-gravitating disk with no velocity dispersion has nothing
    holding it up against itself, and integrating one galaxy for the whole shipped run off
    device showed the price: the disk's RMS thickness went from 0.576 to 4.297 kpc and its
    median radius from 5.194 to 14.194 kpc, with no companion anywhere near it. It evaporated
    on its own. Each galaxy is now two Clouds -- a ``BALL`` halo carrying 80 percent of the
    mass out to 0.7 of the disk radius, and the ``DISK_ANNULUS`` disk carrying the rest -- and
    the same measurement then reports 0.572 to 1.745 kpc of thickness and 5.288 to 5.591 kpc of
    radius. The total mass per galaxy is unchanged, so the encounter orbit is unchanged, and the
    body budget is split rather than raised, so ADR 0076's measured count is unchanged too. The
    cost is seven substeps against five.

    That is affordable only because the halo is *extended*. ``suggest_substeps`` prices a cloud
    by the free-fall time at its mean density, so the compact central bulge that would dominate
    the disk's inner gravity priced at 24 to 132 substeps, while a halo at 0.7 of the disk
    radius prices at seven and still closes every resolvable unstable mode at every radius.

    ``ROTATION_CURVE`` also closes the defect ADR 0081 recorded against the old plan. The
    ``VIRIALIZE`` support this template used to author is normalized against the *unsoftened*
    binding energy, so at the softening the plan shipped, the clouds were super-virial: measured,
    each 10 kpc cloud swelled to 67 kpc and the pair never came within 50.8 kpc of the authored
    10.6 kpc pericentre -- the galaxies did not collide. The new model measures each particle's
    circular speed from the disk's own *softened* self-gravity, in the same field the solver then
    integrates, so support and dynamics cannot disagree by construction. That is also why the
    softening below is not ADR 0081's binding ceiling: there is no mismatch left for a ceiling
    to bound.

    What turns two rotating disks into a collision is the bulk velocity: each galaxy's
    centre-of-mass apocentre speed is written as its Cloud's ``velocity``, which the sampler
    lays down as the whole cloud's drift on top of the rotation
    (``rotation_curve_velocities`` returns ``circular + bulk``). The velocities are perpendicular
    to the separation axis and opposed -- the first galaxy takes ``-Y`` and the second ``+Y``,
    exactly as ``build_binary`` opposes its two Bodies -- so both contribute positive angular
    momentum about ``+Z`` and the encounter swings through pericentre rather than the pair
    translating together. Equal masses, so the barycentre sits on the cursor the positions
    place it on and the two apocentre speeds are equal.

    Unlike ``build_cluster`` the cadence is the *orbital* period, not the internal free-fall
    time: when the two galaxies meet is set by the encounter, and a free-fall time derived
    from one galaxy's own mass and radius would place the collision at an arbitrary frame off
    either end of the timeline. There is correspondingly no controlling cloud to name, so
    neither Cloud carries ``controlling_persistent_id``.

    The point-mass orbit is only a *timing* device: it sets ``seconds_per_frame`` from the
    encounter, and near pericentre -- where the two spheroids overlap at about one radius --
    it stops being a good description of the motion, which is the collision the full N-body
    solver is there to resolve. ``AUTO`` solver so the runtime measures the compiled graph
    once and plays the backend that fits the frame slice (ADR 0056): at this combined body
    count the direct lane still beats the O(N log N) tree on current GPUs, and a pinned
    ``TREE`` would be the backend realtime admission refuses. ``SPEED`` colour so the tidal
    structure reads as it forms.

    No material is assigned: ADR 0037 gives Cloud no physics Material field, and each output
    instances its generated point source plus whatever ordinary Blender material the user
    later puts on it.
    """

    _require_frame_range(frame_start, frame_end)
    if first_persistent_id == second_persistent_id:
        raise ValueError("the two galaxies must be different objects")
    identifiers = (
        first_persistent_id,
        second_persistent_id,
        first_halo_persistent_id,
        second_halo_persistent_id,
    )
    if len(set(identifiers)) != len(identifiers):
        # Four objects now, not two (ADR 0087). A repeated id would silently make one member
        # overwrite another's particles rather than adding a halo to the galaxy.
        raise ValueError("each disk and each halo must be a different object")
    if isinstance(options.count, bool) or not isinstance(options.count, int):
        raise ValueError("count must be an integer")
    if options.count < 2 * GALAXY_HALO_COUNT_DIVISOR:
        # The budget is split (ADR 0087), and both halves have a floor of two: VIRIALIZE has
        # no internal motion to give a single particle once its own centre of mass is removed,
        # and ROTATION_CURVE has no mass but a particle's own to hold it up. Refusing here
        # names the template property rather than surfacing the same limit as a mid-sample
        # error from whichever half happens to round down first.
        raise ValueError(
            f"count must be at least {2 * GALAXY_HALO_COUNT_DIVISOR}: each galaxy's bodies "
            f"are split between a disk and a halo, and each half needs at least two"
        )
    if isinstance(options.seed, bool) or not isinstance(options.seed, int):
        raise ValueError("seed must be an integer")
    if options.seed < 0:
        raise ValueError("seed must not be negative")
    mass = _positive_scalar(options.galaxy_mass, "galaxy_mass")
    radius = _positive_scalar(options.galaxy_radius, "galaxy_radius")
    separation = _positive_scalar(options.separation, "separation")
    eccentricity = _finite_scalar(options.eccentricity, "eccentricity")
    if not 0.0 <= eccentricity <= 0.95:
        raise ValueError("eccentricity must be within [0.0, 0.95]")
    if separation <= 2.0 * radius:
        raise ValueError(
            "the two galaxies overlap at apocentre: separation must exceed the sum of the "
            f"radii ({2.0 * radius:.6g}) so the pair starts as two distinct galaxies"
        )

    orbit = pair_orbit(
        mass,
        mass,
        separation,
        (1.0 - eccentricity) / (1.0 + eccentricity),
        units.gravitational_constant,
    )
    timing = orbit_timing(orbit.period, frame_start, frame_end)

    extents = disk_proxy_extents(radius, GALAXY_THICKNESS)
    # The volume the sampler will actually fill, which is what sets how tightly the disk is
    # packed -- the one quantity a radius cannot express, and the reason the ball rule this
    # template used to call is the wrong rule for a disk (ADR 0083).
    volume = annulus_volume(
        outer_radius=radius,
        inner_radius=GALAXY_INNER_RADIUS,
        half_thickness=0.5 * GALAXY_THICKNESS,
    )
    # ADR 0087. One galaxy is two clouds: a light disk inside a heavy, larger halo. The mass
    # and the body budget are *split*, never added to, so the encounter orbit above and the
    # measured count of ADR 0076 both survive the change untouched.
    halo_count = options.count // GALAXY_HALO_COUNT_DIVISOR
    disk_count = options.count - halo_count
    halo_mass = GALAXY_HALO_MASS_FRACTION * mass
    disk_mass = mass - halo_mass
    halo_radius = GALAXY_HALO_RADIUS_FRACTION * radius
    halo_volume = 4.0 / 3.0 * math.pi * halo_radius**3

    members = []
    for persistent_id, halo_persistent_id, speed, sign in (
        (
            first_persistent_id,
            first_halo_persistent_id,
            orbit.apocentre_speeds[0],
            -1.0,
        ),
        (
            second_persistent_id,
            second_halo_persistent_id,
            orbit.apocentre_speeds[1],
            1.0,
        ),
    ):
        members.append(
            MemberPlan(
                persistent_id=halo_persistent_id,
                role="CLOUD",
                mass=halo_mass,
                radius=halo_radius,
                # The same centre-of-mass drift as its disk: halo and disk are one galaxy on
                # one orbit, and giving them different bulk velocities would tear the galaxy
                # apart before the other one ever arrived.
                velocity=(0.0, sign * speed, 0.0),
                particle_count=halo_count,
                sample_shape="BALL",
                material_key="",
                shell_layout=(),
                settings=(
                    ("velocity_model", "VIRIALIZE"),
                    ("support_fraction", GALAXY_HALO_SUPPORT_FRACTION),
                    ("mass_jitter", 0.0),
                    ("eccentricity_jitter", 0.0),
                    # Retained at their shipped values and inactive: VIRIALIZE reads neither,
                    # and clearing them would present as an authored choice if a user switched
                    # the model to KEPLERIAN.
                    ("keplerian_counter_clockwise", True),
                    ("keplerian_central_persistent_id", ""),
                    ("annulus_inner_radius", 0.0),
                    ("sampling_seed", options.seed),
                    ("count", halo_count),
                    ("total_mass", halo_mass),
                ),
            )
        )
        members.append(
            MemberPlan(
                persistent_id=persistent_id,
                role="CLOUD",
                mass=disk_mass,
                radius=radius,
                velocity=(0.0, sign * speed, 0.0),
                particle_count=disk_count,
                sample_shape="DISK_ANNULUS",
                material_key="",
                shell_layout=(),
                settings=(
                    ("velocity_model", "ROTATION_CURVE"),
                    ("support_fraction", GALAXY_SUPPORT_FRACTION),
                    ("mass_jitter", 0.0),
                    ("eccentricity_jitter", 0.0),
                    # Both disks turn the same way about +Z, which is also the sense of the
                    # orbit: a prograde-prograde encounter is the configuration that produces
                    # the long tidal tails this demo is named for, because the particles on the
                    # outer edge stay in resonance with the companion's pull instead of
                    # sweeping past it (ADR 0083).
                    ("keplerian_counter_clockwise", True),
                    # Retained at its shipped value and inactive: ROTATION_CURVE names no
                    # central object, because it does not orbit one -- it measures the whole
                    # System's field and takes the speed that field asks for, halo included
                    # (ADR 0087). Clearing it would look like an authored choice if a user
                    # switched to KEPLERIAN.
                    ("keplerian_central_persistent_id", ""),
                    ("annulus_inner_radius", GALAXY_INNER_RADIUS),
                    (
                        "annulus_surface_density_exponent",
                        GALAXY_SURFACE_DENSITY_EXPONENT,
                    ),
                    # One seed for the pair: the sampler mixes each Cloud's binding id into the
                    # draw, so two galaxies at the same seed still get distinct particles
                    # rather than one being a mirror image of the other.
                    ("sampling_seed", options.seed),
                    ("count", disk_count),
                    ("total_mass", disk_mass),
                    # Proxy geometry, not a role field: the operator resizes the mesh to these
                    # and from then on the evaluated extents are the authority, exactly as for
                    # the protoplanetary disk.
                    ("proxy_extents", extents),
                ),
            )
        )

    return ScenarioPlan(
        unit_preset="GALACTIC",
        solver="AUTO",
        color_by="SPEED",
        seconds_per_frame=timing.seconds_per_frame,
        # The spacing between neighbouring stars in the disk the sampler is about to fill, not
        # ADR 0081's binding ceiling. That ceiling exists to bound a mismatch between authored
        # support and integrated dynamics, and ROTATION_CURVE has no such mismatch to bound --
        # it reads its speeds out of the same softened field. What is left to choose is the
        # length below which a close pair would scatter off each other instead of orbiting the
        # disk, and that is the nearest-neighbour distance `suggest_neighbour_softening`
        # derives. Measured count-independent: at 750, 1,500 and 3,000 bodies per galaxy the
        # curve it produces is the same curve, 0.13-0.14 code units at 1 kpc rising to 0.26 at
        # the rim, with a seed-to-seed spread of at most 19 percent outside the innermost bin.
        #
        # Two populations now, so the rule is asked twice and the plan takes the larger (ADR
        # 0087): one softening governs the whole System, and the smaller of the two answers
        # would let the sparser population's near pairs scatter off each other. At the shipped
        # split the halo sets it, 0.788 kpc against the disk's 0.474, and that is still below
        # half the disk's own thickness -- which is the bound a bigger halo would cross.
        softening=max(
            suggest_neighbour_softening(volume, disk_count),
            suggest_neighbour_softening(halo_volume, halo_count),
        ),
        # All four clouds, because the step has to resolve the fastest of them and the halo is
        # not automatically the slower one just because it is bigger. At 18 steps per free
        # fall, not the advisor's default 40 (ADR 0147): 3 substeps instead of 7. The encounter
        # ruler passes at 3 and each body lands within a few hundredths of a galaxy radius of
        # a 28-substep run before pericentre; after it, 14 and 28 already disagree by more.
        substeps=suggest_substeps(
            timing.seconds_per_frame,
            [
                (disk_mass, radius),
                (halo_mass, halo_radius),
                (disk_mass, radius),
                (halo_mass, halo_radius),
            ],
            units.gravitational_constant,
            steps_per_free_fall=GALAXY_STEPS_PER_FREE_FALL,
        ),
        members=tuple(members),
        timeline_stages=_evolve_stages(frame_start, frame_end),
        # The halo is drawn, not hidden. It is a *stellar* halo here rather than a dark one:
        # every body the solver pays for is a body the viewer can see, which is the same
        # single-display-path principle ADR 0037 states, and a third of the budget spent on
        # something invisible would be a third of the frame time spent on nothing.
        display_groups=(
            (first_halo_persistent_id, "STAR_POINTS"),
            (first_persistent_id, "STAR_POINTS"),
            (second_halo_persistent_id, "STAR_POINTS"),
            (second_persistent_id, "STAR_POINTS"),
        ),
        result_frame_end=frame_end,
        # 0.8, swept in four windows at the moment this demo is named for (ADR 0105). ADR
        # 0091 withheld a size here because every picture it had was of frame ~194 of 250,
        # long after the merger; ADRs 0101-0104 made the harness photograph the named moment,
        # and the sweep was re-run on four pictures that all land on frame 126 having skipped
        # no frame getting there. 6.10 -> 12.01 -> 19.44 -> 32.16 lit pixels per body against
        # discs of 12.6 -> 50.3 -> 176.7 -> 706.9 px, so the share of each disc that survives
        # overlap halves at every step -- 49 percent at 0.4, 24 at 0.8, 11 at 1.5, 5 at 3.0 --
        # and total lit area grows 97 percent for the first step but only 62 and 65 for the
        # two larger ones. The pictures say the same: at 0.4 the tidal material is there and
        # too faint to read, at 0.8 the purple tail separates from the cyan core and the
        # particles still read as stars, at 1.5 the core fuses into one opaque mass, at 3.0
        # it is overlapping bubbles. This is a denser scene than the other three swept
        # templates -- four clouds overlapping at pericentre -- which is why its disc share
        # at 0.8 is 24 percent against their 45 to 50, and is also why the step up to 1.5
        # costs more here than it does there.
        look=swept_look(0.8),
        tree_theta=GALAXY_TREE_THETA,
    )


# -- Impact builder (Seam 3F) --------------------------------------------------------------

_IMPACT_FRAME_CEILING = 1_048_574
_IMPACT_MAX_BOUNDARY = _IMPACT_FRAME_CEILING - 1
_FRAGMENT_K_MIN = Decimal("0.256")
_FRAGMENT_K_MAX = Decimal("2000")
_PARTICLES_PER_BODY_FLOOR = 128


def _proxy_radius(member: MemberFacts) -> float:
    """Half the maximum finite nonzero evaluated world dimension."""

    extents = tuple(
        _finite_scalar(extent, "dimensions") for extent in member.dimensions
    )
    finite = tuple(extent for extent in extents if extent != 0.0)
    if not finite:
        raise ValueError(
            f"member {member.name!r} requires finite nonzero evaluated dimensions"
        )
    return 0.5 * max(finite)


def _fragments_total(fragments_k: float) -> int:
    value = Decimal(str(_finite_scalar(fragments_k, "fragments_k")))
    if value < _FRAGMENT_K_MIN or value > _FRAGMENT_K_MAX:
        raise ValueError("fragments_k must be within [0.256, 2000.0]")
    return int((value * Decimal("1000")).quantize(Decimal("1"), ROUND_HALF_UP))


def _split_particles(
    total: int, names: tuple[str, ...], radii: tuple[float, ...]
) -> tuple[int, ...]:
    """Volume-weighted largest-remainder split with a 128-per-body reservation.

    Weighting by volume gives the bodies a common particle *volume*, and so comparable
    smoothing lengths across the contact interface -- which is the property the SPH density
    estimate is conditioned by. It does **not** equalise particle *mass*: two bodies of
    different bulk density end up with per-particle masses in that density ratio (see
    ADR 0048, which measures it for the shipped asteroid pair). Per-particle mass is
    per-body throughout the runtime, so that is representable rather than a defect.

    ``names`` is only the tie-break for the largest-remainder pass, so it takes strings
    rather than :class:`MemberFacts`: both impact planners need this rule, and one of them
    identifies its members by persistent id rather than by an evaluated object.
    """

    count = len(names)
    reserved = _PARTICLES_PER_BODY_FLOOR * count
    if total < reserved:
        raise ValueError(
            f"a total of {total} particles cannot reserve "
            f"{_PARTICLES_PER_BODY_FLOOR} particles for each of {count} bodies"
        )
    volumes = tuple(4.0 / 3.0 * math.pi * radius**3 for radius in radii)
    share = total - reserved
    exact = tuple(share * volume / sum(volumes) for volume in volumes)
    floors = tuple(int(math.floor(value)) for value in exact)
    remainder = share - sum(floors)
    order = sorted(
        range(count),
        key=lambda index: (-(exact[index] - floors[index]), names[index]),
    )
    counts = [
        _PARTICLES_PER_BODY_FLOOR + floor for floor in floors
    ]
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _spacing_count(member: MemberFacts, radius: float, units: UnitScale) -> int:
    """D7 SPACING resolution: proxy volume over the converted delta cube."""

    delta_m = _finite_scalar(member.particle_delta_m, "particle_delta_m")
    if delta_m <= 0.0:
        raise ValueError(
            f"member {member.name!r} requires a positive finite particle_delta_m"
        )
    delta = units.length_to_code(delta_m)
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError(
            f"member {member.name!r} particle_delta_m is outside the unit range"
        )
    return max(
        _PARTICLES_PER_BODY_FLOOR,
        math.ceil((4.0 * math.pi / 3.0) * radius**3 / delta**3),
    )


def _shell_layout(member: MemberFacts) -> tuple[tuple[str, int, float], ...]:
    """The accepted layout for one selected object, over :func:`asteroid_shell_layout`.

    What this adds is the one refusal the shared rule cannot make: a selected object can
    carry ``synthetic-fluid``, which is a real material key and so would otherwise reach
    the shared helper's "unknown material" wording. Everything else -- the Synthetic Iron
    core, the outer material id, the fractions -- is the shared rule, not a second copy of
    it, so the Quick Effect and the Add-menu template cannot drift apart.
    """

    outer_key = member.material_key
    if outer_key == "synthetic-fluid":
        raise ValueError(
            "Synthetic Fluid cannot author a solid Impact shell; choose an "
            "accepted outer material"
        )
    if outer_key not in OUTER_MATERIAL_IDS:
        raise ValueError(f"unknown material key {outer_key!r}")
    return asteroid_shell_layout(outer_key)


def build_impact(
    members: tuple[MemberFacts, MemberFacts],
    *,
    options: ImpactOptions,
    units: UnitScale,
    frame_start: int,
    frame_end: int,
) -> ScenarioPlan:
    """Author the two-Solid impact plan (decisions D1, D2, D5, D6, D7).

    Exactly two visible editable Meshes with finite nonzero evaluated dimensions.
    The larger evaluated bounding-sphere radius is the target; ties break by larger
    calculated volume, then Unicode object name.  Both members sample as the
    ``OBJECT_BALL`` convex proxy carrying the default layered shell layout: the
    core Synthetic Iron (id 2) is preserved under each member's chosen outer
    material id.
    """

    pair = _require_member_tuple(members, operator="Impact")
    if len(pair) != 2:
        raise ValueError(f"Impact requires exactly two members, got {len(pair)}")
    _require_frame_range(frame_start, frame_end)
    if frame_end > _IMPACT_FRAME_CEILING:
        raise ValueError(
            f"the impact range exceeds Blender's timeline ceiling {_IMPACT_FRAME_CEILING}"
        )
    if not isinstance(options, ImpactOptions):
        raise ValueError("options must be ImpactOptions")

    for member in pair:
        if member.object_type != "MESH":
            raise ValueError("Impact requires Mesh objects with evaluated dimensions")

    radii = tuple(_proxy_radius(member) for member in pair)
    volumes = tuple(
        member.dimensions[0] * member.dimensions[1] * member.dimensions[2]
        for member in pair
    )
    target_index, impactor_index = sorted(
        range(2),
        key=lambda index: (
            -radii[index],
            -volumes[index],
            pair[index].name,
        ),
    )
    target = pair[target_index]
    impactor = pair[impactor_index]
    target_radius = radii[target_index]
    impactor_radius = radii[impactor_index]

    layouts = (_shell_layout(target), _shell_layout(impactor))
    masses = (
        asteroid_layered_mass(target_radius, layouts[0], units),
        asteroid_layered_mass(impactor_radius, layouts[1], units),
    )

    contact_speed = units.speed_to_code(
        1000.0 * _positive_scalar(options.impact_speed_km_s, "impact_speed_km_s")
    )
    angle = math.radians(
        _finite_scalar(options.impact_angle_degrees, "impact_angle_degrees")
    )
    if not 0.0 <= angle < math.pi / 2.0:
        raise ValueError("impact_angle_degrees must be within [0, 90)")

    initial = solve_encounter_initial_state(
        position_a=target.position,
        position_b=impactor.position,
        mass_a=masses[0],
        mass_b=masses[1],
        radius_a=target_radius,
        radius_b=impactor_radius,
        contact_speed=contact_speed,
        contact_angle_radians=angle,
        gravitational_constant=units.gravitational_constant,
    )

    seconds_per_frame = template_impact_seconds_per_frame(
        contact_time=initial.contact_time, frame_start=frame_start, frame_end=frame_end
    )
    contact_frame = contact_boundary_frame(
        contact_time=initial.contact_time,
        seconds_per_frame=seconds_per_frame,
        frame_start=frame_start,
        maximum_boundary_frame=_IMPACT_MAX_BOUNDARY,
    )
    reaccumulate_frame = reaccumulate_boundary_frame(
        contact_frame=contact_frame,
        contact_speed=initial.contact_speed,
        total_mass=masses[0] + masses[1],
        contact_radius=initial.contact_radius,
        gravitational_constant=units.gravitational_constant,
        seconds_per_frame=seconds_per_frame,
        maximum_boundary_frame=frame_end - 1,
        clamp_to_maximum=True,
    )
    result_frame_end = frame_end

    modes = {member.resolution_mode for member in (target, impactor)}
    if len(modes) > 1:
        raise ValueError(
            "Impact members must share one resolution mode: both TARGET_COUNT "
            "or both SPACING; mixed modes refuse"
        )
    if modes == {"SPACING"}:
        counts = (
            _spacing_count(target, target_radius, units),
            _spacing_count(impactor, impactor_radius, units),
        )
    else:
        counts = _split_particles(
            _fragments_total(options.fragments_k),
            (target.name, impactor.name),
            (target_radius, impactor_radius),
        )

    member_plans = []
    for member, radius, layout, mass, count, velocity in zip(
        (target, impactor),
        (target_radius, impactor_radius),
        layouts,
        masses,
        counts,
        (initial.velocity_a, initial.velocity_b),
        strict=True,
    ):
        # Spin is simulated (ADR 0158): the compile turns the rad/s the member carries into
        # the rigid-rotation starting velocity, so the plan only has to keep it finite.
        for component in member.spin_rad_s:
            _finite_scalar(component, "spin_rad_s")
        if _finite_scalar(member.temperature_k, "temperature_k") != 0.0:
            raise ValueError(
                f"member {member.name!r}: Temperature is nonzero, and starting an "
                "object hot is not simulated -- there is no agreed way to turn "
                "degrees into the heat the solver carries. Set it to zero"
            )
        settings: tuple[tuple[str, AuthoringValue], ...] = (
            ("material", member.material_key),
            ("resolution_mode", member.resolution_mode),
            ("target_count", count),
            ("particle_delta_m", member.particle_delta_m),
            ("spin_rad_s", member.spin_rad_s),
            ("temperature_k", member.temperature_k),
        )
        member_plans.append(
            MemberPlan(
                persistent_id=member.persistent_id,
                role="SOLID",
                mass=mass,
                radius=radius,
                velocity=velocity,
                particle_count=count,
                sample_shape="OBJECT_BALL",
                material_key=member.material_key,
                shell_layout=layout,
                settings=settings,
            )
        )

    stages = (
        TimelineStage(kind=TimelineStageKind.IMPACT, boundary_frame=contact_frame),
        TimelineStage(
            kind=TimelineStageKind.REACCUMULATE, boundary_frame=reaccumulate_frame
        ),
    )
    validate_stage_list(stages, frame_start=frame_start, frame_end=result_frame_end)

    return ScenarioPlan(
        unit_preset="PLANETARY",
        solver="AUTO",
        color_by=options.color_by,
        seconds_per_frame=seconds_per_frame,
        softening=suggest_softening([target_radius, impactor_radius]),
        substeps=suggest_substeps(
            seconds_per_frame,
            [(masses[0], target_radius), (masses[1], impactor_radius)],
            units.gravitational_constant,
        ),
        members=tuple(member_plans),
        timeline_stages=stages,
        # The enum *identifier*, not its UI label. `AstroWarpRole.display_group` is an
        # EnumProperty whose items are ("SPH_POINTS", "SPH Points", ""), so assigning the
        # label raises rather than selecting the item -- latent only because no application
        # seam reads `display_groups` yet, and guaranteed to bite the first one that does.
        display_groups=tuple(
            (member.persistent_id, "SPH_POINTS") for member in (target, impactor)
        ),
        result_frame_end=result_frame_end,
        look=look_for_color_by(options.color_by),
        tree_theta=DEFAULT_TREE_THETA,
    )

