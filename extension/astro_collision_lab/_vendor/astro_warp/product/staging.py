"""Where to point a 3D viewport so authored content is actually on screen.

A template authors in its own physical units -- solar radii and au for the Binary, kpc for
the Cluster, km for the Asteroid Impact -- and those units go into the scene as Blender
units unchanged, because the solver reads them back. That is correct for the physics and it
is why the five shipped templates span 12,750x in size, from a 0.02-unit cluster to a
255-unit impact pair.

Nothing had ever reconciled that with the viewport. The gallery run of 2026-09-05
photographed all five mid-play: the Cluster's fifty thousand particles filled 0.0012 of the
default view and arrived as a single speck, the Galaxy Collision overflowed it 4.7 times
over and ran off two edges, and the Binary was an empty grid. The scenes were simulating
correctly the whole time and there was no way to see it.

This module is the missing half: given the box the content occupies, it says where to stand
and where to put the clip planes. It is pure -- no ``bpy`` -- so the rule that has to hold
across four orders of magnitude can be tested without a window, which is the only way to
know it holds for a template nobody photographed this week.

The rule is scale-free by construction. Every output is a multiple of the content's own
bounding radius, so a template that changes its units, or a user who drags an extent slider,
gets a framing that moves with it rather than one tuned to whatever size shipped.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

__all__ = [
    "Framing",
    "MOTION_FRACTION_OF_EXTENT",
    "MOTION_NOISE_FLOOR",
    "MotionCount",
    "bounds_union",
    "clip_widened_for_view",
    "count_moved",
    "framing_for_bounds",
    "motion_floor_for_radius",
]


# Blender's shipped viewport optics. Named rather than folded into one factor because the
# factor is a consequence of them: someone changing the assumed region shape should see what
# they are changing, not a constant with a comment claiming it came from somewhere.
DEFAULT_LENS_MM = 50.0
DEFAULT_SENSOR_MM = 36.0

# A viewport's height is what constrains framing -- the 3D area is wider than it is tall,
# and wider still once the sidebar and the properties editor take their share. 16:9 is the
# shape to assume: it is close to the common case, and assuming a *wider* region than the
# user has is the failure that puts content off the top and bottom edges. The real region
# aspect is not knowable from here, and asking for it would make the rule untestable
# without a window for no gain -- the margin below covers the spread.
ASSUMED_REGION_ASPECT = 16.0 / 9.0

# Room to breathe between the content's bounding sphere and the edge of the frame. Content
# pinned exactly to the edges reads as overflowing even when it does not, and a simulation
# expands: every one of these scenes is being watched precisely because things move.
FRAMING_MARGIN = 1.15

# How much closer than the nearest content, and further than the furthest, the clip planes
# sit. Two decades of room in each direction, four across the pair: enough that a user can
# push in on an impact or pull back off a galaxy without meeting a clip plane, and near
# enough in scale that the depth buffer keeps its precision. Blender's shipped 0.1/1000
# defaults are a fixed pair, which is why they clip a 0.02-unit cluster and a 255-unit
# impact alike.
CLIP_MARGIN_RATIO = 100.0

# Blender refuses a non-positive near plane, and a denormal one costs all depth precision.
MINIMUM_CLIP_START = 1e-6

# How far a particle has to travel, as a fraction of the radius of the scene it belongs to,
# before an observer would call it moving. A tenth of a percent: below anything a viewer
# could see, and far enough above float32 position noise at these scales that a resting
# particle does not drift across it.
#
# It is a fraction rather than a distance for the same reason the framing above is. The
# gallery run of 2026-09-05 judged all five templates with one absolute distance, 1e-3
# units, and so asked a Galaxy Collision particle to cross 0.0024% of its scene while asking
# a Star Cluster star to cross 5.9% of the entire cluster. 50,000 of the Galaxy's 50,000
# counted as moving and 2,499 of the Cluster's 50,000 did, from one rule that was really
# five. The two templates that rule treated most harshly are the two the owner reported as
# not appearing to simulate anything.
MOTION_FRACTION_OF_EXTENT = 1e-3

# Below this, "it moved" would mean "the last mantissa bit changed". Positions come back
# from the device as float32, whose spacing near 1.0 is about 1.2e-7, so a scene small
# enough that a thousandth of it lands under this floor gets the floor instead. Without it
# the rule would report every particle in a sufficiently tiny scene as moving, forever.
MOTION_NOISE_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class Framing:
    """Where a viewport should stand to see one box of content.

    A value, not a handle: the extension compares framings and hands them across the ``bpy``
    boundary, and two that describe the same view have to be equal for either to be
    testable.
    """

    centre: tuple[float, float, float]
    view_distance: float
    clip_start: float
    clip_end: float
    # The bounding radius the rest of the numbers were derived from. Carried rather than
    # recomputed by callers because more than the viewport is sized against it: the System's
    # controller Empty ships as a fixed one-unit sphere, and Blender's own View All frames
    # *objects*, so an Empty that disagrees with its content is what undoes a staged view the
    # moment a user presses View All. One radius, one source.
    content_radius: float


def framing_for_bounds(
    minimum: tuple[float, float, float] | object,
    maximum: tuple[float, float, float] | object,
    *,
    margin: float = FRAMING_MARGIN,
) -> Framing | None:
    """The view that puts the box between ``minimum`` and ``maximum`` on screen.

    Returns ``None`` when the box carries no usable scale -- a single point, or a corner an
    unevaluated depsgraph handed back as ``inf``. Declining is deliberate: viewport state is
    not undo data, so a ``NaN`` view distance wedges the view in a way the user cannot take
    back, and inventing a scale for content that has none would point the view at a distance
    no evidence supports. The caller keeps the viewport the user already had.

    The corners may arrive in either order; callers assemble them from a bounding-box union
    and should not have to sort first.
    """

    low = _finite_triple(minimum)
    high = _finite_triple(maximum)
    if low is None or high is None:
        return None
    if not math.isfinite(margin) or margin <= 0.0:
        return None

    # Read as a box rather than as an ordered pair, so a caller that unions corners in the
    # other order gets the same view instead of a negative radius that would resolve to a
    # distance behind the viewer.
    centre = tuple((a + b) * 0.5 for a, b in zip(low, high, strict=True))
    radius = 0.5 * math.dist(low, high)
    if not math.isfinite(radius) or radius <= 0.0:
        return None

    # Half the visible height one unit of distance away, from the shipped optics. The
    # height is the constraint; solving the width instead is what leaves content hanging
    # off the top of the frame on any region taller than it is wide.
    half_height_per_unit = (DEFAULT_SENSOR_MM / 2.0) / DEFAULT_LENS_MM / ASSUMED_REGION_ASPECT
    view_distance = radius * margin / half_height_per_unit
    if not math.isfinite(view_distance) or view_distance <= 0.0:
        return None

    # Derived from where the content actually is, not from the content's size alone: the
    # near plane has to clear the front of the bounding sphere, and that is a distance, not
    # a radius. A ratio of the radius alone would clip whenever the margin changed.
    nearest = view_distance - radius
    furthest = view_distance + radius
    clip_start = max(MINIMUM_CLIP_START, nearest / CLIP_MARGIN_RATIO)
    clip_end = furthest * CLIP_MARGIN_RATIO
    if clip_start >= nearest or clip_end <= furthest:
        # Only reachable when the content is so small that the near plane floor overtakes
        # it. Framing it would draw nothing, so leave the viewport alone and say so by
        # declining, rather than hand back a view that clips its own subject.
        return None

    return Framing(
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        view_distance=float(view_distance),
        clip_start=float(clip_start),
        clip_end=float(clip_end),
        content_radius=float(radius),
    )


def clip_widened_for_view(
    clip_start: float,
    clip_end: float,
    view_distance: float,
    content_radius: float,
) -> tuple[float, float] | None:
    """The clip planes a view already standing at ``view_distance`` needs, or ``None``.

    ``framing_for_bounds`` answers "where should the eye go, and what clip planes go with
    it" for a caller that is about to move the view. This answers the other half of the same
    question, for a caller that must not: *the eye is already here; can it see the content
    from here, and if not, what is the smallest change that fixes it?*

    That second question is the one Blender leaves unanswered. Its shipped ``clip_end`` of
    1000 is a fixed distance, and its own View All moves the eye without touching it, so
    framing a scene authored in PLANETARY units puts the eye tens of thousands of units back
    and every body behind the far plane. The scene solves, the draw handler uploads, and the
    viewport is empty -- measured on 2026-09-18 at 19,113 units across with the eye 47,553.9
    units back against a far plane of 1,000.

    **Only ever outward.** The returned pair contains the pair passed in: the far plane can
    move further away and the near plane closer, never the reverse. Clip planes are viewport
    state, which is not undo data, so a rule that could *narrow* them would be able to take
    away a view the user set up deliberately with no way back. Widening can only ever add
    things to the picture, which is why it is safe to apply without asking.

    The near plane is left alone when the eye is inside the content's bounding sphere. There
    is no near plane that clears content surrounding the eye, so the only thing lowering it
    would achieve is to spend the depth buffer's precision -- a near plane at the floor under
    a far plane six decades away is how a correct picture becomes a z-fighting one.

    **Short means short by something the viewport can hold.** A view is asked about at the
    precision a viewport *keeps*, not the precision this module computes at: Blender stores
    both clip planes and the eye's own distance in float32 properties, so the plane written
    from here comes back rounded and the eye is not quite where the framing put it. Demanding
    the float64 answer exactly is how a view that was widened correctly stays permanently one
    representable step short of a plane it already has.

    Returns ``None`` when the view already sees the content, and when any input carries no
    usable scale, so a caller can write the result unconditionally and a declined answer
    means "leave this view exactly as it is".
    """

    current = _finite_pair(clip_start, clip_end)
    if current is None:
        return None
    try:
        distance = float(view_distance)
        radius = float(content_radius)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(distance) or not math.isfinite(radius):
        return None
    if distance <= 0.0 or radius <= 0.0:
        return None

    near, far = current
    # The same two distances ``framing_for_bounds`` derives its pair from, measured from
    # where the eye actually is rather than from where a framing would have put it. Sharing
    # the ratio matters more than sharing the code: a view widened by this rule and a view
    # staged by that one have to agree about how much room a user gets to push in or pull
    # back before they meet a plane, or the two paths would drift apart the first time
    # either constant was tuned.
    nearest = distance - radius
    furthest = distance + radius

    # Each half asks whether the view is short by something the viewport could actually
    # hold, rather than whether it differs from the float64 answer -- see the module note
    # above ``_largest_stored_at_most``.
    wanted_far = furthest * CLIP_MARGIN_RATIO
    far_is_short = far < _largest_stored_at_most(wanted_far)

    wanted_near = near
    if nearest > 0.0:
        wanted_near = max(MINIMUM_CLIP_START, nearest / CLIP_MARGIN_RATIO)
    near_is_high = wanted_near < near and near > _smallest_stored_at_least(wanted_near)

    if not far_is_short and not near_is_high:
        return None
    # Outward only, by construction rather than by a second ``min``/``max``: a plane is
    # replaced exactly when it was found short, and a plane found short is one the wanted
    # value is strictly beyond.
    return (
        float(wanted_near if near_is_high else near),
        float(wanted_far if far_is_short else far),
    )


def bounds_union(
    corner_sets: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The smallest axis-aligned box containing every point in every set given.

    ``corner_sets`` is an iterable of iterables of points -- one set per object, each set
    being that object's corners already in world space. Assembling the box here rather than
    at the call site keeps the arithmetic testable without a window, and keeps the caller to
    the one thing only Blender can do: turn an object into world-space corners.

    Returns ``None`` when nothing usable arrived. Points that are not three finite numbers
    are skipped rather than poisoning the box: one object with an unevaluated transform
    should not cost the framing of the four beside it.
    """

    low: list[float] | None = None
    high: list[float] | None = None
    try:
        sets = list(corner_sets)  # type: ignore[call-overload]
    except TypeError:
        return None
    for corners in sets:
        try:
            points = list(corners)
        except TypeError:
            continue
        for point in points:
            triple = _finite_triple(point)
            if triple is None:
                continue
            if low is None or high is None:
                low = list(triple)
                high = list(triple)
                continue
            for axis in range(3):
                if triple[axis] < low[axis]:
                    low[axis] = triple[axis]
                if triple[axis] > high[axis]:
                    high[axis] = triple[axis]
    if low is None or high is None:
        return None
    return ((low[0], low[1], low[2]), (high[0], high[1], high[2]))


@dataclass(frozen=True, slots=True)
class MotionCount:
    """How many of a population moved, and the yardstick that decided it.

    The floor and the radius travel with the count because a count alone is not evidence:
    "2,499 of 50,000 moved" was a true sentence about the Star Cluster and a false
    impression, and the only thing that would have exposed it at the time is the number it
    was measured against. Anything recording this into a report records all three.
    """

    moved: int
    total: int
    floor: float
    content_radius: float


def motion_floor_for_radius(radius: float) -> float | None:
    """The distance a body in content of this size must cover to count as moving.

    Returns ``None`` when the radius carries no scale -- zero, negative, or non-finite --
    for the same reason ``framing_for_bounds`` declines such content: no fraction of
    nothing is a distance, and inventing one would produce a count that looks like
    evidence and is not.
    """

    try:
        extent = float(radius)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(extent) or extent <= 0.0:
        return None
    return max(extent * MOTION_FRACTION_OF_EXTENT, MOTION_NOISE_FLOOR)


def count_moved(start: object, now: object) -> MotionCount | None:
    """How many points in ``now`` have left where they were in ``start``.

    The scale is taken from ``start``'s own bounding box, so the caller does not supply a
    threshold and cannot pick one that flatters the result. Both sequences are the same
    population sampled at two times; anything else is not comparable.

    Returns ``None`` rather than a number when the pair cannot be measured: different
    lengths (not the same population), nothing to measure, content with no extent, or a
    point that is not three finite numbers.

    That last case is stricter than ``bounds_union``, which skips unusable points. The
    difference is deliberate and follows from what each is for. A framing with one bad
    object still puts the other four on screen, so skipping is the better outcome. A count
    is evidence, and a count that silently omitted the particles it could not read would
    understate motion by exactly the amount it could not see, while still printing a
    confident denominator.
    """

    start_points = _finite_triples(start)
    now_points = _finite_triples(now)
    if start_points is None or now_points is None:
        return None
    if not start_points or len(start_points) != len(now_points):
        return None

    box = bounds_union([start_points])
    if box is None:
        return None
    radius = 0.5 * math.dist(box[0], box[1])
    floor = motion_floor_for_radius(radius)
    if floor is None:
        return None

    moved = sum(
        1
        for before, after in zip(start_points, now_points, strict=True)
        if math.dist(before, after) >= floor
    )
    return MotionCount(
        moved=moved,
        total=len(start_points),
        floor=float(floor),
        content_radius=float(radius),
    )


def _finite_triples(values: object) -> list[tuple[float, float, float]] | None:
    """``values`` as a list of finite triples, or ``None`` if any point is not one."""

    try:
        points = list(values)  # type: ignore[call-overload]
    except TypeError:
        return None
    triples: list[tuple[float, float, float]] = []
    for point in points:
        triple = _finite_triple(point)
        if triple is None:
            return None
        triples.append(triple)
    return triples


#: The largest value a float32 property can hold. Named because ``struct`` will not pack
#: anything beyond it -- it raises ``OverflowError`` rather than handing back an infinity --
#: and a rule whose entire contract is to decline rather than disturb a viewport must not
#: raise out of a playback handler over a scene whose scale it cannot represent.
_LARGEST_STORED = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def _as_stored(value: float) -> float:
    """``value`` as a viewport would keep it: the nearest float32 to it.

    For positive values only, which is all this module has: clip planes are positive by
    ``_finite_pair``, and the distances the targets are derived from are checked positive
    before any of this runs. Anything past the top of the range saturates there, because the
    answer to "what would a viewport keep" for such a value is the largest thing it can keep.
    """

    if not (value < _LARGEST_STORED):
        # Also catches an infinity arriving from a float64 multiplication that overflowed.
        return _LARGEST_STORED
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _stored_neighbour(value: float, direction: int) -> float:
    """The adjacent storable value above (``+1``) or below (``-1``) an exact positive float32.

    Positive floats sort the same way their bit patterns do, which is the whole
    implementation, and is also why this is positive-only: for negative floats the order is
    reversed and stepping the pattern would move the wrong way. Both callers below hand it a
    value that came out of ``_as_stored``.
    """

    pattern = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", pattern + direction))[0]


# Blender keeps `clip_start`, `clip_end` and `RegionView3D.view_distance` in float32
# properties, so this module's float64 arithmetic is not what a viewport ends up holding --
# in either direction. A plane written from here is stored as the nearest float32 to it, and
# the eye's distance is read back the same way, so the far plane a widening asks for the
# *second* time is computed from an eye a fraction further out than the one the framing
# solved for.
#
# Measured on 2026-09-18, staging the by-hand planet-scale System: `framing_for_bounds`
# wanted the eye at 96,641.99791769942 and the far plane at 11,365,939.320320737; the
# viewport stored 96,642.0 and 11,365,939.0. Re-asked from that stored eye, the rule wanted
# 11,365,939.528550794 -- the *adjacent* float32, with nothing representable in between, and
# a far plane already admitting the content 99.99999 times over against a ratio of 100. An
# exact comparison called that view short on every Play for the rest of the session.
#
# So the question is asked in the viewport's own terms: not "does this differ from the
# answer" but "is it short by an amount a viewport could store". A difference finer than one
# representable step is one no assignment could remove and no frame could show.


def _largest_stored_at_most(value: float) -> float:
    """The furthest-out plane a viewport can hold that is no greater than ``value``."""

    stored = _as_stored(value)
    if stored <= value:
        return stored
    return _stored_neighbour(stored, -1)


def _smallest_stored_at_least(value: float) -> float:
    """The nearest-in plane a viewport can hold that is no less than ``value``."""

    stored = _as_stored(value)
    if stored >= value:
        return stored
    return _stored_neighbour(stored, +1)


def _finite_pair(near: object, far: object) -> tuple[float, float] | None:
    """A view's clip pair as two finite floats, or ``None`` if it is not a usable pair.

    Blender refuses a non-positive near plane and orders the two, so a pair that arrives
    reversed or non-positive did not come from a viewport this rule can reason about -- most
    likely a stand-in with a plausible-looking attribute. Declining leaves such a view
    untouched, which is the same answer the rest of this module gives to content it cannot
    measure.
    """

    try:
        low = float(near)  # type: ignore[arg-type]
        high = float(far)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    if low <= 0.0 or high <= low:
        return None
    return (low, high)


def _finite_triple(value: object) -> tuple[float, float, float] | None:
    """``value`` as three finite floats, or ``None``.

    Blender hands corners back as ``bpy_prop_array`` and ``Vector``; both are sequences of
    three, neither is a tuple. Coercing here keeps the caller from having to know that.
    """

    try:
        components = [float(component) for component in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(components) != 3:
        return None
    if not all(math.isfinite(component) for component in components):
        return None
    return (components[0], components[1], components[2])
