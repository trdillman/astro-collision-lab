"""Target-driven bounded service policy for product runtimes.

The policy lives here rather than in the Blender extension because it is the part of the
scheduler that decides *how much work a service call may submit*, and that decision has to
hold as body count and step cost grow. Keeping it as a pure function over measured numbers
means it can be gated by a host suite on the real device instead of only by a stubbed
``bpy`` unit test, and means the extension has no scheduling arithmetic of its own.

Two ideas carry the whole file.

*A target is a position, not an amount of work.* A handler records where a context should
be; it never advances it. One service loop closes the gap across as many ticks as that
takes. A target set while the previous one is still unmet simply replaces it, so a user
scrubbing the timeline cannot queue up work behind themselves.

*The bound is time, not a step count.* A fixed step cap is only correct at the body count
it was chosen for: 64 steps of a two-body scene is nothing, and 64 steps of a hundred
thousand bodies is a frozen viewport. The bound here is a wall-clock budget, and the number
of steps that fits in it is derived from what the runtime just measured. A scene that gets
heavier submits fewer steps per service and stays responsive; a scene that gets lighter
submits more and reaches its target sooner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class ServicePlan:
    """What one service call should do for one context."""

    steps: int
    """Steps to submit now. Zero means there is nothing to submit this service."""

    restore_first: bool
    """The target is behind the current position, so the runtime must be rewound first."""

    reached: bool
    """The target is already met and no further service is needed."""


def plan_service(
    *,
    current_step: int,
    target_step: int | None,
    last_step_ms: float,
    budget_ms: float,
) -> ServicePlan:
    """How much of the gap to ``target_step`` this service may close.

    ``target_step`` is ``None`` for free run, which has no destination: it advances for as
    long as it is playing, so every service submits a full budget's worth.

    ``last_step_ms`` is the runtime's own measurement of its previous step. Zero (or a
    runtime that has never stepped) submits a single step, which measures the cost that
    sizes every service after it.
    """

    if budget_ms <= 0.0:
        raise ValueError("budget_ms must be positive")
    if current_step < 0:
        raise ValueError("current_step must be non-negative")

    chunk = _steps_within_budget(last_step_ms, budget_ms)
    if target_step is None:
        return ServicePlan(steps=chunk, restore_first=False, reached=False)
    if target_step < 0:
        raise ValueError("target_step must be non-negative")
    if target_step < current_step:
        # Rewinding is the caller's business - it restores a checkpoint at or before the
        # target and services the remainder on the next call, from a position this function
        # cannot predict because it depends on which checkpoints exist.
        return ServicePlan(steps=0, restore_first=True, reached=False)
    remaining = target_step - current_step
    if remaining == 0:
        return ServicePlan(steps=0, restore_first=False, reached=True)
    return ServicePlan(steps=min(remaining, chunk), restore_first=False, reached=False)


def _steps_within_budget(last_step_ms: float, budget_ms: float) -> int:
    """Steps that fit in ``budget_ms`` at the last measured cost, at least one.

    At least one because a service that submits nothing makes no progress and never
    measures a new cost, so a runtime whose step is slower than the whole budget would
    stall forever rather than merely run below the target rate.
    """

    if not last_step_ms > 0.0:
        return 1
    return max(1, int(budget_ms // last_step_ms))


# Conservative throughput floors for the data-moving primitives, in bytes per millisecond.
# These are *floors*, not estimates: the admission below is only conservative if real
# hardware is faster than them, so each is set well under what the slowest machine the
# product targets actually sustains. They may be tightened by measurement later --- the
# plan is explicit that measured worst cases may narrow these but never replace them as
# the first-use answer --- and they are never used to size a chunk, only to refuse one.
#
# Device-to-host over PCIe. PCIe 3.0 x16 sustains several GB/s; 500 MB/s is an order of
# magnitude of headroom for a contended bus and an unpinned staging buffer.
DEVICE_TO_HOST_FLOOR_BYTES_PER_MS = 500_000.0
# The record checksum over one contiguous buffer. Set for a software SHA-256, which cores
# with the SHA extensions run at over a GB/s and a portable implementation at several
# hundred MB/s; 200 MB/s is the binding floor here, and deliberately so. The checksum is a
# CRC-32 now (ADR 0140), measured at ~4.4 GB/s inside Blender, and the floor is kept: it is
# a refusal bound, not an estimate.
CHECKSUM_FLOOR_BYTES_PER_MS = 200_000.0
# A host memcpy into the ring buffer. DDR4 sustains well over 10 GB/s.
HOST_COPY_FLOOR_BYTES_PER_MS = 1_000_000.0
# Publishing is a table assignment and a redraw tag: O(1) in body count, and this is a
# generous constant rather than a derived bound because there is nothing to derive it from.
PUBLISH_WORST_CASE_MS = 0.5

#: Blender's own default scene playback rate, and so the rate playback falls back to wherever
#: no scene can be read --- a headless probe, a unit test, a context whose owning scene has
#: been deleted. It replaces a hardcoded 30 that traced to no requirement row and matched no
#: shipped ``.blend``: a fresh Blender file plays at 24 fps, so the fallback now names
#: something the product can point at. Live playback never reaches it --- the budget follows
#: the rate the artist set on their own timeline (ADR 0067 Decision 1, ENGINE-011) --- and
#: reaching it means the scene could not be read, never that 24 was preferred to what it said.
FALLBACK_FRAME_RATE_FPS = 24.0

#: The share of one frame period a timer tick may spend servicing one context. The remainder
#: stays reserved for the draw handler and for Blender's own work, which is why this is a half
#: rather than the whole period: a service that consumed the entire frame would leave nothing
#: to draw the result of it with.
SERVICE_BUDGET_FRACTION = 0.5


def service_budget_ms(frame_rate_fps: float) -> float:
    """The work one timer tick may spend on a context whose scene plays at ``frame_rate_fps``.

    Half of one frame period, in milliseconds. The rate is the artist's --- ``render.fps``
    divided by ``render.fps_base`` --- so the number that paces playback is the number they set
    on their own timeline rather than a constant chosen for them (ADR 0067 Decision 1). A
    caller with no scene to read passes :data:`FALLBACK_FRAME_RATE_FPS` and gets Blender's own
    default rather than a second literal.

    Pure arithmetic over one argument, so the pacing rule has exactly one definition that the
    add-on, the validation harnesses and the tests all read --- rather than each re-deriving a
    literal that then drifts away from the others, which is how ``1.0 / 30.0`` came to be
    written out in four places at once.
    """

    if not frame_rate_fps > 0.0 or not math.isfinite(frame_rate_fps):
        raise ValueError("frame_rate_fps must be positive and finite")
    return (1.0 / frame_rate_fps) * 1000.0 * SERVICE_BUDGET_FRACTION


def service_re_arm_seconds(
    frame_rate_fps: float,
    elapsed_s: float,
    *,
    frame_in_flight: bool = False,
    host_s: float = 0.0,
) -> float:
    """How long after a service tick returns the next one is due, in seconds.

    ``host_s`` is how late the tick fired after the delay it was given (ADR 0142): the time
    Blender spent on its own work -- drawing the last published frame among it -- before
    handing the thread back. It is charged against the period like ``elapsed_s`` and, per
    ADR 0231, also credited against the reservation below, which exists to buy exactly that
    work. A non-finite or negative value charges and credits nothing.

    The other half of :func:`service_budget_ms`, and derived from the same fraction so the
    two cannot drift apart: the budget is what a tick may *spend*, this is what it must
    then *yield*, and together they are exactly one frame period.

    ``bpy.app.timers`` measures the delay a callback returns from the moment the callback
    *returns*, not from when it was called. Returning a whole frame period therefore made
    the delivered period ``work + period``, so the authored rate was unreachable at any
    body count --- a two-particle scene, whose frame indisputably fits one slice, delivered
    18.567 of 24 fps in a windowed run (ADR 0073). That is the reservation being paid twice:
    once by spending only half the period, and again by waiting the whole of it afterwards.

    So the wait is the remainder of the period, floored at the reservation
    :data:`SERVICE_BUDGET_FRACTION` exists to protect. The floor is the half of the point
    that survives: a tick which overran its slice is already late, and re-arming at zero
    would hand the host nothing between services, which is the same invariant pointed the
    other way. A frame costing several ticks therefore still costs several ticks; what it
    stops paying is an idle period on top of each of them.

    An ``elapsed_s`` that is not a finite, non-negative number subtracts nothing and the
    whole period is waited --- the behaviour that predates this rule. A clock that went
    backwards is not a reason to stop yielding to Blender.

    ``frame_in_flight`` says the tick left an authored frame part-solved, and it collapses
    the wait to zero (ADR 0074). The reservation buys the draw handler a chance to put the
    serviced frame on screen; a part-solved frame has published nothing --- ADR 0058
    publishes at commit and never before --- so between two primitives of one frame the
    reservation redraws the *previous* frame and delays the next primitive by half a period
    to do it. Counted against the shipped timer, a 373.1 ms Galaxy-shaped frame spent
    1100.8 ms of wall across 25 ticks, two thirds of it that wait. Zero is one pass of the
    host's event loop rather than no yield at all, so the host still runs between
    primitives; what stops is paying the reservation once per tick instead of once per
    frame. It is deliberately checked before ``elapsed_s``: the zero is structural rather
    than measured, so a clock this function cannot read is not a reason to abandon it.
    """

    if not frame_rate_fps > 0.0 or not math.isfinite(frame_rate_fps):
        raise ValueError("frame_rate_fps must be positive and finite")
    if frame_in_flight:
        return 0.0
    period_s = 1.0 / frame_rate_fps
    reserved_s = period_s * (1.0 - SERVICE_BUDGET_FRACTION)
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        return period_s
    # The reservation buys Blender time to draw the frame this tick published. Time the host
    # already took before the tick could start -- its lateness -- is that same time, already
    # spent on exactly that work, so it is credited against the reservation rather than
    # waited for a second time (ADR 0231). ADR 0135 measured that handing Blender *extra*
    # time only lets its own work grow to fill it, which is why the credit never exceeds
    # what the host actually took and why a stall -- a tick a whole period late or more --
    # still waits the full reservation as ADR 0142 left it: a host that missed a period is
    # not drawing our frames, and a tick that overran its own slice is the case ADR 0135
    # falsified a zero wait for.
    # A negative ``host_s`` is an early tick (ADR 0142): it is still charged, so the wait
    # lengthens by the earliness, but there is no host time to credit.
    charge_s = host_s if math.isfinite(host_s) else 0.0
    credit_s = max(charge_s, 0.0)
    if credit_s >= period_s or elapsed_s > period_s * SERVICE_BUDGET_FRACTION:
        floor_s = reserved_s
    else:
        floor_s = max(reserved_s - credit_s, 0.0)
    return max(period_s - elapsed_s - charge_s, floor_s)


#: Where the cost report changes what it says, per ADR 0067 Decision 3 as amended by ADR 0090.
#: Below one second an over-budget primitive is a stutter: the tick cannot finish the frame, so
#: the frame takes more ticks and the sim plays below realtime. At or above one second the
#: application stops answering the artist between frames, which is a different thing to tell
#: them about. The number is the Miller (1968) / Nielsen response limit --- the boundary where
#: a person stops perceiving an operation as continuous --- not an invented constant, and it is
#: the only number in this module's admissions that is not derived from the artist's own scene.
#:
#: It was a *refusal* threshold until ADR 0090. It is a reporting threshold now: the owner's
#: invariant is that a scene always plays, and a door that opens at 999 ms and closes at
#: 1001 ms puts the addon's behaviour in the artist's hardware rather than in their scene.
#: Blender's own cloth and fluid steps block the interface for exactly as long as one frame
#: takes; this product tells the artist that will happen instead of deciding it for them.
RESPONSIVENESS_CEILING_MS = 1_000.0

_UNPROVEN = "SERVICE_BUDGET_UNPROVEN"

#: Notice marker for a primitive bounded at or above :data:`RESPONSIVENESS_CEILING_MS`. It
#: spans both cost classes below, because the ceiling is a fact about one indivisible unit of
#: host work and not about which admission derived it; the primitive's name in the notice text
#: is what says which class it came from. It is deliberately distinct from the two markers
#: above and below: those name work whose cost is *unknown*, this one names work whose cost is
#: known and large enough to stop the host answering while it runs. ADR 0067 Decision 3, as
#: amended by ADR 0090 --- all three are notices now, and none of them is a refusal.
RESPONSIVENESS_CEILING_EXCEEDED = "RESPONSIVENESS_CEILING_EXCEEDED"

#: The one refusal marker this module still emits (ADR 0090 Decision 4). One frame's record is
#: larger than the entire scrub cache, so no prefix can hold a single frame and there is
#: nothing to play even in principle. It is emphatically *not* a cost marker and must never be
#: confused with the three above: they say a frame will be slow or that nobody knows how slow,
#: this one says the frame cannot exist. It borrowed ``SERVICE_BUDGET_UNPROVEN`` until ADR
#: 0090, which is how a capacity fact reached the artist dressed as a timing one.
#:
#: At the shipped numbers it is unreachable --- a 1 GiB cache against a 2,363,392-byte
#: maximum-width record at the 65,536-body presentation cap --- and that is the point of
#: keeping it rather than deleting the branch: it is the structural floor under the claim that
#: every scene an artist can author plays.
RECORD_EXCEEDS_CACHE_CAPACITY = "RECORD_EXCEEDS_CACHE_CAPACITY"


@dataclass(frozen=True)
class PrimitiveBound:
    """The worst case for one indivisible unit of the frame transaction.

    ``worst_case_ms`` is ``None`` when nothing bounds it. That is a distinct answer from a
    large number and must stay distinct: an unknown cost silently treated as zero is
    exactly the calibration-by-running the admission exists to avoid.
    """

    name: str
    worst_case_ms: float | None
    detail: str


@dataclass(frozen=True)
class FrameAdmission:
    """Whether a frame transaction may be started at all, and what the bounds say it will
    feel like.

    Since ADR 0090 the first half is almost always yes: cost decides how a scene plays, never
    whether it plays, and the only ``admitted=False`` left is one record larger than the whole
    cache. The rest of this type is the second half --- the report the artist gets while the
    scene runs, which is strictly more than the refusal it replaced ever told them.
    """

    admitted: bool
    reason: str
    """Empty when admitted; otherwise says why no frame can exist at this scale.

    Only :data:`RECORD_EXCEEDS_CACHE_CAPACITY` reaches this field. Cost --- unknown, over the
    slice, or over the ceiling --- goes to :attr:`notice` instead, because it is something to
    tell the artist rather than something to stop them with.
    """

    bounds: tuple[PrimitiveBound, ...]

    over_budget: tuple[PrimitiveBound, ...] = ()
    """Admitted primitives bounded above one service slice, worst first.

    Non-empty means the frame was admitted and will play *below* the scene's rate, because a
    primitive that cannot be interrupted is larger than the slice it runs on. That is a
    throughput fact to report, not a permission to withhold (ADR 0067 Decision 2), so it
    travels with the admission rather than becoming a refusal. Always empty on a refusal.
    """

    unbounded: tuple[PrimitiveBound, ...] = ()
    """Admitted primitives nothing has bounded, in the order the transaction runs them.

    The other half of ADR 0067 Decision 2, built by ADR 0090. Unknown cost is absence of
    evidence, not evidence of a freeze: a non-resumable backend or an uncosted module leaves
    no bound to check, so the frame runs and the run is the measurement --- the registry times
    every solve into its own solver-cost record whether the runtime is resumable or not.

    Deliberately not folded into :attr:`over_budget`: that tuple is a claim about known cost,
    and an unknown one must not smuggle a speed claim out in either direction.
    """

    over_ceiling: tuple[PrimitiveBound, ...] = ()
    """Admitted primitives bounded at or above :data:`RESPONSIVENESS_CEILING_MS`, worst first.

    A subset of :attr:`over_budget` --- the ceiling is far above any service slice --- kept
    separate because it is a different sentence to the artist: not "this plays slowly" but
    "Blender will not repaint while each frame solves". ADR 0090 Decision 2.
    """

    notice: str = ""
    """What to tell the artist about cost, empty when there is nothing to say.

    Prefixed with whichever of the three cost markers applies, most severe first, so a caller
    mapping markers to sentences reads one field. Always empty on a refusal, which is not a
    claim about cost at all.
    """

    @property
    def plays_below_realtime(self) -> bool:
        """Does this admitted frame cost more than one slice in a single primitive?

        The question the status line answers for the artist. False on a refusal, which is not
        a claim about speed at all, and false on an unbounded primitive, which is not one
        either: nothing measured it, so nothing may assert it is slow.
        """

        return bool(self.over_budget)

    @property
    def worst_case_ms(self) -> float:
        """The whole transaction's bound, counting only the primitives that have one.

        Reported for the HUD and for diagnostics, never as an admission criterion: a
        transaction is allowed to need several slices, and requiring the sum to fit one
        would refuse the accepted display cap outright.
        """

        return sum(b.worst_case_ms or 0.0 for b in self.bounds)


def admit_frame_transaction(
    *,
    substeps_per_frame: int,
    solver_resumable: bool,
    solver_primitive_worst_case_ms: float | None,
    display_count: int,
    record_bytes: int,
    cache_capacity_bytes: int,
    budget_ms: float,
    solver_work_sliced: bool = False,
    solver_primitive_adaptive: bool = False,
) -> FrameAdmission:
    """May a frame transaction at this scale be started, before any of it has been run?

    A frame is resumable at the seams between its primitives --- solve a substep, copy the
    stage, checksum the record, commit it, publish it --- so what one timer slice is measured
    against is the largest *primitive*, not the frame. A frame needing six slices is fine, and
    so is a single primitive needing two: the tick it runs on overruns, the next frame starts
    later, and the sim plays below the scene's rate. That is a throughput fact this function
    reports in ``over_budget`` rather than a door it closes (ADR 0067 Decision 2, which
    reverses ADR 0056's refusal on the same arithmetic).

    What is still refused is one thing, and it is not a cost: a single frame's record larger
    than the entire playback cache, which no prefix can hold at any speed. A primitive nothing
    bounds is admitted and reported in ``unbounded`` --- the frame runs and the run is the
    measurement. A primitive bounded at or above :data:`RESPONSIVENESS_CEILING_MS` is admitted
    and reported in ``over_ceiling`` --- the host will stop answering while each frame solves,
    which the artist is told plainly and then decides about, rather than being told no
    (ADR 0090 Decisions 1, 2 and 4, amending ADR 0067 Decision 3).

    This function itself never measures: it derives every byte-moving bound from declared
    scale and the conservative floors above, and takes
    ``solver_primitive_worst_case_ms`` --- the one cost no byte count settles, because only
    the runtime's own plan knows what its integrator costs --- from the caller. What it must
    never do is measure the frame *at admission*,
    mid-Play: running the very frame whose cost is in question, at whatever size it happens to
    be, is the stall rather than a test for it. The caller may, however, derive
    ``solver_primitive_worst_case_ms`` from an init-time device calibration of that same
    synchronized primitive, taken once before Play and restored to frame 0
    (:func:`measured_solver_primitive_bound`, ADR 0058) --- a measurement that is not the
    stall because it does not run during the slice it is gating. Absent such a calibration
    the caller supplies the floor; a caller with neither gets an admitted frame carrying an
    ``unbounded`` bound and a notice that says so, rather than a guess.

    ``solver_work_sliced`` is the narrower ADR 0061 identity: a resumable solver whose one
    authored substep is itself a declared sequence of backend-owned synchronized slices.
    ``solver_primitive_adaptive`` is the third one, ADR 0109's: a resumable solver whose
    authored substep is a sequence of *identical* synchronized steps whose length nothing
    authors, because the integrator asks for as many as the physics needs. The two are
    mutually exclusive --- a sequence is either declared or it is not --- and both require a
    resumable solver. Both change the reported primitive name and detail, not the admission
    arithmetic; what they describe is the unit the caller's bound is a bound *of*, which is
    the one thing a cost report must not get wrong. An adaptive frame's bound is one step,
    not one authored substep, so naming it "one of N authored substeps" would attach a
    correct number to the wrong unit and read as a claim that the whole frame fits it.

    The ``notice`` carries the same two markers it used to refuse with, and they still say
    different things. ``SERVICE_BUDGET_UNPROVEN`` claims nothing about speed: it says nothing
    available bounds this primitive, so nothing shows in advance whether it stays interactive.
    ``RESPONSIVENESS_CEILING_EXCEEDED`` claims the opposite --- the cost *is* known, and it is
    past the point where the host still answers the artist. When both apply the known cost is
    said first, because it is the one the artist can make smaller by asking for less. Neither
    is the old "over the slice", which is reported separately in ``over_budget``.
    """

    if substeps_per_frame < 1:
        raise ValueError("substeps_per_frame must be positive")
    if display_count < 1:
        raise ValueError("display_count must be positive")
    if record_bytes < 1:
        raise ValueError("record_bytes must be positive")
    if cache_capacity_bytes < 1:
        raise ValueError("cache_capacity_bytes must be positive")
    if budget_ms <= 0.0:
        raise ValueError("budget_ms must be positive")
    if solver_work_sliced and not solver_resumable:
        raise ValueError("solver_work_sliced requires a resumable solver")
    if solver_primitive_adaptive and not solver_resumable:
        raise ValueError("solver_primitive_adaptive requires a resumable solver")
    if solver_primitive_adaptive and solver_work_sliced:
        raise ValueError(
            "a solver's slice sequence is either declared or adaptive, not both"
        )

    solver_name = (
        "solver_work_slice"
        if solver_work_sliced
        else "solver_adaptive_step"
        if solver_primitive_adaptive
        else "solver_substep"
        if solver_resumable
        else "solver_frame"
    )
    solver_detail = (
        f"one backend-owned work slice within one of {substeps_per_frame} authored substeps"
        if solver_work_sliced
        else "one adaptive solver step; a frame runs as many as the integrator asks for"
        if solver_primitive_adaptive
        else f"one of {substeps_per_frame} authored substeps"
        if solver_resumable
        else f"all {substeps_per_frame} authored substeps in one fallback call"
    )
    stage_ms = record_bytes / DEVICE_TO_HOST_FLOOR_BYTES_PER_MS
    checksum_ms = record_bytes / CHECKSUM_FLOOR_BYTES_PER_MS
    commit_ms = record_bytes / HOST_COPY_FLOOR_BYTES_PER_MS
    # The registry yields after staging. ``cache.append`` canonicalizes at worst once,
    # copies into one final record buffer, hashes that buffer once, freezes it to immutable
    # bytes once, and returns zero-copy read-only channel views for publication. Three copy
    # passes conservatively cover canonicalization, record construction, and the immutable
    # freeze; there is no immediate ``read_exact`` or second hash.
    commit_tail_ms = checksum_ms + 3.0 * commit_ms + PUBLISH_WORST_CASE_MS
    bounds = (
        PrimitiveBound(
            solver_name,
            solver_primitive_worst_case_ms,
            solver_detail,
        ),
        PrimitiveBound(
            "device_stage_copy",
            stage_ms,
            f"{display_count} bodies at {DEVICE_TO_HOST_FLOOR_BYTES_PER_MS:.0f} B/ms",
        ),
        PrimitiveBound(
            "frame_commit_tail",
            commit_tail_ms,
            f"one checksum and three copy passes over {record_bytes} bytes, "
            "then publish the immutable append result",
        ),
    )

    if record_bytes > cache_capacity_bytes:
        # The one thing left that is impossible rather than merely slow or unmeasured.
        return FrameAdmission(
            False,
            f"{RECORD_EXCEEDS_CACHE_CAPACITY}: one {record_bytes}-byte record exceeds the "
            f"whole {cache_capacity_bytes}-byte cache capacity, so no prefix can hold a "
            "frame at this scale",
            bounds,
        )

    # Everything below reports. Worst first in each tuple, so a caller with room for one
    # number names the binding one.
    def _worst_first(candidates: Iterable[PrimitiveBound]) -> tuple[PrimitiveBound, ...]:
        return tuple(sorted(candidates, key=lambda b: b.worst_case_ms or 0.0, reverse=True))

    unbounded = tuple(b for b in bounds if b.worst_case_ms is None)
    over_budget = _worst_first(
        b for b in bounds if b.worst_case_ms is not None and b.worst_case_ms > budget_ms
    )
    over_ceiling = _worst_first(
        b
        for b in bounds
        if b.worst_case_ms is not None and b.worst_case_ms >= RESPONSIVENESS_CEILING_MS
    )

    # Most severe first: a known cost past the ceiling is the sharpest thing that can be said,
    # and it is the only one of the three the artist can make smaller by asking for less.
    sentences: list[str] = []
    if over_ceiling:
        worst = over_ceiling[0]
        assert worst.worst_case_ms is not None
        sentences.append(
            f"{RESPONSIVENESS_CEILING_EXCEEDED}: {worst.name} is indivisible and bounded at "
            f"{worst.worst_case_ms:.3f} ms ({worst.detail}), "
            f"{worst.worst_case_ms - RESPONSIVENESS_CEILING_MS:.3f} ms over the "
            f"{RESPONSIVENESS_CEILING_MS:.3f} ms responsiveness ceiling, so the host will "
            "stop answering while each frame solves rather than merely play slowly"
        )
    if unbounded:
        first = unbounded[0]
        sentences.append(
            f"{_UNPROVEN}: nothing bounds {first.name} ({first.detail}), so nothing shows in "
            f"advance whether it stays under the {RESPONSIVENESS_CEILING_MS:.3f} ms "
            "responsiveness ceiling; the frame runs and the run is the measurement"
        )
    return FrameAdmission(
        True,
        "",
        bounds,
        over_budget,
        unbounded,
        over_ceiling,
        ". ".join(sentences),
    )


# Conservative floor on solver throughput, in body-interactions per millisecond. One
# interaction is one body reading one source of acceleration --- another body, a multipole,
# a neighbour --- and doing the handful of flops that follow from it.
#
# A floor, on the same terms as the byte rates above: the product requires a CUDA device
# (``DeviceDescriptor.cuda_required``), and the weakest card that satisfies that requirement
# sustains well over 10^10 such interactions per second on the kernels here, which are
# arithmetic-dense and read almost entirely from cache or shared memory. 10^7 per
# millisecond is 10^10 per second, so this stays a floor rather than a guess even on a
# contended device sharing the GPU with the host compositor.
SOLVER_FLOOR_INTERACTIONS_PER_MS = 10_000_000.0
# Interactions one body pays per level of the tree it walks. The asymptotics of Barnes-Hut
# are ``log N``, but the constant is what actually decides whether a scene is interactive:
# a body accepts a few hundred cells and leaves on the way down at the opening angles this
# product uses. Counting 256 per level is comfortably above what a walk at those angles
# touches, which is the direction this has to err in.
TREE_WALK_INTERACTIONS_PER_LEVEL = 256.0
# Interactions one body pays for a bounded neighbour list. The hash grid exists precisely
# to make this a constant rather than a function of the particle count; 128 is well over
# the neighbour counts the particle engines' kernels are tuned for.
NEIGHBOUR_INTERACTIONS_PER_BODY = 128.0

# Margin applied to an init-time device measurement of the real solver primitive, when one is
# available, to bound that primitive at admission (:func:`measured_solver_primitive_bound`). It
# measures the *real* descriptor's *real* captured graph --- the exact graph Play replays. The
# residual uncertainty is therefore only live-driver scheduling, viewport draw contention on
# the shared GPU, and thermal drift, which 1.25x covers while still admitting a frame the
# floor's weakest-card bound would refuse. It is a margin on a measurement, never a substitute
# for one: with no measurement the floor stays the fail-closed default.
SOLVER_MEASUREMENT_MARGIN = 1.25


class SolverCostClass(StrEnum):
    """How the work of one solver substep grows with the number of bodies.

    A cost class, rather than a measured number, because the admission has to answer
    before anything has run. What a compiled plan costs is decided by which sources of
    acceleration each body reads, and that is fixed the moment the graph compiles.

    ``UNKNOWN`` is the fail-closed answer and the default for anything not classified: a
    plan built from a module nobody has costed is exactly the case where a guess is worst,
    because it would be made once, silently, and never revisited.
    """

    ALL_PAIRS = "all_pairs"
    """Every body reads every other body. Direct summation."""

    TREE_WALK = "tree_walk"
    """Every body walks a hierarchy and reads accepted cells. Barnes-Hut."""

    NEIGHBOUR = "neighbour"
    """Every body reads a bounded neighbour list. Anything behind a hash grid."""

    PER_BODY = "per_body"
    """Every body reads only itself. Integration, equations of state, diagnostics."""

    UNKNOWN = "unknown"
    """Nothing declares how this scales, so nothing bounds it."""


def solver_call_bound(
    *,
    body_count: int,
    cost_class: SolverCostClass,
    substeps_per_call: int = 1,
) -> float | None:
    """Worst case for one uninterruptible solver call, or ``None`` if nothing bounds it.

    One *call*, not one substep, because that is the unit the host actually invokes:
    ``ProductRuntime.step`` runs every substep of a frame inside a single call and nothing
    can interrupt it part-way, so a frame subdivided into eight substeps presents the host
    with one primitive eight times the size rather than eight primitives it can spread
    across ticks. A caller that can genuinely stop between substeps passes ``1`` and asks
    again.

    This is the one primitive an admission cannot bound by inspecting a byte count. Every
    other part of a frame transaction moves a record of known size, so a throughput floor
    settles it; what a solve costs depends on how many bodies read how many sources of
    acceleration, which only the compiled plan knows.
    """

    if body_count < 1:
        raise ValueError("body_count must be positive")
    if substeps_per_call < 1:
        raise ValueError("substeps_per_call must be positive")
    per_body = _interactions_per_body(body_count, cost_class)
    if per_body is None:
        return None
    interactions = body_count * per_body * substeps_per_call
    return interactions / SOLVER_FLOOR_INTERACTIONS_PER_MS


def measured_solver_primitive_bound(measured_primitive_ms: float) -> float | None:
    """Bound one synchronized solver primitive from an init-time device measurement.

    The sanctioned counterpart to :func:`solver_call_bound`'s floor. Where the floor answers
    the admission from declared scale and a weakest-supported-card throughput --- so it can
    answer before anything runs, at the cost of mis-estimating a real GPU in both directions
    --- this answers from what the runtime measured its own real captured-graph frame to cost
    once at initialize time (``ProductRuntime.measure_solver_primitive_cost_ms``). For a
    resumable Direct runtime it is one captured substep; for an ADR 0061 Tree runtime it is
    one backend-owned work slice; and for a fallback runtime it is the whole authored frame.

    This is the one place the frame-transaction admission is allowed to bound the solver call
    by a measurement rather than a floor. It is fail-closed: a non-positive or non-finite
    measurement is no measurement, so it returns ``None``. Legacy callers may then fall back
    to their plan floor; work-sliced callers must refuse because that floor does not bound one
    internal slice.
    """

    if not math.isfinite(measured_primitive_ms) or measured_primitive_ms <= 0.0:
        return None
    return measured_primitive_ms * SOLVER_MEASUREMENT_MARGIN


# How many recent measurements of one frame-tail primitive (stage or commit) the packing loop
# keeps (ADR 0141). The worst of them, margined, is what it plans on: a trailing window so a
# one-off stall stops shaping the packing after a few frames, short enough to follow a scene
# that got heavier, and still several frames of evidence rather than the last one alone.
TAIL_MEASUREMENT_WINDOW = 8

# What the packing loop adds to a frame tail's worst recent measurement (ADR 0146). None: the
# worst of the last eight already is the margin. The two mistakes it chooses between are not
# the same size. A tail that runs past the budget shortens Blender's half of that one period by
# the overrun, and the re-arm's reserved floor still holds. A tail held back to the next tick
# costs that frame a whole 41.7 ms tick. At 100,000 on the Disk, 1.25 held back a 4.5 ms tail
# with 6 ms of budget left, so a frame of 19.3 ms of work took two ticks.
TAIL_PACKING_MARGIN = 1.0


def narrowed_primitive_bound(
    admitted_ms: float | None, measured_ms: Iterable[float]
) -> float | None:
    """Tighten one admitted primitive bound by what that primitive recently measured.

    The floors above "may be tightened by measurement later"; this is where the packing
    loop does it for a frame's stage and commit (ADR 0141). At 100,000 bodies the commit
    tail's floor alone exceeds a 24 fps service budget, so planning on the floor made every
    frame span two ticks, each paying Blender's own time and the re-arm floor. It only
    ever narrows: the worst recent measurement, times :data:`TAIL_PACKING_MARGIN`
    (ADR 0146), replaces the bound only when it is smaller. No usable
    measurement (none yet, or only non-finite or non-positive ones) keeps the admitted
    bound, which stays the first-use answer, and an unbounded primitive stays unbounded.
    Admission itself is untouched: this decides how a tick is packed, not what is admitted.
    """

    if admitted_ms is None:
        return None
    samples = [m for m in measured_ms if math.isfinite(m) and m > 0.0]
    if not samples:
        return admitted_ms
    return min(admitted_ms, max(samples) * TAIL_PACKING_MARGIN)


def tail_fits_frame_period(spent_ms: float, tail_bound_ms: float, frame_rate_fps: float) -> bool:
    """Whether a frame's tail may still run in the tick that has spent ``spent_ms``.

    Asked only once the frame's last solve primitive has run and the stage or the commit is
    next (ADR 0223, amending ADR 0141 for the packing of the tail only). Every other
    primitive is still planned against the slice, :func:`service_budget_ms`. The tail is
    planned against one whole frame period measured from the start of the tick:
    ``spent_ms + tail_bound_ms`` must not exceed it, and the edge is inclusive.

    The period is the slice divided by :data:`SERVICE_BUDGET_FRACTION`, so the two cannot
    drift apart. Deferring a tail to a second tick costs more than the tail: a zero re-arm
    (ADR 0074), one full pass of the host's event loop that redraws the old frame because
    nothing new was published (ADR 0058), then a committing tick that ADR 0094 charges the
    whole span since admission, so it overruns and waits the full reservation. On the Star
    Cluster at 100,000 a 22 ms frame took 55-68 ms of wall that way and 34-37 ms in one tick.

    This does not move the reservation. A tick that runs past its slice still re-arms by
    :func:`service_re_arm_seconds`, which waits at least the half period the fraction
    reserves (ADR 0135), so Blender still gets that half after every published frame.
    What grows is the longest span the host can be uninterruptible: up to one period
    rather than half of one.

    A non-finite or negative ``spent_ms`` or ``tail_bound_ms`` is no evidence that the tail
    fits, so it answers ``False`` and the tail waits, the conservative direction. An invalid
    ``frame_rate_fps`` raises, exactly as :func:`service_budget_ms` does.
    """

    period_ms = service_budget_ms(frame_rate_fps) / SERVICE_BUDGET_FRACTION
    if not (math.isfinite(spent_ms) and math.isfinite(tail_bound_ms)):
        return False
    if spent_ms < 0.0 or tail_bound_ms < 0.0:
        return False
    return spent_ms + tail_bound_ms <= period_ms


# Compatibility for third-party callers compiled against ADR 0056. Shipped code uses the
# precise primitive name; keeping the alias does not keep the ambiguous runtime state.
measured_solver_call_bound = measured_solver_primitive_bound


def _interactions_per_body(body_count: int, cost_class: SolverCostClass) -> float | None:
    """Sources of acceleration one body reads in one substep, or ``None`` if unknown."""

    if cost_class is SolverCostClass.ALL_PAIRS:
        return float(body_count)
    if cost_class is SolverCostClass.TREE_WALK:
        # ``max(..., 2)`` because a one-body tree still costs a body one look, and ``log2``
        # of anything below two is zero or undefined --- a bound of nothing is the one
        # answer this must never produce.
        return TREE_WALK_INTERACTIONS_PER_LEVEL * math.log2(max(body_count, 2))
    if cost_class is SolverCostClass.NEIGHBOUR:
        return NEIGHBOUR_INTERACTIONS_PER_BODY
    if cost_class is SolverCostClass.PER_BODY:
        return 1.0
    return None


SOLVER_COST_CLASS_BY_MODULE: dict[str, SolverCostClass] = {
    # Gravity: the cost lives in whichever SPATIAL_SERVICE step computed the
    # acceleration, not in the FORCE step that applies it to a body. Every-body-reads-
    # every-body for the direct sum, log-height tree descent for Barnes-Hut, and
    # ``auto_select`` is bounded as if it will pick the direct sum because it decides
    # live and a bound taken before the first step cannot know which backend it lands
    # on --- the conservative reading is the only safe one.
    "astro.gravity.direct_all_pairs": SolverCostClass.ALL_PAIRS,
    "astro.gravity.barnes_hut_tree": SolverCostClass.TREE_WALK,
    "astro.gravity.auto_select": SolverCostClass.ALL_PAIRS,
    "astro.gravity.newtonian": SolverCostClass.PER_BODY,
    "astro.gravity.tree_newtonian": SolverCostClass.PER_BODY,
    "astro.gravity.auto_newtonian": SolverCostClass.PER_BODY,
    # The shearing box (ADR 0202) sums the same all-pairs gravity over the box and its nearest
    # images -- nine times the sources of the plain sum at the default one shell -- so its
    # step is priced as a second all-pairs sum rather than as free. That is still below the
    # true cost (nine times, not two); the runtime's own measured primitive bound, not this
    # declared floor, is what a resumable Direct runtime is admitted on.
    "astro.frame.shearing_box": SolverCostClass.ALL_PAIRS,
    # Giant impact (ADR 0094): the pressure constraint reads the one neighbour grid, the
    # gravity is the direct all-pairs sum, and the moon guide reads only the particle it
    # steers. Its frame is one primitive, so the runtime measures what that frame costs rather
    # than trusting this bound for the pressure substeps inside it.
    "astro.giant_impact.neighbor_grid": SolverCostClass.NEIGHBOUR,
    "astro.giant_impact.pbf_pressure": SolverCostClass.NEIGHBOUR,
    "astro.giant_impact.tiled_gravity": SolverCostClass.ALL_PAIRS,
    "astro.giant_impact.moon_guide": SolverCostClass.PER_BODY,
    # Rock impact (ADR 0193): the pressure solve and the bond springs read the one bounded
    # neighbour or bond row per body, and gravity is the direct all-pairs sum. A frame is one
    # primitive holding several outer and inner steps, so the runtime measures what that frame
    # costs rather than trusting this bound for the steps inside it.
    "astro.spacesim.neighbor_grid": SolverCostClass.NEIGHBOUR,
    "astro.spacesim.iisph_pressure": SolverCostClass.NEIGHBOUR,
    "astro.spacesim.bond_strength": SolverCostClass.NEIGHBOUR,
    "astro.spacesim.tiled_gravity": SolverCostClass.ALL_PAIRS,
}
"""Cost class of every module a compiled plan may schedule into the solve.

A module absent from this table costs nothing toward the sum in :func:`plan_solver_call_bound`
--- it is host-side bookkeeping (context, initial state, integration, diagnostics, output),
not a source of per-body work. A module that *is* solver work but is missing here would
silently bound as free, which is exactly the failure this table exists to prevent; the
standing coverage test in ``tests/unit/test_scheduler.py`` guards every gravity and Solid-engine
module against that gap.
"""


def plan_solver_call_bound(
    *,
    modules: Iterable[str],
    body_count: int,
    substeps_per_call: int = 1,
) -> float | None:
    """Worst case for one uninterruptible solver call across a whole compiled plan.

    ``modules`` is the caller's own SPATIAL_SERVICE and FORCE steps --- the steps that
    actually cost per-body work --- already picked out of the compiled plan; this function
    has no notion of step family and sums exactly what it is given. Summing only the FORCE
    steps of a gravity plan would under-bound a direct-summation or tree-walk plan as if it
    were the cost of its cheap force-application step alone, because the O(N^2)/O(N log N)
    work lives in the SPATIAL_SERVICE step that computed the acceleration those FORCE steps
    merely apply.

    A module this table has no cost class for defaults to :attr:`SolverCostClass.UNKNOWN`,
    which makes the whole sum ``None``: one uncosted step must not let the plan look bounded
    on the strength of its costed neighbours.
    """

    total = 0.0
    for module in modules:
        cost_class = SOLVER_COST_CLASS_BY_MODULE.get(module, SolverCostClass.UNKNOWN)
        bound = solver_call_bound(
            body_count=body_count,
            cost_class=cost_class,
            substeps_per_call=substeps_per_call,
        )
        if bound is None:
            return None
        total += bound
    return total