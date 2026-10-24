"""Pure value objects for authored timeline-stage intent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TimelineStageKind(StrEnum):
    """Stage vocabulary exposed by the native authoring surface."""

    STABILIZE = "Stabilize"
    IMPACT = "Impact"
    REACCUMULATE = "Reaccumulate"
    EVOLVE = "Evolve"


#: The kinds a user can pick and a template may list (ADR 0161). ``Stabilize`` is retired:
#: nothing ran it, and ADR 0086 says a stage nothing reads is not a stage. It stays in
#: :class:`TimelineStageKind` and in what ``validate_stage_list`` accepts, because a file
#: saved while it was offered must still load; the Blender enum keeps it for the same
#: reason (removing an item from a stored enum shifts the saved integers).
OFFERED_STAGE_KINDS: tuple[TimelineStageKind, ...] = (
    TimelineStageKind.IMPACT,
    TimelineStageKind.REACCUMULATE,
    TimelineStageKind.EVOLVE,
)
RETIRED_STAGE_KINDS: tuple[TimelineStageKind, ...] = (TimelineStageKind.STABILIZE,)


@dataclass(frozen=True)
class TimelineStage:
    """An authored stage and the first timeline frame where it applies."""

    kind: TimelineStageKind
    boundary_frame: int | None


def validate_stage_list(
    stages: tuple[TimelineStage, ...], *, frame_start: int, frame_end: int
) -> None:
    """Validate stage types, off-timeline placement, and ordered frame boundaries.

    A ``kind`` may repeat by design: the only uniqueness-like constraint is that
    ``STABILIZE`` is confined to index 0. Every other kind (``IMPACT``,
    ``REACCUMULATE``, ``EVOLVE``) may appear more than once so long as the
    strictly increasing ``boundary_frame`` ordering holds -- e.g. a multi-impact
    timeline with two ``IMPACT`` stages at different frames is valid. Ordering,
    not kind identity, defines a well-formed stage list.
    """

    if not isinstance(stages, tuple):
        raise TypeError("stages must be a tuple")
    _require_integer("frame_start", frame_start)
    _require_integer("frame_end", frame_end)
    if frame_end < frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start")

    previous_boundary: int | None = None
    for index, stage in enumerate(stages):
        field = f"stages[{index}]"
        if not isinstance(stage, TimelineStage):
            raise TypeError(f"{field} must be a TimelineStage")
        if not isinstance(stage.kind, TimelineStageKind):
            raise TypeError(f"{field}.kind must be a TimelineStageKind")

        if stage.kind is TimelineStageKind.STABILIZE:
            if index != 0:
                raise ValueError(f"{field}.kind Stabilize is only valid at index 0")
            if stage.boundary_frame is not None:
                raise ValueError(f"{field}.boundary_frame must be None for Stabilize")
            continue

        boundary = stage.boundary_frame
        if not _is_integer(boundary):
            raise TypeError(f"{field}.boundary_frame must be an integer")
        if not frame_start <= boundary <= frame_end:
            raise ValueError(
                f"{field}.boundary_frame must be within [{frame_start}, {frame_end}]"
            )
        if previous_boundary is not None and boundary <= previous_boundary:
            raise ValueError(f"{field}.boundary_frame must be strictly increasing")
        previous_boundary = boundary


#: The solver ("backend") names that read each stage pair. Empty: the only reader was the
#: stage-boundary handoff of the detailed impact engine, which was deleted on the owner's
#: order of 2026-10-13. On every engine that remains (Direct, Tree, the cracking-rock engine
#: and the moon engine) a stage name is a label for a moment on the timeline, and no check
#: runs at its boundary. The table is kept, empty, as the one place a future consumer is
#: declared; :func:`inert_stage_boundaries` and the Timeline overlay both read it.
BOUNDARY_CONSUMER_BACKENDS: dict[tuple[TimelineStageKind, TimelineStageKind], frozenset[str]] = {}

#: The adjacent ``(from_kind, to_kind)`` stage-kind pairs some part of the product acts on,
#: on at least one solver (the keys of :data:`BOUNDARY_CONSUMER_BACKENDS`). None today.
#:
#: Every pair absent from this set is *inert*: authoring it is legal, it round-trips, and it
#: moves both state fingerprints -- so editing it forces a full re-solve -- but no solver,
#: service or display path reads it. That combination is the trap
#: :func:`inert_stage_boundaries` exists to surface: a boundary the user pays for and does
#: not get.
CONSUMED_STAGE_TRANSITIONS: frozenset[tuple[TimelineStageKind, TimelineStageKind]] = frozenset(
    BOUNDARY_CONSUMER_BACKENDS
)


def boundary_is_consumed(
    from_kind: TimelineStageKind, to_kind: TimelineStageKind, backend: str | None = None
) -> bool:
    """Whether the ``from_kind`` -> ``to_kind`` boundary is read on ``backend``.

    ``backend=None`` means the solver is not known (a descriptor that names none, or a draw
    callback with no compiled row yet). It answers exactly as the pair alone does. With no
    consumer declared in :data:`BOUNDARY_CONSUMER_BACKENDS` the answer is ``False`` for every
    pair and every solver.
    """

    if backend is not None and not isinstance(backend, str):
        raise TypeError("backend must be a string or None")
    consumers = BOUNDARY_CONSUMER_BACKENDS.get((from_kind, to_kind))
    if consumers is None:
        return False
    return backend is None or backend in consumers


@dataclass(frozen=True)
class InertBoundary:
    """An authored stage boundary that nothing downstream acts on.

    ``index`` is the position of the *later* stage in the list -- the stage the
    boundary begins -- matching how ``boundary_frame`` is stored and how
    ``validate_stage_list`` numbers its own messages.
    """

    index: int
    from_kind: TimelineStageKind
    to_kind: TimelineStageKind
    boundary_frame: int | None


def inert_stage_boundaries(
    stages: tuple[TimelineStage, ...],
    *,
    backend: str | None = None,
) -> tuple[InertBoundary, ...]:
    """Every adjacent stage pair in ``stages`` that no consumer reads, in list order.

    A boundary is a *transition between two stages*, so the first stage is never
    reported: nothing transitions into it, and the single leading ``Evolve`` that every
    non-staged template authors declares no interior moment at all. Reporting that one
    would fire on all five shipped templates and mean nothing, which is how a real signal
    gets trained away.

    ``backend`` is the solver the stages will run on, when it is known; it narrows a
    consumer declared for particular solvers, and ``None`` keeps the answer the pair alone
    gives.

    This is deliberately a pure query and not a refusal. The authoring is well-formed --
    ``validate_stage_list`` is unchanged and still accepts it -- and a stage list is
    forward-looking authoring intent that may find its consumer later. The caller decides
    what to do with the answer; :class:`~.compiler.GraphCompiler` raises it as an
    ``ADVISORY``, which is surfaced and never blocks.
    """

    if not isinstance(stages, tuple):
        raise TypeError("stages must be a tuple")
    inert: list[InertBoundary] = []
    for index, (earlier, later) in enumerate(zip(stages, stages[1:], strict=False), start=1):
        if not isinstance(earlier, TimelineStage) or not isinstance(later, TimelineStage):
            raise TypeError(f"stages[{index}] and its predecessor must be TimelineStage values")
        if boundary_is_consumed(earlier.kind, later.kind, backend):
            continue
        inert.append(
            InertBoundary(
                index=index,
                from_kind=earlier.kind,
                to_kind=later.kind,
                boundary_frame=later.boundary_frame,
            )
        )
    return tuple(inert)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_integer(field: str, value: object) -> None:
    if not _is_integer(value):
        raise TypeError(f"{field} must be an integer")
