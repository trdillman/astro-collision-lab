"""What an edit to a property does: one decision, stated once (ADR 0211).

An ``update=`` callback in the add-on has to answer the same few questions about the edit it
was just handed: is there a compiled plan to go stale, does the System's engine read this
field at all, is a runtime playing that could take the new value now, and, for the context
row, is the add-on itself the writer. The answers used to be spread over eight callbacks in
two files; each carried its own copy of the branch it needed and none could be compared with
another. This module is the branch, with no Blender in it:

* :class:`EditKind` is what the declaration site said the property is -- a role setting that
  waits for a restart, one a running simulation can adopt, a member's Role or System, or a
  context-row property by how much of the solved state it reaches;
* :class:`EditFact` is what the session looks like at the moment of the edit;
* :func:`classify` maps the two to a :class:`Route`: an :class:`Effect` to perform, and the
  effect to fall back to if the first one is declined.

The add-on's ``edit_router`` gathers the facts, performs the effect through its injectable
seams, and owns the never-raise wrapping. Nothing here reads or writes a scene.

**Which fields are live** is not decided here. ``invalidation.PROSPECTIVE_*`` already says
which solver inputs no solved step has read, so a running simulation can adopt them going
forward; :data:`LIVE_ROLE_PROPERTIES` is derived from those sets, and a test requires the
add-on's declarations to equal it. A field added to a ``PROSPECTIVE_*`` set but left on the
restart callback, or wired live without being prospective, fails that test.

**Two fallbacks differ on purpose.** A *role* live edit that the runtime declines (or that has
no runtime to take it) marks the System for a restart and leaves the stored descriptor where
the running simulation is (ADR 0098, ADR 0165). A *row* live edit that is declined invalidates
the plan and the runtime (ADR 0021): a row that disagrees with what plays is worse than a lost
plan. They also read the runtime's answer differently (:func:`role_live_accepted`,
:func:`settle_row`). Unifying them would change what an artist sees on a refusal, so the table
keeps them apart.

This is not ADR 0066's build. No field becomes live or keyframable here; this names the
decisions the callbacks already made.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .invalidation import (
    PROSPECTIVE_FIELD_BINDING_FIELDS,
    PROSPECTIVE_TIMELINE_FIELDS,
    PROSPECTIVE_UNITS_FIELDS,
)

__all__ = [
    "LIVE_ROLE_PROPERTIES",
    "LIVE_ROW_PROPERTIES",
    "EditFact",
    "EditKind",
    "Effect",
    "Route",
    "RowSettle",
    "classify",
    "role_edit_kind",
    "role_live_accepted",
    "row_edit_kind",
    "settle_row",
]


class EditKind(StrEnum):
    """What the declaration site says the edited property is."""

    #: A role setting a running simulation cannot adopt: the System waits for a restart.
    ROLE_RESTART = "role_restart"
    #: A role setting that is prospective: offered to the runtime, a restart mark if declined.
    ROLE_LIVE = "role_live"
    #: A member's Role or System changed: every compiled System it is in or joining rebuilds.
    ROLE_MEMBER_SYSTEM = "role_member_system"
    #: A context-row property that reaches a solved step: the plan, runtime and state go.
    ROW_INVALIDATE = "row_invalidate"
    #: The frame range: only the frame-to-step mapping moves, so the run is truncated, not lost.
    ROW_TIMELINE = "row_timeline"
    #: A prospective parameter on the context row: offered to the runtime, invalidated if declined.
    ROW_PARAMETERS = "row_parameters"
    #: A display or output-only value: written into the stored descriptor, nothing else moves.
    ROW_PRESENTATION = "row_presentation"
    #: A viewport-look value: the stream republishes, no solver input is touched.
    ROW_RENDER = "row_render"


class Effect(StrEnum):
    """What the router performs."""

    #: Nothing is marked and nothing is written. (The start-frame preview is told separately.)
    NOTHING = "nothing"
    #: Mark the System as waiting for a restart, under the edit's label.
    MARK_RESTART = "mark_restart"
    #: Role tier: hand the runtime the new values; accepted only when :func:`role_live_accepted`.
    OFFER_RUNTIME = "offer_runtime"
    #: Row tier: put the row's copy back to the value that is playing (ADR 0098 addendum).
    RESTORE_ROW = "restore_row"
    #: Row tier: discard the compiled plan and the runtime, and say the whole context is suspect.
    INVALIDATE = "invalidate"
    #: Row tier: install the new frame range and retime the runtime; see :func:`settle_row`.
    RETIME = "retime"
    #: Row tier: install the new parameters and reparameterize the runtime; see :func:`settle_row`.
    REPARAMETERIZE = "reparameterize"
    #: Row tier: refresh the stored descriptor's display and output values.
    REFRESH_PRESENTATION = "refresh_presentation"
    #: Row tier: republish the stream the edited look belongs to.
    REPUBLISH = "republish"


@dataclass(frozen=True)
class EditFact:
    """The session at the moment of one edit. Facts a kind does not read are ignored.

    ``has_plan`` is "there is a compiled plan that can go stale": the System's context row for a
    role edit, the edited row for a row edit, and, for a member edit, at least one compiled
    System to mark.
    """

    kind: EditKind
    has_plan: bool = False
    #: Role tier: the System's Solid engine never reads the edited field.
    engine_ignores: bool = False
    #: Role tier: a runtime is playing the plan.
    live_runtime: bool = False
    #: Row tier: the add-on itself is writing the row's copy of a role System's values.
    restoring: bool = False
    #: Row tier: the row is the copy of a role System that is playing now.
    playing_role_row: bool = False


@dataclass(frozen=True)
class Route:
    """The effect to perform, and the one to perform instead if it is declined."""

    effect: Effect
    fallback: Effect = Effect.NOTHING


class RowSettle(StrEnum):
    """How a row edit reads the runtime's answer (the earliest step it reaches)."""

    #: The runtime would not take it (answered step 0): the plan goes with it.
    REBUILD = "rebuild"
    #: The runtime took it and has solved nothing yet (answered ``None``).
    NO_SOLVED_STATE = "no_solved_state"
    #: The runtime took it; steps before the answer stand.
    TRUNCATED = "truncated"


#: Role properties a running simulation adopts going forward: ``PROSPECTIVE_*``, by property name.
LIVE_ROLE_PROPERTIES: frozenset[str] = frozenset(
    (
        *PROSPECTIVE_UNITS_FIELDS,
        *PROSPECTIVE_TIMELINE_FIELDS,
        *PROSPECTIVE_FIELD_BINDING_FIELDS,
    )
)

#: Context-row properties routed to the live parameter path: the prospective ones that exist on
#: the row, plus ``gravitational_constant``. G left ``PROSPECTIVE_*`` at ADR 0040, but its row
#: property still takes this path on purpose: ``refresh_solver_parameters`` refuses a moved G,
#: and the row then invalidates, which is the right outcome for a retroactive input.
LIVE_ROW_PROPERTIES: frozenset[str] = frozenset(
    (*PROSPECTIVE_UNITS_FIELDS, *PROSPECTIVE_TIMELINE_FIELDS, "gravitational_constant")
)

_ROW_TIMELINE_PROPERTIES = frozenset({"frame_start", "frame_end"})
_ROW_PRESENTATION_PROPERTIES = frozenset({"display_scale"})


def role_edit_kind(prop: str) -> EditKind:
    """The kind of a role setting (a simulation setting, not a member's Role or System)."""

    return EditKind.ROLE_LIVE if prop in LIVE_ROLE_PROPERTIES else EditKind.ROLE_RESTART


def row_edit_kind(prop: str) -> EditKind:
    """The kind of a solver-facing context-row property.

    The viewport-look properties are declared by the render annotation table
    (``ROW_RENDER``) and are not enumerated here; every other row property that is wired at all
    invalidates unless it is named below.
    """

    if prop in _ROW_PRESENTATION_PROPERTIES:
        return EditKind.ROW_PRESENTATION
    if prop in _ROW_TIMELINE_PROPERTIES:
        return EditKind.ROW_TIMELINE
    if prop in LIVE_ROW_PROPERTIES:
        return EditKind.ROW_PARAMETERS
    return EditKind.ROW_INVALIDATE


def role_live_accepted(earliest: int | None) -> bool:
    """Whether a role live edit counts the runtime's answer as taken.

    Any non-zero step is. ``None`` (a runtime that answered nothing) and ``0`` (one that would
    have to start over) both fall back to the restart mark; the row tier reads these two
    answers differently (:func:`settle_row`).
    """

    return bool(earliest)


def settle_row(earliest: int | None) -> RowSettle:
    """How a row timeline or parameter edit reads the runtime's answer."""

    if earliest == 0:
        return RowSettle.REBUILD
    if earliest is None:
        return RowSettle.NO_SOLVED_STATE
    return RowSettle.TRUNCATED


def classify(fact: EditFact) -> Route:
    """The route for one edit. Behaviour-for-behaviour the callbacks it replaced."""

    kind = fact.kind
    if kind is EditKind.ROLE_RESTART:
        if not fact.has_plan or fact.engine_ignores:
            return Route(Effect.NOTHING)
        return Route(Effect.MARK_RESTART)
    if kind is EditKind.ROLE_LIVE:
        if not fact.has_plan or fact.engine_ignores:
            return Route(Effect.NOTHING)
        if fact.live_runtime:
            return Route(Effect.OFFER_RUNTIME, Effect.MARK_RESTART)
        return Route(Effect.MARK_RESTART)
    if kind is EditKind.ROLE_MEMBER_SYSTEM:
        return Route(Effect.MARK_RESTART if fact.has_plan else Effect.NOTHING)
    if kind is EditKind.ROW_INVALIDATE:
        return Route(Effect.INVALIDATE)
    if kind is EditKind.ROW_TIMELINE:
        if fact.has_plan:
            return Route(Effect.RETIME, Effect.INVALIDATE)
        return Route(Effect.INVALIDATE)
    if kind is EditKind.ROW_PARAMETERS:
        if fact.restoring:
            return Route(Effect.NOTHING)
        if fact.playing_role_row:
            return Route(Effect.RESTORE_ROW)
        if fact.has_plan:
            return Route(Effect.REPARAMETERIZE, Effect.INVALIDATE)
        return Route(Effect.INVALIDATE)
    if kind is EditKind.ROW_PRESENTATION:
        return Route(Effect.REFRESH_PRESENTATION)
    if kind is EditKind.ROW_RENDER:
        return Route(Effect.REPUBLISH)
    raise ValueError(f"unrouted edit kind: {kind!r}")
