"""Carrying legacy-panel Advanced edits onto the role System (Slice 5 Chunk 5).

Slice 5 Task 11 moved the D10 Advanced tier -- Precision, Collision, the tree parameters, the
Auto accuracy policy, the SPH group -- onto ``AstroWarpSystemSettings``, and pointed the
compile bridge at it. Every default is byte-identical to the legacy context row's, so an
untouched System compiles exactly what it always did.

One case that change does not cover, recorded at the time in ``DEVELOPMENT_STATE.md`` section
9: a user who edited those values *through the legacy Solver panel* on a role-authored System.
Their edits live on the context row, the compile no longer reads them, and once Chunk 5
deletes the panels there is no surface that shows them either. The values would still be in
the file, describing a simulation nobody can see and nothing runs.

So this has to exist before the deletion, not after. It decides what to carry; the Blender
half applies it.

**The rule is deliberately conservative.** A row value is carried only when the user changed
it *and* the matching role field is still untouched. When both were changed and they
disagree, nothing is carried and the disagreement is reported: the role settings are the
accepted surface, and silently overwriting an explicitly authored value with a legacy one
would replace a decision the user made on the surface that survives with one they made on a
surface that is being removed. There is no reading of that which is safe to guess.

Pure: dicts in, a report out, no ``bpy``. The extension gathers the two value sets and
applies the result.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


def _f32(value: object) -> object:
    """Round a Python float to the single precision Blender stores it at.

    Blender holds every ``FloatProperty`` as a C 32-bit float, so a value read back
    with ``getattr`` is the float32-widened form of what was stored -- an untouched
    default of ``0.6`` comes back as ``0.6000000238418579``. Non-floats (ints, enums,
    bools, strings) are not widened and pass through unchanged.
    """

    if isinstance(value, float):
        try:
            return struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error):
            return value
    return value


def _is_edit(value: object, default: object) -> bool:
    """True when ``value`` genuinely differs from ``default`` at storage precision.

    The only evidence of an edit is a value that differs from the one the field
    shipped with, but the live read is float32-widened while ``default`` is a float64
    literal: comparing them with a raw ``!=`` reports every untouched float field as
    edited, which turns a legacy edit that should be carried into a false conflict.
    Quantising both to float32 first makes an untouched default compare equal to its
    shipped value, while a real edit still lands on a different float32.
    """

    return _f32(value) != _f32(default)


@dataclass(frozen=True)
class AdvancedMigrationPlan:
    """What a legacy row has to say about one System's Advanced settings.

    Four buckets rather than one dict, because they call for four different actions and
    collapsing them would hide the only one that needs a person.
    """

    #: The user edited the row and left the role field alone: carry it.
    carried: dict[str, object] = field(default_factory=dict)
    #: Both were edited and they disagree. Nothing is carried; the user decides.
    conflicts: dict[str, tuple[object, object]] = field(default_factory=dict)
    #: Both were edited to the same value. Nothing to do, and worth distinguishing from a
    #: conflict so a report does not raise an alarm about an agreement.
    agreed: tuple[str, ...] = ()
    #: Fields the legacy row never carried, so there was nothing to compare.
    unknown: tuple[str, ...] = ()

    @property
    def has_work(self) -> bool:
        return bool(self.carried)

    @property
    def needs_a_decision(self) -> bool:
        return bool(self.conflicts)

    def message(self) -> str:
        """One line for the status area, saying what happened and what did not.

        A migration that reported only its successes would leave the conflicts invisible,
        and a conflict is precisely the case where the user has to look.
        """

        if not self.carried and not self.conflicts:
            return "No legacy Advanced settings to carry over."
        parts = []
        if self.carried:
            parts.append(
                f"carried {len(self.carried)} legacy Advanced setting(s): "
                + ", ".join(sorted(self.carried))
            )
        if self.conflicts:
            parts.append(
                f"left {len(self.conflicts)} alone because the System already sets them "
                "differently: " + ", ".join(sorted(self.conflicts))
            )
        # Not `str.capitalize()`: it lowercases everything after the first character, so
        # "Advanced" and "System" came out mangled in the one message a user reads about
        # their own file.
        sentence = "; ".join(parts)
        return sentence[:1].upper() + sentence[1:] + "."


def advanced_migration_plan(
    row_values: dict[str, object],
    role_values: dict[str, object],
    defaults: dict[str, object],
) -> AdvancedMigrationPlan:
    """Decide what to carry from a legacy context row onto a role System.

    ``defaults`` is the shipped default of each role field, and it is what makes "the user
    changed this" answerable at all: a Blender property always holds a value, so the only
    evidence of an edit is that the value differs from the one it shipped with. That is why
    Task 11 made every role default byte-identical to the row's -- without that, every
    System would look edited on both sides and every field would be a conflict.

    Fields absent from ``row_values`` are reported as unknown rather than skipped: a legacy
    row saved before a field existed is a real thing to find in a user's file, and silently
    treating it as "unchanged" would be a guess dressed as a fact.
    """

    carried: dict[str, object] = {}
    conflicts: dict[str, tuple[object, object]] = {}
    agreed: list[str] = []
    unknown: list[str] = []

    for name, default in sorted(defaults.items()):
        if name not in row_values:
            unknown.append(name)
            continue
        row_value = row_values[name]
        role_value = role_values.get(name, default)

        # Compared at single precision: a float read back from Blender is float32-widened
        # while `default` is a float64 literal, so a raw `!=` flags every untouched float
        # as edited (see `_is_edit`). `row_value == role_value` below stays a plain compare
        # -- both are live reads at the same precision, so it is already exact.
        row_edited = _is_edit(row_value, default)
        role_edited = _is_edit(role_value, default)

        if not row_edited:
            # Nothing was authored on the legacy side, so there is nothing to carry --
            # whatever the role side says.
            continue
        if not role_edited:
            carried[name] = row_value
        elif row_value == role_value:
            agreed.append(name)
        else:
            conflicts[name] = (row_value, role_value)

    return AdvancedMigrationPlan(
        carried=carried,
        conflicts=conflicts,
        agreed=tuple(agreed),
        unknown=tuple(unknown),
    )
