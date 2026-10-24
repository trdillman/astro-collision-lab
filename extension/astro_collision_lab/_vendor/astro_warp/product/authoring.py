"""Pure membership validation for role records (no bpy).

The extension's panels convert scene objects to RoleRecords and display these
messages verbatim; compile-time enforcement arrives with the Slice 2 bridge.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# The seven role identifiers a scene object may carry (ADR 0037 SS2 / addendum SS3).
# The extension's PropertyGroups and the Slice 2 compile bridge both draw from this
# set rather than re-declaring it.
ROLES = ("NONE", "SYSTEM", "BODY", "SOLID", "CLOUD", "EMITTER", "FORCE")

# Roles that must belong to a System carrier: every role except NONE (unassigned)
# and SYSTEM (the carrier itself).
MEMBER_ROLES = tuple(role for role in ROLES if role not in ("NONE", "SYSTEM"))


# AW-004 / ADR 0159: the System the add-on makes for an artist who gave an object a member
# Role in a scene that has none. The extension creates it; these names live here so the
# decision and the wording are shared by the creator, Remove from Scene and the tests.
AUTO_SYSTEM_NAME = "Astro System"
AUTO_SYSTEM_COLLECTION_NAME = "Astro Warp"


@dataclass(frozen=True)
class RoleRecord:
    object_name: str
    role: str  # UPPER_SNAKE identifier, "NONE" when unassigned.
    system_name: str | None = None


def list_systems(records: Sequence[RoleRecord]) -> list[str]:
    """Return every System carrier's object name, in record order.

    This exists precisely so callers can tell "zero Systems" apart from
    "several Systems" -- a distinction ``default_system`` deliberately
    collapses, since both states mean there is no unambiguous implicit
    default.
    """
    return [r.object_name for r in records if r.role == "SYSTEM"]


def default_system(records: Sequence[RoleRecord]) -> str | None:
    """Return the scene's sole System carrier's object name, or ``None``.

    This answers "is there an unambiguous implicit default?" -- ``None`` is
    returned both when zero Systems exist (there is nothing to default to)
    and when several exist (the default would be ambiguous -- the caller
    must point at one explicitly via ``RoleRecord.system_name``). Only when
    exactly one System is present is it unambiguous, and that name is
    returned. Callers that need to distinguish the zero and several cases
    (to report *why* there is no default) should use ``list_systems``
    instead.
    """
    systems = list_systems(records)
    return systems[0] if len(systems) == 1 else None


def auto_system_needed(
    prior_role: str, new_role: str, records: Sequence[RoleRecord]
) -> bool:
    """Whether giving an object ``new_role`` should also make the scene its System (AW-004).

    True only for the first Role an object is given (``prior_role`` is ``NONE``), when that
    Role is a member role and ``records`` -- which already include the object itself -- hold
    no System. Every other case answers False: the System Role itself is the thing being
    asked for, clearing a Role or moving between member Roles is not a first assignment,
    and a scene that already has one System (or several, which is the artist's own choice to
    resolve) never gets another.

    An unknown role is a programming error, as in :func:`validate_membership`.
    """
    for role in (prior_role, new_role):
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}'")
    return prior_role == "NONE" and new_role in MEMBER_ROLES and not list_systems(records)


def validate_membership(records: Sequence[RoleRecord]) -> list[str]:
    """Validate that every member-role record resolves to exactly one System.

    Returns human-readable messages intended for direct display in the
    authoring panel, one per offending object, in the order the records were
    given; an empty list means every member role validly resolves to a
    System. ``NONE`` and ``SYSTEM`` records are never offending: ``NONE`` is
    unassigned and ignored, and ``SYSTEM`` records are carriers, not members.

    A member record with an explicit ``system_name`` is valid only if that
    name is a System carrier in ``records``. A member record with no
    ``system_name`` relies on the scene's default System: valid when exactly
    one System exists, an error when zero exist (nothing to default to) or
    when several exist (the default is ambiguous and must be resolved by
    pointing at one explicitly).

    A record whose role is not one of the module's known identifiers is a
    programming error in the caller, not a user-facing validation outcome:
    it raises ``ValueError`` naming the offending object and role.
    """
    for record in records:
        if record.role not in ROLES:
            raise ValueError(
                f"{record.object_name}: unknown role '{record.role}'"
            )

    systems = list_systems(records)
    system_names = set(systems)
    errors: list[str] = []
    for record in records:
        if record.role not in MEMBER_ROLES:
            continue
        if record.system_name is not None:
            if record.system_name not in system_names:
                errors.append(
                    f"{record.object_name}: System reference "
                    f"'{record.system_name}' is not a System carrier; "
                    f"point it at an object whose Role is System"
                )
        elif len(systems) == 0:
            errors.append(
                f"{record.object_name}: no System in the scene; "
                f"set some object's Role to System"
            )
        elif len(systems) > 1:
            errors.append(
                f"{record.object_name}: multiple Systems exist; choose one explicitly"
            )
    return errors


def split_membership_message(message: str) -> tuple[str, str | None]:
    """Split a membership error into its problem statement and optional remedy.

    Splits at the *last* ``"; "`` so an object name that legally contains ``"; "``
    (e.g. ``"Cloud; A"``) is preserved in the problem half.
    """
    problem, separator, remedy = message.rpartition("; ")
    return (problem, remedy) if separator else (message, None)
