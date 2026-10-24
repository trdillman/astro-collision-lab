"""D13's Color By fields on the immutable bake, and derivation from the selected bake alone.

The scrub cache holds live frames and is rebuilt freely; a bake is immutable and shipped. So
the same D13 requirement lands differently here, and the difference is the whole module.

**Switching Color By must not touch the bake.** Not a rewrite, not a new revision, not a
solver start. The way to guarantee that is not to be careful at the call site -- it is for the
selected mode never to enter the bake at all. A bake records which *sources* it carries; the
mode a user happens to be looking through is a viewing decision, and viewing decisions do not
belong in a content-addressed artefact. `bake.py` folds ``provenance`` into ``revision_key``,
so a selected mode stored there would mint a new ``bake_id`` on every dropdown change --
turning a display switch into a rebake, which is exactly what D13 forbids. :func:`source_note`
refuses to record one.

**Absent means absent.** A body with no authoritative damage gets no damage *field*, not a
zero. The v2 payload derives its columns from the body dicts it is handed, so omitting the key
omits the column, and a bake from a Direct-backend System is byte-identical to one written
before these fields existed. A zero would be a column full of "undamaged", which is a claim.

**Velocity and internal energy are reused, never duplicated.** They are already in the bake;
Speed and Energy read them where they are. Only Solid damage and persistent material ID are
genuinely missing, so only those two are added.

Derivation reads the bake and nothing else -- no runtime, no scrub cache, no solver state.
:mod:`astro_warp.product.color_by` owns the formulas and the availability rules; this owns
where the numbers live once a bake exists.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from .color_by import (
    INTERNAL_ENERGY_BACKENDS,
    ColorByMode,
    PartitionSources,
    availability,
    damage_value,
    material_value,
    specific_energy,
    speed,
)
from .color_carriage import ColorProvenance

#: Body-dict keys. Namespaced so a column cannot collide with a user's own field, and fixed
#: forever once a bake carries one: renaming would make an existing bake decode as missing.
BODY_DAMAGE = "astro_damage"
BODY_MATERIAL_ID = "astro_material_id"

#: Existing body keys these fields deliberately do *not* duplicate.
REUSED_BODY_KEYS = ("velocity", "internal_energy")

#: Where the source note lives inside ``BakeRevision.provenance``.
PROVENANCE_KEY = "color_by_sources"
_FORBIDDEN_PROVENANCE_SELECTION_KEYS = frozenset(
    {"selected", "selected_mode", "mode", "color_by"}
)
_FORBIDDEN_TYPED_STORAGE_KEYS = frozenset(
    {
        "astro_color",
        "astro_color_rgba",
        "astro_color_value",
        "color",
        "color_by",
        "color_by_sources",
        "color_ramp",
        "color_rgba",
        "color_source",
        "color_value",
        "color_values",
        "colors",
        "mode",
        "ramp",
        "rgba",
        "selected",
        "selected_mode",
    }
)


class ColorBakeError(ValueError):
    """A Color By field could not be written to, or read from, an immutable bake."""


@dataclass(frozen=True)
class ColorBakeFrame:
    """One ordinary bake snapshot plus its authoritative missing D13 source arrays.

    Velocity and specific internal energy stay in the sample body records because the bake
    already owns those columns. Damage and persistent material identity stay separate until
    preflight succeeds, preventing partial mutation of the caller's snapshot.
    """

    sample: dict[str, Any]
    solid_damage: Any = None
    material_ids: Any = None


def _frame_values(values: Any, *, label: str, count: int) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bool)):
        raise ColorBakeError(f"{label} must be a one-dimensional sequence")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ColorBakeError(f"{label} must be a one-dimensional sequence") from error
    if len(result) != count:
        raise ColorBakeError(
            f"{label} carries {len(result)} values for a baked frame with {count} bodies"
        )
    return result


def _damage_values(values: Any, *, count: int) -> tuple[float, ...]:
    result = _frame_values(values, label="damage", count=count)
    converted = []
    for value in result:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ColorBakeError("damage accepts finite real values only")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ColorBakeError("damage accepts finite real values only")
        converted.append(scalar)
    return tuple(converted)


def _material_values(values: Any, *, count: int) -> tuple[int, ...]:
    result = _frame_values(values, label="material_ids", count=count)
    converted = []
    maximum = (1 << 63) - 1
    for value in result:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ColorBakeError("material_ids accepts exact integers only")
        material_id = int(value)
        if material_id < 0:
            raise ColorBakeError(
                "material_ids contains a negative persistent material ID"
            )
        if material_id > maximum:
            raise ColorBakeError("material_ids contains a value above INT64_MAX")
        converted.append(material_id)
    return tuple(converted)


def _materialize_owned(value: Any, *, path: str, active: set[int]) -> Any:
    """Recursively own one typed value, including one-shot iterables.

    The binary writer serializes a frame immediately, but the legacy writer materializes all
    frames before serialization. Owning both through this one seam prevents a caller from
    changing bytes or revision identity after a frame has passed preflight.
    """

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ColorBakeError(f"typed bake sample contains a cycle at {path}")
        active.add(identity)
        try:
            return {
                deepcopy(key): _materialize_owned(
                    item, path=f"{path}.{key}", active=active
                )
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ColorBakeError(f"typed bake sample contains a cycle at {path}")
        active.add(identity)
        try:
            return [
                _materialize_owned(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        raise ColorBakeError(f"typed bake sample contains unordered values at {path}")
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        identity = id(value)
        if identity in active:
            raise ColorBakeError(f"typed bake sample contains a cycle at {path}")
        active.add(identity)
        try:
            try:
                items = list(value)
            except Exception as error:
                raise ColorBakeError(
                    f"typed bake sample iterable failed at {path}: {error}"
                ) from error
            return [
                _materialize_owned(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(items)
            ]
        finally:
            active.remove(identity)
    try:
        return deepcopy(value)
    except Exception as error:
        raise ColorBakeError(
            f"typed bake sample value cannot be detached at {path}: {error}"
        ) from error


def _reject_typed_color_storage(value: Any, *, path: str = "sample") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_TYPED_STORAGE_KEYS:
                raise ColorBakeError(
                    f"typed bake selection/output display output key {key!r} is forbidden "
                    f"at {path}"
                )
            _reject_typed_color_storage(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_typed_color_storage(item, path=f"{path}[{index}]")


def _velocity(value: Any, *, body_index: int) -> list[float]:
    components = _frame_values(
        value, label=f"body {body_index} velocity", count=3
    )
    result = []
    for component in components:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ColorBakeError(
                f"body {body_index} velocity needs exactly three finite non-boolean reals"
            )
        scalar = float(component)
        if not math.isfinite(scalar):
            raise ColorBakeError(
                f"body {body_index} velocity needs exactly three finite non-boolean reals"
            )
        result.append(scalar)
    return result


def _internal_energy(value: Any, *, body_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ColorBakeError(
            f"body {body_index} internal_energy must be one finite non-boolean real"
        )
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ColorBakeError(
            f"body {body_index} internal_energy must be one finite non-boolean real"
        )
    return scalar


def carry_bake_frame(
    frame: ColorBakeFrame, *, provenance: ColorProvenance
) -> dict[str, Any]:
    """Preflight and copy one frame into the existing homogeneous bake column schema."""

    if not isinstance(frame, ColorBakeFrame):
        raise ColorBakeError(f"expected ColorBakeFrame, got {type(frame).__name__}")
    if not isinstance(provenance, ColorProvenance):
        raise ColorBakeError("a typed Color bake frame needs exact recorded provenance")
    if not isinstance(frame.sample, dict):
        raise ColorBakeError("a Color bake frame sample must be a dict")
    owned_sample = _materialize_owned(frame.sample, path="sample", active=set())
    _reject_typed_color_storage(owned_sample)
    raw_bodies = owned_sample.get("bodies")
    if not isinstance(raw_bodies, list) or not all(
        isinstance(body, dict) for body in raw_bodies
    ):
        raise ColorBakeError("a Color bake frame needs a list of body records")
    bodies = list(raw_bodies)
    for index, body in enumerate(bodies):
        existing = {BODY_DAMAGE, BODY_MATERIAL_ID}.intersection(body)
        if existing:
            raise ColorBakeError(
                f"baked body {index} already contains typed Color source columns: "
                + ", ".join(sorted(existing))
            )
    available = {
        mode for partition in provenance.partitions for mode in partition.available
    }

    requires_velocity = "SPEED" in available or "ENERGY" in available
    requires_internal_energy = any(
        "ENERGY" in partition.available
        and partition.backend in INTERNAL_ENERGY_BACKENDS
        for partition in provenance.partitions
    )
    for index, body in enumerate(bodies):
        if "velocity" in body:
            body["velocity"] = _velocity(body["velocity"], body_index=index)
        elif requires_velocity:
            raise ColorBakeError(
                "recorded Speed/Energy availability needs velocity on every baked body"
            )
        if "internal_energy" in body:
            body["internal_energy"] = _internal_energy(
                body["internal_energy"], body_index=index
            )
        elif requires_internal_energy:
            raise ColorBakeError(
                "recorded SPH/impact Energy availability needs internal_energy on every "
                "baked body"
            )

    damage = None
    if frame.solid_damage is not None:
        if "DAMAGE" not in available:
            raise ColorBakeError(
                "damage was supplied but Damage is not recorded as available"
            )
        damage = _damage_values(frame.solid_damage, count=len(bodies))
    elif "DAMAGE" in available:
        raise ColorBakeError("Damage is recorded as available but its raw damage is missing")

    material_ids = None
    if frame.material_ids is not None:
        if "MATERIAL" not in available:
            raise ColorBakeError(
                "material_ids were supplied but Material is not recorded as available"
            )
        material_ids = _material_values(frame.material_ids, count=len(bodies))
    elif "MATERIAL" in available:
        raise ColorBakeError(
            "Material is recorded as available but persistent material_ids are missing"
        )

    carried = dict(owned_sample)
    carried["bodies"] = [
        extend_bake_body(
            body,
            damage=None if damage is None else damage[index],
            material_id=None if material_ids is None else material_ids[index],
        )
        for index, body in enumerate(bodies)
    ]
    return carried


def merge_source_note(
    bake_provenance: dict[str, Any] | None, color_provenance: ColorProvenance
) -> dict[str, Any]:
    """Return manifest provenance with one exact, non-conflicting D13 source note."""

    if bake_provenance is not None and not isinstance(bake_provenance, dict):
        raise ColorBakeError("bake provenance must be a JSON object")
    merged = dict(bake_provenance or {})
    forbidden = _FORBIDDEN_PROVENANCE_SELECTION_KEYS.intersection(merged)
    if forbidden:
        raise ColorBakeError(
            "Color By selection is a viewing decision and cannot enter bake provenance: "
            + ", ".join(sorted(forbidden))
        )
    note = source_note(color_provenance)[PROVENANCE_KEY]
    existing = merged.get(PROVENANCE_KEY)
    if existing is not None:
        decoded = read_source_note({PROVENANCE_KEY: existing})
        if decoded != color_provenance:
            raise ColorBakeError(
                "bake provenance conflicts with the authoritative Color source provenance"
            )
    merged[PROVENANCE_KEY] = note
    return merged


def extend_bake_body(
    body: dict[str, Any],
    *,
    damage: float | None = None,
    material_id: int | None = None,
) -> dict[str, Any]:
    """A copy of `body` carrying only the D13 fields that are authoritatively present.

    Returns a new dict rather than mutating: a bake body is written once and read many
    times, and an in-place extension would let a caller's later reuse of the same dict change
    what a frame already claimed.

    Passing ``None`` is how a caller says "this System has no such source". It is not the same
    as passing ``0.0`` or ``-1``, and neither of those is accepted as a stand-in.
    """

    extended = dict(body)
    if damage is not None:
        if isinstance(damage, bool) or not isinstance(damage, Real):
            raise ColorBakeError("damage accepts finite real values only")
        value = float(damage)
        if not math.isfinite(value):
            raise ColorBakeError(
                "damage is not finite; an unmeasured body omits the field entirely"
            )
        extended[BODY_DAMAGE] = value
    if material_id is not None:
        if isinstance(material_id, bool) or not isinstance(material_id, Integral):
            raise ColorBakeError(
                "material_id must be an int persistent registry/shell ID, not "
                f"{type(material_id).__name__}"
            )
        material_id = int(material_id)
        if material_id < 0:
            raise ColorBakeError(
                f"material_id {material_id} is not a valid persistent ID; a body with no "
                "material omits the field rather than baking a negative one"
            )
        if material_id > (1 << 63) - 1:
            raise ColorBakeError("material_id is above INT64_MAX")
        extended[BODY_MATERIAL_ID] = material_id
    return extended


def source_note(provenance: ColorProvenance) -> dict[str, Any]:
    """The ``BakeRevision.provenance`` entry describing what sources this bake carries.

    Refuses anything that looks like a *selected* mode. `bake.py` folds provenance into
    ``revision_key``, so a selection recorded here would give every dropdown change its own
    ``bake_id`` -- a display switch becoming a rebake. The refusal is here rather than in a
    comment because the mistake is natural: "record what Color By was set to" reads like
    provenance right up until you notice it changes the artefact's identity.
    """

    if not isinstance(provenance, ColorProvenance):
        raise ColorBakeError(
            f"source_note needs a ColorProvenance, got {type(provenance).__name__}"
        )
    document = provenance.as_document()
    for forbidden in _FORBIDDEN_PROVENANCE_SELECTION_KEYS:
        if forbidden in document:  # pragma: no cover - structural guard
            raise ColorBakeError(
                f"{forbidden!r} is a viewing decision and must not enter the bake's identity"
            )
    return {PROVENANCE_KEY: document}


def read_source_note(provenance: dict[str, Any] | None) -> ColorProvenance | None:
    """The Color By source note out of a revision's provenance, or ``None``.

    ``None`` for a bake written before these fields existed, which is a real thing to find
    and is distinguished from a bake that recorded no available modes.
    """

    if not provenance:
        return None
    document = provenance.get(PROVENANCE_KEY)
    if document is None:
        return None
    if not isinstance(document, dict):
        raise ColorBakeError(f"{PROVENANCE_KEY} is not a JSON object")
    try:
        return ColorProvenance.from_document(document)
    except ValueError as error:
        raise ColorBakeError(f"invalid {PROVENANCE_KEY}: {error}") from error


def sources_from_bake(
    bodies: list[dict[str, Any]], *, provenance: ColorProvenance, name: str | None = None
) -> PartitionSources:
    """What a baked frame authoritatively offers, read off the bodies themselves.

    Deliberately derived from the bake rather than taken from the live System: a bake is a
    record of what was true when it was written, and a System's role or damage setting can
    have changed since. Asking the bake is the only way to get an answer about the bake.
    """

    if not bodies:
        raise ColorBakeError("a baked frame with no bodies offers no Color By source")
    if not isinstance(provenance, ColorProvenance):
        raise ColorBakeError("a baked frame needs its recorded Color By provenance")
    try:
        partition = provenance.resolve_partition(name)
    except (KeyError, TypeError) as error:
        raise ColorBakeError(str(error)) from error
    has_velocity = bool({"SPEED", "ENERGY"}.intersection(partition.available)) and all(
        "velocity" in body for body in bodies
    )
    has_damage = "DAMAGE" in partition.available and all(
        BODY_DAMAGE in body for body in bodies
    )
    has_material = "MATERIAL" in partition.available and all(
        BODY_MATERIAL_ID in body for body in bodies
    )
    has_internal_energy = "ENERGY" in partition.available and all(
        "internal_energy" in body for body in bodies
    )
    return PartitionSources(
        role=partition.role,
        backend=partition.backend,
        has_committed_velocity=has_velocity,
        damage_active=has_damage,
        damage_model="BAKED" if has_damage else "NONE",
        has_material_registry=has_material,
        has_persistent_material_id=has_material,
        has_specific_internal_energy=has_internal_energy,
        name=partition.name,
    )


def bake_color_values(
    mode: ColorByMode,
    bodies: list[dict[str, Any]],
    *,
    provenance: ColorProvenance,
    name: str | None = None,
) -> list[float]:
    """``astro_color_value`` per body, derived from the selected bake alone.

    Refuses through the same availability rules the live path uses, so a mode unavailable in
    the viewport cannot become available by going through a bake. That symmetry is the point:
    the alternative is a user who sees Damage refused live, bakes, and finds it "works".
    """

    mode = ColorByMode(mode)
    if not isinstance(provenance, ColorProvenance):
        raise ColorBakeError("a bake needs recorded provenance for its selected partition")
    try:
        partition = provenance.resolve_partition(name)
    except (KeyError, TypeError) as error:
        raise ColorBakeError(str(error)) from error
    recorded_refusal = partition.reason_for(mode.value)
    if recorded_refusal is not None:
        raise ColorBakeError(recorded_refusal)
    sources = sources_from_bake(bodies, provenance=provenance, name=name)
    verdict = availability(mode, sources)
    if not verdict.available:
        raise ColorBakeError(verdict.refusal)

    if mode is ColorByMode.SPEED:
        return [speed(tuple(body["velocity"])) for body in bodies]
    if mode is ColorByMode.ENERGY:
        # The same INTERNAL_ENERGY_BACKENDS the live path uses, rather than a second literal
        # tuple: two lists of backend names would be one edit away from disagreeing about
        # which Energy definition a bake carries.
        if (
            sources.has_specific_internal_energy
            and sources.backend in INTERNAL_ENERGY_BACKENDS
        ):
            return [
                specific_energy(
                    tuple(body["velocity"]),
                    specific_internal_energy=float(body["internal_energy"]),
                )
                for body in bodies
            ]
        return [specific_energy(tuple(body["velocity"])) for body in bodies]
    if mode is ColorByMode.DAMAGE:
        return [damage_value(body[BODY_DAMAGE]) for body in bodies]
    return [material_value(int(body[BODY_MATERIAL_ID])) for body in bodies]
