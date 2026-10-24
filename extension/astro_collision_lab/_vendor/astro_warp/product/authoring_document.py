"""``AstroAuthoringDocumentV1``: the portable record of what a user *authored*.

Slice 5 ships two documents that look superficially similar and must never be confused,
because confusing them is how stale derived values get treated as authority.

**The Effective Descriptor** is the full compiled `SimulationDescriptor` JSON, offered
read-only for inspection, diffing and bug reports. It contains everything -- including
values Blender derived at compile time: a Disk Annulus object's evaluated basis and
extents, and a ``FROM_OBJECT`` cloud's measured velocity and sampled frame pair. It is
explicitly **not importable**, and this module refuses it by name.

**The authoring document** is this schema. It carries only what a person typed or chose:
role and membership, and the authored fields of each role's settings. It carries no
derived byte at all. That is the whole point -- a derived value is a *measurement of a
scene at a moment*, and importing one into a different scene, or the same scene later,
would assert a measurement that was never taken.

Why this matters concretely. Import a document into a scene where the disk object has been
rotated since export. If the document carried the old basis, compile would have two
candidate answers and the stale one would look just as authoritative as the live one. By
carrying no basis at all, there is exactly one answer: the current evaluation. The old
observation still has a use -- as a *drift baseline*, so the UI can say "this compiled from
a different orientation than last time" -- but a baseline is a report, never an input. That
asymmetry is the contract, and :func:`drift_report` is its other half.

Everything here is pure: dicts in, dicts out, no ``bpy``. The extension gathers role state
into :func:`build_authoring_document` and applies a validated
:class:`AuthoringDocument` back onto role objects; both halves of that are Blender's, and
neither is allowed to relax what this module refuses.

Refusal is total. :func:`load_authoring_document` either returns a fully validated document
or raises, and it validates every object before returning any of them, so a caller that
applies only after a successful return cannot half-apply a bad file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

SCHEMA_NAME = "AstroAuthoringDocumentV1"
"""The only schema tag this module accepts. A different tag is refused, not migrated."""

SCHEMA_VERSION = 1

#: Roles that may appear in an authoring document. Every one of them now carries authored
#: fields: ``FORCE`` gained its pair in WP-D2 (ADR 0045) and ``EMITTER`` its release
#: schedule in WP-D3 (ADR 0047), retiring the v1 state in which both were representable as
#: a role assignment and nothing more.
ROLES = ("SYSTEM", "BODY", "CLOUD", "SOLID", "EMITTER", "FORCE")

DISPLAY_GROUPS = (
    "AUTO",
    "BODIES",
    "SPH_POINTS",
    "TRAILS",
    "GAS_VOLUME",
    "STAR_POINTS",
    "FRAGMENTS",
)

STAGE_KINDS = ("Stabilize", "Impact", "Reaccumulate", "Evolve")


def _floats(count: int):
    return ("vector", count)


#: The authored field of each role, and its type tag. These sets are the schema: a key not
#: listed here is refused rather than ignored, because silently dropping an unknown key is
#: how a document from a newer version imports as a *subset* of itself with no indication.
#:
#: What is deliberately absent from each set matters as much as what is present:
#:
#: - ``CLOUD`` has no ``sample_basis``, ``sample_extent`` or ``sample_frame_provenance``.
#:   All three are host measurements; ``annulus_inner_radius`` is present because ADR 0040
#:   section 5 makes it the one disk quantity no object can imply.
#: - ``SOLID`` has no ``mass``, ``radius`` or ``initial_velocity``. Those are written by
#:   plan application from the material and the proxy geometry -- derived, despite living
#:   on the same settings group as the authored fields.
#: - ``BODY`` *does* carry ``mass`` and ``radius``, because for a Body they are typed, not
#:   derived. The same field name is authored on one role and derived on another, which is
#:   exactly why this is a per-role table rather than one shared list.
ROLE_FIELDS: dict[str, dict[str, Any]] = {
    "SYSTEM": {
        "unit_preset": ("enum", ("PLANETARY", "STELLAR", "GALACTIC", "CUSTOM")),
        "length_scale_m": ("positive", None),
        "mass_scale_kg": ("positive", None),
        "time_scale_s": ("positive", None),
        "solver": ("str", None),
        "color_by": ("str", None),
        "seconds_per_frame": ("positive", None),
        "softening": ("nonnegative", None),
        "substeps": ("positive_int", None),
        # ADR 0193 decision 6. Authored, not derived: a System exported as Rock and imported
        # as Auto would play a different simulation from the same file. A document from
        # before the field existed simply does not carry it, and an import leaves the
        # System's own value alone. A document that says ``DETAILED`` (the compressible
        # engine, deleted 2026-10-13) is read as ROCK: see ``RETIRED_ENUM_VALUES``.
        "impact_model": ("enum", ("AUTO", "ROCK", "GIANT")),
        # ADR 0202. Authored, for the reason impact_model is: a System exported as a shearing box
        # and imported as open space would play a different simulation from the same file.
        "shearing_box": ("bool", None),
        "shear_omega": ("positive", None),
        "shear_box_x": ("positive", None),
        "shear_box_y": ("positive", None),
        "timeline_stages": ("stages", None),
    },
    "BODY": {
        "mass": ("positive", None),
        "radius": ("nonnegative", None),
        "initial_velocity": _floats(3),
        "spin": _floats(3),
        "pinned": ("bool", None),
    },
    "CLOUD": {
        "shape": ("str", None),
        "count": ("positive_int", None),
        "total_mass": ("positive", None),
        "seed": ("nonnegative_int", None),
        "mass_jitter": ("unit_interval", None),
        "velocity_model": ("str", None),
        "support_fraction": ("nonnegative", None),
        "keplerian_central_persistent_id": ("str", None),
        "keplerian_companion_persistent_id": ("str", None),
        "keplerian_counter_clockwise": ("bool", None),
        "eccentricity_jitter": ("unit_interval", None),
        "annulus_inner_radius": ("nonnegative", None),
        "annulus_surface_density_exponent": ("nonnegative", None),
        # ADR 0202. Authored, not derived: an Expansion cloud exported at one speed and
        # imported at another would play a different shell from the same file.
        "expansion_speed": ("nonnegative", None),
        "expansion_proportional": ("bool", None),
        "bulk_velocity": _floats(3),
    },
    "SOLID": {
        "material": ("str", None),
        "resolution_mode": ("str", None),
        "target_count": ("positive_int", None),
        "particle_delta_m": ("positive", None),
        "spin": _floats(3),
        "temperature": ("nonnegative", None),
    },
    "EMITTER": {
        # No ``shape``: an Emitter's nozzle volume is the object's own half-size, the
        # way a Ball cloud's is, so there is no shape to author and nothing for a
        # document to carry. ``capacity`` is authored rather than derived from
        # ``rate`` and the range because it is the allocation, and the compiler
        # checks the release budget against it.
        "rate": ("nonnegative", None),
        "capacity": ("positive_int", None),
        "frame_start": ("nonnegative_int", None),
        "frame_end": ("nonnegative_int", None),
        "total_mass": ("positive", None),
        "seed": ("nonnegative_int", None),
        "initial_velocity": _floats(3),
    },
    "FORCE": {
        # ``strength`` is a plain finite number with no range clause: its sign is
        # meaning (a negative Uniform pushes backward along the axis, a negative
        # attractor repels), so the only invalid strengths are the non-finite ones.
        "force_kind": ("enum", ("UNIFORM", "POINT_ATTRACTOR")),
        "strength": ("number", None),
    },
}

#: Keys whose presence anywhere in a candidate document proves it is not an authoring
#: document. Split into two groups because the two failures deserve different messages: a
#: compiled descriptor is a *recognisable* wrong document a user could plausibly have
#: exported from the inspector, while a lone derived key is a hand-edited or
#: machine-generated file trying to smuggle a measurement in.
COMPILED_DESCRIPTOR_KEYS = ("bindings", "graph", "runtime_plan", "stages", "context_id")

DERIVED_KEYS = (
    "sample_basis",
    "sample_extent",
    "sample_frame_provenance",
    "binding_id",
    "velocity_model_provenance",
    "code_fingerprint",
    "gravitational_constant",
)


EFFECTIVE_DESCRIPTOR_SCHEMA = "AstroEffectiveDescriptorV1"
"""The inspector document's tag. Never accepted by :func:`load_authoring_document`."""

EFFECTIVE_DESCRIPTOR_NOTE = (
    "Read-only. This is the full compiled descriptor, including values Blender derived "
    "from the scene at compile time (Disk Annulus evaluated basis and extents, FROM_OBJECT "
    "measured velocity and frame pair). It is for inspection, diffing and bug reports. To "
    "write a file you can import, use Export Authoring."
)


def effective_descriptor_document(descriptor_json: str) -> str:
    """Wrap a compiled descriptor for the read-only inspector.

    The descriptor's own JSON is nested **unchanged**. It has to be: the release proof
    compares role-built and legacy-built descriptor bytes, so adding a key inside it would
    move bytes that a gate is pinning. The `importable=false` label the plan requires
    therefore lives in the wrapper, where it can be added without touching what it
    describes -- and where :func:`load_authoring_document` can see it and refuse by name.
    """

    return (
        json.dumps(
            {
                "schema": EFFECTIVE_DESCRIPTOR_SCHEMA,
                "importable": False,
                "note": EFFECTIVE_DESCRIPTOR_NOTE,
                "descriptor": json.loads(descriptor_json),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


class AuthoringDocumentError(ValueError):
    """A candidate document is not a valid ``AstroAuthoringDocumentV1``.

    A ``ValueError`` subclass so existing refusal handling on the compile path catches it
    without a new branch, and its own type so an importer can tell a schema refusal from
    an arithmetic one.
    """


@dataclass(frozen=True)
class AuthoringObject:
    """One role-carrying object's authored state."""

    name: str
    persistent_id: str
    role: str
    #: The persistent id of the System this object belongs to. Empty for a System itself
    #: and for an unassigned object -- a name would break on rename, and an object
    #: reference does not serialize.
    system_persistent_id: str = ""
    display_group: str = "AUTO"
    settings: dict[str, Any] = field(default_factory=dict)


#: Enum values retired from a role field, mapped to what a document carrying one now means.
#:
#: ``MASS`` left ``color_by`` on 2026-08-28 (``adr/0044-color-by-enum-retirement-of-mass.md``).
#: The ``.blend`` property needs no migration -- Blender resets an unknown enum item to the
#: default on load, and since nothing has ever read ``color_by`` the reset changes nothing a
#: user can see. A *document* is different: it is applied by ``setattr`` onto a live enum, which
#: refuses an unknown identifier with ``TypeError`` partway through the write. Normalising here,
#: during validation, is what keeps ``apply_document``'s all-or-nothing guarantee intact.
#:
#: ``DETAILED`` left ``impact_model`` on 2026-10-13 with the compressible impact engine it
#: selected. Unlike ``MASS`` it was a *working* choice, so the replacement is not a reset to a
#: default: ROCK is the realtime engine every collision that is not a giant impact plays on
#: (ADR 0201), which is what a System left at Detailed would have been routed to had it been
#: Auto. The ``.blend`` property is handled by :func:`current_impact_model`, which every
#: reader of a stored value goes through.
RETIRED_ENUM_VALUES: dict[str, dict[str, str]] = {
    "color_by": {"MASS": "SPEED"},
    "impact_model": {"DETAILED": "ROCK"},
}

#: What Blender hands back for an enum whose stored integer no longer matches any item: the
#: empty string (it logs "current value ... matches no enum" and returns ``''``). The retired
#: ``DETAILED`` was the last item of ``impact_model``, so a saved ``DETAILED`` reads back this
#: way once the item is gone, and :func:`current_impact_model` maps it the same way.
_UNMATCHED_STORED_ENUM = ""


def current_impact_model(value: object) -> str:
    """The Impact Model a stored or authored value means in this build.

    ``AUTO``, ``ROCK`` and ``GIANT`` mean themselves. The retired ``DETAILED`` and the empty
    string a saved ``DETAILED`` reads back as (see :data:`_UNMATCHED_STORED_ENUM`) mean ``ROCK``.
    Any other string is returned unchanged, so a value this does not recognise is still refused
    by the caller that knows what to say about it, instead of being silently turned into rock.
    """

    text = str(value)
    if text == _UNMATCHED_STORED_ENUM:
        return RETIRED_ENUM_VALUES["impact_model"]["DETAILED"]
    return RETIRED_ENUM_VALUES["impact_model"].get(text, text)


@dataclass(frozen=True)
class AuthoringDocument:
    """A validated authoring document. Construct it through this module, not directly."""

    objects: tuple[AuthoringObject, ...] = ()
    schema: str = SCHEMA_NAME
    version: int = SCHEMA_VERSION
    #: One line per retired enum value substituted during validation. Empty for any document
    #: written by this build. Reported rather than applied silently: the user authored a value
    #: this build no longer has, and a substitution they are not told about is a value they
    #: will look for later and not find.
    normalisations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "objects": [
                {
                    "name": obj.name,
                    "persistent_id": obj.persistent_id,
                    "role": obj.role,
                    "system_persistent_id": obj.system_persistent_id,
                    "display_group": obj.display_group,
                    "settings": dict(obj.settings),
                }
                for obj in self.objects
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# -- export -------------------------------------------------------------------------


def build_authoring_document(objects: tuple[AuthoringObject, ...]) -> AuthoringDocument:
    """Validate gathered role state into a document.

    Runs the same validation the import path runs. Exporting a document the importer would
    refuse is a bug worth failing on at export, where the scene that caused it is still in
    front of the user, rather than at import in some other session.
    """

    # Round-tripped through the dict form on purpose: it is the import path's own
    # validator, so export cannot pass something import would reject.
    return load_authoring_document(AuthoringDocument(objects=objects).to_dict())


# -- import -------------------------------------------------------------------------


def load_authoring_document(data: Any) -> AuthoringDocument:
    """Validate a candidate document, or raise.

    Accepts a mapping or a JSON string. Every object is validated before any is returned,
    so a caller that applies only on success cannot partially mutate a scene from a file
    that turns out to be bad three objects in.
    """

    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise AuthoringDocumentError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AuthoringDocumentError(
            f"an authoring document is a JSON object, got {type(data).__name__}"
        )

    _refuse_compiled_descriptor(data)

    schema = data.get("schema")
    if schema != SCHEMA_NAME:
        raise AuthoringDocumentError(
            f"expected schema {SCHEMA_NAME!r}, got {schema!r}. Export Authoring writes "
            "this schema; the Effective Descriptor inspector's JSON is a different "
            "document and is not importable"
        )
    version = data.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise AuthoringDocumentError(
            f"{SCHEMA_NAME} version {version!r} is not supported; this build reads "
            f"version {SCHEMA_VERSION}"
        )

    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        raise AuthoringDocumentError("'objects' must be a list")

    normalisations: list[str] = []
    objects = tuple(
        _validated_object(_as_mapping(item), index, normalisations)
        for index, item in enumerate(raw_objects)
    )
    _validate_membership(objects)
    return AuthoringDocument(
        objects=objects,
        schema=SCHEMA_NAME,
        version=SCHEMA_VERSION,
        normalisations=tuple(normalisations),
    )


def _as_mapping(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise AuthoringDocumentError(
            f"each entry in 'objects' is a JSON object, got {type(item).__name__}"
        )
    return item


def _refuse_compiled_descriptor(data: dict[str, Any]) -> None:
    """Name the specific wrong document rather than failing on a missing schema key.

    A user who exports the Effective Descriptor and tries to import it has done something
    reasonable-looking; "expected schema AstroAuthoringDocumentV1" would leave them
    guessing which of the two menu items they wanted.
    """

    if data.get("schema") == EFFECTIVE_DESCRIPTOR_SCHEMA or data.get("importable") is False:
        raise AuthoringDocumentError(
            "this is an Effective Descriptor, which is labelled importable=false: it "
            "carries values Blender derived from the scene at compile time, and importing "
            "them would assert a measurement that was never taken in this scene. Use "
            "Export Authoring to write an importable document"
        )
    present = [key for key in COMPILED_DESCRIPTOR_KEYS if key in data]
    if len(present) >= 2:
        raise AuthoringDocumentError(
            "this looks like a compiled Effective Descriptor (it carries "
            f"{', '.join(sorted(present))}), which is read-only and not importable: it "
            "contains values Blender derived from the scene at compile time. Use Export "
            "Authoring to write an importable document"
        )
    for key in DERIVED_KEYS:
        if key in data:
            raise AuthoringDocumentError(
                f"'{key}' is a derived value, not an authored one, so it may not appear "
                "in an authoring document. Compile recomputes it from the current scene"
            )


def _validated_object(
    item: dict[str, Any], index: int, normalisations: list[str]
) -> AuthoringObject:
    """Validate one object, appending to `normalisations` for any retired value substituted.

    The list is threaded in rather than returned alongside because the caller builds every
    object before accepting any: a substitution found in the last object still has to be
    reportable, and a per-object return would need collecting anyway.
    """

    where = f"objects[{index}]"

    role = item.get("role")
    if role not in ROLES:
        raise AuthoringDocumentError(
            f"{where}: role {role!r} is not one of {', '.join(ROLES)}"
        )

    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise AuthoringDocumentError(f"{where}: 'name' must be a non-empty string")

    persistent_id = item.get("persistent_id")
    if not isinstance(persistent_id, str) or not persistent_id:
        raise AuthoringDocumentError(
            f"{where} ({name}): 'persistent_id' must be a non-empty string -- it is how "
            "import finds the object again after a rename"
        )

    system_persistent_id = item.get("system_persistent_id", "")
    if not isinstance(system_persistent_id, str):
        raise AuthoringDocumentError(
            f"{where} ({name}): 'system_persistent_id' must be a string"
        )

    display_group = item.get("display_group", "AUTO")
    if display_group not in DISPLAY_GROUPS:
        raise AuthoringDocumentError(
            f"{where} ({name}): display group {display_group!r} is not one of "
            f"{', '.join(DISPLAY_GROUPS)}"
        )

    raw_settings = item.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise AuthoringDocumentError(f"{where} ({name}): 'settings' must be an object")

    allowed = ROLE_FIELDS[role]
    settings: dict[str, Any] = {}
    for key, value in raw_settings.items():
        if key in DERIVED_KEYS:
            raise AuthoringDocumentError(
                f"{where} ({name}): '{key}' is derived from the scene at compile time and "
                "may not be authored. Remove it; compile recomputes it"
            )
        if key not in allowed:
            raise AuthoringDocumentError(
                f"{where} ({name}): {role} has no authored field '{key}'. Allowed: "
                f"{', '.join(sorted(allowed)) or '(none)'}"
            )
        # A retired value is replaced *before* it is validated: it is no longer in the field's
        # accepted set (an enum field would otherwise refuse the very value this is about to
        # replace), and the replacement is then validated like any other value. `isinstance(str)`
        # before the lookup, not for tidiness: a settings value may be a list or a dict
        # (`timeline_stages`, vectors), and an unhashable value in a `dict` membership test
        # raises TypeError. Only an enum identifier can be retired anyway.
        retired = RETIRED_ENUM_VALUES.get(key, {})
        if isinstance(value, str) and value in retired:
            replacement = retired[value]
            normalisations.append(
                f"{name}: '{key}' was {value!r}, which this build no longer offers; "
                f"applied {replacement!r} instead"
            )
            value = replacement
        settings[key] = _validated_value(where, name, key, allowed[key], value)

    return AuthoringObject(
        name=name,
        persistent_id=persistent_id,
        role=role,
        system_persistent_id=system_persistent_id,
        display_group=display_group,
        settings=settings,
    )


def _validated_value(where: str, name: str, key: str, spec: Any, value: Any) -> Any:
    kind, extra = spec
    label = f"{where} ({name}): '{key}'"

    if kind == "bool":
        if not isinstance(value, bool):
            raise AuthoringDocumentError(f"{label} must be true or false")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise AuthoringDocumentError(f"{label} must be a string")
        return value
    if kind == "enum":
        if value not in extra:
            raise AuthoringDocumentError(
                f"{label} must be one of {', '.join(extra)}, got {value!r}"
            )
        return value
    if kind == "vector":
        # A bool is an int in Python, so the isinstance chain has to exclude it explicitly
        # or `[True, False, True]` would import as a velocity of (1, 0, 1).
        if (
            not isinstance(value, (list, tuple))
            or len(value) != extra
            or any(isinstance(component, bool) for component in value)
            or not all(isinstance(component, (int, float)) for component in value)
        ):
            raise AuthoringDocumentError(f"{label} must be {extra} numbers")
        numbers = tuple(float(component) for component in value)
        if not all(math.isfinite(component) for component in numbers):
            raise AuthoringDocumentError(f"{label} must be finite")
        return numbers
    if kind in ("positive_int", "nonnegative_int"):
        if not isinstance(value, int) or isinstance(value, bool):
            raise AuthoringDocumentError(f"{label} must be a whole number")
        if kind == "positive_int" and value < 1:
            raise AuthoringDocumentError(f"{label} must be at least 1, got {value}")
        if kind == "nonnegative_int" and value < 0:
            raise AuthoringDocumentError(f"{label} must not be negative, got {value}")
        return value
    if kind == "stages":
        return _validated_stages(label, value)

    # number / positive / nonnegative / unit_interval (number has no range clause)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthoringDocumentError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AuthoringDocumentError(f"{label} must be finite, got {value!r}")
    if kind == "positive" and number <= 0.0:
        raise AuthoringDocumentError(f"{label} must be positive, got {number}")
    if kind == "nonnegative" and number < 0.0:
        raise AuthoringDocumentError(f"{label} must not be negative, got {number}")
    if kind == "unit_interval" and not 0.0 <= number < 1.0:
        raise AuthoringDocumentError(
            f"{label} must satisfy 0 <= value < 1, got {number}"
        )
    return number


def _validated_stages(label: str, value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise AuthoringDocumentError(f"{label} must be a list of stages")
    stages: list[dict[str, Any]] = []
    for position, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise AuthoringDocumentError(f"{label}[{position}] must be an object")
        extra_keys = set(entry) - {"kind", "boundary_frame"}
        if extra_keys:
            raise AuthoringDocumentError(
                f"{label}[{position}] has unexpected keys: "
                f"{', '.join(sorted(extra_keys))}"
            )
        kind = entry.get("kind")
        if kind not in STAGE_KINDS:
            raise AuthoringDocumentError(
                f"{label}[{position}]: stage {kind!r} is not one of "
                f"{', '.join(STAGE_KINDS)}"
            )
        frame = entry.get("boundary_frame")
        if frame is not None and (not isinstance(frame, int) or isinstance(frame, bool)):
            raise AuthoringDocumentError(
                f"{label}[{position}]: 'boundary_frame' must be a whole number or null"
            )
        stages.append({"kind": kind, "boundary_frame": frame})
    return tuple(stages)


def _validate_membership(objects: tuple[AuthoringObject, ...]) -> None:
    """Membership must resolve inside the document, unambiguously.

    Checked here rather than at apply time because a document that cannot resolve its own
    membership is malformed regardless of what scene it lands in -- and because refusing
    before any object is touched is what makes "no partial mutation" true.
    """

    seen: dict[str, str] = {}
    for obj in objects:
        if obj.persistent_id in seen:
            raise AuthoringDocumentError(
                f"persistent id {obj.persistent_id!r} appears twice ({seen[obj.persistent_id]} "
                f"and {obj.name}); it identifies exactly one object"
            )
        seen[obj.persistent_id] = obj.name

    systems = {obj.persistent_id for obj in objects if obj.role == "SYSTEM"}
    for obj in objects:
        if obj.role == "SYSTEM":
            if obj.system_persistent_id:
                raise AuthoringDocumentError(
                    f"{obj.name} is a System, so it cannot itself belong to a System"
                )
            continue
        if not obj.system_persistent_id:
            continue
        if obj.system_persistent_id not in systems:
            raise AuthoringDocumentError(
                f"{obj.name} belongs to System {obj.system_persistent_id!r}, which is not "
                "in this document. Export the whole System, or clear the membership"
            )

    for obj in objects:
        if obj.role != "CLOUD":
            continue
        # Both halves of the centre (ADR 0085), by the same rule: a document that names a
        # body it does not contain cannot be reloaded into the scene it describes.
        for key, label in (
            ("keplerian_central_persistent_id", "Keplerian Central"),
            ("keplerian_companion_persistent_id", "Keplerian Companion"),
        ):
            referenced = obj.settings.get(key, "")
            if not referenced:
                continue
            if referenced not in seen:
                raise AuthoringDocumentError(
                    f"{obj.name} orbits central object {referenced!r}, which is not in "
                    f"this document. Export the whole System, or clear the {label}"
                )


# -- drift --------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftEntry:
    """One derived field whose recomputed value differs from a prior observation."""

    field: str
    prior: Any
    recomputed: Any


@dataclass(frozen=True)
class DriftReport:
    """What changed between a prior compiled observation and this compile.

    Purely informational. The recomputed value is always the one that is kept: this report
    describes a difference, it never resolves one. That is the asymmetry the module
    docstring describes, and the reason ``entries`` carries both values rather than a
    "correct" one.
    """

    baseline_available: bool
    entries: tuple[DriftEntry, ...] = ()

    @property
    def has_drift(self) -> bool:
        return bool(self.entries)

    def message(self) -> str:
        """One line for the HUD status area.

        The no-baseline case says so explicitly rather than reporting "no drift". They are
        different facts, and reporting the second when the first is true would tell a user
        that a value was confirmed unchanged when nothing was compared.
        """

        if not self.baseline_available:
            return (
                "No prior compiled observation was available, so no drift comparison was "
                "made. Derived values come from the current scene."
            )
        if not self.entries:
            return "Derived values match the prior compiled observation."
        parts = ", ".join(
            f"{entry.field}: {entry.prior!r} -> {entry.recomputed!r}"
            for entry in self.entries
        )
        return (
            f"Derived values changed since the prior compile ({parts}). The current "
            "scene's values were used."
        )


#: The derived fields worth reporting drift on. Exactly the two ADR 0040 introduces plus
#: the annulus geometry they depend on: these are the values a user can move without
#: touching any authored field, which is what makes a silent change possible at all.
DRIFT_FIELDS = (
    "sample_basis",
    "sample_extent",
    "sample_frame_provenance",
    "velocity",
)


def drift_report(prior: Any, recomputed: Any, *, fields: tuple[str, ...] = DRIFT_FIELDS):
    """Compare a prior compiled observation against this compile's derived values.

    ``prior`` is ``None`` when nothing has been compiled yet. That is reported as *no
    baseline*, never as *no drift* -- see :meth:`DriftReport.message`.
    """

    if prior is None:
        return DriftReport(baseline_available=False)

    entries = []
    for name in fields:
        if not hasattr(prior, name) or not hasattr(recomputed, name):
            continue
        before, after = getattr(prior, name), getattr(recomputed, name)
        if before != after:
            entries.append(DriftEntry(field=name, prior=before, recomputed=after))
    return DriftReport(baseline_available=True, entries=tuple(entries))
