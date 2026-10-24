"""Slice 5's D13 channels, carried through Slice 4's optional-tag seam.

D13 requires four things in the scrub cache that Slice 4 does not store: Solid damage,
persistent material ID, SPH/impact specific internal energy, and the provenance saying which
Color By modes were available and why the rest were not. It also requires that adding them
changes nothing about records written before they existed.

Those two requirements decide the whole design. This module adds **no required channel** and
does not touch the v1 codec's required path. Everything here rides Slice 4's shipped
optional-field mechanism -- ``encode_record``'s ``extra_tags`` with ``TAG_OPTIONAL_FLAG`` --
which a v1 reader already skips by construction. So "a pre-Slice-5 record decodes exactly as
it always did" is not a behaviour under test; it is a consequence of never having written to
that path.

**Absent is a first-class answer, and it carries a reason.** D13: "absent physical channels
remain absent with a reason." A record from a Direct-backend System has no damage because
damage does not exist there; a record from an impact System with damage switched off has no
damage for an entirely different reason, and the user can act on the second. Storing only the
arrays would collapse both into a missing key. So the provenance blob travels with them, and
:func:`extract_carriage` reports absence and its reason together.

**Nothing here stores a colour.** No RGBA, no mode-specific ``astro_color_value``, no
pre-resolved scalar. Recolouring an exact cached frame re-derives the current selection from
these immutable raw records, which is what makes switching Color By on a cached frame a
display operation with no cache write. A cached RGBA would make the cache mode-specific and
force a rewrite on every switch -- the exact thing D13 forbids.

Pure codec support: no ``bpy``, no Warp, no runtime. :mod:`astro_warp.product.color_by` owns
what the numbers *mean*; this owns how they travel.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .display import SCRUB_PRESENTATION_CAP, ScrubSampleSchemaV1
from .scrub_cache import SCRUB_CACHE_CAPACITY_BYTES, TAG_OPTIONAL_FLAG, SchemaError

#: Wire tag ids. 1-4 are the required v1 channels and 5 is the T5a boundary diagnostic, so
#: Slice 5 starts at 6. Each is fixed forever once a record carries it: a reassignment would
#: make an old cache decode as the wrong quantity, which is worse than failing to decode.
TAG_SOLID_DAMAGE = 6
TAG_MATERIAL_ID = 7
TAG_SPECIFIC_INTERNAL_ENERGY = 8
TAG_COLOR_PROVENANCE = 9
#: The per-particle mass density an impact engine publishes (not a Color By mode: it feeds the
#: viewport's Density look, so a rewound frame keeps that colour).
TAG_DENSITY = 10

#: Wire dtypes, and they are not interchangeable. Material ID is ``<i8`` because it is the
#: exact persistent identity Step 3 owns. Narrowing it to int32 would refuse or wrap valid
#: registry IDs above 2^31 even though the authoritative device state preserved them.
_DAMAGE_DTYPE = "<f4"
_MATERIAL_DTYPE = "<i8"
_INTERNAL_ENERGY_DTYPE = "<f4"
_DENSITY_DTYPE = "<f4"

_ITEM_BYTES = {
    TAG_SOLID_DAMAGE: 4,
    TAG_MATERIAL_ID: 8,
    TAG_SPECIFIC_INTERNAL_ENERGY: 4,
    TAG_DENSITY: 4,
}

#: Provenance framing: a ``<Q`` length in bytes, then that many UTF-8 bytes, then zero
#: padding up to an 8-byte boundary. The length prefix is what makes the padding removable;
#: without it a decoder could not tell a trailing zero byte from padding, and JSON would
#: silently fail to parse for reasons invisible in the bytes.
_LENGTH = struct.Struct("<Q")

#: Inclusive ceiling on the encoded provenance blob. A blob is a bounded description of one
#: System's roles, backends and refusals, never per-particle data; capping it keeps the
#: worst-case record arithmetic below exact rather than aspirational.
PROVENANCE_CEILING_BYTES = 8_192
#: Maximum bytes placed on the optional-tag wire: bounded JSON plus its ``<Q`` framing.
PROVENANCE_WIRE_CEILING_BYTES = PROVENANCE_CEILING_BYTES + _LENGTH.size


@dataclass(frozen=True)
class ColorPartitionProvenance:
    """Recorded D13 availability for one geometry-bearing output partition."""

    name: str
    role: str
    backend: str
    available: tuple[str, ...] = ()
    refusals: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "role", "backend"):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"provenance partition {field_name} must be a string")
        if not self.name:
            raise ValueError("a Color By provenance partition needs a name")
        if not self.role or not self.backend:
            raise ValueError("a Color By provenance partition needs a role and backend")
        if isinstance(self.available, str) or not all(
            isinstance(value, str) for value in self.available
        ):
            raise ValueError("provenance partition available modes must be strings")
        if len(self.available) != len(set(self.available)):
            raise ValueError("provenance partition available modes contain a duplicate")
        if not isinstance(self.refusals, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.refusals.items()
        ):
            raise ValueError("provenance partition refusal modes and reasons must be strings")
        overlap = set(self.available).intersection(self.refusals)
        if overlap:
            raise ValueError(
                "provenance mode is both available and refused: " + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "available", tuple(sorted(self.available)))
        object.__setattr__(
            self,
            "refusals",
            dict(sorted(self.refusals.items())),
        )

    def as_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "backend": self.backend,
            "available": list(self.available),
            "refusals": dict(self.refusals),
        }

    def reason_for(self, mode: str) -> str | None:
        if mode in self.available:
            return None
        return self.refusals.get(
            mode, f"{mode} was neither available nor refused when this frame was cached."
        )


@dataclass(frozen=True)
class ColorProvenance:
    """Why each Color By mode was or was not derivable when this frame was cached.

    Recorded at cache time rather than recomputed at read time, and that is the point: the
    System's role, backend and damage settings can all change after a frame is cached, so
    recomputing would answer about the *present* while showing a frame from the past. A user
    scrubbing back to a frame where damage was off should be told damage was off then.
    """

    # The singular fields remain a decode/construction compatibility surface for pre-D13
    # records. New records use ``partitions`` and never collapse several outputs into one.
    role: str = ""
    backend: str = ""
    #: Mode names derivable for this frame. Canonicalised to sorted order on construction.
    available: tuple[str, ...] = ()
    #: Mode name -> the refusal that applied when this frame was cached.
    refusals: dict[str, str] = field(default_factory=dict)
    partitions: tuple[ColorPartitionProvenance, ...] = ()

    def __post_init__(self) -> None:
        """Canonicalise member order here rather than only in ``to_bytes``.

        Sorting only at encode time would make a decoded provenance unequal to the one that
        produced it whenever the caller passed an unsorted tuple -- so a round-trip test
        would pass or fail depending on the fixture's spelling, and any caching keyed on
        equality would see spurious changes. Normalising in the type makes round-trip
        identity a property of the type instead of a property of the caller.
        """

        if not isinstance(self.role, str) or not isinstance(self.backend, str):
            raise ValueError("singular provenance role and backend must be strings")
        if isinstance(self.available, str) or not all(
            isinstance(value, str) for value in self.available
        ):
            raise ValueError("singular provenance available modes must be strings")
        if not isinstance(self.refusals, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.refusals.items()
        ):
            raise ValueError("singular provenance refusal modes and reasons must be strings")
        object.__setattr__(self, "available", tuple(sorted(self.available)))
        object.__setattr__(
            self,
            "refusals",
            dict(sorted(self.refusals.items())),
        )
        if self.partitions:
            if self.role or self.backend or self.available or self.refusals:
                raise ValueError("partition-aware provenance cannot also carry singular fields")
            partitions = tuple(
                sorted(self.partitions, key=lambda item: (item.name, item.role, item.backend))
            )
            if len(partitions) == 1:
                only = partitions[0]
                object.__setattr__(self, "role", only.role)
                object.__setattr__(self, "backend", only.backend)
                object.__setattr__(self, "available", only.available)
                object.__setattr__(self, "refusals", only.refusals)
        else:
            if not self.role or not self.backend:
                raise ValueError("singular Color By provenance needs a role and backend")
            partitions = (
                ColorPartitionProvenance(
                    name="output",
                    role=self.role,
                    backend=self.backend,
                    available=self.available,
                    refusals=self.refusals,
                ),
            )
        names = [item.name for item in partitions]
        if len(names) != len(set(names)):
            raise ValueError("Color By provenance partition names must be unique")
        object.__setattr__(self, "partitions", partitions)

    def to_bytes(self) -> bytes:
        """Length-prefixed, zero-padded UTF-8 JSON."""
        document = self.as_document()
        # `sort_keys` plus the canonical member order from `__post_init__` make the encoding
        # a function of the content alone -- two equal provenances produce equal bytes, so a
        # record's checksum does not change because a dict was built in a different order.
        raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(raw) > PROVENANCE_CEILING_BYTES and len(self.partitions) > 1:
            # A System may legitimately own many authored BODY objects whose Color By
            # policy is identical. Repeating the same refusal sentences once per object
            # made that System look particle-sized and could exhaust the fixed wire budget
            # before playback began. Keep the public/bake document expanded, but deduplicate
            # equal policies on this bounded scrub wire and reconstruct every exact name on
            # decode. Existing records that already fit retain byte-identical encoding.
            compact = self._compact_wire_document()
            raw = json.dumps(compact, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(raw) > PROVENANCE_CEILING_BYTES:
            raise SchemaError(
                f"provenance is {len(raw)} bytes, above the {PROVENANCE_CEILING_BYTES}-byte "
                "ceiling; provenance describes a System, never its particles"
            )
        padded = raw + b"\x00" * (-len(raw) % 8)
        return _LENGTH.pack(len(raw)) + padded

    def as_document(self) -> dict[str, Any]:
        """Canonical partition-aware JSON value, without wire framing."""
        return {"partitions": [partition.as_document() for partition in self.partitions]}

    def _compact_wire_document(self) -> dict[str, Any]:
        """Deduplicate equal partition policies without collapsing partition identity."""

        grouped: dict[tuple[Any, ...], list[str]] = {}
        for partition in self.partitions:
            refusal_patterns = tuple(
                (mode, tuple(reason.split(partition.name)))
                for mode, reason in partition.refusals.items()
            )
            policy = (
                partition.role,
                partition.backend,
                partition.available,
                refusal_patterns,
            )
            grouped.setdefault(policy, []).append(partition.name)
        groups = []
        for (role, backend, available, refusal_patterns), names in sorted(grouped.items()):
            groups.append(
                {
                    "names": sorted(names),
                    "role": role,
                    "backend": backend,
                    "available": list(available),
                    "refusal_patterns": {
                        mode: list(segments) for mode, segments in refusal_patterns
                    },
                }
            )
        return {"partition_groups": groups}

    @classmethod
    def from_document(cls, document: Any) -> ColorProvenance:
        """Validate either canonical partition JSON or the existing singular form."""
        if not isinstance(document, dict):
            raise ValueError("provenance is not a JSON object")
        if not all(isinstance(key, str) for key in document):
            raise ValueError("provenance field names must be strings")
        if "partition_groups" in document:
            if set(document) != {"partition_groups"}:
                raise ValueError(
                    "compact partition provenance contains non-canonical top-level fields"
                )
            raw_groups = document["partition_groups"]
            if not isinstance(raw_groups, list) or not raw_groups:
                raise ValueError("provenance partition_groups is not a non-empty JSON array")
            expanded = []
            for index, item in enumerate(raw_groups):
                if not isinstance(item, dict):
                    raise ValueError(f"provenance partition group {index} is not a JSON object")
                if not all(isinstance(key, str) for key in item):
                    raise ValueError(
                        f"provenance partition group {index} field names must be strings"
                    )
                missing = {"names", "role", "backend"} - set(item)
                if missing:
                    raise ValueError(
                        f"provenance partition group {index} is missing "
                        + ", ".join(sorted(missing))
                    )
                unknown = set(item) - {
                    "names",
                    "role",
                    "backend",
                    "available",
                    "refusal_patterns",
                }
                if unknown:
                    raise ValueError(
                        f"provenance partition group {index} contains unknown fields: "
                        + ", ".join(sorted(unknown))
                    )
                names = item["names"]
                if (
                    not isinstance(names, list)
                    or not names
                    or not all(isinstance(name, str) for name in names)
                ):
                    raise ValueError(
                        f"provenance partition group {index} names is not a non-empty string array"
                    )
                patterns = item.get("refusal_patterns", {})
                if not isinstance(patterns, dict) or not all(
                    isinstance(mode, str)
                    and isinstance(segments, list)
                    and bool(segments)
                    and all(isinstance(segment, str) for segment in segments)
                    for mode, segments in patterns.items()
                ):
                    raise ValueError(
                        f"provenance partition group {index} refusal_patterns is not a "
                        "JSON object of non-empty string arrays"
                    )
                for name in names:
                    expanded.append(
                        {
                            "name": name,
                            "role": item["role"],
                            "backend": item["backend"],
                            "available": item.get("available", []),
                            "refusals": {
                                mode: name.join(segments) for mode, segments in patterns.items()
                            },
                        }
                    )
            return cls.from_document({"partitions": expanded})

        if "partitions" in document:
            if set(document) != {"partitions"}:
                raise ValueError("partition provenance contains non-canonical top-level fields")
            raw_partitions = document["partitions"]
            if not isinstance(raw_partitions, list) or not raw_partitions:
                raise ValueError("provenance partitions is not a non-empty JSON array")
            partitions = []
            for index, item in enumerate(raw_partitions):
                if not isinstance(item, dict):
                    raise ValueError(f"provenance partition {index} is not a JSON object")
                if not all(isinstance(key, str) for key in item):
                    raise ValueError(
                        f"provenance partition {index} field names must be strings"
                    )
                missing = {"name", "role", "backend"} - set(item)
                if missing:
                    raise ValueError(
                        f"provenance partition {index} is missing "
                        + ", ".join(sorted(missing))
                    )
                unknown = set(item) - {"name", "role", "backend", "available", "refusals"}
                if unknown:
                    raise ValueError(
                        f"provenance partition {index} contains unknown fields: "
                        + ", ".join(sorted(unknown))
                    )
                available = item.get("available", [])
                refusals = item.get("refusals", {})
                if not isinstance(available, list):
                    raise ValueError(
                        f"provenance partition {index} available is not a JSON array"
                    )
                if not isinstance(refusals, dict):
                    raise ValueError(
                        f"provenance partition {index} refusals is not a JSON object"
                    )
                partitions.append(
                    ColorPartitionProvenance(
                        name=item["name"],
                        role=item["role"],
                        backend=item["backend"],
                        available=tuple(available),
                        refusals=refusals,
                    )
                )
            return cls(partitions=tuple(partitions))

        missing = {"role", "backend"} - set(document)
        if missing:
            raise ValueError(f"provenance is missing {', '.join(sorted(missing))}")
        unknown = set(document) - {"role", "backend", "available", "refusals"}
        if unknown:
            raise ValueError(
                "singular provenance contains unknown fields: " + ", ".join(sorted(unknown))
            )
        available = document.get("available", [])
        refusals = document.get("refusals", {})
        if not isinstance(available, list):
            raise ValueError("provenance available is not a JSON array")
        if not isinstance(refusals, dict):
            raise ValueError("provenance refusals is not a JSON object")
        return cls(
            role=document["role"],
            backend=document["backend"],
            available=tuple(available),
            refusals=refusals,
        )

    @classmethod
    def from_bytes(cls, blob: bytes) -> ColorProvenance:
        """Decode, or refuse. Never guess at a partially readable blob."""
        raw = bytes(blob)
        if len(raw) > PROVENANCE_WIRE_CEILING_BYTES:
            raise SchemaError(
                f"provenance blob is {len(raw)} bytes, above the "
                f"{PROVENANCE_WIRE_CEILING_BYTES}-byte wire ceiling"
            )
        if len(raw) < _LENGTH.size:
            raise SchemaError("provenance blob is shorter than its length prefix")
        (length,) = _LENGTH.unpack(raw[: _LENGTH.size])
        body = raw[_LENGTH.size :]
        if length > len(body):
            raise SchemaError(
                f"provenance declares {length} bytes but only {len(body)} follow"
            )
        expected_length = _LENGTH.size + length + (-length % 8)
        if len(raw) != expected_length:
            raise SchemaError(
                f"provenance blob is {len(raw)} bytes, not its canonical padded length "
                f"{expected_length}"
            )
        if any(body[length:]):
            # Padding must be zero. A non-zero tail means the blob is not what it claims,
            # and accepting it would let a truncated or overlapping payload decode as valid.
            raise SchemaError("provenance padding is not zero-filled")
        try:
            document = json.loads(body[:length].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SchemaError(f"provenance is not valid UTF-8 JSON: {error}") from error
        try:
            return cls.from_document(document)
        except ValueError as error:
            raise SchemaError(f"invalid provenance: {error}") from error

    def resolve_partition(self, name: str | None) -> ColorPartitionProvenance:
        """Resolve an exact name, or the sole legacy partition when no name was supplied."""
        if name is None:
            if len(self.partitions) == 1:
                return self.partitions[0]
            raise KeyError("Color By provenance needs an explicit partition name")
        if not isinstance(name, str):
            raise TypeError("Color By provenance partition name must be a string or None")
        matches = tuple(item for item in self.partitions if item.name == name)
        if len(matches) != 1:
            raise KeyError(f"Color By provenance has no unique partition named {name!r}")
        return matches[0]

    def partition(self, name: str) -> ColorPartitionProvenance:
        """Compatibility spelling for exact named partition lookup."""
        return self.resolve_partition(name)

    def reason_for(self, mode: str, *, partition: str | None = None) -> str | None:
        """Why ``mode`` is absent across the frame, or on one named partition."""
        if partition is not None:
            return self.resolve_partition(partition).reason_for(mode)
        reasons = [
            (item.name, item.reason_for(mode))
            for item in self.partitions
            if item.reason_for(mode) is not None
        ]
        if not reasons:
            return None
        if len(self.partitions) == 1:
            return reasons[0][1]
        return " ".join(f"{name}: {reason}" for name, reason in reasons)


def _as_wire(values: Any, dtype: str, tag: int, display_count: int | None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 1:
        raise SchemaError(f"tag {tag} must be a 1-D per-sample array, got shape {array.shape}")
    if display_count is not None and array.shape[0] != display_count:
        raise SchemaError(
            f"tag {tag} has {array.shape[0]} rows but the record displays {display_count}"
        )
    return array


def _material_ids_wire(values: Any, display_count: int | None) -> np.ndarray:
    """Validate persistent identity before conversion so no coercion can change it."""

    if isinstance(values, np.ndarray):
        source = values
    else:
        if isinstance(values, (str, bytes, bool, np.bool_)):
            raise SchemaError("material_ids accepts a one-dimensional sequence of integers")
        try:
            sequence = tuple(values)
        except TypeError as error:
            raise SchemaError(
                "material_ids accepts a one-dimensional sequence of integers"
            ) from error
        for value in sequence:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise SchemaError("material_ids accepts exact integers only")
            integer = int(value)
            if integer < 0:
                raise SchemaError(
                    "material_ids contains a negative persistent ID; a particle with no "
                    "material is carried by omitting it from the array's System, never as -1"
                )
            if integer > np.iinfo(np.int64).max:
                raise SchemaError("material_ids contains a value above INT64_MAX")
        source = np.asarray(sequence, dtype=_MATERIAL_DTYPE)
    if source.ndim != 1:
        raise SchemaError(
            f"tag {TAG_MATERIAL_ID} must be a 1-D per-sample array, got shape {source.shape}"
        )
    if display_count is not None and source.shape[0] != display_count:
        raise SchemaError(
            f"tag {TAG_MATERIAL_ID} has {source.shape[0]} rows but the record displays "
            f"{display_count}"
        )
    if source.size == 0:
        return np.ascontiguousarray(source, dtype=_MATERIAL_DTYPE)

    maximum = np.iinfo(np.int64).max
    if source.dtype.kind in {"i", "u"}:
        if source.dtype.kind == "i" and int(source.min()) < 0:
            raise SchemaError(
                "material_ids contains a negative persistent ID; a particle with no material "
                "is carried by omitting it from the array's System, never as -1"
            )
        if int(source.max()) > maximum:
            raise SchemaError("material_ids contains a value above INT64_MAX")
    elif source.dtype.kind == "O":
        for value in source:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise SchemaError("material_ids accepts exact integers only")
            integer = int(value)
            if integer < 0:
                raise SchemaError(
                    "material_ids contains a negative persistent ID; a particle with no "
                    "material is carried by omitting it from the array's System, never as -1"
                )
            if integer > maximum:
                raise SchemaError("material_ids contains a value above INT64_MAX")
    else:
        raise SchemaError("material_ids accepts exact integers only")
    return np.ascontiguousarray(source, dtype=_MATERIAL_DTYPE)


def carriage_extra_tags(
    *,
    display_count: int | None = None,
    solid_damage: Any = None,
    material_ids: Any = None,
    specific_internal_energy: Any = None,
    provenance: ColorProvenance | None = None,
    densities: Any = None,
) -> list[tuple[int, int, np.ndarray]]:
    """The ``extra_tags`` list for one record's D13 carriage.

    Every argument is optional and omitting one means the channel is genuinely absent, not
    zero. A caller with no damage passes nothing; it does not pass an array of zeros, which
    would be indistinguishable from a fully intact body and is the fabricated-physics case
    D13 rules out.

    ``display_count`` is checked when given, because a per-sample array of the wrong length
    is the one error that would otherwise survive encoding and surface as a misaligned colour
    much later, on a different machine.
    """

    tags: list[tuple[int, int, np.ndarray]] = []
    if solid_damage is not None:
        tags.append(
            (
                TAG_SOLID_DAMAGE,
                TAG_OPTIONAL_FLAG,
                _as_wire(solid_damage, _DAMAGE_DTYPE, TAG_SOLID_DAMAGE, display_count),
            )
        )
    if material_ids is not None:
        array = _material_ids_wire(material_ids, display_count)
        tags.append((TAG_MATERIAL_ID, TAG_OPTIONAL_FLAG, array))
    if specific_internal_energy is not None:
        tags.append(
            (
                TAG_SPECIFIC_INTERNAL_ENERGY,
                TAG_OPTIONAL_FLAG,
                _as_wire(
                    specific_internal_energy,
                    _INTERNAL_ENERGY_DTYPE,
                    TAG_SPECIFIC_INTERNAL_ENERGY,
                    display_count,
                ),
            )
        )
    if densities is not None:
        tags.append(
            (
                TAG_DENSITY,
                TAG_OPTIONAL_FLAG,
                _as_wire(densities, _DENSITY_DTYPE, TAG_DENSITY, display_count),
            )
        )
    if provenance is not None:
        blob = provenance.to_bytes()
        tags.append(
            (
                TAG_COLOR_PROVENANCE,
                TAG_OPTIONAL_FLAG,
                np.frombuffer(blob, dtype="<u8"),
            )
        )
    return tags


@dataclass(frozen=True)
class Carriage:
    """What one decoded record actually carries, with absence stated rather than implied."""

    solid_damage: np.ndarray | None = None
    material_ids: np.ndarray | None = None
    specific_internal_energy: np.ndarray | None = None
    provenance: ColorProvenance | None = None
    #: Per-particle mass density, when the producing engine published one. Not a Color By
    #: source, so it is not listed by bsent.
    densities: np.ndarray | None = None

    def absent(self) -> tuple[str, ...]:
        names = []
        if self.solid_damage is None:
            names.append("solid_damage")
        if self.material_ids is None:
            names.append("material_ids")
        if self.specific_internal_energy is None:
            names.append("specific_internal_energy")
        return tuple(names)

    def reason_for(self, mode: str) -> str:
        """Why a mode is not derivable from this record.

        Falls back to a statement about the record's own age rather than inventing one: a
        pre-Slice-5 record carries no provenance at all, and saying so is more useful than
        implying a refusal nobody recorded.
        """

        if self.provenance is None:
            return (
                f"This frame was cached before Color By provenance existed, so no reason for "
                f"{mode} was recorded. Re-cache the frame to record one."
            )
        return self.provenance.reason_for(mode) or ""


def extract_carriage(
    payloads: Any, *, display_count: int | None = None, strict: bool = True
) -> Carriage:
    """Rebuild the D13 carriage from a decoded record's optional payloads.

    ``payloads`` is ``DecodedRecord.optional_payloads`` -- raw bytes by tag id, which is what
    the v1 decoder hands back for optional extension tags. Reinterpreting rather than adding
    required channels keeps this layer additive: the v1 schema and legacy record framing do
    not change, while the current reader can strictly validate the reserved D13 forms.

    ``strict`` refuses a payload whose byte length is not a whole number of items. That is
    the fail-closed case D13 asks for, and it must be an error rather than a truncation --
    a short damage array silently paired with a full position array recolours the wrong
    particles, and every frame still renders.
    """

    if not payloads:
        return Carriage()

    def array_for(tag: int, dtype: str) -> np.ndarray | None:
        raw = payloads.get(tag)
        if raw is None:
            return None
        item = _ITEM_BYTES[tag]
        if len(raw) % item:
            if strict:
                raise SchemaError(
                    f"tag {tag} payload is {len(raw)} bytes, not a whole number of "
                    f"{item}-byte items"
                )
            return None
        array = np.frombuffer(raw, dtype=dtype)
        if display_count is not None and array.shape[0] != display_count:
            if strict:
                raise SchemaError(
                    f"tag {tag} carries {array.shape[0]} items but the record displays "
                    f"{display_count}"
                )
            return None
        return array

    provenance = None
    raw_provenance = payloads.get(TAG_COLOR_PROVENANCE)
    if raw_provenance is not None:
        try:
            provenance = ColorProvenance.from_bytes(raw_provenance)
        except SchemaError:
            if strict:
                raise
            provenance = None

    return Carriage(
        solid_damage=array_for(TAG_SOLID_DAMAGE, _DAMAGE_DTYPE),
        material_ids=array_for(TAG_MATERIAL_ID, _MATERIAL_DTYPE),
        specific_internal_energy=array_for(
            TAG_SPECIFIC_INTERNAL_ENERGY, _INTERNAL_ENERGY_DTYPE
        ),
        provenance=provenance,
        densities=array_for(TAG_DENSITY, _DENSITY_DTYPE),
    )


class ScrubSampleSchemaV1WithCarriage:
    """The worst-case arithmetic after the widening, recomputed rather than assumed.

    D13 requires Slice 4's capacity forecast to be recalculated once the new channels exist.
    Deliberately *not* a subclass of ``ScrubSampleSchemaV1`` and not passed to the codec: the
    record format is unchanged and v1 remains the schema every record is written and read
    against. This is arithmetic about the widest record Slice 5 can now produce, which is a
    different question from what the codec validates -- and conflating them would turn a
    forecast into a format change.
    """

    #: v1's four required channels, unchanged.
    base_per_sample_bytes = ScrubSampleSchemaV1.per_sample_bytes
    #: Damage + material ID + specific internal energy, all per-sample.
    carriage_per_sample_bytes = sum(_ITEM_BYTES.values())
    per_sample_bytes = base_per_sample_bytes + carriage_per_sample_bytes
    presentation_cap = SCRUB_PRESENTATION_CAP
    #: v1's header ceiling plus the complete provenance wire value: the bounded JSON is
    #: preceded by an eight-byte length. Provenance is per-record, not per-sample.
    record_header_ceiling = (
        ScrubSampleSchemaV1.record_header_ceiling + PROVENANCE_WIRE_CEILING_BYTES
    )
    max_record_bytes = presentation_cap * per_sample_bytes + record_header_ceiling


def widened_capacity_forecast(
    capacity_bytes: int = SCRUB_CACHE_CAPACITY_BYTES,
) -> Any:
    """Slice 4's forecast recomputed against the widest Slice 5 record.

    Fewer frames fit than before, and that is the honest answer rather than a regression:
    the same 1 GiB budget now has to hold records that can carry three more per-sample
    channels. A forecast still quoting the narrower record would over-promise the scrub
    range by roughly a third, and the first user to find out would find out by hitting the
    capacity refusal mid-scrub.
    """

    from .display import scrub_capacity_forecast

    return scrub_capacity_forecast(
        capacity_bytes, ScrubSampleSchemaV1WithCarriage.max_record_bytes
    )
