"""Custom presets: one Body or Cloud saved as a file and added again later (AW-007, ADR 0203).

Pure Python, no Blender. The Blender half (``custom_presets`` in the add-on) reads the active
object and writes the new one; this module decides what a saved file holds, reads it back, and
keeps every file operation inside the presets folder.

**There is no second settings format.** A preset's role settings are an ordinary
``AstroAuthoringDocumentV1`` holding exactly one object, validated by the same
``authoring_document`` rules Export/Import Authoring uses. What a preset adds around it is only
what that document deliberately does not carry: the *scale* of the System the object was made in
(so it is added to, or makes, a System of the same scale), the *pace* a new System is made with,
and the stand-in mesh the object is drawn on (a sphere, a disk, or, for a Cloud that fills its own
mesh, that mesh). Pointers to other objects never travel (``POINTER_KEYS``): they mean nothing in
another scene, and a Keplerian cloud takes the active Body, or the heaviest, as its centre again
when it is added.

Nothing here is tied to a scene or to the catalogue: any Body or Cloud, of any shape, velocity
model and scale the add-on offers, can be saved.

Every message an artist can read is a full plain sentence. A damaged or older file is never
fatal: :func:`scan` skips it and says why, and :func:`from_json` raises :class:`PresetError`.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import struct
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .authoring_document import (
    AuthoringDocumentError,
    AuthoringObject,
    build_authoring_document,
    load_authoring_document,
)

PRESET_FORMAT = "AstroWarpPresetV1"
"""The ``format`` key of a preset file. A file with any other value is not one of ours."""

PRESET_VERSION = 1
"""The ``version`` key. Only this one is read; any other number is skipped with its number."""

PRESET_SUFFIX = ".json"
SUPPORTED_ROLES: tuple[str, ...] = ("BODY", "CLOUD")
SUPPORTED_UNIT_PRESETS: tuple[str, ...] = ("PLANETARY", "STELLAR", "GALACTIC", "CUSTOM")
"""Every scale a System can have. A Custom one carries its three numbers in the preset."""

MESH_SHAPES: tuple[str, ...] = ("OBJECT_VOLUME", "OBJECT_SURFACE")
"""The Cloud Shapes that fill the object's own mesh rather than an analytic ball or disk."""

SPHERE_SHAPES: tuple[str, ...] = ("BALL", "SHELL")
"""Analytic shapes sized by the object's half-size. A Shell's hole is its Inner Radius setting,
which the authoring document already carries."""

DISK_SHAPES: tuple[str, ...] = ("DISK_ANNULUS",)
"""Analytic shapes sized by the object's outer radius and thickness. Every Cloud Shape the
add-on accepts must be in one of these three tuples: ``test_custom_preset_operators`` compares
them with ``role_capabilities.ACCEPTED_CLOUD_SHAPES``, so a shape added later cannot silently
become one a preset refuses."""

POINTER_KEYS = frozenset({"keplerian_central_persistent_id", "keplerian_companion_persistent_id"})

MAX_NAME_LENGTH = 64
MAX_FILE_BYTES = 4 << 20
"""A preset is a few kilobytes, or under a megabyte with a mesh. Past 4 MiB it is not one."""

MAX_MESH_TRIANGLES = 20_000
MAX_MESH_VERTICES = 60_000
"""The most a mesh Cloud's preset keeps; the vertex bound is three per triangle, so it is
never the first one reached by a real mesh."""

OVERWRITE_LABEL = "Replace Existing"
"""The label of the checkbox the save dialog shows; the refusal text points at it by name."""

FALLBACK_BODY_PROXY_RADIUS = 1e-3
"""The stand-in sphere for a Body whose object has no measurable size and no Radius."""

_PLACEHOLDER_ID = "preset"
_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)

_PROXY_KINDS = ("SPHERE", "ANNULUS", "MESH")
_BAD_PROXY = "The preset's proxy size is missing or not valid."
_NO_MESH_REFUSAL = (
    "This Cloud fills the shape of its own mesh, but there is no mesh to copy. Give the "
    "object geometry, or set its Shape to Ball or Disk Annulus, then save it again."
)


class PresetError(ValueError):
    """A preset cannot be saved or read. The message is a full sentence for the artist."""


class PresetExists(PresetError):
    """Saving would replace a preset that is already there, and replacing was not asked for."""


@dataclass(frozen=True)
class SystemFacts:
    """The System a preset was made in: its scale, and the pace a new one is made with."""

    unit_preset: str
    seconds_per_frame: float
    softening: float
    substeps: int
    length_scale_m: float = 0.0
    mass_scale_kg: float = 0.0
    time_scale_s: float = 0.0
    """The three scales, kept only for ``CUSTOM`` (zero otherwise): the named presets define
    their own."""

    def pacing(self) -> dict[str, float]:
        """The shape ``add_objects._make_system`` takes for a System it has to make."""

        return {
            "seconds_per_frame": self.seconds_per_frame,
            "softening": self.softening,
            "substeps": self.substeps,
        }

    def custom_scales(self) -> tuple[float, float, float]:
        return (self.length_scale_m, self.mass_scale_kg, self.time_scale_s)


@dataclass(frozen=True)
class ProxyShape:
    """The stand-in mesh the object is drawn on, in Blender units (the System's length unit)."""

    kind: str  # "SPHERE" | "ANNULUS" | "MESH"
    radius: float
    inner_radius: float = 0.0
    thickness: float = 0.0
    vertices: tuple[tuple[float, float, float], ...] = ()
    triangles: tuple[tuple[int, int, int], ...] = ()
    """``MESH`` only: the object's own triangulated mesh, scale applied, origin at its origin.
    ``radius`` is then half its largest extent."""


@dataclass(frozen=True)
class Preset:
    name: str
    role: str
    system: SystemFacts
    proxy: ProxyShape
    settings: dict[str, Any] = field(default_factory=dict)
    display_group: str = "AUTO"

    @property
    def needs_central(self) -> bool:
        """A Keplerian cloud orbits another object, which a preset cannot carry."""

        return self.role == "CLOUD" and self.settings.get("velocity_model") == "KEPLERIAN"


@dataclass(frozen=True)
class ScanResult:
    presets: tuple[tuple[str, Preset], ...] = ()
    """``(file name, preset)`` in display-name order."""
    skipped: tuple[tuple[str, str], ...] = ()
    """``(file name, reason)`` for every file that looks like a preset and could not be used."""


# -- names ----------------------------------------------------------------------------------


def clean_name(raw: Any) -> str:
    """The name an artist sees: trimmed, one space between words, printable, capped."""

    if not isinstance(raw, str):
        raise PresetError("Give the preset a name.")
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(ch if ch.isprintable() or ch.isspace() else " " for ch in text)
    text = " ".join(text.split())
    text = text[:MAX_NAME_LENGTH].rstrip()
    if not any(ch.isalnum() for ch in text):
        raise PresetError("Give the preset a name with at least one letter or number.")
    return text


def file_stem(name: Any) -> str:
    """A file name stem that cannot leave the folder, hide, or be a Windows device name."""

    text = clean_name(name)
    mapped = "".join(ch if (ch.isalnum() or ch in " -_") else "_" for ch in text)
    stem = re.sub(r"_+", "_", mapped).strip(" _-")
    stem = stem[:MAX_NAME_LENGTH].strip(" _-")
    if not stem:
        raise PresetError("Give the preset a name with at least one letter or number.")
    if stem.upper() in _RESERVED_STEMS:
        stem += "_preset"
    return stem


def file_name_for(name: Any) -> str:
    return file_stem(name) + PRESET_SUFFIX


# -- what may be saved ----------------------------------------------------------------------


def refusal_for_role(role: Any, object_name: str) -> str | None:
    """``None`` for a Body or a Cloud, else why this object cannot be saved, in plain words."""

    if role in SUPPORTED_ROLES:
        return None
    if role == "SYSTEM":
        return (
            f"'{object_name}' is a System, which holds a whole simulation. "
            "Only a single Body or a Cloud can be saved as a preset."
        )
    if role == "SOLID":
        return (
            f"'{object_name}' is a Solid. Solids cannot be saved as presets yet; "
            "only a Body or a Cloud can."
        )
    if role in ("NONE", "", None):
        return f"'{object_name}' has no Role yet. Give it a Body or Cloud Role first, then save it."
    label = str(role).replace("_", " ").title()
    return f"'{object_name}' is a {label}. Only a Body or a Cloud can be saved as a preset."


def needs_mesh(role: Any, shape: Any) -> bool:
    """Whether saving this object has to read its mesh (a Cloud that fills its own shape)."""

    return role == "CLOUD" and shape in MESH_SHAPES


def proxy_for(
    role: str,
    shape: str,
    dimensions: Any,
    *,
    body_radius: float = 0.0,
    inner_radius: float = 0.0,
    mesh: Any = None,
) -> ProxyShape:
    """The stand-in mesh that gives the added object the size this one has.

    Sized the way the compile sizes it: a Ball cloud by ``max(dimensions) / 2``, a Disk
    Annulus by its outer radius and thickness (ADR 0040 section 5), and a cloud that fills its
    own mesh by that mesh. A Body is drawn on a sphere of its own half-size, else its Radius.
    The object's position and rotation are not kept: an added object arrives level at the 3D
    cursor.

    ``mesh`` is ``(vertices, triangles)`` for a mesh-filling Cloud: the evaluated, triangulated
    mesh with the object's scale applied. Only that shape reads it.
    """

    extents = [0.0, 0.0, 0.0]
    for index, value in enumerate(list(dimensions)[:3]):
        try:
            number = abs(float(value))
        except (TypeError, ValueError):
            continue
        extents[index] = number if math.isfinite(number) else 0.0

    if role == "BODY":
        half = max(extents) / 2.0
        if half > 0.0:
            return ProxyShape("SPHERE", half)
        if body_radius > 0.0 and math.isfinite(body_radius):
            return ProxyShape("SPHERE", float(body_radius))
        return ProxyShape("SPHERE", FALLBACK_BODY_PROXY_RADIUS)

    if role != "CLOUD":
        raise PresetError(refusal_for_role(role, "This object") or "This object cannot be saved.")
    if shape in SPHERE_SHAPES:
        half = max(extents) / 2.0
        if half <= 0.0:
            raise PresetError(
                "This Cloud has no size to copy. Scale the object up, then save it again."
            )
        return ProxyShape("SPHERE", half)
    if shape in DISK_SHAPES:
        outer = max(extents[0], extents[1]) / 2.0
        if outer <= 0.0:
            raise PresetError(
                "This Cloud has no size to copy. Scale the object up, then save it again."
            )
        thickness = extents[2] if extents[2] > 0.0 else 0.02 * outer
        inner = float(inner_radius) if 0.0 <= float(inner_radius) < outer else 0.0
        return ProxyShape("ANNULUS", outer, inner, thickness)
    if shape in MESH_SHAPES:
        if mesh is None:
            raise PresetError(_NO_MESH_REFUSAL)
        vertices, triangles = mesh
        return _validated_mesh_proxy(
            ProxyShape("MESH", 1.0, vertices=vertices, triangles=triangles)
        )
    raise PresetError(f"This Cloud's Shape '{shape}' cannot be saved in a preset.")


# -- validation helpers ---------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validated_system(facts: Any) -> SystemFacts:
    unit = getattr(facts, "unit_preset", None)
    if unit not in SUPPORTED_UNIT_PRESETS:
        raise PresetError(
            f"The System scale {unit!r} is not one a preset can hold. The System's Unit "
            "Preset must be Planetary, Stellar, Galactic or Custom."
        )
    scales = (0.0, 0.0, 0.0)
    if unit == "CUSTOM":
        given = tuple(
            getattr(facts, key, None) for key in ("length_scale_m", "mass_scale_kg", "time_scale_s")
        )
        if not all(_is_number(v) and math.isfinite(v) and v > 0.0 for v in given):
            raise PresetError(
                "A Custom scale needs its length, mass and time scales set above zero."
            )
        scales = tuple(float(v) for v in given)  # type: ignore[assignment]
    pace = getattr(facts, "seconds_per_frame", None)
    if not _is_number(pace) or not math.isfinite(pace) or pace <= 0.0:
        raise PresetError("The System's seconds per frame must be a number above zero.")
    softening = getattr(facts, "softening", None)
    if not _is_number(softening) or not math.isfinite(softening) or softening < 0.0:
        raise PresetError("The System's softening must be a number, zero or more.")
    substeps = getattr(facts, "substeps", None)
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise PresetError("The System's substeps must be a whole number, 1 or more.")
    return SystemFacts(str(unit), float(pace), float(softening), int(substeps), *scales)


def check_mesh_size(vertex_count: int, triangle_count: int) -> None:
    """Refuse a mesh too big for a preset, by counts alone.

    Called before anything is converted or copied, so a huge mesh costs a sentence and not a
    minute. The Blender side calls it as soon as it has read the counts.
    """

    for count, limit, noun in (
        (triangle_count, MAX_MESH_TRIANGLES, "triangles"),
        (vertex_count, MAX_MESH_VERTICES, "vertices"),
    ):
        if count > limit:
            raise PresetError(
                f"This object's mesh has {count:,} {noun}, and a preset can hold up to "
                f"{limit:,}. Simplify the mesh first, or set the Cloud's Shape to "
                "Ball or Disk Annulus."
            )


def _float32_vertices(vertices: Any) -> tuple[tuple[float, float, float], ...]:
    """The vertices as the float32 values the file stores, or :class:`PresetError`."""

    try:
        flat = [float(component) for vertex in vertices for component in vertex]
        if len(flat) % 3:
            raise ValueError("not a whole number of vertices")
        packed = struct.pack(f"<{len(flat)}f", *flat)
    except (TypeError, ValueError, OverflowError, struct.error) as exc:
        raise PresetError(_BAD_PROXY) from exc
    rounded = struct.unpack(f"<{len(flat)}f", packed)
    if not all(math.isfinite(component) for component in rounded):
        raise PresetError(_BAD_PROXY)
    return tuple(tuple(rounded[index : index + 3]) for index in range(0, len(rounded), 3))  # type: ignore[misc]


def _validated_mesh_proxy(proxy: Any) -> ProxyShape:
    """A mesh stand-in with its counts, indices and size checked; its radius is recomputed."""

    raw_vertices = getattr(proxy, "vertices", ())
    raw_triangles = getattr(proxy, "triangles", ())
    try:
        check_mesh_size(len(raw_vertices), len(raw_triangles))
    except TypeError as exc:
        raise PresetError(_BAD_PROXY) from exc
    vertices = _float32_vertices(raw_vertices)
    try:
        triangles = tuple((int(a), int(b), int(c)) for a, b, c in raw_triangles)
    except (TypeError, ValueError) as exc:
        raise PresetError(_BAD_PROXY) from exc
    if len(vertices) < 3 or not triangles:
        raise PresetError(_BAD_PROXY)
    top = len(vertices)
    if any(index < 0 or index >= top for triangle in triangles for index in triangle):
        raise PresetError(_BAD_PROXY)
    spans = [
        max(vertex[axis] for vertex in vertices) - min(vertex[axis] for vertex in vertices)
        for axis in range(3)
    ]
    radius = max(spans) / 2.0
    if not radius > 0.0 or not math.isfinite(radius):
        raise PresetError(_BAD_PROXY)
    return ProxyShape("MESH", radius, 0.0, 0.0, vertices, triangles)


def _validated_proxy(proxy: Any) -> ProxyShape:
    bad = PresetError(_BAD_PROXY)
    kind = getattr(proxy, "kind", None)
    if kind == "MESH":
        return _validated_mesh_proxy(proxy)
    radius = getattr(proxy, "radius", None)
    if kind not in _PROXY_KINDS or not _is_number(radius):
        raise bad
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise bad
    if kind == "SPHERE":
        return ProxyShape("SPHERE", radius)
    inner = getattr(proxy, "inner_radius", None)
    thickness = getattr(proxy, "thickness", None)
    if not _is_number(inner) or not _is_number(thickness):
        raise bad
    inner, thickness = float(inner), float(thickness)
    if not (math.isfinite(inner) and math.isfinite(thickness)):
        raise bad
    if thickness <= 0.0 or not 0.0 <= inner < radius:
        raise bad
    return ProxyShape("ANNULUS", radius, inner, thickness)


def _document(name: str, role: str, settings: dict[str, Any], display_group: str):
    entry = AuthoringObject(
        name=name,
        persistent_id=_PLACEHOLDER_ID,
        role=role,
        display_group=display_group,
        settings={key: value for key, value in settings.items() if key not in POINTER_KEYS},
    )
    try:
        return build_authoring_document((entry,))
    except AuthoringDocumentError as exc:
        raise PresetError(f"These settings cannot be saved in a preset: {exc}") from exc


def build_preset(
    *,
    name: Any,
    role: Any,
    system: SystemFacts,
    proxy: ProxyShape,
    settings: dict[str, Any],
    display_group: str = "AUTO",
) -> Preset:
    """Validate everything an object gave and return the preset, or raise :class:`PresetError`."""

    cleaned = clean_name(name)
    refusal = refusal_for_role(role, cleaned)
    if refusal is not None:
        raise PresetError(refusal)
    facts = _validated_system(system)
    shape = _validated_proxy(proxy)
    document = _document(cleaned, role, dict(settings), display_group)
    entry = document.objects[0]
    return Preset(
        name=cleaned,
        role=role,
        system=facts,
        proxy=shape,
        settings=dict(entry.settings),
        display_group=entry.display_group,
    )


# -- the file -------------------------------------------------------------------------------


def _encode_mesh(proxy: ProxyShape) -> dict[str, str]:
    """The mesh as two base64 arrays, little-endian ``float32`` and ``uint32``.

    Packed rather than listed: a 20,000-triangle mesh is under half a megabyte this way and
    several megabytes as JSON numbers, and the menu reads every preset file each time it opens.
    """

    flat_vertices = [component for vertex in proxy.vertices for component in vertex]
    flat_triangles = [index for triangle in proxy.triangles for index in triangle]
    return {
        "vertices": base64.b64encode(struct.pack(f"<{len(flat_vertices)}f", *flat_vertices)).decode(
            "ascii"
        ),
        "triangles": base64.b64encode(
            struct.pack(f"<{len(flat_triangles)}I", *flat_triangles)
        ).decode("ascii"),
    }


def _decode_mesh(block: Any) -> tuple[tuple, tuple]:
    if not isinstance(block, dict):
        raise PresetError(_BAD_PROXY)
    decoded = []
    for key, code, limit in (
        ("vertices", "f", MAX_MESH_VERTICES * 3),
        ("triangles", "I", MAX_MESH_TRIANGLES * 3),
    ):
        text = block.get(key)
        if not isinstance(text, str):
            raise PresetError(_BAD_PROXY)
        try:
            raw = base64.b64decode(text.encode("ascii"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PresetError(_BAD_PROXY) from exc
        count = len(raw) // 4
        if len(raw) % 12 or count > limit:
            raise PresetError(_BAD_PROXY)
        decoded.append(struct.unpack(f"<{count}{code}", raw))
    flat_vertices, flat_triangles = decoded
    vertices = tuple(tuple(flat_vertices[i : i + 3]) for i in range(0, len(flat_vertices), 3))
    triangles = tuple(tuple(flat_triangles[i : i + 3]) for i in range(0, len(flat_triangles), 3))
    return vertices, triangles


def to_json(preset: Preset) -> str:
    document = _document(preset.name, preset.role, preset.settings, preset.display_group)
    system: dict[str, Any] = {
        "unit_preset": preset.system.unit_preset,
        "seconds_per_frame": preset.system.seconds_per_frame,
        "softening": preset.system.softening,
        "substeps": preset.system.substeps,
    }
    if preset.system.unit_preset == "CUSTOM":
        system["length_scale_m"] = preset.system.length_scale_m
        system["mass_scale_kg"] = preset.system.mass_scale_kg
        system["time_scale_s"] = preset.system.time_scale_s
    proxy: dict[str, Any] = {
        "kind": preset.proxy.kind,
        "radius": preset.proxy.radius,
        "inner_radius": preset.proxy.inner_radius,
        "thickness": preset.proxy.thickness,
    }
    if preset.proxy.kind == "MESH":
        proxy["mesh"] = _encode_mesh(preset.proxy)
    payload = {
        "format": PRESET_FORMAT,
        "version": PRESET_VERSION,
        "name": preset.name,
        "role": preset.role,
        "system": system,
        "proxy": proxy,
        "authoring": document.to_dict(),
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def from_json(text: str | bytes) -> Preset:
    """A preset from file text, or :class:`PresetError` saying in plain words what is wrong."""

    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise PresetError(
            "The file is not valid JSON, so it may be damaged or may not be a preset."
        ) from exc
    if not isinstance(data, dict) or data.get("format") != PRESET_FORMAT:
        raise PresetError("This is not an Astro Warp preset file.")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise PresetError("The preset's version number is missing or damaged.")
    if version != PRESET_VERSION:
        raise PresetError(
            f"This preset was saved as preset version {version}, and this Astro Warp "
            f"reads version {PRESET_VERSION}."
        )

    role = data.get("role")
    if role not in SUPPORTED_ROLES:
        raise PresetError(f"The preset's Role is {role!r}; a preset holds a Body or a Cloud.")
    name = clean_name(data.get("name"))

    system = data.get("system")
    if not isinstance(system, dict):
        raise PresetError("The preset has no System information.")
    facts = _validated_system(
        SystemFacts(
            unit_preset=system.get("unit_preset"),
            seconds_per_frame=system.get("seconds_per_frame"),
            softening=system.get("softening"),
            substeps=system.get("substeps"),
            length_scale_m=system.get("length_scale_m"),
            mass_scale_kg=system.get("mass_scale_kg"),
            time_scale_s=system.get("time_scale_s"),
        )
    )

    proxy = data.get("proxy")
    if not isinstance(proxy, dict):
        raise PresetError(_BAD_PROXY)
    vertices: tuple = ()
    triangles: tuple = ()
    if proxy.get("kind") == "MESH":
        vertices, triangles = _decode_mesh(proxy.get("mesh"))
    shape = _validated_proxy(
        ProxyShape(
            kind=proxy.get("kind"),
            radius=proxy.get("radius"),
            inner_radius=proxy.get("inner_radius", 0.0),
            thickness=proxy.get("thickness", 0.0),
            vertices=vertices,
            triangles=triangles,
        )
    )

    try:
        document = load_authoring_document(data.get("authoring"))
    except AuthoringDocumentError as exc:
        raise PresetError(f"The preset's settings cannot be read ({exc}).") from exc
    if len(document.objects) != 1:
        raise PresetError("The preset must hold exactly one object.")
    entry = document.objects[0]
    if entry.role != role:
        raise PresetError("The preset's settings are for a different Role than the preset says.")
    return Preset(
        name=name,
        role=role,
        system=facts,
        proxy=shape,
        settings={k: v for k, v in entry.settings.items() if k not in POINTER_KEYS},
        display_group=entry.display_group,
    )


# -- the folder -----------------------------------------------------------------------------


def exists(directory: str | os.PathLike[str], name: Any) -> bool:
    """Whether saving ``name`` would land on a preset that is already there."""

    if not str(directory):
        return False
    try:
        return (Path(directory) / file_name_for(name)).is_file()
    except PresetError:
        return False


def save(directory: str | os.PathLike[str], preset: Preset, *, overwrite: bool = False) -> Path:
    """Write ``preset`` into ``directory``; refuse to replace one unless ``overwrite``."""

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name_for(preset.name)
    text = to_json(preset)
    if not overwrite:
        try:
            with open(path, "x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except FileExistsError:
            raise PresetExists(
                f"A preset with this name is already saved ({path.name}). Choose another "
                f"name, or tick {OVERWRITE_LABEL} to replace it."
            ) from None
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _read(path: Path) -> Preset:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise PresetError("The file is too large to be a preset.")
        raw = path.read_bytes()
    except OSError as exc:
        raise PresetError(
            f"The file could not be read ({exc.strerror or type(exc).__name__})."
        ) from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PresetError("The file is not readable text, so it is not a preset.") from exc
    return from_json(text)


def _is_preset_file_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and not name.startswith(".")
        and "/" not in name
        and "\\" not in name
        and name.lower().endswith(PRESET_SUFFIX)
        and Path(name).name == name
    )


def load_file(directory: str | os.PathLike[str], file_name: Any) -> Preset:
    """One saved preset by its bare file name. A name with a path in it is refused."""

    if not _is_preset_file_name(file_name) or not str(directory):
        raise PresetError("That is not the name of a saved preset file.")
    path = Path(directory) / file_name
    if not path.is_file():
        raise PresetError(f"The preset file '{file_name}' is not in the presets folder any more.")
    try:
        return _read(path)
    except PresetError as exc:
        raise PresetError(f"The preset file '{file_name}' could not be used. {exc}") from exc


_scan_cache: dict[str, tuple[tuple, ScanResult]] = {}
"""Last result per folder, with the listing it was made from. A menu redraws often while it is
open, and a preset with a mesh is large; re-reading only when a file's name, size or time
changes keeps the menu light and can never show stale presets."""


def scan(directory: str | os.PathLike[str]) -> ScanResult:
    """Every preset in ``directory``, and every preset-looking file that had to be skipped."""

    if not str(directory):
        return ScanResult()
    folder = Path(directory)
    if not folder.is_dir():
        return ScanResult()
    try:
        entries = sorted(folder.iterdir(), key=lambda item: item.name.casefold())
        files = [
            (entry, entry.stat())
            for entry in entries
            if _is_preset_file_name(entry.name) and entry.is_file()
        ]
    except OSError as exc:
        reason = f"The presets folder could not be read ({exc.strerror or type(exc).__name__})."
        return ScanResult(skipped=(("(presets folder)", reason),))
    signature = tuple((entry.name, info.st_size, info.st_mtime_ns) for entry, info in files)
    key = str(folder)
    cached = _scan_cache.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    presets: list[tuple[str, Preset]] = []
    skipped: list[tuple[str, str]] = []
    for entry, _info in files:
        try:
            presets.append((entry.name, _read(entry)))
        except PresetError as exc:
            skipped.append((entry.name, str(exc)))
    presets.sort(key=lambda pair: (pair[1].name.casefold(), pair[0]))
    result = ScanResult(tuple(presets), tuple(skipped))
    if len(_scan_cache) > 4:
        _scan_cache.clear()
    _scan_cache[key] = (signature, result)
    return result
