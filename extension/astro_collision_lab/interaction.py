"""Pure authored-revision and preparation-cache contracts.

Signatures describe ordered sources, not Blender names. Only equilibrium
qualification removes each assembly's global position and bulk velocity;
orientation, relative component state, numerical settings and support remain.
No helper imports Blender, Warp, or initializes a device.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .simulation import BodySpec, SimulationConfig


def _record(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("expected dataclass instance or mapping")


def _payload(config: SimulationConfig, bodies: Sequence[BodySpec]) -> dict:
    groups: dict[str, str] = {}
    sources = []
    for body in bodies:
        source = _record(body)
        source.pop("name", None)
        label = source.get("galaxy_id", "")
        if label:
            source["galaxy_id"] = groups.setdefault(label, f"group-{len(groups)}")
        sources.append(source)
    return {"config": _record(config), "bodies": sources}


def _json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def physical_signature(config: SimulationConfig, bodies: Sequence[BodySpec]) -> str:
    """Canonical JSON of all physical inputs, ignoring display names only.

    Nonempty assembly labels are numbered in first-occurrence order so renaming
    an assembly preserves its revision, while regrouping components changes it.
    Source order is retained because it participates in impact realization.
    """
    return _json(_payload(config, bodies))


def same_sources(previous, current):
    """Keep renames live, but recognize replacement of Blender-owned sources."""
    if len(previous) != len(current):
        return False
    for old, new in zip(previous, current):
        if old is new:
            continue
        try:
            if old.as_pointer() != new.as_pointer():
                return False
        except (AttributeError, ReferenceError):
            return False
    return True


def equilibrium_signature(config: SimulationConfig, bodies: Sequence[BodySpec]) -> str:
    """Canonical isolated-model inputs, with each group's COM drift removed.

    Empty membership labels represent independent bodies, not a shared group.
    Subtraction uses an anchor to avoid losing precision to a large world-space
    origin. Only the centered coordinates are rounded to 12 significant digits
    to absorb floating-point subtraction drift; support and rotations are exact.
    """
    payload = _payload(config, bodies)
    groups: dict[tuple[str, object], list[dict]] = {}
    for index, body in enumerate(payload["bodies"]):
        label = body["galaxy_id"]
        key = ("group", label) if label else ("body", index)
        groups.setdefault(key, []).append(body)
    for group in groups.values():
        mass = math.fsum(body["mass"] for body in group)
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("equilibrium groups require positive finite mass")
        for field in ("position", "velocity"):
            anchor = tuple(group[0][field])
            offsets = [tuple(body[field][axis] - anchor[axis] for axis in range(3))
                       for body in group]
            center = tuple(math.fsum(body["mass"] * offset[axis]
                                     for body, offset in zip(group, offsets)) / mass
                           for axis in range(3))
            for body, offset in zip(group, offsets):
                centered = [float(format(offset[axis] - center[axis], ".12g"))
                            for axis in range(3)]
                body[field] = [0.0 if value == 0.0 else value for value in centered]
    return _json(payload)


def resolved_mode(mode: str, bodies: Sequence[BodySpec], tagged_galaxy: bool) -> str:
    """Compatibility mode suggestion; callers must still validate the scene.

    New authoring entry points should use strict ``resolve_mode`` instead.
    """
    mode = mode.upper()
    records = tuple(_record(body) for body in bodies)
    if (mode == "IMPACT" and tagged_galaxy and records
            and all(str(body.get("kind", "")).upper() in ("DISK", "BULGE", "HALO")
                    and bool(body.get("galaxy_id"))
                    for body in records)):
        return "GALAXY"
    return mode


def resolve_mode(mode: str, bodies: Sequence[BodySpec], tagged_galaxy: bool = False) -> str:
    """Recover only associated, explicitly tagged Galaxy components in IMPACT.

    Reject incompatible sources before expensive warmup. Explicit COMPRESSIBLE
    and valid sphere-only IMPACT are never changed by a stale preset tag.
    """
    mode = mode.upper()
    if mode not in ("IMPACT", "GALAXY", "COMPRESSIBLE"):
        raise ValueError("unknown dynamics mode")
    records = tuple(_record(body) for body in bodies)
    kinds = tuple(str(body.get("kind", "")).upper() for body in records)
    components = {"DISK", "BULGE", "HALO"}
    if not records:
        raise ValueError("at least one authored primitive is required")
    if any(kind not in components | {"SPHERE"} for kind in kinds):
        raise ValueError("unknown primitive kind")
    if (mode == "IMPACT" and tagged_galaxy is True
            and all(kind in components and bool(body.get("galaxy_id"))
                    for kind, body in zip(kinds, records))):
        mode = "GALAXY"
    if mode != "GALAXY" and any(kind != "SPHERE" for kind in kinds):
        raise ValueError("impact models accept SPHERE primitives only")
    return mode


def code_fingerprint(package: Path) -> str:
    """Hash actual owned/vendor Python and manifest bytes, never build claims."""
    package = Path(package)
    paths = [path for path in package.rglob("*.py") if path.is_file()]
    paths.append(package / "blender_manifest.toml")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(package).as_posix()):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


SCHEMA_VERSION = 2


def runtime_identity(wp, app, module_name):
    """Bind preparation to the actual Blender/Warp build and selected CUDA GPU.

    Called only after input validation. Device discovery does not compile or
    step a simulation; expensive preparation remains in the owned worker.
    """
    device = wp.get_device("cuda:0")
    build = getattr(app, "build_hash", b"")
    if isinstance(build, bytes):
        build = build.decode("ascii", errors="replace")
    return (module_name, tuple(app.version), str(build), wp.__version__,
            sys.platform, str(device.name), str(getattr(device, "arch", "")),
            str(getattr(device, "uuid", "")))


def warmup_identity(*, signature, code_fingerprint, runtime):
    """Content-bound request identity copied into worker ready receipts.

    Runtime is explicit (e.g. Blender, Warp, CUDA and device identity), supplied
    by the caller. Request IDs detect stale/mismatched content, not malicious
    tampering by a writer with full access to both request and receipt.
    """
    if not isinstance(signature, str) or not signature:
        raise ValueError("nonempty physical signature is required")
    if not isinstance(code_fingerprint, str) or not code_fingerprint:
        raise ValueError("nonempty code fingerprint is required")
    if not isinstance(runtime, (tuple, list)) or not runtime:
        raise ValueError("nonempty runtime tuple is required")
    identity = dict(schema=SCHEMA_VERSION, signature=signature,
                    code_fingerprint=code_fingerprint, runtime=json.loads(_json(runtime)))
    identity["request_id"] = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()
    return identity


def ready_marker(identity):
    """Create receipt after successful preparation using verified actual inputs."""
    expected = warmup_identity(signature=identity["signature"],
                               code_fingerprint=identity["code_fingerprint"],
                               runtime=identity["runtime"])
    if identity != expected:
        raise ValueError("invalid warmup identity")
    return {**expected, "pass": True}


def marker_matches(marker, identity):
    """Reject failed, malformed, old-schema, wrong-code/runtime/model receipts."""
    if not isinstance(marker, Mapping) or marker.get("pass") is not True:
        return False
    try:
        expected = ready_marker(identity)
        actual = {key: marker[key] for key in expected}
        return _json(actual) == _json(expected)
    except (KeyError, TypeError, ValueError):
        return False


def marker_valid(path: Path, identity) -> bool:
    """Read and validate a content-bound receipt; old key-only markers fail closed."""
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    return marker_matches(receipt, identity)
