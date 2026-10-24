"""USD time-sampled point export from an immutable bake revision (EXPORT-001).

``12_IMPLEMENTATION_ROADMAP.md:139`` names USD point-instancer/time-sample export and
``01_SCOPE_AND_NON_GOALS.md:62`` puts Geometry Nodes and USD workflows in scope "through
stable exported data contracts". The stable contract already exists: an immutable bake
revision. This module is a *reader* of one, and nothing else — it never touches a runtime,
never steps a solver, and takes no context. Its whole input is a manifest path.

Two properties matter more than the format details:

**It streams.** The exporter consumes ``cache.iter_frame_columns``, which yields one
frame's typed columns at a time. A production bake is a hundred thousand bodies over two
hundred fifty frames; materializing it to build a text file would reintroduce exactly the
residency defect ``STAB-005`` removed from the bake path. Peak residency here is one
frame's columns plus one frame's formatted text.

**It is lossy, and says so.** ``UsdGeomPoints.points`` is typed ``point3f[]`` — float32 —
by the USD schema itself. The bake is float64. Every exported position is therefore a
correctly-rounded float32 of the baked value, and a round-trip gate must compare against
the float32 cast rather than against the raw bake column. The written layer declares this
in its own ``customLayerData`` so a consumer reading the file alone still learns it. The
bake remains the float64 record of authority; the ``.usda`` is an interchange artifact.

``pxr`` is not importable inside Blender's Python, so the layer is written as ``.usda``
ASCII text rather than through the USD API. That is a supported serialization of the same
format, and ``bpy.ops.wm.usd_import`` reads it — which is what ``EXPORT-002`` gates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .cache import SCHEMA_V2, filesystem_path, iter_frame_columns, read_payload_header

#: The prim the exporter writes. One ``UsdGeomPoints`` prim holds the whole bake.
PRIM_NAME = "AstroWarpBake"
POSITION_COLUMN = "position"
RADIUS_COLUMN = "radius"
PRECISION_NOTE = "positions are float32: UsdGeomPoints.points is point3f[] by USD schema"


class UsdExportError(ValueError):
    """A bake revision could not be exported to USD."""


def _require_column(
    header: dict[str, Any], name: str, kind: str, width: int
) -> dict[str, Any] | None:
    """Return the named column if the payload carries it in the expected shape.

    The v2 codec derives its columns from the body dicts it was handed, so the column set
    is a property of the bake rather than of this module. Looking a column up by name and
    then holding it to a kind and a width is what keeps a body-schema change into a failed
    export instead of a silently mis-typed one.
    """

    for column in header["columns"]:
        if column["name"] != name:
            continue
        if column["kind"] != kind or int(column["width"]) != width:
            raise UsdExportError(
                f"bake column {name!r} is {column['kind']}/{column['width']}, "
                f"expected {kind}/{width}"
            )
        return column
    return None


def _usd_string(value: Any) -> str:
    """A double-quoted USD string literal.

    USD's ASCII grammar reads UTF-8 and the escapes ``\\\\ \\" \\n \\r \\t``; it has no
    ``\\uXXXX`` escape, which is what ``json.dumps`` writes for every non-ASCII character
    (SR-49). Non-ASCII is therefore written as itself, and any other control character,
    which has no USD spelling, is replaced rather than emitted as an invalid escape.
    """

    out = ['"']
    for char in str(value):
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append("�")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _floats(values: np.ndarray) -> np.ndarray:
    """Shortest round-trip text for each value, at the float32 the schema stores.

    ``astype(str)`` on a float32 array is numpy's shortest-repr formatting, so re-parsing
    the text as a double and narrowing it back to float32 is exact. Formatting per value in
    Python instead would cost tens of millions of calls on a production bake.
    """

    return values.astype(np.float32).astype("U")


def _payload_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    if manifest.get("compatible_schema") != SCHEMA_V2:
        raise UsdExportError(
            f"USD export needs an {SCHEMA_V2} bake; this revision is "
            f"{manifest.get('compatible_schema')!r}. A pre-cache JSON bake cannot be "
            "streamed and must be re-baked."
        )
    stem = manifest_path.name[: -len(".manifest.json")]
    payload = manifest_path.parent / f"{stem}.frames.bin"
    if not payload.is_file():
        raise UsdExportError(f"bake payload is missing: {payload}")
    return payload


def _layer_metadata(manifest: dict[str, Any], frame_start: int, frame_end: int, fps: float) -> str:
    provenance = {
        "bakeId": manifest.get("bake_id", ""),
        "contextId": manifest.get("context_id", ""),
        "graphHash": manifest.get("graph_hash", ""),
        "payloadSha256": manifest.get("payload_sha256", ""),
        "manifestSha256": manifest.get("manifest_sha256", ""),
        "backend": manifest.get("backend", ""),
        "integrator": manifest.get("integrator", ""),
        "compatibleSchema": manifest.get("compatible_schema", ""),
        "positionPrecision": PRECISION_NOTE,
    }
    entries = "".join(
        f"            string {key} = {_usd_string(value)}\n"
        for key, value in provenance.items()
    )
    return (
        "#usda 1.0\n"
        "(\n"
        f'    defaultPrim = "{PRIM_NAME}"\n'
        # Declared, not defaulted: USD's own default up axis is Y, and a consumer that
        # applies the Y-to-Z conversion to a bake that was always Z-up silently rotates the
        # scene. The bake is in Blender's world, in metres.
        '    upAxis = "Z"\n'
        "    metersPerUnit = 1\n"
        f"    timeCodesPerSecond = {fps!r}\n"
        f"    startTimeCode = {frame_start}\n"
        f"    endTimeCode = {frame_end}\n"
        "    customLayerData = {\n"
        "        dictionary astroWarp = {\n"
        f"{entries}"
        "        }\n"
        "    }\n"
        ")\n\n"
    )


def export_points_usda(
    *,
    manifest_path: Path,
    output_path: Path,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Write one immutable bake revision to ``output_path`` as a ``.usda`` layer.

    Returns a summary of what was written. Raises ``UsdExportError`` rather than writing a
    partial layer if the revision is the wrong schema or carries no usable position column.
    """

    manifest_path = filesystem_path(manifest_path)
    output_path = Path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = _payload_path(manifest_path, manifest)

    header = read_payload_header(payload_path)
    if header.get("context_id") != manifest.get("context_id"):
        raise UsdExportError("bake payload and manifest disagree on context ID")
    if _require_column(header, POSITION_COLUMN, "vector-float", 3) is None:
        raise UsdExportError(f"bake carries no {POSITION_COLUMN!r} column to export")
    has_radius = _require_column(header, RADIUS_COLUMN, "scalar-float", 1) is not None

    frame_count = int(header["frame_count"])
    frame_start = int(manifest.get("frame_start", 0))
    frame_end = frame_start + max(0, frame_count - 1)
    stable_ids = [int(value) for value in manifest.get("stable_ids", ())]

    # Written beside the target and renamed into place only once the frame count has been
    # checked, so a failed export never leaves a truncated layer at ``output_path`` and never
    # destroys the previous good one (SR-50).
    temporary_path = output_path.with_name(output_path.name + ".partial")

    # ``ids`` is written once rather than per frame: ``bake._check_sample_ids`` rejects any
    # frame whose stable IDs differ from the runtime ordering, so the correspondence is a
    # bake invariant and time-sampling it would be a quarter-gigabyte of restated constant.
    try:
        _write_layer(
            temporary_path,
            manifest=manifest,
            payload_path=payload_path,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_count=frame_count,
            fps=fps,
            stable_ids=stable_ids,
            has_radius=has_radius,
        )
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "output_path": str(output_path),
        "bake_id": manifest.get("bake_id", ""),
        "context_id": manifest.get("context_id", ""),
        "graph_hash": manifest.get("graph_hash", ""),
        "payload_sha256": manifest.get("payload_sha256", ""),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_count,
        "body_count": len(stable_ids),
        "widths_exported": has_radius,
        "byte_count": output_path.stat().st_size,
        "position_precision": PRECISION_NOTE,
    }


def _write_layer(
    path: Path,
    *,
    manifest: dict[str, Any],
    payload_path: Path,
    frame_start: int,
    frame_end: int,
    frame_count: int,
    fps: float,
    stable_ids: list[int],
    has_radius: bool,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_layer_metadata(manifest, frame_start, frame_end, fps))
        handle.write(f'def Points "{PRIM_NAME}"\n{{\n')
        handle.write("    int64[] ids = [" + ", ".join(str(value) for value in stable_ids) + "]\n")
        if has_radius:
            _write_time_samples(
                handle,
                payload_path,
                declaration="    float[] widths.timeSamples = {",
                frame_start=frame_start,
                frame_count=frame_count,
                # USD point widths are diameters; the bake column is a radius (SR-13,
                # ADR 0187). Doubled in float64 before the float32 narrowing.
                render=lambda columns: ", ".join(
                    _floats(np.asarray(columns[RADIUS_COLUMN], dtype=np.float64) * 2.0)
                ),
            )
        _write_time_samples(
            handle,
            payload_path,
            declaration="    point3f[] points.timeSamples = {",
            frame_start=frame_start,
            frame_count=frame_count,
            render=lambda columns: ", ".join(
                f"({x}, {y}, {z})" for x, y, z in _floats(columns[POSITION_COLUMN])
            ),
        )
        handle.write("}\n\n")


def _write_time_samples(
    handle: Any,
    payload_path: Path,
    *,
    declaration: str,
    frame_start: int,
    frame_count: int,
    render: Any,
) -> None:
    """Stream one attribute's ``timeSamples`` block, one frame per pass of the payload.

    A ``.usda`` attribute owns a contiguous ``timeSamples`` block, so two attributes cannot
    be interleaved in one pass over the frames. Re-reading the payload per attribute is the
    cheap half of that trade: the alternative is buffering a whole attribute in memory,
    which is the residency this exporter exists to avoid.
    """

    handle.write(declaration + "\n")
    written = 0
    for index, (_meta, columns) in enumerate(iter_frame_columns(payload_path)):
        # No separator after the last entry: a trailing comma is accepted by some readers
        # and not others, and the frame count is known from the header before the walk.
        tail = ",\n" if index + 1 < frame_count else "\n"
        handle.write(f"        {frame_start + index}: [{render(columns)}]{tail}")
        written += 1
    if written != frame_count:
        raise UsdExportError(f"bake header declares {frame_count} frames, payload held {written}")
    handle.write("    }\n")


def _demo() -> None:
    """Self-check: a streamed bake exports, and the layer restates the bake exactly."""

    import tempfile

    from .bake import write_bake_revision

    rng = np.random.default_rng(11)
    bodies, frames = 40, 6
    stable_ids = list(range(bodies))
    samples = [
        {
            "frame": 1 + frame,
            "bodies": [
                {
                    "stable_id": index,
                    "active": True,
                    "mass": float(rng.random()),
                    "radius": float(rng.random()),
                    "position": [float(value) for value in rng.random(3)],
                    "velocity": [float(value) for value in rng.random(3)],
                }
                for index in range(bodies)
            ],
        }
        for frame in range(frames)
    ]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        revision, manifest_path = write_bake_revision(
            directory=root / "bakes",
            name="usd-demo",
            context_id="ctx-usd",
            graph_hash="graph-usd",
            frame_start=1,
            backend="direct",
            integrator="velocity-verlet",
            dt=0.01,
            substeps=1,
            stable_ids=stable_ids,
            units={"length": "m"},
            samples=iter(samples),
            compatible_schema=SCHEMA_V2,
        )
        layer = root / "bake.usda"
        summary = export_points_usda(manifest_path=manifest_path, output_path=layer)
        assert summary["frame_count"] == frames
        assert summary["body_count"] == bodies

        text = layer.read_text(encoding="utf-8")
        assert text.startswith("#usda 1.0")
        assert revision.graph_hash in text
        assert revision.payload_sha256 in text

        # Every exported sample equals the baked value narrowed to the schema's float32.
        parsed = _parse_points_time_samples(text)
        assert sorted(parsed) == [1 + frame for frame in range(frames)]
        for frame, sample in enumerate(samples):
            expected = np.asarray(
                [body["position"] for body in sample["bodies"]], dtype=np.float64
            ).astype(np.float32)
            assert np.array_equal(parsed[1 + frame], expected), frame

    print(f"usd self-check OK: {frames} frames x {bodies} bodies, {summary['byte_count']} bytes")


def _parse_points_time_samples(text: str) -> dict[int, np.ndarray]:
    """Read back the ``points.timeSamples`` block. Test/self-check helper, not an importer."""

    samples: dict[int, np.ndarray] = {}
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("point3f[] points.timeSamples"):
            inside = True
            continue
        if inside:
            if stripped == "}":
                break
            time_code, _, body = stripped.partition(":")
            values = [
                [float(part) for part in point.split(",")]
                for point in re.findall(r"\(([^)]*)\)", body)
            ]
            samples[int(time_code)] = np.asarray(values, dtype=np.float32)
    return samples


if __name__ == "__main__":
    _demo()
