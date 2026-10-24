"""Versioned chunked typed binary bake frames (``astro-warp-bake-v2``).

STAB-005's cache half. The Phase 0 bake payload was a JSON document holding one dict
per body per frame; at a hundred thousand bodies over two hundred fifty frames that is
tens of gigabytes of text that no reader can parse into a production cache. This module
replaces the *encoding* — not the content — with a chunked, columnar, little-endian
binary layout: a self-describing JSON header followed by one contiguous typed block per
column per frame. The bake stays lossless and byte-addressable, and a frame's columns
can be read without decoding the frames before it.

The codec is schema-agnostic on purpose. It discovers the per-body field set from the
data (a Direct, Tree, or SPH snapshot carry different fields) and records enough type
information in the header to reconstruct each body dict exactly — ints as ints, floats
as floats, bools as bools, and fixed-width vectors as lists of floats. Floats are stored
as IEEE-754 float64 so the round trip is exact in the precision lane the runtime records
its state in; a narrower or compressed lane is a future compatibility option, not a
silent default (see ``adr/0025-phase35-stabilization-milestone.md``).

The identity model is unchanged from ``bake.py``: ``payload_sha256`` is the SHA-256 of
the whole binary file, so an immutable revision is content-addressed exactly as the JSON
payload was, and a single flipped byte fails integrity.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_V2 = "astro-warp-bake-v2"
CODEC = "chunked-columnar-le-v1"
BODIES_KEY = "bodies"

# Explicit little-endian dtypes so the payload is identical on any host that writes it.
_DTYPE = {"int64": "<i8", "float64": "<f8", "uint8": "<u1"}
_HEADER_PREFIX = struct.Struct("<Q")  # uint64 header length


class BakeCacheError(ValueError):
    """A binary bake payload could not be encoded or decoded."""


def filesystem_path(path: Path | str) -> Path:
    """Return a path Windows filesystem APIs can use beyond ``MAX_PATH``.

    The extension keeps the artist-authored, ordinary path in scene data and bake
    provenance. Only filesystem calls use the extended-length spelling, so cache
    identity does not change merely because Windows needs a transport prefix.
    """

    candidate = Path(path)
    text = str(candidate)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return candidate
    absolute = str(candidate.resolve())
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def logical_path(path: Path | str) -> Path:
    """Remove a Windows extended-length transport prefix from a path."""

    candidate = Path(path)
    if os.name != "nt":
        return candidate
    text = str(candidate)
    upper = text.upper()
    if upper.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + text[8:])
    if upper.startswith("\\\\?\\"):
        return Path(text[4:])
    return candidate


def _classify(value: Any) -> tuple[str, str, int]:
    """Return ``(kind, dtype, width)`` for one body field value.

    ``bool`` is checked before ``int`` because ``bool`` is an ``int`` subclass and must
    round-trip back to a bool, not a 0/1 int.
    """

    if isinstance(value, bool):
        return "scalar-bool", "uint8", 1
    if isinstance(value, int):
        return "scalar-int", "int64", 1
    if isinstance(value, float):
        return "scalar-float", "float64", 1
    if isinstance(value, (list, tuple)):
        numeric = all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        )
        if not value or not numeric:
            raise BakeCacheError(f"vector body field is empty or non-numeric: {value!r}")
        return "vector-float", "float64", len(value)
    raise BakeCacheError(f"unsupported body field type for binary cache: {type(value).__name__}")


def _columns_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    columns = []
    for name, value in body.items():
        kind, dtype, width = _classify(value)
        columns.append({"name": name, "kind": kind, "dtype": dtype, "width": width})
    return columns


def _reconstruct(kind: str, cells: np.ndarray) -> Any:
    if kind == "scalar-bool":
        return bool(cells[0])
    if kind == "scalar-int":
        return int(cells[0])
    if kind == "scalar-float":
        return float(cells[0])
    if kind == "vector-float":
        return [float(item) for item in cells]
    raise BakeCacheError(f"unknown column kind: {kind}")


_SCALAR_PYTYPE = {"scalar-bool": bool, "scalar-int": int, "scalar-float": float}


def _is_subtype(value: Any, expected: type) -> bool:
    # bool is an int subclass and must not pass as an int column (or vice versa).
    return isinstance(value, expected) and not (expected is int and isinstance(value, bool))


def _frame_blocks(bodies: list[dict[str, Any]], columns: list[dict[str, Any]]) -> list[bytes]:
    """One contiguous typed block per column for a single frame's bodies."""

    blocks = []
    for column in columns:
        name, dtype, width = column["name"], column["dtype"], column["width"]
        if width == 1:
            values = [body[name] for body in bodies]
            expected = _SCALAR_PYTYPE[column["kind"]]
            if any(type(value) is not expected and not _is_subtype(value, expected)
                   for value in values):
                raise BakeCacheError(
                    f"field {name!r} changes type across bodies (column is {column['kind']})"
                )
        else:
            values = []
            for body in bodies:
                row = body[name]
                if len(row) != width:
                    raise BakeCacheError(f"vector field {name!r} changed width mid-bake")
                values.extend(row)
        blocks.append(np.asarray(values, dtype=_DTYPE[dtype]).tobytes())
    return blocks


def _pack_header(
    *,
    context_id: str,
    columns: list[dict[str, Any]],
    field_order: list[str],
    frames_meta: list[dict[str, Any]],
) -> bytes:
    header = {
        "schema": SCHEMA_V2,
        "codec": CODEC,
        "byte_order": "little",
        "context_id": context_id,
        "frame_count": len(frames_meta),
        "columns": columns,
        "body_field_order": field_order,
        "frames": frames_meta,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _HEADER_PREFIX.pack(len(header_bytes)) + header_bytes


class _FrameEncoder:
    """Shared frame-by-frame encoding state for the in-memory and streaming writers.

    The column set is discovered from the first non-empty frame and every later frame is
    held to it, so a payload always describes exactly one body schema and the decoder's
    offset walk is a fixed stride per frame.
    """

    def __init__(self) -> None:
        self.columns: list[dict[str, Any]] = []
        self.field_order: list[str] = []
        self.frames_meta: list[dict[str, Any]] = []

    def add(self, sample: dict[str, Any]) -> list[bytes]:
        bodies = sample.get(BODIES_KEY, [])
        meta = {key: value for key, value in sample.items() if key != BODIES_KEY}
        self.frames_meta.append({"meta": meta, "body_count": len(bodies)})
        if not bodies:
            return []
        if not self.columns:
            self.columns = _columns_from_body(bodies[0])
            self.field_order = [column["name"] for column in self.columns]
        for body in bodies:
            if list(body.keys()) != self.field_order:
                raise BakeCacheError("bake bodies are not homogeneous across bodies or frames")
        return _frame_blocks(bodies, self.columns)

    def header(self, context_id: str) -> bytes:
        return _pack_header(
            context_id=context_id,
            columns=self.columns,
            field_order=self.field_order,
            frames_meta=self.frames_meta,
        )


def encode_frames(*, context_id: str, samples: list[dict[str, Any]]) -> bytes:
    """Encode a list of snapshot samples into the ``astro-warp-bake-v2`` binary payload.

    Each sample is a snapshot dict: a ``bodies`` list of homogeneous per-body dicts plus
    any number of frame-level scalar/JSON keys (``frame``, ``step``, ``simulation_time``,
    …), which are carried verbatim in the header so the decoded sample equals the input.

    Every frame is held in memory here. A production-scale bake streams instead — see
    ``BinaryBakeWriter``, which produces byte-identical payloads one frame at a time.
    """

    encoder = _FrameEncoder()
    blocks: list[bytes] = []
    for sample in samples:
        blocks.extend(encoder.add(sample))
    return b"".join([encoder.header(context_id), *blocks])


def decode_frames(payload: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Decode an ``astro-warp-bake-v2`` payload back into ``(context_id, samples)``."""

    if len(payload) < _HEADER_PREFIX.size:
        raise BakeCacheError("binary bake payload is truncated before its header length")
    (header_len,) = _HEADER_PREFIX.unpack_from(payload)
    start = _HEADER_PREFIX.size
    end = start + header_len
    if end > len(payload):
        raise BakeCacheError("binary bake payload is truncated inside its header")
    header = json.loads(payload[start:end].decode("utf-8"))
    if header.get("schema") != SCHEMA_V2:
        raise BakeCacheError(f"not an {SCHEMA_V2} payload: {header.get('schema')!r}")
    columns = header["columns"]
    samples: list[dict[str, Any]] = []
    offset = end
    for frame in header["frames"]:
        body_count = frame["body_count"]
        decoded_columns: dict[str, list[Any]] = {}
        for column in columns:
            name, kind = column["name"], column["kind"]
            dtype, width = column["dtype"], column["width"]
            count = body_count * width
            itemsize = np.dtype(_DTYPE[dtype]).itemsize
            span = count * itemsize
            if offset + span > len(payload):
                raise BakeCacheError(f"binary bake payload is truncated in column {name!r}")
            cells = np.frombuffer(payload, dtype=_DTYPE[dtype], count=count, offset=offset)
            offset += span
            decoded_columns[name] = [
                _reconstruct(kind, cells[index * width : (index + 1) * width])
                for index in range(body_count)
            ]
        bodies = [
            {column["name"]: decoded_columns[column["name"]][index] for column in columns}
            for index in range(body_count)
        ]
        samples.append({**frame["meta"], BODIES_KEY: bodies})
    if offset != len(payload):
        raise BakeCacheError("binary bake payload has trailing bytes past its last frame")
    return header["context_id"], samples


def payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Chunked digest of a payload file; equals ``payload_sha256`` of its bytes."""

    digest = hashlib.sha256()
    with filesystem_path(path).open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


#: Blocks are staged beside the payload while the header is still growing.
_BLOCKS_SUFFIX = ".blocks"
_COPY_CHUNK = 8 << 20


class BinaryBakeWriter:
    """Stream frames into one ``astro-warp-bake-v2`` file without holding the bake.

    A production bake is a hundred thousand bodies over two hundred fifty frames. Encoding
    that in one call means every frame's bodies are live at once — tens of gigabytes of
    host objects — so the acceptance scale is only reachable by writing each frame as it is
    produced. The header has to precede the blocks (a reader locates a frame's columns by
    arithmetic rather than by scanning), and it is not complete until the last frame is
    known, so blocks are staged in a sibling ``.blocks`` file and copied in behind the
    header on close. That copy is one sequential pass and computes the digest as it goes,
    so the payload is content-addressed without a second read of the file.

    The bytes are identical to ``encode_frames`` on the same samples: both share the frame
    encoder and the header packer, so streaming is an execution strategy, not a format.
    """

    def __init__(self, path: Path, *, context_id: str) -> None:
        self.path = filesystem_path(path)
        self.context_id = context_id
        self.payload_sha256 = ""
        self.byte_count = 0
        self._blocks_path = self.path.with_name(self.path.name + _BLOCKS_SUFFIX)
        self._encoder = _FrameEncoder()
        self._handle = self._blocks_path.open("xb")
        self._complete = False

    @property
    def frame_count(self) -> int:
        return len(self._encoder.frames_meta)

    @property
    def first_frame_body_count(self) -> int:
        """Bodies in the first frame — what the manifest records as the bake's body count."""

        meta = self._encoder.frames_meta
        return int(meta[0]["body_count"]) if meta else 0

    def add_sample(self, sample: dict[str, Any]) -> None:
        for block in self._encoder.add(sample):
            self._handle.write(block)

    def close(self) -> None:
        """Assemble header + staged blocks into ``path`` and record digest and size."""

        self._handle.close()
        header = self._encoder.header(self.context_id)
        digest = hashlib.sha256(header)
        total = len(header)
        try:
            with self.path.open("xb") as payload, self._blocks_path.open("rb") as blocks:
                payload.write(header)
                while chunk := blocks.read(_COPY_CHUNK):
                    digest.update(chunk)
                    payload.write(chunk)
                    total += len(chunk)
        except BaseException:
            self.abort()
            raise
        self._blocks_path.unlink()
        self._complete = True
        self.payload_sha256 = digest.hexdigest()
        self.byte_count = total

    def abort(self) -> None:
        """Drop the staged blocks; a half-written bake must never look committed."""

        if not self._handle.closed:
            self._handle.close()
        self._blocks_path.unlink(missing_ok=True)
        if not self._complete:
            # A close() that died mid-copy leaves a partial (potentially multi-GB) payload.
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> BinaryBakeWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _read_header(handle: Any) -> dict[str, Any]:
    """Consume the header from an open payload, leaving the handle on the first block."""

    prefix = handle.read(_HEADER_PREFIX.size)
    if len(prefix) < _HEADER_PREFIX.size:
        raise BakeCacheError("binary bake payload is truncated before its header length")
    (header_len,) = _HEADER_PREFIX.unpack(prefix)
    raw = handle.read(header_len)
    if len(raw) < header_len:
        raise BakeCacheError("binary bake payload is truncated inside its header")
    header = json.loads(raw.decode("utf-8"))
    if header.get("schema") != SCHEMA_V2:
        raise BakeCacheError(f"not an {SCHEMA_V2} payload: {header.get('schema')!r}")
    return header


def read_payload_header(path: Path) -> dict[str, Any]:
    """Read a payload's self-describing header without reading its blocks."""

    with filesystem_path(path).open("rb") as handle:
        return _read_header(handle)


def iter_frame_columns(path: Path) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Yield ``(frame_meta, columns)`` one frame at a time, in file order.

    The counterpart to ``BinaryBakeWriter``: a reader for bakes too large to decode whole.
    Columns arrive as the typed arrays they were written as — shape ``(bodies,)`` for
    scalars and ``(bodies, width)`` for vectors — so a consumer that wants arrays never
    pays for the per-body dicts ``decode_frames`` builds.
    """

    with filesystem_path(path).open("rb") as handle:
        header = _read_header(handle)
        columns = header["columns"]
        for frame in header["frames"]:
            body_count = int(frame["body_count"])
            arrays: dict[str, np.ndarray] = {}
            for column in columns:
                name, dtype, width = column["name"], column["dtype"], column["width"]
                count = body_count * width
                cells = np.fromfile(handle, dtype=_DTYPE[dtype], count=count)
                if cells.size != count:
                    raise BakeCacheError(f"binary bake payload is truncated in column {name!r}")
                arrays[name] = cells if width == 1 else cells.reshape(body_count, width)
            yield dict(frame["meta"]), arrays
        if handle.read(1):
            raise BakeCacheError("binary bake payload has trailing bytes past its last frame")


def _demo() -> None:
    """Round-trip self-check: identity, hash stability, and tamper detection."""

    rng = np.random.default_rng(7)
    frames = 10
    bodies = 2000
    samples = []
    for frame in range(frames):
        records = []
        for index in range(bodies):
            records.append(
                {
                    "index": index,
                    "stable_id": index - 3,
                    "active": bool(index % 2),
                    "mass": float(rng.random()),
                    "position": [float(x) for x in rng.random(3)],
                    "velocity": [float(x) for x in rng.random(3)],
                }
            )
        samples.append(
            {
                "schema": "astro-warp-checkpoint-v1",
                "backend": "giant_impact",
                "frame": 1000 + frame,
                "simulation_time": float(frame) * 0.5,
                "count": bodies,
                "recomputed_on_restore": ["neighbour_grid"],
                "bodies": records,
            }
        )

    payload = encode_frames(context_id="ctx-demo", samples=samples)
    context_id, decoded = decode_frames(payload)
    assert context_id == "ctx-demo"
    assert decoded == samples, "round trip changed the samples"

    # Re-encoding the decoded samples reproduces identical bytes: the payload is a pure
    # function of its content, which is what content-addressed immutability needs.
    assert encode_frames(context_id="ctx-demo", samples=decoded) == payload

    # An empty-frame bake is legal and round-trips.
    empty = encode_frames(context_id="ctx-empty", samples=[{"frame": 5, "bodies": []}])
    assert decode_frames(empty) == ("ctx-empty", [{"frame": 5, "bodies": []}])

    # One flipped byte in the body block changes the digest.
    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    assert payload_sha256(bytes(tampered)) != payload_sha256(payload)

    # Streaming a bake frame by frame produces the same payload as encoding it whole, and
    # the streamed file reads back column by column without decoding the frames before it.
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        streamed = Path(temporary) / "streamed.frames.bin"
        with BinaryBakeWriter(streamed, context_id="ctx-demo") as writer:
            for sample in samples:
                writer.add_sample(sample)
        assert streamed.read_bytes() == payload, "streamed payload differs from encode_frames"
        assert writer.payload_sha256 == payload_sha256(payload)
        assert writer.byte_count == len(payload)
        frames_read = list(iter_frame_columns(streamed))
        assert len(frames_read) == frames
        assert frames_read[3][1]["position"].shape == (bodies, 3)
        assert frames_read[3][0]["frame"] == 1003

    json_estimate = len(json.dumps({"samples": samples}).encode("utf-8"))
    print(
        f"cache self-check OK: {frames} frames x {bodies} bodies, "
        f"binary {len(payload)} bytes vs JSON ~{json_estimate} bytes "
        f"({json_estimate / len(payload):.1f}x)"
    )


if __name__ == "__main__":
    _demo()
