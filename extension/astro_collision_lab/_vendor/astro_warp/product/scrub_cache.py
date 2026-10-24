"""Immutable exact-frame scrub cache (Slice 4, T2--T4).

A scrub cache is a dense, append-only prefix of authored Blender frames
``[frame_start, frontier]`` held entirely in process memory. It copies the
mutable host buffers a refresh reuses into owned little-endian record bytes,
checksums every record, and hands back read-only retained samples. It never
substitutes a nearest frame, never evicts to make room, and performs no
filesystem writes.

This module is deliberately a pure data structure: it imports the fixed record
schema from :mod:`astro_warp.product.display` and nothing else from the product
runtime. It does not import ``bpy``, Warp, the bake writer/reader, or the runtime
factory, so the codec can be exercised and reviewed in isolation.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .display import (
    TAG_POSITION,
    TAG_SIZE,
    TAG_STABLE_ID,
    TAG_VELOCITY,
    DisplaySample,
    ScrubChannelSpec,
    ScrubSampleSchemaV1,
)

#: 1 GiB per live context for v1, allocated on demand rather than reserved (T4).
SCRUB_CACHE_CAPACITY_BYTES = 1_073_741_824

#: Field-directory flag marking a structurally valid but non-required tag. Such a
#: tag is skipped by a v1 reader, stays unavailable, and cannot alias a known tag.
TAG_OPTIONAL_FLAG = 0x01

_MAGIC = b"ASCB"  # Astro Scrub Cache Block

# Explicit little-endian dtype codes so the payload is identical on any host.
#
# ``<i4`` (code 3) and ``<i8`` (code 4) are additive optional-tag codes: records written
# before either existed cannot contain them, so every legacy record still decodes exactly.
# Code 4 preserves D13's persistent material identity at its authoritative int64 width;
# code 3 remains decodable for other generic optional-tag users but is not a valid Material
# carriage type. See ``color_carriage.py``.
_DTYPE_CODES = {"<f4": 1, "<u8": 2, "<i4": 3, "<i8": 4}
_CODE_DTYPES = {code: dtype for dtype, code in _DTYPE_CODES.items()}

# A CRC-32 appended to every record, little-endian. It is there to catch a record that
# changed after it was sealed --- a stray write into a buffer, a torn copy --- and a CRC-32
# catches every burst up to 32 bits and misses random damage once in 2^32. A cryptographic
# digest bought nothing more: records live in this process's memory only, where anything
# able to rewrite one could rewrite its digest too. Inside Blender 5.1 a 100,000-body record
# (3.6 MB) took 5.09 ms to hash with BLAKE2b and 0.82 ms with CRC-32 (ADR 0140), and before
# that SHA-256 took ~22 ms at 50,000 (ADR 0094). No record outlives the process, so none
# carries an older checksum.
_CHECKSUM = struct.Struct("<I")
_CHECKSUM_SIZE = _CHECKSUM.size


def _record_digest(body: Any) -> bytes:
    return _CHECKSUM.pack(zlib.crc32(body))

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
# frame, step_index, body_count, display_count, stride, availability_mask
_BODY = struct.Struct("<qqQQQQ")


# ---------------------------------------------------------------------------
# Errors --- every malformed input fails visibly rather than degrading silently.
# ---------------------------------------------------------------------------
class ScrubCacheError(Exception):
    """Base class for every scrub-cache refusal."""


class CodecError(ScrubCacheError):
    """A record's framing, identity, or checksum is wrong."""


class SchemaError(ScrubCacheError):
    """A record or append input violates the v1 field schema."""


class CapacityError(ScrubCacheError):
    """The next record would exceed the byte budget; the cache never evicts."""


class SequenceError(ScrubCacheError):
    """An append is not the dense ``frontier + 1`` frame."""


class LifecycleError(ScrubCacheError):
    """The cache has been closed or discarded."""


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheRecord:
    """The receipt an ``append`` returns: which frame, and how many bytes it cost."""

    frame: int
    byte_length: int


@dataclass(frozen=True)
class DecodedRecord:
    """A decoded record's metadata plus the retained known channels it carried.

    ``optional_payloads`` carries the raw bytes of every structurally valid
    optional-unknown tag in the record (tags outside the schema's known set,
    see ``_validate_directory``), keyed by tag id --- populated in the same
    decode pass rather than silently dropped, so a schema-unaware payload
    can still be recovered by a caller that knows its tag id. It never contains
    a known-channel tag.
    """

    frame: int
    step_index: int
    body_count: int
    display_count: int
    stride: int
    availability_mask: int
    channels: dict[int, np.ndarray]
    optional_payloads: Mapping[int, bytes]

    @property
    def available(self) -> frozenset[int]:
        return frozenset(self.channels)

    @property
    def positions(self) -> np.ndarray | None:
        return self.channels.get(TAG_POSITION)

    @property
    def stable_ids(self) -> np.ndarray | None:
        return self.channels.get(TAG_STABLE_ID)

    @property
    def velocities(self) -> np.ndarray | None:
        return self.channels.get(TAG_VELOCITY)

    @property
    def sizes(self) -> np.ndarray | None:
        return self.channels.get(TAG_SIZE)

    @property
    def color_carriage(self) -> Any:
        """Strictly decode the optional raw Color By sources carried by this record."""

        from .color_carriage import extract_carriage

        return extract_carriage(
            self.optional_payloads, display_count=self.display_count, strict=True
        )

    @property
    def densities(self) -> np.ndarray | None:
        """The per-particle mass density the producing engine published, or `None`.

        The channel the Density look ramps (`display.ramp_scalars`). `None` is a fact about
        the producer --- the gravity-only engines have no density --- not an error, so the look
        says it draws flat rather than inventing a field.
        """

        from .color_carriage import TAG_DENSITY

        if TAG_DENSITY not in self.optional_payloads:
            return None
        return self.color_carriage.densities

    @property
    def damage(self) -> np.ndarray | None:
        """The per-particle damage the producing engine published, or `None` (ADR 0236).

        Guarded by tag presence like ``densities``: a record that never carried damage answers
        `None` without decoding its colour carriage, so a malformed carriage cannot abort a
        render that never asked for damage.
        """

        from .color_carriage import TAG_SOLID_DAMAGE

        if TAG_SOLID_DAMAGE not in self.optional_payloads:
            return None
        return self.color_carriage.solid_damage

    @property
    def internal_energies(self) -> np.ndarray | None:
        """The per-particle specific internal energy published, or `None` (ADR 0236).

        Guarded by tag presence, as ``damage`` and ``densities`` are.
        """

        from .color_carriage import TAG_SPECIFIC_INTERNAL_ENERGY

        if TAG_SPECIFIC_INTERNAL_ENERGY not in self.optional_payloads:
            return None
        return self.color_carriage.specific_internal_energy


@dataclass(frozen=True)
class FrozenDisplaySample:
    """Immutable retained data --- not the mutable host buffer behind ``DisplaySample``.

    Every channel is a read-only array; Speed is derived from the committed
    velocity on demand and is never a stored channel. ``optional_payloads``
    surfaces every schema-unaware extra-tag blob the underlying record carried,
    keyed by tag id; it is an empty dict when the record carried none. The
    mapping itself is read-only, matching the channel arrays.

    ``colors`` is not part of the record and is never sealed: it is the display
    stage's device-ramped colour (ADR 0217), attached only to the sample a commit
    publishes so the viewport skips its host ramp. A frame read back from the cache
    has none, and the viewport derives its colours as before.
    """

    step_index: int
    body_count: int
    display_count: int
    stride: int
    positions: np.ndarray
    stable_ids: np.ndarray
    velocities: np.ndarray
    sizes: np.ndarray
    availability_mask: int
    available: frozenset[int]
    optional_payloads: Mapping[int, bytes]
    colors: np.ndarray | None = None

    @property
    def speed(self) -> np.ndarray:
        speed = np.linalg.norm(self.velocities, axis=1).astype("<f4")
        speed.flags.writeable = False
        return speed

    @property
    def color_carriage(self) -> Any:
        """Strictly decode immutable raw Color By sources; never cache a selected value."""

        from .color_carriage import extract_carriage

        return extract_carriage(
            self.optional_payloads, display_count=self.display_count, strict=True
        )

    @property
    def densities(self) -> np.ndarray | None:
        """The per-particle mass density the producing engine published, or `None`.

        The channel the Density look ramps (`display.ramp_scalars`). `None` is a fact about
        the producer --- the gravity-only engines have no density --- not an error, so the look
        says it draws flat rather than inventing a field.
        """

        from .color_carriage import TAG_DENSITY

        if TAG_DENSITY not in self.optional_payloads:
            return None
        return self.color_carriage.densities

    @property
    def damage(self) -> np.ndarray | None:
        """The per-particle damage the producing engine published, or `None` (ADR 0236).

        Guarded by tag presence like ``densities``: a record that never carried damage answers
        `None` without decoding its colour carriage, so a malformed carriage cannot abort a
        render that never asked for damage.
        """

        from .color_carriage import TAG_SOLID_DAMAGE

        if TAG_SOLID_DAMAGE not in self.optional_payloads:
            return None
        return self.color_carriage.solid_damage

    @property
    def internal_energies(self) -> np.ndarray | None:
        """The per-particle specific internal energy published, or `None` (ADR 0236).

        Guarded by tag presence, as ``damage`` and ``densities`` are.
        """

        from .color_carriage import TAG_SPECIFIC_INTERNAL_ENERGY

        if TAG_SPECIFIC_INTERNAL_ENERGY not in self.optional_payloads:
            return None
        return self.color_carriage.specific_internal_energy


@dataclass(frozen=True)
class CacheAppendResult(CacheRecord):
    """Receipt plus the immutable frame backed by the exact sealed cache bytes.

    The subtype preserves the original ``CacheRecord`` interface while giving the writer
    the publication-ready sample it already constructed. Historical callers still use
    :meth:`ScrubCache.read_exact`, which independently verifies the stored checksum.
    """

    sample: FrozenDisplaySample


@dataclass(frozen=True)
class _EncodedEntry:
    tag: int
    code: int
    flags: int
    bit: int
    shape: tuple[int, ...]
    offset: int
    length: int
    array: np.ndarray


@dataclass(frozen=True)
class _SealedRecord:
    blob: bytes
    payload_start: int
    entries: tuple[_EncodedEntry, ...]
    frame: int
    step_index: int
    body_count: int
    display_count: int
    stride: int
    availability_mask: int


def _freeze(array: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(array)
    if frozen is array:
        frozen = array.copy()
    frozen.flags.writeable = False
    return frozen


# ---------------------------------------------------------------------------
# Input validation (append/encode)
# ---------------------------------------------------------------------------
def _validate_sample(sample: DisplaySample, stable_ids: Any) -> None:
    dc = int(sample.display_count)
    if dc < 0:
        # Zero rows are a real frame, not a malformed one. ADR 0046 compacts the display
        # lane over ``active``, and ADR 0047's emitter reaches a population where nothing
        # is live yet on purpose: an Emitter authored to release from frame 3 has every
        # slot inactive at frames 0..2. Every shape check below reads ``dc``, so a zero-row
        # record is self-consistent by the same rules a full one is -- four channels of
        # zero length at offset 0, a payload of zero bytes, and a header that says so. The
        # codec never needed widening; only this refusal did.
        raise SchemaError("display_count must not be negative")
    if sample.positions is None:
        raise SchemaError("record requires a positions channel")
    if sample.velocities is None:
        raise SchemaError("record requires a velocities channel")
    if sample.sizes is None:
        raise SchemaError("record requires a sizes channel")
    ids = np.asarray(stable_ids)
    if ids.dtype != np.dtype("<u8") and ids.dtype != np.dtype(np.uint64):
        raise SchemaError(f"stable_ids must be uint64, got {ids.dtype}")
    if ids.shape != (dc,):
        raise SchemaError(f"stable_ids length {ids.shape} != display_count {dc}")
    if np.asarray(sample.positions).shape != (dc, 3):
        raise SchemaError("positions must have shape [display_count, 3]")
    if np.asarray(sample.velocities).shape != (dc, 3):
        raise SchemaError("velocities must have shape [display_count, 3]")
    if np.asarray(sample.sizes).shape != (dc,):
        raise SchemaError("sizes must have shape [display_count]")


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------
def _channel_arrays(sample: DisplaySample, stable_ids: Any) -> dict[int, np.ndarray]:
    dc = int(sample.display_count)
    return {
        TAG_POSITION: np.ascontiguousarray(sample.positions, dtype="<f4").reshape(dc, 3),
        TAG_STABLE_ID: np.ascontiguousarray(stable_ids, dtype="<u8").reshape(dc),
        TAG_VELOCITY: np.ascontiguousarray(sample.velocities, dtype="<f4").reshape(dc, 3),
        TAG_SIZE: np.ascontiguousarray(sample.sizes, dtype="<f4").reshape(dc),
    }


def _carriage_tags(
    *,
    display_count: int,
    extra_tags: Any,
    solid_damage: Any,
    material_ids: Any,
    specific_internal_energy: Any,
    color_provenance: Any,
    densities: Any = None,
) -> Any:
    """Compose typed Color By carriage with the existing generic optional-tag seam.

    The lazy import is intentional: ``color_carriage`` builds on this module's v1
    optional-tag codec, so importing it at module load would reverse that dependency.
    Keeping ``None`` unchanged also preserves the exact legacy encode path.
    """

    has_carriage = any(
        value is not None
        for value in (
            solid_damage,
            material_ids,
            specific_internal_energy,
            color_provenance,
            densities,
        )
    )
    if not has_carriage:
        return extra_tags

    from .color_carriage import carriage_extra_tags

    existing = list(extra_tags) if extra_tags is not None else []
    existing.extend(
        carriage_extra_tags(
            display_count=display_count,
            solid_damage=solid_damage,
            material_ids=material_ids,
            specific_internal_energy=specific_internal_energy,
            provenance=color_provenance,
            densities=densities,
        )
    )
    return existing


def encode_record(
    *,
    context_id: str,
    descriptor_fingerprint: str,
    frame: int,
    sample: DisplaySample,
    stable_ids: Any,
    schema: Any = ScrubSampleSchemaV1,
    extra_tags: Any = None,
    corrupt_mask_extra_bit: bool = False,
    override_display_count: int | None = None,
    corrupt_optional_shape_rows: int | None = None,
) -> bytes:
    """Encode one immutable, self-describing, checksummed record.

    ``extra_tags``, ``corrupt_mask_extra_bit``, ``override_display_count`` and
    ``corrupt_optional_shape_rows`` exist so reviewers and tests can craft
    optional-unknown, mask-disagreement, header-lie and malformed-shape records;
    production callers pass none of them and get exactly the four required v1
    channels with a self-consistent header.
    """

    return _seal_record(
        context_id=context_id,
        descriptor_fingerprint=descriptor_fingerprint,
        frame=frame,
        sample=sample,
        stable_ids=stable_ids,
        schema=schema,
        extra_tags=extra_tags,
        corrupt_mask_extra_bit=corrupt_mask_extra_bit,
        override_display_count=override_display_count,
        corrupt_optional_shape_rows=corrupt_optional_shape_rows,
    ).blob


def _seal_record(
    *,
    context_id: str,
    descriptor_fingerprint: str,
    frame: int,
    sample: DisplaySample,
    stable_ids: Any,
    schema: Any,
    extra_tags: Any = None,
    corrupt_mask_extra_bit: bool = False,
    override_display_count: int | None = None,
    corrupt_optional_shape_rows: int | None = None,
) -> _SealedRecord:
    """Build one checksummed immutable record without full-payload intermediates."""

    _validate_sample(sample, stable_ids)
    known = _channel_arrays(sample, stable_ids)
    raw_entries: list[tuple[int, int, int, int, tuple[int, ...], np.ndarray]] = []
    mask = 0
    for spec in schema.channels:
        array = known[spec.tag]
        bit = schema.bit_for(spec.tag)
        raw_entries.append((spec.tag, _DTYPE_CODES[spec.dtype], 0, bit, tuple(array.shape), array))
        mask |= 1 << bit

    next_bit = 4
    for tag, flags, raw in extra_tags or []:
        array = np.ascontiguousarray(raw)
        code = _DTYPE_CODES.get(array.dtype.str)
        if code is None:
            array = np.ascontiguousarray(array, dtype="<f4")
            code = _DTYPE_CODES["<f4"]
        bit = next_bit
        next_bit += 1
        raw_entries.append((int(tag), code, int(flags), bit, tuple(array.shape), array))
        mask |= 1 << bit

    if corrupt_optional_shape_rows is not None and raw_entries:
        tag, code, flags, bit, shape, array = raw_entries[-1]
        broken = (int(corrupt_optional_shape_rows),) + tuple(shape[1:])
        raw_entries[-1] = (tag, code, flags, bit, broken, array)
    if corrupt_mask_extra_bit:
        mask |= 1 << 40

    payload_size = sum(int(array.nbytes) for *_, array in raw_entries)
    entries: list[_EncodedEntry] = []
    directory = bytearray(_U16.pack(len(raw_entries)))
    payload_offset = 0
    for tag, code, flags, bit, shape, array in raw_entries:
        length = int(array.nbytes)
        entries.append(
            _EncodedEntry(
                tag=tag,
                code=code,
                flags=flags,
                bit=bit,
                shape=shape,
                offset=payload_offset,
                length=length,
                array=array,
            )
        )
        directory += _U16.pack(tag)
        directory += _U8.pack(code)
        directory += _U8.pack(flags)
        directory += _U8.pack(bit)
        directory += _U8.pack(len(shape))
        for dim in shape:
            directory += _U32.pack(int(dim))
        directory += _U64.pack(payload_offset)
        directory += _U64.pack(length)
        payload_offset += length

    ctx = context_id.encode("utf-8")
    fingerprint = descriptor_fingerprint.encode("utf-8")
    declared_display_count = int(
        sample.display_count if override_display_count is None else override_display_count
    )
    header = bytearray()
    header += _MAGIC
    header += _U16.pack(schema.version)
    header += _U16.pack(len(ctx))
    header += ctx
    header += _U16.pack(len(fingerprint))
    header += fingerprint
    header += _BODY.pack(
        int(frame),
        int(sample.step_index),
        int(sample.body_count),
        declared_display_count,
        int(sample.stride),
        int(mask),
    )
    header += directory
    header += _U64.pack(payload_size)
    payload_start = len(header)

    # Header, then each channel's bytes in entry order (offsets are cumulative lengths, a
    # zero-length channel adds nothing). Hashed chunk by chunk and joined once into
    # immutable bytes: no zeroed scratch buffer, no second freezing copy (ADR 0222).
    chunks: list[Any] = [header]
    chunks.extend(memoryview(entry.array).cast("B") for entry in entries if entry.length)
    crc = 0
    for chunk in chunks:
        crc = zlib.crc32(chunk, crc)
    checksum = _CHECKSUM.pack(crc)
    blob = b"".join((*chunks, checksum))
    if len(blob) != payload_start + payload_size + _CHECKSUM_SIZE:
        raise CodecError("record checksum seal failed")
    return _SealedRecord(
        blob=blob,
        payload_start=payload_start,
        entries=tuple(entries),
        frame=int(frame),
        step_index=int(sample.step_index),
        body_count=int(sample.body_count),
        display_count=declared_display_count,
        stride=int(sample.stride),
        availability_mask=int(mask),
    )


def _frozen_from_sealed_record(
    sealed: _SealedRecord, *, schema: Any = ScrubSampleSchemaV1
) -> FrozenDisplaySample:
    """Expose read-only zero-copy channel views into bytes sealed by :func:`_seal_record`."""

    directory = [
        {
            "tag": entry.tag,
            "code": entry.code,
            "flags": entry.flags,
            "bit": entry.bit,
            "shape": entry.shape,
            "offset": entry.offset,
            "length": entry.length,
        }
        for entry in sealed.entries
    ]
    payload_len = len(sealed.blob) - sealed.payload_start - _CHECKSUM_SIZE
    _validate_directory(directory, sealed.availability_mask, payload_len, schema)
    _validate_color_directory(directory, sealed.display_count)

    channels: dict[int, np.ndarray] = {}
    optional_payloads: dict[int, bytes] = {}
    known_tags = set(schema.known_tags())
    for entry in sealed.entries:
        start = sealed.payload_start + entry.offset
        if entry.tag not in known_tags:
            optional_payloads[entry.tag] = sealed.blob[start : start + entry.length]
            continue
        spec = schema.spec_for(entry.tag)
        dtype = _CODE_DTYPES.get(entry.code)
        if dtype is None or dtype != spec.dtype:
            raise SchemaError(f"tag {entry.tag} has the wrong dtype")
        expected_ndim = 2 if spec.columns > 1 else 1
        if len(entry.shape) != expected_ndim:
            raise SchemaError(f"tag {entry.tag} has the wrong rank")
        if spec.columns > 1 and entry.shape[1] != spec.columns:
            raise SchemaError(f"tag {entry.tag} has the wrong column count")
        if entry.shape[0] != sealed.display_count:
            raise SchemaError(
                f"tag {entry.tag} has {entry.shape[0]} rows but header declares "
                f"{sealed.display_count}"
            )
        array = np.frombuffer(
            sealed.blob,
            dtype=spec.dtype,
            count=entry.length // np.dtype(spec.dtype).itemsize,
            offset=start,
        ).reshape(entry.shape)
        array.flags.writeable = False
        channels[entry.tag] = array
    for spec in schema.channels:
        if spec.tag not in channels:
            raise SchemaError(f"record is missing required tag {spec.tag}")
    _validate_color_payloads(optional_payloads, sealed.display_count)
    return FrozenDisplaySample(
        step_index=sealed.step_index,
        body_count=sealed.body_count,
        display_count=sealed.display_count,
        stride=sealed.stride,
        positions=channels[TAG_POSITION],
        stable_ids=channels[TAG_STABLE_ID],
        velocities=channels[TAG_VELOCITY],
        sizes=channels[TAG_SIZE],
        availability_mask=sealed.availability_mask,
        available=frozenset(channels),
        optional_payloads=MappingProxyType(optional_payloads),
    )


class _Reader:
    def __init__(self, buffer: bytes) -> None:
        self._buffer = buffer
        self.pos = 0

    def take(self, count: int) -> bytes:
        end = self.pos + count
        if end > len(self._buffer):
            raise CodecError("record is truncated")
        chunk = self._buffer[self.pos : end]
        self.pos = end
        return chunk

    def u8(self) -> int:
        return _U8.unpack(self.take(1))[0]

    def u16(self) -> int:
        return _U16.unpack(self.take(2))[0]

    def u32(self) -> int:
        return _U32.unpack(self.take(4))[0]

    def u64(self) -> int:
        return _U64.unpack(self.take(8))[0]


def decode_record(
    blob: bytes,
    *,
    context_id: str,
    descriptor_fingerprint: str,
    schema: Any = ScrubSampleSchemaV1,
    expected_version: int | None = None,
) -> DecodedRecord:
    """Decode and fully validate one record, or refuse it visibly."""

    if len(blob) < _CHECKSUM_SIZE:
        raise CodecError("record is shorter than its checksum")
    body = blob[:-_CHECKSUM_SIZE]
    if _record_digest(body) != blob[-_CHECKSUM_SIZE:]:
        raise CodecError("record checksum mismatch")

    reader = _Reader(body)
    if reader.take(len(_MAGIC)) != _MAGIC:
        raise CodecError("bad record magic")

    version = reader.u16()
    target = schema.version if expected_version is None else int(expected_version)
    if version != target:
        raise CodecError(f"unknown record version {version}")

    ctx = reader.take(reader.u16()).decode("utf-8", "replace")
    if ctx != context_id:
        raise CodecError("record context id mismatch")
    fingerprint = reader.take(reader.u16()).decode("utf-8", "replace")
    if fingerprint != descriptor_fingerprint:
        raise CodecError("record descriptor fingerprint mismatch")

    frame, step_index, body_count, display_count, stride, mask = _BODY.unpack(
        reader.take(_BODY.size)
    )

    # Field directory.
    entry_count = reader.u16()
    raw_entries: list[dict[str, Any]] = []
    for _ in range(entry_count):
        tag = reader.u16()
        code = reader.u8()
        flags = reader.u8()
        mask_bit = reader.u8()
        ndim = reader.u8()
        shape = tuple(reader.u32() for _ in range(ndim))
        offset = reader.u64()
        length = reader.u64()
        raw_entries.append(
            {
                "tag": tag,
                "code": code,
                "flags": flags,
                "bit": mask_bit,
                "shape": shape,
                "offset": offset,
                "length": length,
            }
        )

    payload_len = reader.u64()
    payload = reader.take(payload_len)
    if reader.pos != len(body):
        raise CodecError("record has trailing bytes")

    _validate_directory(raw_entries, mask, payload_len, schema)
    _validate_color_directory(raw_entries, int(display_count))

    channels: dict[int, np.ndarray] = {}
    optional_payloads: dict[int, bytes] = {}
    known_tags = set(schema.known_tags())
    for entry in raw_entries:
        tag = entry["tag"]
        if tag not in known_tags:
            # A structurally valid unknown tag is skipped as a *channel* only
            # when explicitly optional (enforced by ``_validate_directory``
            # above); it stays unavailable and cannot alias a known tag. Its
            # raw bytes are still captured here, in the same decode pass,
            # rather than dropped, so a schema-unaware payload carried via
            # ``extra_tags`` can be recovered by tag id.
            offset = entry["offset"]
            length = entry["length"]
            optional_payloads[tag] = payload[offset : offset + length]
            continue
        spec = schema.spec_for(tag)
        channels[tag] = _decode_channel(entry, payload, spec)
        # The header's display_count is authoritative; every known channel must
        # carry exactly that many rows, or the record's shape contradicts itself.
        if channels[tag].shape[0] != display_count:
            raise SchemaError(
                f"tag {tag} has {channels[tag].shape[0]} rows but header declares {display_count}"
            )

    for spec in schema.channels:
        if spec.tag not in channels:
            raise SchemaError(f"record is missing required tag {spec.tag}")
    _validate_color_payloads(optional_payloads, int(display_count))

    return DecodedRecord(
        frame=int(frame),
        step_index=int(step_index),
        body_count=int(body_count),
        display_count=int(display_count),
        stride=int(stride),
        availability_mask=int(mask),
        channels=channels,
        optional_payloads=optional_payloads,
    )


def _validate_directory(
    entries: list[dict[str, Any]], mask: int, payload_len: int, schema: Any
) -> None:
    seen_tags: set[int] = set()
    intervals: list[tuple[int, int]] = []
    entry_bits: set[int] = set()
    known_tags = set(schema.known_tags())

    for entry in entries:
        tag = entry["tag"]
        if tag in seen_tags:
            raise SchemaError(f"duplicate tag {tag} in field directory")
        seen_tags.add(tag)

        offset = entry["offset"]
        length = entry["length"]
        if offset < 0 or length < 0 or offset + length > payload_len:
            raise SchemaError(f"tag {tag} bounds fall outside the payload")
        intervals.append((offset, offset + length))

        # Every entry -- known or optional-unknown, before any decode or skip --
        # must be structurally self-consistent: a resolvable dtype, at least one
        # dimension, and a declared shape whose element count exactly accounts for
        # its byte length. This refuses malformed optional entries that would
        # otherwise be silently skipped.
        code_dtype = _CODE_DTYPES.get(entry["code"])
        if code_dtype is None:
            raise SchemaError(f"tag {tag} has an unknown dtype code {entry['code']}")
        shape = entry["shape"]
        if len(shape) < 1:
            raise SchemaError(f"tag {tag} has no dimensions")
        rows = 1
        for dim in shape:
            rows *= int(dim)
        if rows * np.dtype(code_dtype).itemsize != length:
            raise SchemaError(f"tag {tag} shape does not account for its byte length")

        bit = entry["bit"]
        if bit in entry_bits:
            raise SchemaError(f"two directory entries claim mask bit {bit}")
        entry_bits.add(bit)
        if not (mask >> bit) & 1:
            raise SchemaError(f"tag {tag} is not marked in the availability mask")

        if tag not in known_tags and not (entry["flags"] & TAG_OPTIONAL_FLAG):
            raise SchemaError(f"unknown required tag {tag}")

    intervals.sort()
    for (_, a_end), (b_start, _) in zip(intervals, intervals[1:], strict=False):
        if b_start < a_end:
            raise SchemaError("field directory ranges overlap")

    mask_bits = {bit for bit in range(mask.bit_length()) if (mask >> bit) & 1}
    if mask_bits != entry_bits:
        raise SchemaError("availability mask and field directory disagree")


def _validate_color_directory(entries: list[dict[str, Any]], display_count: int) -> None:
    """Validate reserved D13 optional tags without widening the required v1 schema.

    Arbitrary optional tags retain v1's forward-compatible behavior. Tags 6--9 are no
    longer arbitrary, however: accepting a two-dimensional Damage array or float Material
    IDs would decode plausible bytes against the wrong particle/source contract.
    """

    from .color_carriage import (
        PROVENANCE_WIRE_CEILING_BYTES,
        TAG_COLOR_PROVENANCE,
        TAG_DENSITY,
        TAG_MATERIAL_ID,
        TAG_SOLID_DAMAGE,
        TAG_SPECIFIC_INTERNAL_ENERGY,
    )

    per_sample_codes = {
        TAG_SOLID_DAMAGE: _DTYPE_CODES["<f4"],
        TAG_MATERIAL_ID: _DTYPE_CODES["<i8"],
        TAG_SPECIFIC_INTERNAL_ENERGY: _DTYPE_CODES["<f4"],
        TAG_DENSITY: _DTYPE_CODES["<f4"],
    }
    for entry in entries:
        tag = int(entry["tag"])
        if tag in per_sample_codes:
            if entry["code"] != per_sample_codes[tag]:
                raise SchemaError(f"Color By tag {tag} has the wrong dtype")
            if tuple(entry["shape"]) != (display_count,):
                raise SchemaError(
                    f"Color By tag {tag} must be one-dimensional with {display_count} samples"
                )
        elif tag == TAG_COLOR_PROVENANCE:
            if entry["code"] != _DTYPE_CODES["<u8"]:
                raise SchemaError("Color By provenance tag has the wrong dtype")
            if len(entry["shape"]) != 1:
                raise SchemaError("Color By provenance tag must be one-dimensional")
            if int(entry["length"]) > PROVENANCE_WIRE_CEILING_BYTES:
                raise SchemaError("Color By provenance tag exceeds its 8200-byte wire ceiling")


def _validate_color_payloads(payloads: Mapping[int, bytes], display_count: int) -> None:
    """Parse reserved Color By payloads now so a ScrubCache never retains malformed state."""

    from .color_carriage import extract_carriage

    carriage = extract_carriage(payloads, display_count=display_count, strict=True)
    material_ids = carriage.material_ids
    if material_ids is not None and material_ids.size and int(material_ids.min()) < 0:
        raise SchemaError("Color By material IDs must be persistent non-negative identifiers")


def _decode_channel(entry: dict[str, Any], payload: bytes, spec: ScrubChannelSpec) -> np.ndarray:
    dtype = _CODE_DTYPES.get(entry["code"])
    if dtype is None or dtype != spec.dtype:
        raise SchemaError(f"tag {entry['tag']} has the wrong dtype")
    expected_ndim = 2 if spec.columns > 1 else 1
    if len(entry["shape"]) != expected_ndim:
        raise SchemaError(f"tag {entry['tag']} has the wrong rank")
    if spec.columns > 1 and entry["shape"][1] != spec.columns:
        raise SchemaError(f"tag {entry['tag']} has the wrong column count")
    raw = payload[entry["offset"] : entry["offset"] + entry["length"]]
    array = np.frombuffer(raw, dtype=dtype)
    if array.nbytes != entry["length"]:
        raise SchemaError(f"tag {entry['tag']} byte length is inconsistent")
    return array.reshape(entry["shape"])


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
class ScrubCache:
    """A dense, append-only, process-memory prefix of authored frames."""

    def __init__(
        self,
        context_id: str,
        descriptor_fingerprint: str,
        frame_start: int,
        capacity_bytes: int,
        sample_schema: Any,
    ) -> None:
        self._context_id = context_id
        self._fingerprint = descriptor_fingerprint
        self._frame_start = int(frame_start)
        self._capacity = int(capacity_bytes)
        self._schema = sample_schema
        self._records: dict[int, bytes] = {}
        self._order: list[int] = []
        self._bytes = 0
        self._alive = True
        self._open = True

    @classmethod
    def create(
        cls,
        *,
        context_id: str,
        descriptor_fingerprint: str,
        frame_start: int,
        capacity_bytes: int,
        sample_schema: Any,
    ) -> ScrubCache:
        return cls(context_id, descriptor_fingerprint, frame_start, capacity_bytes, sample_schema)

    # -- scalar properties --------------------------------------------------
    @property
    def frame_start(self) -> int:
        return self._frame_start

    @property
    def frontier(self) -> int | None:
        return self._order[-1] if self._order else None

    @property
    def record_count(self) -> int:
        return len(self._order)

    @property
    def held_frames(self) -> tuple[int, ...]:
        """Every frame currently held, in frame order.

        ``frontier`` answers "how far has this been solved", which is what playback and the
        HUD ask. A caller that needs a *history* rather than a single frame --- the trail
        curve a render writes (ADR 0043) --- needs the frames themselves, and probing for
        them with ``read_exact`` and a caught ``KeyError`` would dress an ordinary answer up
        as an error path. Sorted rather than in append order: append order is ascending
        today, but the contract callers want is the timeline's, not the writer's.
        """

        return tuple(sorted(self._order))

    @property
    def bytes_used(self) -> int:
        return self._bytes

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def admission_record_ceiling(self) -> int:
        """Conservative widest admitted record, including all raw Color By carriage."""

        from .color_carriage import ScrubSampleSchemaV1WithCarriage

        return int(ScrubSampleSchemaV1WithCarriage.max_record_bytes)

    @property
    def capacity_forecast(self) -> Any:
        """Conservative frame forecast used for admission messaging and planning."""

        from .color_carriage import widened_capacity_forecast

        return widened_capacity_forecast(self._capacity)

    @property
    def descriptor_fingerprint(self) -> str:
        return self._fingerprint

    # -- operations ---------------------------------------------------------
    def append(
        self,
        frame: int,
        sample: DisplaySample,
        stable_ids: Any,
        *,
        extra_tags: Any = None,
        solid_damage: Any = None,
        material_ids: Any = None,
        specific_internal_energy: Any = None,
        color_provenance: Any = None,
        densities: Any = None,
    ) -> CacheAppendResult:
        """Append one record and return its immutable publication-ready sample.

        ``extra_tags`` passes through the same sealed-record encoder as ``encode_record`` (a
        sequence of ``(tag, flags, raw)`` triples, where ``raw`` is array-like
        of a wire-supported dtype); omitted or ``None``
        preserves the exact prior four-required-channel behavior. The four typed Color By
        arguments add only authoritative raw sources and partition provenance through that
        same optional lane. They never store a selected mode, scalar, or RGBA value.
        """

        self._require_alive()
        if not self._open:
            raise LifecycleError("cannot append to a closed scrub cache")

        expected = self._frame_start if not self._order else self._order[-1] + 1
        if int(frame) != expected:
            raise SequenceError(f"append must be frame {expected}, got {frame}")

        # Validate before encoding so a bad input leaves index and frontier intact.
        _validate_sample(sample, stable_ids)
        combined_extra_tags = _carriage_tags(
            display_count=int(sample.display_count),
            extra_tags=extra_tags,
            solid_damage=solid_damage,
            material_ids=material_ids,
            specific_internal_energy=specific_internal_energy,
            color_provenance=color_provenance,
            densities=densities,
        )
        sealed = _seal_record(
            context_id=self._context_id,
            descriptor_fingerprint=self._fingerprint,
            frame=int(frame),
            sample=sample,
            stable_ids=stable_ids,
            schema=self._schema,
            extra_tags=combined_extra_tags,
        )

        # Preflight capacity: the cache never evicts, so a full cache keeps the
        # prior frontier rather than dropping the oldest record. The caller shows the
        # refused frame anyway and plays on without keeping it (ADR 0095).
        blob = sealed.blob
        if self._bytes + len(blob) > self._capacity:
            raise CapacityError(
                f"record {frame} needs {len(blob)} bytes; {self._bytes}/{self._capacity} used"
            )

        # Construct the zero-copy view before mutation too: a schema or ownership defect
        # must leave the dense prefix untouched just like validation/capacity failures do.
        frozen = _frozen_from_sealed_record(sealed, schema=self._schema)
        frame_key = int(frame)
        result = CacheAppendResult(
            frame=frame_key,
            byte_length=len(blob),
            sample=frozen,
        )

        # Atomic commit: both immutable bytes and their publication view exist before the
        # index advertises them. The two collection mutations still need rollback as a unit:
        # allocation failure in ``_order.append`` must not leave a hidden readable record
        # outside the public dense-prefix counters.
        prior_order_length = len(self._order)
        prior_bytes = self._bytes
        try:
            self._records[frame_key] = blob
            self._order.append(frame_key)
            self._bytes = prior_bytes + len(blob)
        except BaseException:
            self._records.pop(frame_key, None)
            if len(self._order) > prior_order_length:
                del self._order[prior_order_length:]
            self._bytes = prior_bytes
            raise
        return result

    def read_exact(self, frame: int) -> FrozenDisplaySample:
        self._require_alive()
        blob = self._records.get(int(frame))
        if blob is None:
            raise KeyError(f"frame {frame} is not in the cached prefix")
        record = decode_record(
            blob,
            context_id=self._context_id,
            descriptor_fingerprint=self._fingerprint,
            schema=self._schema,
        )
        return FrozenDisplaySample(
            step_index=record.step_index,
            body_count=record.body_count,
            display_count=record.display_count,
            stride=record.stride,
            positions=_freeze(record.channels[TAG_POSITION]),
            stable_ids=_freeze(record.channels[TAG_STABLE_ID]),
            velocities=_freeze(record.channels[TAG_VELOCITY]),
            sizes=_freeze(record.channels[TAG_SIZE]),
            availability_mask=record.availability_mask,
            available=record.available,
            optional_payloads=MappingProxyType(dict(record.optional_payloads)),
        )

    def truncate_from(self, frame: int) -> None:
        self._require_alive()
        cut = int(frame)
        removed = [existing for existing in self._order if existing >= cut]
        for existing in removed:
            self._bytes -= len(self._records.pop(existing))
        self._order = [existing for existing in self._order if existing < cut]

    def close(self) -> None:
        self._open = False

    def discard(self) -> None:
        self._records.clear()
        self._order.clear()
        self._bytes = 0
        self._alive = False
        self._open = False

    def _require_alive(self) -> None:
        if not self._alive:
            raise LifecycleError("scrub cache has been discarded")
