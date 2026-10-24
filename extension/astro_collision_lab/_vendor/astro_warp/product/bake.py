"""Immutable bake revision manifests."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cache import (
    SCHEMA_V2,
    BakeCacheError,
    BinaryBakeWriter,
    decode_frames,
    file_sha256,
    filesystem_path,
    logical_path,
    payload_sha256,
)
from .color_bake import (
    ColorBakeError,
    ColorBakeFrame,
    carry_bake_frame,
    merge_source_note,
)
from .color_carriage import ColorProvenance

# A bake's payload encoding is chosen by ``compatible_schema``. ``astro-warp-bake-v1`` is
# the original canonical-JSON payload, kept as a reader so pre-cache bakes stay verifiable
# (ADR 0025 validation-impact). ``astro-warp-bake-v2`` is the chunked typed binary payload
# STAB-005 ships for production-scale bakes; the product runtime writes it.
_BINARY_SCHEMA = SCHEMA_V2


@dataclass(frozen=True)
class BakeRevision:
    bake_id: str
    name: str
    context_id: str
    graph_hash: str
    created_at_utc: str
    frame_start: int
    frame_end: int
    backend: str
    integrator: str
    dt: float
    substeps: int
    stable_ids: tuple[int, ...]
    units: dict[str, Any]
    channels: tuple[str, ...]
    body_count: int
    step_count: int
    payload_sha256: str
    manifest_sha256: str
    compatible_schema: str
    # Resolved cache root this revision's files were written under. Not part of the
    # content-addressed identity below (``revision_key``/``bake_id``): the same physical
    # bake written under two different roots is still the same bake, so this must never
    # mint a second revision for identical content. Defaults to "" for manifests written
    # before this field existed; ``_revision_from_dict`` derives a real value for those on
    # read rather than leaving the empty string standing in for a resolved location.
    cache_root: str = ""
    validity: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def write_bake_revision(
    *,
    directory: Path,
    name: str,
    context_id: str,
    graph_hash: str,
    frame_start: int,
    backend: str,
    integrator: str,
    dt: float,
    substeps: int,
    stable_ids: list[int],
    units: dict[str, Any],
    samples: Iterable[dict[str, Any] | ColorBakeFrame],
    compatible_schema: str,
    provenance: dict[str, Any] | None = None,
    color_provenance: ColorProvenance | None = None,
    observations: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    cache_root: str = "",
) -> tuple[BakeRevision, Path]:
    """Write one immutable revision.

    ``samples`` may be any iterable of snapshots and is consumed exactly once: a bake at
    the acceptance scale is streamed to disk frame by frame rather than materialized. A
    caller carrying ADR 0059 sources supplies typed Color bake frames together with one
    immutable source provenance; plain snapshot dictionaries retain the legacy path.

    ``provenance`` states what produced the revision and is part of its identity, so two
    bakes that differ only in precision lane, Warp version, or device stay separate
    revisions instead of collapsing onto one. ``observations`` records what was measured
    while baking — diagnostic scalars, timings — and is deliberately outside the identity:
    those values move at round-off between runs, and keying on them would mint a new
    revision for every re-bake of identical inputs. It may be a callable, resolved after
    the payload is written, for measurements a streaming caller only has once its last
    frame has been produced.
    """

    filesystem_path(directory).mkdir(parents=True, exist_ok=True)
    color_provenance = _immutable_color_provenance(color_provenance)
    if color_provenance is not None:
        provenance = merge_source_note(provenance, color_provenance)
    binary = compatible_schema == _BINARY_SCHEMA
    if binary:
        # ``payload_sha256`` hashes exactly the bytes written to disk, so the binary file
        # is content-addressed with no canonicalization step between disk and digest. The
        # name a content-addressed revision lands under is only known once that digest is,
        # so the payload is streamed to a staging file first and renamed into place below.
        # Streaming is what makes the acceptance scale reachable: the bake is never held
        # in memory, only the frame being written. ``BakeSession`` is the same streaming
        # writer, driven one sample per call instead of by exhausting ``samples`` here, so
        # a modal caller shares this exact staging/digest/commit path.
        session = BakeSession(
            directory=directory,
            context_id=context_id,
            stable_ids=stable_ids,
            color_provenance=color_provenance,
        )
        try:
            for sample in samples:
                session.add_sample(sample)
        except BaseException:
            session.cancel()
            raise
        return session.finalize(
            name=name,
            graph_hash=graph_hash,
            frame_start=frame_start,
            backend=backend,
            integrator=integrator,
            dt=dt,
            substeps=substeps,
            units=units,
            compatible_schema=compatible_schema,
            provenance=provenance,
            observations=observations,
            cache_root=cache_root,
        )
    materialized = [
        _prepare_color_sample(sample, color_provenance=color_provenance)
        for sample in samples
    ]
    for sample in materialized:
        _check_sample_ids(sample, stable_ids)
    payload = {
        "schema": compatible_schema,
        "context_id": context_id,
        "samples": materialized,
    }
    # v1's digest is over the compact canonical form while the file is written pretty,
    # so the reader re-canonicalizes on load. Preserved verbatim for existing bakes.
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    frame_count = len(materialized)
    body_count = len(materialized[0]["bodies"]) if materialized else 0
    return _finalize_revision(
        directory=directory,
        name=name,
        context_id=context_id,
        graph_hash=graph_hash,
        frame_start=frame_start,
        backend=backend,
        integrator=integrator,
        dt=dt,
        substeps=substeps,
        stable_ids=stable_ids,
        units=units,
        compatible_schema=compatible_schema,
        payload_hash=payload_hash,
        frame_count=frame_count,
        body_count=body_count,
        staged_path=None,
        data_bytes=data_bytes,
        provenance=provenance,
        observations=observations,
        cache_root=cache_root,
    )


class BakeSession:
    """Incrementally drivable counterpart to ``write_bake_revision``'s binary path.

    ``write_bake_revision`` exhausts ``samples`` in one call; a modal operator instead adds
    one sample per tick and finalizes once the caller decides all samples are in. Both
    share this class (``write_bake_revision`` itself is a thin loop over it), so the
    staging file, the digest, and the atomic commit are one code path either way — a modal
    bake and a scripted one land byte-identical revisions.

    ``cancel()`` gives the same "nothing committed" guarantee ``write_bake_revision``'s own
    ``except BaseException`` clause gives a failed bake: the staged file is dropped before
    it is ever named, so no reader — immutable or otherwise — can see a partial bake.
    """

    def __init__(
        self,
        *,
        directory: Path,
        context_id: str,
        stable_ids: list[int],
        color_provenance: ColorProvenance | None = None,
    ) -> None:
        logical_directory = logical_path(directory)
        self._directory = filesystem_path(logical_directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._stable_ids = stable_ids
        self._color_provenance = _immutable_color_provenance(color_provenance)
        self.staged_path = self._directory / f".incoming-{uuid.uuid4()}.frames.bin"
        self._writer = BinaryBakeWriter(self.staged_path, context_id=context_id)
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._writer.frame_count

    @property
    def first_frame_body_count(self) -> int:
        return self._writer.first_frame_body_count

    @property
    def payload_sha256(self) -> str:
        return self._writer.payload_sha256

    def add_sample(self, sample: dict[str, Any] | ColorBakeFrame) -> None:
        try:
            sample = _prepare_color_sample(
                sample, color_provenance=self._color_provenance
            )
            _check_sample_ids(sample, self._stable_ids)
            self._writer.add_sample(sample)
        except BaseException:
            self.cancel()
            raise

    def cancel(self) -> None:
        """Drop the staged bytes; a cancelled bake must never look committed."""

        if not self._closed:
            self._writer.abort()
            self._closed = True

    def finalize(
        self,
        *,
        name: str,
        graph_hash: str,
        frame_start: int,
        backend: str,
        integrator: str,
        dt: float,
        substeps: int,
        units: dict[str, Any],
        compatible_schema: str,
        provenance: dict[str, Any] | None = None,
        observations: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
        cache_root: str = "",
    ) -> tuple[BakeRevision, Path]:
        if self._color_provenance is not None:
            try:
                provenance = merge_source_note(provenance, self._color_provenance)
            except BaseException:
                self.cancel()
                raise
        try:
            self._writer.close()
        except BaseException:
            self.cancel()
            raise
        self._closed = True
        return _finalize_revision(
            directory=self._directory,
            name=name,
            context_id=self._writer.context_id,
            graph_hash=graph_hash,
            frame_start=frame_start,
            backend=backend,
            integrator=integrator,
            dt=dt,
            substeps=substeps,
            stable_ids=self._stable_ids,
            units=units,
            compatible_schema=compatible_schema,
            payload_hash=self.payload_sha256,
            frame_count=self.frame_count,
            body_count=self.first_frame_body_count,
            staged_path=self.staged_path,
            data_bytes=None,
            provenance=provenance,
            observations=observations,
            cache_root=cache_root,
        )


def _prepare_color_sample(
    sample: dict[str, Any] | ColorBakeFrame,
    *,
    color_provenance: ColorProvenance | None,
) -> dict[str, Any]:
    """Keep legacy samples exact and make the typed path impossible without provenance."""

    if isinstance(sample, ColorBakeFrame):
        if color_provenance is None:
            raise ColorBakeError(
                "a ColorBakeFrame cannot be written without Color source provenance"
            )
        return carry_bake_frame(sample, provenance=color_provenance)
    if not isinstance(sample, dict):
        raise ColorBakeError(f"bake sample must be a dict, got {type(sample).__name__}")
    if color_provenance is None:
        return sample
    return carry_bake_frame(ColorBakeFrame(sample=sample), provenance=color_provenance)


def _immutable_color_provenance(
    provenance: ColorProvenance | None,
) -> ColorProvenance | None:
    """Detach nested mutable refusal dictionaries from the caller before streaming."""

    if provenance is None:
        return None
    if not isinstance(provenance, ColorProvenance):
        raise ColorBakeError(
            f"color_provenance must be ColorProvenance, got {type(provenance).__name__}"
        )
    return ColorProvenance.from_document(provenance.as_document())


def _finalize_revision(
    *,
    directory: Path,
    name: str,
    context_id: str,
    graph_hash: str,
    frame_start: int,
    backend: str,
    integrator: str,
    dt: float,
    substeps: int,
    stable_ids: list[int],
    units: dict[str, Any],
    compatible_schema: str,
    payload_hash: str,
    frame_count: int,
    body_count: int,
    staged_path: Path | None,
    data_bytes: bytes | None,
    provenance: dict[str, Any] | None,
    observations: dict[str, Any] | Callable[[], dict[str, Any]] | None,
    cache_root: str = "",
) -> tuple[BakeRevision, Path]:
    """Digest, name, and commit one revision from an already-written payload.

    Shared tail for ``write_bake_revision``'s blocking path and ``BakeSession``'s
    incrementally-driven one, so a content-addressed name, an immutable commit, and a
    peer-collision reading exist in exactly one place.
    """

    directory = filesystem_path(directory)
    binary = compatible_schema == _BINARY_SCHEMA
    provenance = provenance or {}
    # Resolved only now: a streaming caller measures the bake's exit state as the last frame
    # goes to disk, and observations are outside the revision key, so nothing above this
    # line depends on them.
    resolved_observations = observations() if callable(observations) else (observations or {})
    revision_key = json.dumps(
        {
            "name": name,
            "context_id": context_id,
            "graph_hash": graph_hash,
            "frame_start": frame_start,
            "backend": backend,
            "integrator": integrator,
            "dt": dt,
            "substeps": substeps,
            "stable_ids": stable_ids,
            "units": units,
            "compatible_schema": compatible_schema,
            "payload_sha256": payload_hash,
            "provenance": provenance,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    bake_id = f"bake-{uuid.uuid5(uuid.NAMESPACE_URL, revision_key)}"
    data_path = directory / (f"{bake_id}.frames.bin" if binary else f"{bake_id}.samples.json")
    manifest_path = directory / f"{bake_id}.manifest.json"
    orphan = data_path.exists() and not manifest_path.exists()
    if manifest_path.exists() or (data_path.exists() and not orphan):
        # The staged payload is byte-identical to what is already committed — the digest it
        # was named by proves that — so drop it rather than rewriting an immutable file.
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        return _load_existing_revision(data_path, manifest_path, payload_hash, bake_id)

    created_at_utc = datetime.now(UTC).isoformat()
    channels = ("transforms", "points", "velocities", "stable_ids", "trails")
    manifest_data = {
        "bake_id": bake_id,
        "name": name,
        "context_id": context_id,
        "cache_root": cache_root,
        "graph_hash": graph_hash,
        "created_at_utc": created_at_utc,
        "frame_start": frame_start,
        "frame_end": frame_start + max(0, frame_count - 1),
        "backend": backend,
        "integrator": integrator,
        "dt": dt,
        "substeps": substeps,
        "stable_ids": tuple(stable_ids),
        "units": units,
        "channels": channels,
        "body_count": body_count,
        "step_count": frame_count,
        "payload_sha256": payload_hash,
        "compatible_schema": compatible_schema,
        "validity": {
            "integrity": {
                "manifest": "sha256-canonical-json-self-envelope",
                "payload": "sha256-chunked-binary-frames" if binary else "sha256-canonical-json",
            },
            "currentness": {"graph_hash": graph_hash},
            "compatibility": {
                "schema": compatible_schema,
                "channels": channels,
                # The reader a consumer needs to decode ``payload_sha256``'s bytes: the v2
                # payload is a chunked columnar little-endian binary file, the v1 payload
                # is canonical JSON. Recorded on the compatibility axis so a mismatch is a
                # compatibility failure, not a silent misread.
                "payload_encoding": "chunked-columnar-le-v1" if binary else "canonical-json",
            },
        },
        "provenance": provenance,
        "observations": resolved_observations,
    }
    revision = BakeRevision(
        **manifest_data,
        manifest_sha256=manifest_sha256(manifest_data),
    )
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        if orphan:
            # SR-05: a crash between payload rename and manifest write left a payload with
            # no manifest. The manifest is the commit marker, so adopt the orphan only if
            # its bytes are exactly this revision's payload; otherwise refuse as a collision.
            if _committed_payload_hash(data_path) != payload_hash:
                raise FileExistsError(f"immutable bake payload collision for {bake_id}")
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
        elif staged_path is not None:
            # `rename` refuses an existing destination on Windows and is atomic where the
            # platform supports it, so a peer that committed first is detected here rather
            # than overwritten.
            staged_path.rename(data_path)
        else:
            with data_path.open("xb") as handle:
                handle.write(data_bytes or b"")
        # The manifest is the commit marker: written whole to a temp name, renamed last, so a
        # crash never leaves a half-written manifest and a manifest never precedes its payload.
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            handle.write(revision.to_json())
        if manifest_path.exists():
            raise FileExistsError(str(manifest_path))
        manifest_tmp.rename(manifest_path)
    except FileExistsError:
        manifest_tmp.unlink(missing_ok=True)
        # A concurrent writer already committed (or is committing) this content-addressed
        # revision. Bakes are immutable, so validate what is on disk instead of deleting a
        # peer's already-committed files out from under it.
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        return _load_existing_revision(data_path, manifest_path, payload_hash, bake_id)
    except Exception:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        if not manifest_path.exists():
            data_path.unlink(missing_ok=True)
        raise
    return revision, logical_path(manifest_path)


def _check_sample_ids(sample: dict[str, Any], stable_ids: list[int]) -> None:
    """Every frame must present the same bodies in the same order as the runtime."""

    sample_ids = [int(body["stable_id"]) for body in sample.get("bodies", ())]
    if sample_ids != stable_ids:
        raise ValueError("bake sample stable IDs do not match runtime ordering")


def _load_existing_revision(
    data_path: Path, manifest_path: Path, payload_hash: str, bake_id: str
) -> tuple[BakeRevision, Path]:
    """Validate an already-committed revision's files and return it."""

    if not data_path.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"incomplete immutable bake revision {bake_id}")
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Re-hash the committed payload in its own encoding: binary frames are hashed as the
    # raw bytes on disk, the v1 JSON payload is re-canonicalized then hashed.
    existing_hash = _committed_payload_hash(data_path)
    if existing_hash != payload_hash:
        raise FileExistsError(f"immutable bake payload collision for {bake_id}")
    if manifest_sha256(existing_manifest) != existing_manifest.get("manifest_sha256"):
        raise FileExistsError(f"immutable bake manifest integrity failure for {bake_id}")
    return (
        _revision_from_dict(
            existing_manifest, fallback_cache_root=str(logical_path(manifest_path.parent))
        ),
        logical_path(manifest_path),
    )


def read_bake_samples(manifest_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read a committed bake's payload back into ``(context_id, samples)``.

    Dispatches on the manifest's ``compatible_schema`` so the v2 binary payload and the v1
    JSON payload decode through one entry point. ADR 0025 requires the v1 reader to survive
    the cache change so pre-cache bakes stay decodable, not only hash-verifiable.
    """

    manifest_path = filesystem_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stem = manifest_path.name[: -len(".manifest.json")]
    directory = manifest_path.parent
    expected = manifest.get("payload_sha256")
    if manifest.get("compatible_schema") == _BINARY_SCHEMA:
        data = (directory / f"{stem}.frames.bin").read_bytes()
        if expected is not None and payload_sha256(data) != expected:
            raise BakeCacheError(f"bake payload hash mismatch for {stem}")
        return decode_frames(data)
    payload = json.loads((directory / f"{stem}.samples.json").read_text(encoding="utf-8"))
    if expected is not None and _json_sha256(payload) != expected:
        raise BakeCacheError(f"bake payload hash mismatch for {stem}")
    return payload["context_id"], payload["samples"]


def _committed_payload_hash(path: Path) -> str:
    """Digest a committed payload in its own encoding; ``.bin`` streams in chunks (SR-06)."""

    if path.suffix == ".bin":
        return file_sha256(path)
    return _json_sha256(json.loads(path.read_text(encoding="utf-8")))


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_sha256(value: dict[str, Any]) -> str:
    unhashed = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return _json_sha256(unhashed)


def _revision_from_dict(
    data: dict[str, Any], *, fallback_cache_root: str = ""
) -> BakeRevision:
    return BakeRevision(
        **{
            **data,
            "channels": tuple(str(channel) for channel in data.get("channels", ())),
            "stable_ids": tuple(int(stable_id) for stable_id in data.get("stable_ids", ())),
            "units": dict(data.get("units") or {}),
            "validity": dict(data.get("validity") or {}),
            "provenance": dict(data.get("provenance") or {}),
            # Revisions written before observations existed simply carry none.
            "observations": dict(data.get("observations") or {}),
            # Triple-identity provenance predates this field on-disk for any manifest
            # written before it existed (or by a writer that never supplied a resolved
            # root). Derived from where the manifest actually was found rather than left
            # as an empty string, so a legacy bake still names a real, usable location.
            "cache_root": data.get("cache_root") or fallback_cache_root,
        }
    )


# --------------------------------------------------------------------------------------
# Free Bake: crash-safe deletion engine (Seam 6 sub-commit 3a)
#
# Deleting a bake revision's committed files is destructive, so it goes through a
# write-ahead journal: every file that will be removed is first atomically renamed to a
# "tombstone" location (durably recorded in a fsynced journal before and after each
# rename), and only once every rename has landed is the journal marked ``COMMITTED`` and
# the tombstones actually unlinked. If the process dies at any point before ``COMMITTED``,
# ``recover_free_bakes`` -- run at the next startup -- finds the journal and rolls the
# rename(s) back, restoring the original files exactly as if Free Bake had never run. If
# the process dies after ``COMMITTED`` but before the tombstones/journal are unlinked,
# recovery simply finishes that cleanup. This module is intentionally pure: it knows
# nothing about ``bpy``, operators, or the runtime registry -- callers hand it a resolved
# cache root and an identity triple and it does filesystem work only.
# --------------------------------------------------------------------------------------

FREEBAKE_JOURNAL_SCHEMA = 1
_FREEBAKE_DIRNAME = ".astro_warp_freebake"

# Journal lifecycle. ``PAYLOAD_MOVED``/``BOTH_MOVED`` name the common two-file case
# (payload then manifest); a plan with more than two moves still progresses through the
# same two named milestones -- "at least one move landed" / "every move landed" -- rather
# than minting a new state name per file, so the state machine recovery reasons about
# never grows with the file count.
STATE_PREPARED = "PREPARED"
STATE_PAYLOAD_MOVED = "PAYLOAD_MOVED"
STATE_BOTH_MOVED = "BOTH_MOVED"
STATE_COMMITTED = "COMMITTED"
STATE_RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class FreeBakeError(Exception):
    """Base class for Free Bake engine errors."""


class FreeBakeContainmentError(FreeBakeError):
    """A Free Bake target path does not resolve inside its cache root."""


class FreeBakeOwnershipError(FreeBakeError):
    """The requested ``(cache_root, context_id, bake_id)`` triple does not match the
    on-disk manifest."""


class FreeBakeRecoveryError(FreeBakeError):
    """Rolling back (or finishing) a partially executed Free Bake itself failed.

    The journal is left durably marked ``RECOVERY_REQUIRED`` before this is raised, so a
    later :func:`recover_free_bakes` call can retry.
    """


class _SimulatedFreeBakeCrash(FreeBakeError):
    """Test-only: represents a hard process crash at a specific Free Bake step.

    Raised only when the private ``_fail_after`` test hook names a step at which a real
    crash would leave no in-process handler to run, so :func:`execute_free_bake`
    deliberately does not attempt its own best-effort rollback for this exception --
    only a later, separate :func:`recover_free_bakes` call (simulating the process
    restart) may repair the on-disk state, exactly as it would after a genuine crash.
    """


@dataclass(frozen=True)
class FreeBakeMove:
    """One file this Free Bake will atomically rename out of the way."""

    original: Path
    tombstone: Path


@dataclass(frozen=True)
class FreeBakePlan:
    """A validated, not-yet-executed Free Bake: the files it will remove and where."""

    cache_root: Path
    context_id: str
    bake_id: str
    moves: tuple[FreeBakeMove, ...]

    @property
    def journal_path(self) -> Path:
        return _freebake_journal_path(self.cache_root, self.bake_id)


@dataclass(frozen=True)
class RecoveryOutcome:
    """What :func:`recover_free_bakes` did (or failed to do) for one journalled bake."""

    bake_id: str
    journal_path: Path
    action: str
    ok: bool
    error: str | None = None


def _freebake_dir(cache_root: Path) -> Path:
    return Path(cache_root) / _FREEBAKE_DIRNAME


def _freebake_journal_path(cache_root: Path, bake_id: str) -> Path:
    return _freebake_dir(cache_root) / f"{bake_id}.journal.json"


def _check_containment(target: Path, root: Path) -> None:
    """Refuse ``target`` if its resolved location is not inside ``root``.

    Both paths are run through :func:`os.path.realpath`, which resolves symlinks (and,
    on Windows, junctions), so an attacker-controlled path component -- whether a literal
    ``..`` segment or a symlink/junction planted on disk -- is caught by the same check.
    Comparison is case-normalized so this holds on case-insensitive filesystems too.
    """

    root_real = _realpath_key(root)
    target_real = _realpath_key(target)
    if target_real != root_real and not target_real.startswith(root_real + os.sep):
        raise FreeBakeContainmentError(
            f"Free Bake target {target} resolves outside cache root {root}"
        )


def _realpath_key(path: Path | str) -> str:
    """Comparable real path without Windows' filesystem-only transport prefix."""

    return os.path.normcase(os.path.realpath(str(logical_path(path))))


def plan_free_bake(cache_root: Path | str, context_id: str, bake_id: str) -> FreeBakePlan:
    """Resolve and validate the files one bake revision's Free Bake would delete.

    Performs no filesystem mutation. Mirrors the manifest/payload directory layout the
    product runtime registry already resolves bakes under
    (``<cache_root>/bakes/<context_id>/<bake_id>.manifest.json`` plus its payload).

    Raises ``FreeBakeContainmentError`` if any resolved target would land outside
    ``cache_root`` (including via a symlink/junction escape), and
    ``FreeBakeOwnershipError`` if the requested triple does not match what the on-disk
    manifest records (a different root or context using the same ``bake_id`` text is not
    the same bake and must not be deleted for it).
    """

    root = filesystem_path(Path(cache_root).resolve())
    if not bake_id or Path(bake_id).name != bake_id:
        raise FreeBakeContainmentError("bake ID must be a single non-empty path component")

    directory = root / "bakes" / context_id
    manifest_path = directory / f"{bake_id}.manifest.json"
    _check_containment(manifest_path, root)

    if not manifest_path.is_file():
        raise FreeBakeOwnershipError(
            f"no bake manifest for bake_id={bake_id!r} under context_id={context_id!r}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FreeBakeOwnershipError(
            f"could not read manifest for bake_id={bake_id!r}: {exc}"
        ) from exc

    stored_bake_id = manifest.get("bake_id")
    stored_context_id = manifest.get("context_id")
    stored_cache_root = manifest.get("cache_root") or ""
    if stored_bake_id != bake_id or stored_context_id != context_id:
        raise FreeBakeOwnershipError(
            f"manifest identity (context_id={stored_context_id!r}, bake_id={stored_bake_id!r}) "
            f"does not match requested (context_id={context_id!r}, bake_id={bake_id!r})"
        )
    # A legacy manifest predating the ``cache_root`` field (or written by a caller that
    # never supplied a resolved root) has no recorded value to compare, so this axis of
    # the ownership check is skipped for it -- mirroring the same pre-field tolerance
    # ``_revision_from_dict`` applies elsewhere (line ~472). Unlike that read path,
    # deriving a fallback here from "where the manifest was physically found" would only
    # ever reproduce ``root`` itself (the manifest was located via
    # ``root / "bakes" / context_id``), making the comparison tautologically pass rather
    # than meaningfully close the gap -- so it is not a clean derivation for THIS check
    # and is deliberately not attempted. The residual gap this leaves: a legacy manifest
    # physically duplicated (e.g. by an out-of-band copy or migration) under a second,
    # unrelated cache_root with the same (context_id, bake_id) text would be deleted from
    # whichever root Free Bake is invoked against, since it carries no self-reported root
    # of its own to contradict the request.
    logical_root = logical_path(root)
    if stored_cache_root and os.path.normcase(
        os.path.realpath(stored_cache_root)
    ) != os.path.normcase(os.path.realpath(str(logical_root))):
        raise FreeBakeOwnershipError(
            f"manifest cache_root {stored_cache_root!r} does not match requested "
            f"cache_root {str(logical_root)!r}"
        )

    payload_suffix = (
        ".frames.bin" if manifest.get("compatible_schema") == _BINARY_SCHEMA else ".samples.json"
    )
    payload_path = directory / f"{bake_id}{payload_suffix}"
    _check_containment(payload_path, root)

    freebake_dir = _freebake_dir(root)
    moves: list[FreeBakeMove] = []
    if payload_path.exists() or payload_path.is_symlink():
        moves.append(
            FreeBakeMove(original=payload_path, tombstone=freebake_dir / f"{bake_id}.0.tombstone")
        )
    moves.append(
        FreeBakeMove(
            original=manifest_path,
            tombstone=freebake_dir / f"{bake_id}.{len(moves)}.tombstone",
        )
    )
    return FreeBakePlan(cache_root=root, context_id=context_id, bake_id=bake_id, moves=tuple(moves))


def _fsync_dir(directory: Path) -> None:
    if os.name == "nt":
        # Windows offers no directory file descriptor to fsync; directory-entry
        # durability there rests on NTFS's own metadata journal, so there is nothing
        # additional to flush.
        return
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_and_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def _write_journal(
    plan: FreeBakePlan, *, state: str, moved: int, pending_action: str = "rollback"
) -> None:
    payload = {
        "schema": FREEBAKE_JOURNAL_SCHEMA,
        "bake_id": plan.bake_id,
        "context_id": plan.context_id,
        "cache_root": str(logical_path(plan.cache_root)),
        "state": state,
        "moved": moved,
        "pending_action": pending_action,
        "moves": [
            {
                "original": str(logical_path(move.original)),
                "tombstone": str(logical_path(move.tombstone)),
            }
            for move in plan.moves
        ],
    }
    _atomic_write_and_fsync(
        plan.journal_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _read_journal(journal_path: Path) -> dict[str, Any]:
    return json.loads(journal_path.read_text(encoding="utf-8"))


def _plan_from_journal(
    data: dict[str, Any], *, fallback_bake_id: str, recovery_root: Path
) -> FreeBakePlan:
    recovery_root = filesystem_path(Path(recovery_root).resolve())
    journal_root = filesystem_path(Path(data.get("cache_root", "")).resolve())
    if _realpath_key(journal_root) != _realpath_key(recovery_root):
        raise FreeBakeContainmentError(
            f"Free Bake journal cache root {logical_path(journal_root)} does not match "
            f"recovery root {logical_path(recovery_root)}"
        )
    moves = tuple(
        FreeBakeMove(
            original=filesystem_path(Path(move["original"])),
            tombstone=filesystem_path(Path(move["tombstone"])),
        )
        for move in data.get("moves", ())
    )
    tombstone_root = _freebake_dir(recovery_root)
    for move in moves:
        _check_containment(move.original, recovery_root)
        _check_containment(move.tombstone, tombstone_root)
    return FreeBakePlan(
        cache_root=recovery_root,
        context_id=str(data.get("context_id", "")),
        bake_id=str(data.get("bake_id") or fallback_bake_id),
        moves=moves,
    )


def _remove_tombstone(path: Path) -> None:
    path.unlink(missing_ok=True)


def _progress_state(completed: int, total: int) -> str:
    return STATE_BOTH_MOVED if completed >= total else STATE_PAYLOAD_MOVED


def execute_free_bake(plan: FreeBakePlan, *, _fail_after: str | None = None) -> None:
    """Journalled, crash-safe deletion of every file in ``plan``.

    Every rename is durably journalled before it is attempted and again once it lands,
    so a crash at any point before the journal reaches ``COMMITTED`` leaves enough
    information for :func:`recover_free_bakes` to restore the originals exactly, and a
    crash after ``COMMITTED`` leaves enough for it to finish deleting the tombstones.

    ``_fail_after`` is a private test-only hook naming the step after which to raise a
    synthetic failure, so tests can exercise every crash/recovery path without real disk
    faults: ``"first_rename"``, ``"second_rename"``, and ``"before_commit"`` simulate a
    hard process crash (bypassing this function's own best-effort rollback, exactly as a
    real crash would); ``"rollback"`` simulates a genuine pre-commit failure that this
    function does attempt to roll back in-process, with that rollback attempt itself
    made to fail once, to exercise the ``RECOVERY_REQUIRED`` path deterministically.
    """

    moves = plan.moves
    total = len(moves)
    journal_path = plan.journal_path
    completed = 0
    _write_journal(plan, state=STATE_PREPARED, moved=0)
    try:
        for index, move in enumerate(moves):
            os.replace(move.original, move.tombstone)
            completed = index + 1
            _write_journal(plan, state=_progress_state(completed, total), moved=completed)
            if _fail_after == "first_rename" and completed == 1:
                raise _SimulatedFreeBakeCrash("first_rename")
            if _fail_after == "second_rename" and completed == 2:
                raise _SimulatedFreeBakeCrash("second_rename")
        if _fail_after == "before_commit":
            raise _SimulatedFreeBakeCrash("before_commit")
        if _fail_after == "rollback":
            # Not a crash: a genuine failure the process itself observes just before
            # commit, so the in-process rollback below is attempted -- and is, in turn,
            # made to fail once, landing on RECOVERY_REQUIRED without a real restart.
            raise RuntimeError("synthetic pre-commit failure (Free Bake rollback test)")
        _write_journal(plan, state=STATE_COMMITTED, moved=total, pending_action="finish")
    except _SimulatedFreeBakeCrash:
        raise
    except Exception:
        _rollback(plan, journal_path, fail_once=(_fail_after == "rollback"))
        raise
    _finish(plan, journal_path)


def _exists_or_symlink(path: Path) -> bool:
    """``True`` if ``path`` names anything on disk, including a broken symlink.

    Plain ``Path.exists()`` follows symlinks and reports ``False`` for a dangling one;
    recovery must still recognize a dangling symlink as "something is here" (mirrors the
    same convention :func:`plan_free_bake` already uses for payload detection).
    """

    return path.exists() or path.is_symlink()


def _pending_restore_count(plan: FreeBakePlan) -> int:
    """Non-authoritative hint: how many of ``plan.moves`` still need restoring.

    Derived from real on-disk state at call time, purely for the journal's informational
    ``moved`` field -- nothing in recovery correctness depends on this number.
    """

    return sum(1 for move in plan.moves if not _exists_or_symlink(move.original))


def _rollback(plan: FreeBakePlan, journal_path: Path, *, fail_once: bool = False) -> None:
    """Restore every journalled move whose tombstone is on disk, driven by real
    on-disk state rather than a "moved" counter.

    The journal's durable ``moved`` counter is written *after* the physical rename it
    describes, so a crash can always land in the gap between the rename landing on disk
    and that count being journalled -- leaving a journal that durably lags reality by up
    to one move. Gating rollback on that counter (as opposed to actual disk state) would
    silently skip restoring a file that was, in fact, already tombstoned, orphaning it
    once the journal is deleted. Instead, every move in the full ``plan.moves`` list
    (always written in full at ``PREPARED``, before any rename happens) is inspected
    directly: a move whose original is already present is left alone (never moved, or
    already restored by an earlier attempt); a move whose tombstone is present and whose
    original is absent is restored. This makes rollback idempotent and safe to retry
    from any on-disk state, including immediately after another rollback attempt failed
    partway through.
    """

    for move in reversed(plan.moves):
        if _exists_or_symlink(move.original):
            continue  # never moved, or already restored by an earlier attempt
        if fail_once:
            fail_once = False
            _write_journal(
                plan,
                state=STATE_RECOVERY_REQUIRED,
                moved=_pending_restore_count(plan),
                pending_action="rollback",
            )
            raise FreeBakeRecoveryError(
                f"rollback of free bake {plan.bake_id} failed with move(s) still "
                "pending restoration (synthetic failure)"
            )
        if not _exists_or_symlink(move.tombstone):
            # Neither the original nor its tombstone exists: an anomaly outside the
            # documented crash window (e.g. something else removed the file out of
            # band), not something rollback can safely paper over by declaring success.
            _write_journal(
                plan,
                state=STATE_RECOVERY_REQUIRED,
                moved=_pending_restore_count(plan),
                pending_action="rollback",
            )
            raise FreeBakeRecoveryError(
                f"rollback of free bake {plan.bake_id} cannot find {move.original} or "
                f"its tombstone {move.tombstone}"
            )
        try:
            os.replace(move.tombstone, move.original)
        except OSError as exc:
            _write_journal(
                plan,
                state=STATE_RECOVERY_REQUIRED,
                moved=_pending_restore_count(plan),
                pending_action="rollback",
            )
            raise FreeBakeRecoveryError(
                f"rollback of free bake {plan.bake_id} failed restoring {move.original}"
            ) from exc
    journal_path.unlink(missing_ok=True)


def _finish(plan: FreeBakePlan, journal_path: Path) -> None:
    """Delete every tombstone in ``plan.moves``, then the journal itself."""

    for move in plan.moves:
        try:
            _remove_tombstone(move.tombstone)
        except OSError as exc:
            _write_journal(
                plan, state=STATE_RECOVERY_REQUIRED, moved=len(plan.moves), pending_action="finish"
            )
            raise FreeBakeRecoveryError(
                f"finishing free bake {plan.bake_id} failed removing {move.tombstone}"
            ) from exc
    journal_path.unlink(missing_ok=True)


def recover_free_bakes(cache_root: Path | str) -> list[RecoveryOutcome]:
    """Scan ``<cache_root>/.astro_warp_freebake`` and resolve every journalled Free Bake.

    Only journal files under that dedicated subdirectory are ever touched -- nothing
    else in ``cache_root`` is inspected or mutated. Safe to call repeatedly: a bake with
    no journal (already fully resolved) is simply absent from the scan on the next call.
    """

    root = filesystem_path(Path(cache_root).resolve())
    freebake_dir = _freebake_dir(root)
    if not freebake_dir.is_dir():
        return []
    outcomes = []
    for journal_path in sorted(freebake_dir.glob("*.journal.json")):
        outcomes.append(_recover_one(journal_path, recovery_root=root))
    return outcomes


def _recover_one(journal_path: Path, *, recovery_root: Path) -> RecoveryOutcome:
    fallback_bake_id = journal_path.name
    if fallback_bake_id.endswith(".journal.json"):
        fallback_bake_id = fallback_bake_id[: -len(".journal.json")]

    try:
        data = _read_journal(journal_path)
        plan = _plan_from_journal(
            data, fallback_bake_id=fallback_bake_id, recovery_root=recovery_root
        )
    except (OSError, ValueError, FreeBakeError, KeyError, TypeError, AttributeError) as exc:
        return RecoveryOutcome(
            bake_id=fallback_bake_id,
            journal_path=journal_path,
            action="unreadable",
            ok=False,
            error=str(exc),
        )

    state = data.get("state")
    pending_action = data.get("pending_action") or (
        "finish" if state == STATE_COMMITTED else "rollback"
    )

    action_name = "finished" if pending_action == "finish" else "rolled_back"
    try:
        if pending_action == "finish":
            _finish(plan, journal_path)
        else:
            _rollback(plan, journal_path)
        return RecoveryOutcome(
            bake_id=plan.bake_id, journal_path=journal_path, action=action_name, ok=True
        )
    except FreeBakeRecoveryError as exc:
        return RecoveryOutcome(
            bake_id=plan.bake_id,
            journal_path=journal_path,
            action="recovery_required",
            ok=False,
            error=str(exc),
        )


def has_pending_free_bake(cache_root: Path | str, bake_id: str) -> bool:
    """True iff a not-yet-resolved Free Bake journal exists for ``bake_id``.

    Any journal presence at all counts as pending -- including ``RECOVERY_REQUIRED`` and
    a lingering post-``COMMITTED`` journal whose tombstone cleanup has not yet finished --
    since in every one of those states the revision's files are not in their final,
    fully-resolved shape yet.
    """

    root = filesystem_path(Path(cache_root).resolve())
    return _freebake_journal_path(root, bake_id).is_file()
