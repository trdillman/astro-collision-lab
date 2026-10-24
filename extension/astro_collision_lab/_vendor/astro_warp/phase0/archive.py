"""Blender extension archive verification."""

from __future__ import annotations

import hashlib
import posixpath
import tomllib
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .constants import (
    BLENDER_PLATFORM,
    BLENDER_VERSION_MIN,
    EXTENSION_ID,
    EXTENSION_VERSION,
    WARP_WHEEL_FILENAME,
    WARP_WHEEL_SHA256,
)


class ArchiveVerificationError(RuntimeError):
    """Raised when a built extension archive fails Phase 0 package gates."""


@dataclass(frozen=True)
class ArchiveVerification:
    archive_path: str
    sha256: str
    entries: int
    wheel_sha256: str
    required_files_present: bool
    unsafe_paths: tuple[str, ...]
    duplicate_entries: tuple[str, ...]


REQUIRED_FILES = (
    "blender_manifest.toml",
    "__init__.py",
    "operators.py",
    "properties.py",
    "ui.py",
    "draw.py",
    f"wheels/{WARP_WHEEL_FILENAME}",
)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_unsafe_zip_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    # A directory entry ("pkg/") is a normal ZIP member; the trailing slash is not an escape
    # (SR-51). Strip one before the checks so it is judged by the path it names.
    if normalized.endswith("/") and normalized != "/":
        normalized = normalized[:-1]
    path = PurePosixPath(normalized)
    if normalized.startswith("/") or normalized.startswith("../"):
        return True
    if path.is_absolute():
        return True
    if any(part in {"", ".", ".."} for part in path.parts):
        return True
    if not path.parts or ":" in path.parts[0]:
        return True
    return posixpath.normpath(normalized) != normalized


# The code a release ships, as opposed to everything a release contains. The wheel is
# hash-gated on its own and would otherwise dominate the digest; `_build_info.py` carries a
# build timestamp, so including it would make every rebuild of identical source look like a
# different release; `__pycache__` is a byproduct of running the code, not the code.
CODE_SUFFIXES = (".py", ".toml")
CODE_EXCLUDED_NAMES = ("_build_info.py",)
CODE_EXCLUDED_DIRS = ("wheels", "__pycache__")


def _is_code_entry(relative_posix_path: str) -> bool:
    parts = relative_posix_path.split("/")
    if any(part in CODE_EXCLUDED_DIRS for part in parts[:-1]):
        return False
    if parts[-1] in CODE_EXCLUDED_NAMES:
        return False
    return relative_posix_path.endswith(CODE_SUFFIXES)


def _fingerprint(entries: Iterable[tuple[str, bytes]]) -> str:
    """One digest over path-and-content pairs, order-independent by construction.

    Each pair contributes its path and the digest of its bytes rather than the bytes
    themselves, so a file cannot be renamed without changing the result and content can be
    streamed rather than concatenated.

    CRLF is read as LF. Git stores this code with LF (`.gitattributes`) and calls a CRLF
    checkout clean, so without this two checkouts of one commit name different releases
    (ADR 0138).
    """

    digest = hashlib.sha256()
    for path, payload in sorted(entries):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload.replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()


def _directory_entries(root) -> Iterator[tuple[str, bytes]]:
    base = Path(root)
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        if _is_code_entry(relative):
            yield relative, path.read_bytes()


def directory_code_fingerprint(root) -> str:
    """The code fingerprint of an unpacked extension, staged or installed."""

    return _fingerprint(_directory_entries(root))


def archive_code_fingerprint(path) -> str:
    """The code fingerprint of a built extension archive.

    Equal to :func:`directory_code_fingerprint` of anything the archive is extracted to,
    which is what lets a proof run against an installed copy and still name the release it
    came from.
    """

    with zipfile.ZipFile(path, "r") as zf:
        return _fingerprint(
            (name, zf.read(name)) for name in zf.namelist() if _is_code_entry(name)
        )


def verify_extension_archive(path) -> ArchiveVerification:
    archive_hash = sha256_file(path)
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        counts = Counter(names)
        duplicates = tuple(sorted(name for name, count in counts.items() if count > 1))
        unsafe = tuple(sorted(name for name in names if _is_unsafe_zip_path(name)))
        missing = tuple(name for name in REQUIRED_FILES if name not in counts)
        if duplicates:
            raise ArchiveVerificationError(f"duplicate ZIP entries: {duplicates}")
        if unsafe:
            raise ArchiveVerificationError(f"unsafe ZIP paths: {unsafe}")
        if missing:
            raise ArchiveVerificationError(f"missing required files: {missing}")

        manifest = tomllib.loads(zf.read("blender_manifest.toml").decode("utf-8"))
        if manifest.get("id") != EXTENSION_ID:
            raise ArchiveVerificationError("manifest id mismatch")
        if manifest.get("blender_version_min") != BLENDER_VERSION_MIN:
            raise ArchiveVerificationError("manifest blender_version_min mismatch")
        if manifest.get("version") != EXTENSION_VERSION:
            raise ArchiveVerificationError(
                f"manifest version {manifest.get('version')!r} does not match "
                f"EXTENSION_VERSION {EXTENSION_VERSION!r}"
            )
        if BLENDER_PLATFORM not in manifest.get("platforms", []):
            raise ArchiveVerificationError("manifest does not include windows-x64 platform")
        wheel_manifest_path = f"./wheels/{WARP_WHEEL_FILENAME}"
        if wheel_manifest_path not in manifest.get("wheels", []):
            raise ArchiveVerificationError("manifest does not list required Warp wheel")

        wheel_digest = hashlib.sha256(zf.read(f"wheels/{WARP_WHEEL_FILENAME}")).hexdigest()
        if wheel_digest != WARP_WHEEL_SHA256:
            raise ArchiveVerificationError(
                f"Warp wheel hash mismatch: expected {WARP_WHEEL_SHA256}, found {wheel_digest}"
            )

    return ArchiveVerification(
        archive_path=str(path),
        sha256=archive_hash,
        entries=len(names),
        wheel_sha256=wheel_digest,
        required_files_present=True,
        unsafe_paths=unsafe,
        duplicate_entries=duplicates,
    )
