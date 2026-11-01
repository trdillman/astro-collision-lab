"""Install one exact, hash-checked wheel into a private folder, without pip.

Made for apps that embed Python (Blender add-ons, plug-ins, portable tools) and must load an
exact version of a heavy dependency such as NVIDIA Warp, without bundling the wheel and
without touching the host's site-packages. Generalized from the Astro Collision Lab add-on's
Warp installer. Standard library only.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import pathlib
import sys
import urllib.request
import zipfile
from dataclasses import dataclass

__version__ = "0.1.0"
__all__ = ["PinnedWheel", "WheelError", "activate", "install", "isolated_blender_command",
           "pypi_url", "verify_archive"]

PYPI = "https://pypi.org/pypi"


class WheelError(RuntimeError):
    """The wheel, its metadata or the installed module did not match the pin."""


@dataclass(frozen=True)
class PinnedWheel:
    """One exact wheel: PyPI project name, version, file name and SHA-256."""

    project: str
    version: str
    filename: str
    sha256: str
    module: str = ""

    def __post_init__(self):
        digest = self.sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be 64 hex characters")
        if not self.filename.endswith(".whl") or "/" in self.filename or "\\" in self.filename:
            raise ValueError("filename must be a bare .whl file name")
        object.__setattr__(self, "sha256", digest)
        if not self.module:
            object.__setattr__(self, "module", self.project.split("-")[0])


def verify_archive(data: bytes, sha256: str) -> zipfile.ZipFile:
    """Return the wheel as a zip only if its hash matches and no entry escapes the folder."""
    if hashlib.sha256(data).hexdigest() != sha256.lower():
        raise WheelError("wheel hash mismatch")
    archive = zipfile.ZipFile(io.BytesIO(data))
    for name in archive.namelist():
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise WheelError("unsafe archive entry: " + name)
    return archive


def pypi_url(wheel: PinnedWheel, *, index: str = PYPI, timeout: float = 30, opener=None) -> str:
    """Look up the download URL on PyPI and check PyPI reports the same SHA-256."""
    open_url = opener or urllib.request.urlopen
    with open_url(f"{index}/{wheel.project}/{wheel.version}/json", timeout=timeout) as response:
        meta = json.load(response)
    item = next((x for x in meta.get("urls", ()) if x.get("filename") == wheel.filename), None)
    if item is None:
        raise WheelError(f"{wheel.filename} is not published for {wheel.project} {wheel.version}")
    if item.get("digests", {}).get("sha256", "").lower() != wheel.sha256:
        raise WheelError("PyPI metadata differs from the pinned hash")
    return item["url"]


def install(wheel: PinnedWheel, destination, *, data: bytes | None = None, index: str = PYPI,
            timeout: float = 120, opener=None) -> pathlib.Path:
    """Download (unless ``data`` is given), verify and unpack the wheel into ``destination``."""
    target = pathlib.Path(destination)
    if data is None:
        open_url = opener or urllib.request.urlopen
        url = pypi_url(wheel, index=index, timeout=min(timeout, 30), opener=open_url)
        with open_url(url, timeout=timeout) as response:
            data = response.read()
    archive = verify_archive(data, wheel.sha256)
    target.mkdir(parents=True, exist_ok=True)
    archive.extractall(target)
    return target


def activate(wheel: PinnedWheel, directory, *, required: bool = False) -> bool:
    """Put ``directory`` on ``sys.path`` and check the importable module is the pinned version.

    Returns False (or raises ``WheelError`` when ``required``) if the module is missing or a
    different version was already imported by something else in the process.
    """
    folder = pathlib.Path(directory)
    if (folder / wheel.module / "__init__.py").exists() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        module = importlib.import_module(wheel.module)
    except ImportError:
        if required:
            raise WheelError(f"install {wheel.project} {wheel.version} first") from None
        return False
    found = getattr(module, "__version__", None)
    if found != wheel.version:
        if required:
            raise WheelError(f"{wheel.module} {found} is already loaded; {wheel.version} is required. "
                             "Use a clean process or profile.")
        return False
    return True


def isolated_blender_command(blender: str, script, args=(), *, scratch) -> tuple[list[str], dict]:
    """Command and environment to run ``script`` in a throw-away background Blender.

    Every Blender user folder points inside ``scratch``, so the user's add-ons, startup
    scripts and preferences cannot interfere (``--factory-startup`` alone is not enough).
    """
    root = pathlib.Path(scratch)
    env = os.environ.copy()
    for key in ("RESOURCES", "CONFIG", "EXTENSIONS", "SCRIPTS", "DATAFILES"):
        folder = root / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        env["BLENDER_USER_" + key] = str(folder)
    argv = [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", str(script), "--", *map(str, args)]
    return argv, env
