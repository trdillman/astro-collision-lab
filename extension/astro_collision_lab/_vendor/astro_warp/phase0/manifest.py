"""JSON environment manifest helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contracts import EnvironmentReport, Phase0Config, RuntimeStatus


def source_revision(repo_root: Path) -> tuple[str, bool | None]:
    """Return Git revision and dirty flag for a repo, tolerating no-commit repos."""

    root = Path(repo_root)
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ("unknown", None)

    revision = rev.stdout.strip() if rev.returncode == 0 else "unborn-or-unknown"
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return (revision, None)

    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return (revision, dirty)


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest atomically so an interrupted write never leaves an unparseable file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parseable_manifest(path: Path) -> bool:
    try:
        read_manifest(path)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def local_environment_manifest(
    repo_root: Path,
    config: Phase0Config | None = None,
) -> dict[str, Any]:
    config = config or Phase0Config()
    config.validate()
    revision, dirty = source_revision(repo_root)
    report = EnvironmentReport.empty(
        status=RuntimeStatus.UNLOADED,
        config=config,
        source_revision=revision,
        source_dirty=dirty,
    )
    data = report.to_json_dict()
    data["python"] = {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": sys.implementation.name,
    }
    return data
