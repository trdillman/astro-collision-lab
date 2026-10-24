"""Package complete source, current installer, and retained qualification evidence.

Run tools/build.py first. This packages a reviewable candidate; it never turns
host tests or historical native runs into qualification of a new installer.
"""

import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (
    ".gitignore", "README.md", "requirements-dev.txt", "launch-lab.cmd",
    "SOURCE_COMMIT.txt", "CANDIDATE_STATUS.json", "CHANGES_WITH_LINES.md",
)
DIRECTORIES = ("extension", "tests", "tools", "validation", "docs", "status")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def package():
    source = ROOT / "extension" / "astro_collision_lab"
    version = tomllib.loads((source / "blender_manifest.toml").read_text("utf-8"))["version"]
    artifacts = ROOT / "artifacts"
    installer = artifacts / f"astro_collision_lab-{version}-windows-x64.zip"
    build_report = installer.with_suffix(".build.json")
    if not installer.is_file() or not build_report.is_file():
        raise RuntimeError("Build the current installer with python tools/build.py first.")

    installer_bytes = installer.read_bytes()
    report = json.loads(build_report.read_text("utf-8"))
    if report.get("sha256") != sha256(installer_bytes):
        raise RuntimeError("Installer and build report do not match; rebuild before packaging.")
    with zipfile.ZipFile(installer) as archive:
        entries = archive.namelist()
        if len(entries) != len(set(entries)):
            raise RuntimeError("Installer contains duplicate entries.")
        metadata = json.loads(archive.read("_build.json"))
        if metadata.get("version") != version:
            raise RuntimeError("Installer version does not match the current source.")
        if set(entries) != set(metadata["files"]) | {"_build.json"}:
            raise RuntimeError("Installer manifest coverage is incomplete.")
        for name in entries:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise RuntimeError(f"Unsafe installer member: {name}")
            content = archive.read(name)
            if content != (source / relative).read_bytes():
                raise RuntimeError(f"Installer differs from current source: {name}")
            if name != "_build.json" and sha256(content) != metadata["files"][name]:
                raise RuntimeError(f"Installer member hash is incorrect: {name}")

    files = {}
    for name in TOP_LEVEL:
        path = ROOT / name
        if path.is_file():
            files[name] = path.read_bytes()
    for folder in DIRECTORIES:
        for path in sorted((ROOT / folder).rglob("*")):
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                raise RuntimeError(f"Refusing to package a symlink: {relative}")
            if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
                files[relative.as_posix()] = path.read_bytes()
    files[f"artifacts/{installer.name}"] = installer_bytes
    files[f"artifacts/{build_report.name}"] = build_report.read_bytes()
    inventory = {
        "project": "astro_collision_lab", "version": version,
        "qualified": False,
        "qualification_note": "See CANDIDATE_STATUS.json. Historical runs do not qualify this installer.",
        "installer_sha256": sha256(installer_bytes),
        "files": {name: sha256(data) for name, data in sorted(files.items())},
    }
    files["SOURCE_ARCHIVE_MANIFEST.json"] = (json.dumps(inventory, indent=2) + "\n").encode("utf-8")
    out = artifacts / f"astro_collision_lab-{version}-source-and-evidence.zip"
    prefix = f"astro_collision_lab-{version}"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{prefix}/{name}", (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    result = {
        "archive": str(out), "sha256": sha256(out.read_bytes()),
        "bytes": out.stat().st_size, "members": len(files),
        "installer_sha256": sha256(installer_bytes), "qualified": False,
    }
    out.with_suffix(".package.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return out


if __name__ == "__main__":
    package()
