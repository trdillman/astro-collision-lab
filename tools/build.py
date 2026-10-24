"""Build a slim extension without bundled runtime wheels or binaries."""

import argparse
import hashlib
import json
import shutil
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = "warp_lang-1.15.0-py3-none-win_amd64.whl"
WHEEL_SHA = "06ef42c3ce522749268056653ea253645eb4a96ca164af74ae5113f0d55a6970"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(wheel=None):
    source = ROOT / "extension" / "astro_collision_lab"
    stage = ROOT / "build" / "astro_collision_lab"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "wheels", "*.whl", "*.dll", "*.pyd", "_build.json"))
    version = tomllib.loads((stage / "blender_manifest.toml").read_text(encoding="utf-8"))["version"]
    sources = {
        p.relative_to(stage).as_posix(): digest(p)
        for p in sorted(stage.rglob("*")) if p.is_file() and p.suffix != ".whl"
    }
    # Match runtime's actual Python/manifest fingerprint without importing Blender.
    identity = hashlib.sha256()
    for name in sorted(n for n in sources if n.endswith(".py") or n == "blender_manifest.toml"):
        relative = name.encode("utf-8")
        content = (stage / name).read_bytes()
        identity.update(len(relative).to_bytes(8, "big")); identity.update(relative)
        identity.update(len(content).to_bytes(8, "big")); identity.update(content)
    fingerprint = identity.hexdigest()
    manifest = json.dumps({
        "project": "astro_collision_lab", "version": version,
        "source_fingerprint": fingerprint, "qualification_status": "candidate_not_qualified",
        "files": sources, "warp_wheel_sha256": WHEEL_SHA, "dependencies_bundled": False,
    }, indent=2)
    (stage / "_build.json").write_text(manifest, encoding="utf-8")
    (source / "_build.json").write_text(manifest, encoding="utf-8")
    out = ROOT / "artifacts" / f"astro_collision_lab-{version}-windows-x64.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        for p in sorted(stage.rglob("*")):
            if not p.is_file():
                continue
            info = zipfile.ZipInfo(p.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            compression = zipfile.ZIP_STORED if p.suffix == ".whl" else zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes(), compress_type=compression)
    report = {"archive": str(out), "sha256": digest(out), "bytes": out.stat().st_size,
              "source_fingerprint": fingerprint, "wheel_sha256": WHEEL_SHA, "dependencies_bundled": False}
    out.with_suffix(".build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, help="Deprecated; never bundled")
    build(parser.parse_args().wheel)
