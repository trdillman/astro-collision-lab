"""Isolated installed-package probe support, owned by this project."""

import importlib
import json
import os
from types import SimpleNamespace


def resolve_probe_paths(*, repo_root, report_dir, warp_cache_dir):
    return SimpleNamespace(repo_root=repo_root, report_dir=report_dir,
                           warp_cache_dir=warp_cache_dir)


def configure_environment(paths):
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.warp_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(paths.warp_cache_dir)
    os.environ["ACL_KERNEL_CACHE"] = str(paths.warp_cache_dir)


def import_extension_module(module_name, *, mode):
    if mode != "installed":
        raise ValueError("this acceptance probe requires an installed extension")
    return importlib.import_module(module_name)


def extension_code_fingerprint(module):
    from pathlib import Path

    return json.loads((Path(module.__file__).parent / "_build.json").read_text())["source_fingerprint"]
