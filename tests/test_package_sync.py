"""The add-on ships byte-identical copies of the reusable packages' shared modules."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("sync_packages", ROOT / "tools" / "sync_packages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_modules_are_identical():
    tool = load_tool()
    assert tool.drifted() == [], "run: python tools/sync_packages.py"


def test_every_shared_source_exists():
    tool = load_tool()
    for src, dst in tool.SHARED.items():
        assert (ROOT / src).is_file(), src
        assert (tool.ADDON / dst).is_file(), dst
