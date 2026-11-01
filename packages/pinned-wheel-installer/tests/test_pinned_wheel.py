import hashlib
import io
import json
import sys
import zipfile

import pytest

import pinned_wheel as pw


def make_wheel(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, text in entries.items():
            z.writestr(name, text)
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def pin(sha, **kw):
    return pw.PinnedWheel("demo-pkg", "1.2.3", "demo_pkg-1.2.3-py3-none-any.whl", sha, **kw)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(data, sha, published_sha=None):
    meta = {"urls": [{"filename": "demo_pkg-1.2.3-py3-none-any.whl", "url": "https://files/x.whl",
                      "digests": {"sha256": published_sha or sha}}]}
    def opener(url, timeout):
        return FakeResponse(json.dumps(meta).encode() if url.endswith("/json") else data)
    return opener


def test_module_name_defaults_from_project():
    assert pin("a" * 64).module == "demo"
    assert pw.PinnedWheel("warp-lang", "1.15.0", "w.whl", "A" * 64).sha256 == "a" * 64


@pytest.mark.parametrize("sha,filename", [("abc", "x.whl"), ("g" * 64, "x.whl"), ("a" * 64, "x.zip"),
                                          ("a" * 64, "../x.whl")])
def test_bad_pins_rejected(sha, filename):
    with pytest.raises(ValueError):
        pw.PinnedWheel("p", "1", filename, sha)


def test_hash_mismatch_rejected():
    data, _ = make_wheel({"demo/__init__.py": ""})
    with pytest.raises(pw.WheelError, match="hash"):
        pw.verify_archive(data, "0" * 64)


@pytest.mark.parametrize("name", ["../evil.py", "/abs.py", "a\\b.py", "C:evil.py"])
def test_unsafe_entries_rejected(name):
    data, sha = make_wheel({name: "x"})
    with pytest.raises(pw.WheelError, match="unsafe"):
        pw.verify_archive(data, sha)


def test_install_from_fake_index_and_activate(tmp_path, monkeypatch):
    data, sha = make_wheel({"demo/__init__.py": "__version__ = '1.2.3'\n"})
    target = pw.install(pin(sha), tmp_path / "deps", opener=opener_for(data, sha))
    assert (target / "demo" / "__init__.py").exists()
    monkeypatch.setattr(sys, "path", list(sys.path))
    sys.modules.pop("demo", None)
    try:
        assert pw.activate(pin(sha), target, required=True)
    finally:
        sys.modules.pop("demo", None)


def test_pypi_hash_disagreement_rejected(tmp_path):
    data, sha = make_wheel({"demo/__init__.py": ""})
    with pytest.raises(pw.WheelError, match="PyPI"):
        pw.install(pin(sha), tmp_path, opener=opener_for(data, sha, published_sha="1" * 64))


def test_activate_wrong_version(tmp_path, monkeypatch):
    data, sha = make_wheel({"demo/__init__.py": "__version__ = '9.9.9'\n"})
    pw.install(pin(sha), tmp_path, data=data)
    monkeypatch.setattr(sys, "path", list(sys.path))
    sys.modules.pop("demo", None)
    try:
        assert pw.activate(pin(sha), tmp_path) is False
        with pytest.raises(pw.WheelError, match="already loaded"):
            pw.activate(pin(sha), tmp_path, required=True)
    finally:
        sys.modules.pop("demo", None)


def test_activate_missing(tmp_path):
    missing = pw.PinnedWheel("nothing-here-xyz", "1", "n.whl", "a" * 64)
    assert pw.activate(missing, tmp_path) is False
    with pytest.raises(pw.WheelError):
        pw.activate(missing, tmp_path, required=True)


def test_isolated_blender_command(tmp_path):
    argv, env = pw.isolated_blender_command("blender", "install.py", ["out"], scratch=tmp_path)
    assert argv[:3] == ["blender", "--background", "--factory-startup"]
    assert argv[-2:] == ["--", "out"]
    for key in ("RESOURCES", "CONFIG", "EXTENSIONS", "SCRIPTS", "DATAFILES"):
        assert env["BLENDER_USER_" + key].startswith(str(tmp_path))


def test_data_purelib_layout_rejected():
    data, sha = make_wheel({"demo-1.2.3.data/purelib/demo/__init__.py": ""})
    with pytest.raises(pw.WheelError, match="purelib"):
        pw.verify_archive(data, sha)


def test_activate_rejects_module_loaded_from_elsewhere(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "demo").mkdir(parents=True)
    (elsewhere / "demo" / "__init__.py").write_text("__version__ = '1.2.3'\n")
    private = tmp_path / "private"
    data, sha = make_wheel({"demo/__init__.py": "__version__ = '1.2.3'\n"})
    pw.install(pin(sha), private, data=data)
    monkeypatch.setattr(sys, "path", [str(elsewhere), *sys.path])
    sys.modules.pop("demo", None)
    try:
        __import__("demo")
        assert pw.activate(pin(sha), private) is False
        with pytest.raises(pw.WheelError, match="not from"):
            pw.activate(pin(sha), private, required=True)
    finally:
        sys.modules.pop("demo", None)
