import importlib.util
from pathlib import Path
import pytest
from astro_collision_lab.install_dependency import verified_archive


def test_dependency_installer_rejects_unpinned_bytes():
    with pytest.raises(RuntimeError,match='hash mismatch'):
        verified_archive(b'not the certified wheel')


def test_slim_build_contains_no_dependency_payload(tmp_path):
    import zipfile
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('acl_build_slim',root/'tools/build.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # Build against real source into an owned temporary project.
    import shutil
    shutil.copytree(root/'extension',tmp_path/'extension')
    mod.ROOT=tmp_path
    archive=mod.build()
    with zipfile.ZipFile(archive) as z:
        assert not any(n.endswith(('.whl','.dll','.pyd')) or n.startswith('wheels/') for n in z.namelist())
        assert b'wheels =' not in z.read('blender_manifest.toml')
        import hashlib
        import json
        import tomllib
        manifest=json.loads(z.read('_build.json'))
        assert '_build.json' not in manifest['files']
        assert set(z.namelist())==set(manifest['files'])|{'_build.json'}
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected
        version=tomllib.loads(z.read('blender_manifest.toml').decode())['version']
        assert archive.name==f'astro_collision_lab-{version}-windows-x64.zip'
        from astro_collision_lab.interaction import code_fingerprint
        assert manifest['source_fingerprint']==code_fingerprint(tmp_path/'extension/astro_collision_lab')
    first=archive.read_bytes()
    assert mod.build().read_bytes()==first
