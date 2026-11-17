"""Installer failure injection: real archives/files, with no network or Blender."""
import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from astro_collision_lab import dependencies as deps
from astro_collision_lab import install_dependency as installer


def request_url(request):
    return getattr(request, 'full_url', request)


class WheelResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}

    def getcode(self):
        return 200


def wheel(entries=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, content in (entries or {'warp/__init__.py': '__version__="1.18.0"',
                                           'warp/bin/warp.dll': 'binary',
                                           'warp_lang-1.18.0.dist-info/METADATA': 'Version: 1.18.0'}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def download(monkeypatch):
    def configure(data=None, metadata=None, failure=None):
        data = wheel() if data is None else data
        digest = hashlib.sha256(data).hexdigest()
        monkeypatch.setattr(installer, 'SHA', digest)
        meta = metadata or {'urls': [{'filename': installer.NAME, 'digests': {'sha256': digest},
                                     'size': len(data),
                                     'url': 'https://files.pythonhosted.org/packages/warp.whl'}]}
        def open_url(url, **kwargs):
            if failure:
                raise failure
            return (io.BytesIO(json.dumps(meta).encode()) if request_url(url).endswith('/json')
                    else WheelResponse(data))
        monkeypatch.setattr(installer.urllib.request, 'urlopen', open_url)
        return data
    return configure


def preserved(destination):
    (destination / 'warp').mkdir()
    (destination / 'warp/__init__.py').write_text('old working install')
    (destination / 'unrelated.txt').write_text('sentinel')


def assert_preserved(destination):
    assert (destination / 'warp/__init__.py').read_text() == 'old working install'
    assert (destination / 'unrelated.txt').read_text() == 'sentinel'


def test_success_publishes_owned_version_and_preserves_other_files(tmp_path, download, capsys):
    preserved(tmp_path)
    download()
    installer.main(tmp_path)
    log = capsys.readouterr().out
    assert installer.NAME in log and installer.SHA in log and str(tmp_path) in log
    assert 'Starting PyPI metadata' in log and 'Starting wheel download' in log
    assert (tmp_path / 'warp-1.18.0/warp/bin/warp.dll').read_text() == 'binary'
    assert_preserved(tmp_path)
    installer.main(tmp_path)  # Safe retry after a completed publication.
    assert not list(tmp_path.glob('.acl-warp-stage-*'))


@pytest.mark.parametrize('failure', [OSError('download unavailable'), KeyboardInterrupt()])
def test_download_failure_preserves_install(tmp_path, download, failure):
    preserved(tmp_path)
    download(failure=failure)
    with pytest.raises(type(failure)):
        installer.main(tmp_path)
    assert_preserved(tmp_path)
    assert not (tmp_path / 'warp-1.18.0').exists()


def test_extraction_interruption_cleans_own_staging_and_retry_works(tmp_path, download, monkeypatch):
    preserved(tmp_path)
    download()
    real = zipfile.ZipFile.extractall
    def interrupt(self, path, *args, **kwargs):
        (Path(path) / 'partial').write_text('incomplete')
        raise KeyboardInterrupt()
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', interrupt)
    with pytest.raises(KeyboardInterrupt):
        installer.main(tmp_path)
    assert_preserved(tmp_path)
    assert not list(tmp_path.glob('.acl-warp-stage-*'))
    assert not (tmp_path / 'partial').exists()
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', real)
    installer.main(tmp_path)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/drive', 'warp\\escape',
                                  'unrelated.txt', 'warp/../../escape'])
def test_archive_rejects_unowned_or_unsafe_entries(download, name):
    # ZipInfo normalizes Windows separators when creating a ZIP. Replace both
    # filename records to exercise an actual hostile backslash on Windows too.
    encoded = wheel({'warp/__init__.py': '', name: 'bad'})
    if '\\' in name:
        encoded = encoded.replace(name.replace('\\', '/').encode(), name.encode())
    data = download(encoded)
    with pytest.raises(RuntimeError, match='archive entry'):
        installer.verified_archive(data)


def test_missing_warp_package_fails_without_publication(tmp_path, download):
    download(wheel({'warp_lang-1.18.0.dist-info/METADATA': 'Version: 1.18.0'}))
    with pytest.raises(RuntimeError, match='Warp package'):
        installer.main(tmp_path)
    assert not (tmp_path / 'warp-1.18.0').exists()


def test_wrong_download_hash_preserves_existing_install(tmp_path, download, monkeypatch):
    preserved(tmp_path)
    data = download()
    original = installer.urllib.request.urlopen
    monkeypatch.setattr(installer.urllib.request, 'urlopen',
                        lambda url, **kwargs: original(url, **kwargs) if request_url(url).endswith('/json')
                        else WheelResponse(data[:-1] + bytes([data[-1] ^ 1])))
    with pytest.raises(RuntimeError, match='hash mismatch'):
        installer.main(tmp_path)
    assert_preserved(tmp_path)


def test_unexpected_target_and_abandoned_staging_are_preserved(tmp_path, download):
    preserved(tmp_path)
    download()
    orphan = tmp_path / '.acl-warp-stage-abandoned'
    orphan.mkdir()
    (orphan / 'partial').write_text('interrupted')
    target = tmp_path / 'warp-1.18.0'
    target.mkdir()
    (target / 'sentinel').write_text('unknown ownership')
    with pytest.raises(RuntimeError, match='without certification'):
        installer.main(tmp_path)
    assert (target / 'sentinel').read_text() == 'unknown ownership'
    assert (orphan / 'partial').read_text() == 'interrupted'
    assert_preserved(tmp_path)


def test_symlink_archive_entry_is_rejected(download):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('warp/__init__.py', '')
        link = zipfile.ZipInfo('warp/link')
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        archive.writestr(link, '../../sentinel')
    with pytest.raises(RuntimeError, match='archive entry'):
        installer.verified_archive(download(buffer.getvalue()))


def test_changed_hash_fails_before_extraction(tmp_path, download, monkeypatch):
    download()
    monkeypatch.setattr(installer, 'SHA', '0' * 64)
    with pytest.raises(RuntimeError, match='metadata'):
        installer.main(tmp_path)
    assert not (tmp_path / 'warp-1.18.0').exists()


@pytest.mark.parametrize('url', ['http://files.pythonhosted.org/a', 'https://example.com/a'])
def test_metadata_download_origin_is_pinned(tmp_path, download, url):
    data = wheel()
    download(data, {'urls': [{'filename': installer.NAME,
                             'digests': {'sha256': hashlib.sha256(data).hexdigest()}, 'url': url}]})
    with pytest.raises(RuntimeError, match='download URL'):
        installer.main(tmp_path)


@pytest.fixture
def dependency_state(monkeypatch, tmp_path):
    import tempfile
    real_mkdtemp = tempfile.mkdtemp
    def owned_mkdtemp(suffix=None, prefix=None, dir=None):
        return real_mkdtemp(suffix=suffix, prefix=prefix, dir=tmp_path if dir is None else dir)
    monkeypatch.setattr(tempfile, 'mkdtemp', owned_mkdtemp)
    monkeypatch.setattr(deps, '_ENABLED', True, raising=False)
    monkeypatch.setattr(deps, '_PROCESS', None)
    monkeypatch.setattr(deps, '_LOG', None)
    monkeypatch.setattr(deps, '_TEMP_ROOT', None, raising=False)
    monkeypatch.setattr(deps, '_ERROR', '')
    monkeypatch.setattr(deps, 'directory', lambda: tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    callbacks = set()
    timers = types.SimpleNamespace(register=lambda fn, **kwargs: callbacks.add(fn),
                                   is_registered=lambda fn: fn in callbacks,
                                   unregister=lambda fn: callbacks.remove(fn))
    handlers = types.SimpleNamespace(exit_pre=[], persistent=lambda fn: fn)
    bpy = types.SimpleNamespace(app=types.SimpleNamespace(binary_path='blender',
                               timers=timers, handlers=handlers))
    monkeypatch.setitem(sys.modules, 'bpy', bpy)
    yield tmp_path
    if deps._LOG:
        deps._LOG.close()
        deps._LOG = None
    deps._cleanup_temp()


def test_loaded_wrong_version_is_actionable_and_never_shadowed(dependency_state, monkeypatch):
    monkeypatch.setitem(sys.modules, 'warp', types.SimpleNamespace(__version__='1.14.0'))
    (dependency_state / 'warp-1.18.0/warp').mkdir(parents=True)
    (dependency_state / 'warp-1.18.0/warp/__init__.py').write_text('')
    before = list(sys.path)
    assert not deps.activate()
    assert '1.14.0' in deps.status()
    assert 'Restart Blender' in deps.status()
    assert sys.path == before


def test_spawn_failure_closes_log_and_allows_retry(dependency_state, monkeypatch):
    import subprocess
    def fail(*args, **kwargs):
        assert not kwargs['stdout'].closed
        fail.log = kwargs['stdout']
        raise OSError('cannot launch Blender')
    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='cannot launch'):
        deps.install()
    assert fail.log.closed
    assert deps._PROCESS is None
    assert 'cannot launch Blender' in deps.status()


def test_poll_reports_error_tail_and_log_location(dependency_state):
    path = dependency_state / 'install.log'
    deps._LOG = path.open('w', encoding='utf-8')
    deps._LOG.write('Traceback\nRuntimeError: Certified Warp wheel hash mismatch\n')
    deps._PROCESS = types.SimpleNamespace(poll=lambda: 1, returncode=1)
    assert deps._poll() is None
    assert deps._PROCESS is None and deps._LOG is None
    assert 'hash mismatch' in deps.status()
    assert str(path) in deps.status()
    assert 'Retry' in deps.status()
    assert deps._poll() is None  # An obsolete timer must be harmless.


def test_install_uses_isolated_hidden_child_and_ignores_duplicate_request(dependency_state, monkeypatch):
    import os
    import subprocess
    calls = []
    process = types.SimpleNamespace(poll=lambda: None)
    def spawn(args, **kwargs):
        calls.append((args, kwargs))
        return process
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    try:
        deps.install()
        deps.install()
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert '--factory-startup' in args and '--background' in args
        assert kwargs['creationflags'] == (subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for name in ('RESOURCES', 'CONFIG', 'EXTENSIONS', 'SCRIPTS', 'DATAFILES'):
            assert Path(kwargs['env']['BLENDER_USER_' + name]).is_dir()
        assert deps._poll() == .5
    finally:
        if deps._LOG:
            deps._LOG.close()


def test_successful_poll_requires_compatible_loaded_version(dependency_state, monkeypatch):
    monkeypatch.setitem(sys.modules, 'warp', types.SimpleNamespace(__version__='1.14.0'))
    deps._LOG = (dependency_state / 'install.log').open('w')
    deps._PROCESS = types.SimpleNamespace(poll=lambda: 0, returncode=0)
    assert deps._poll() is None
    assert '1.14.0' in deps.status() and 'Restart Blender' in deps.status()


def test_uncertified_version_directory_is_not_activated(dependency_state, monkeypatch):
    monkeypatch.setitem(sys.modules, 'warp', types.SimpleNamespace(__version__='1.18.0'))
    package = dependency_state / 'warp-1.18.0/warp'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('__version__="1.18.0"')
    deps.activate()
    assert str(package.parent) not in sys.path


@pytest.mark.parametrize('exiting', [False, True])
@pytest.mark.parametrize('needs_kill', [False, True])
def test_disable_or_exit_stops_exact_owned_child_and_closes_log(dependency_state, monkeypatch, exiting, needs_kill):
    import subprocess
    events = []
    def wait(timeout):
        events.append(('wait', timeout))
        if needs_kill and 'kill' not in events:
            raise subprocess.TimeoutExpired('owned-blender', timeout)
    process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: events.append('terminate'),
                                    wait=wait, kill=lambda: events.append('kill'))
    deps._PROCESS = process
    log = (dependency_state / 'install.log').open('w')
    deps._LOG = log
    bpy = sys.modules['bpy']
    deps.register()
    deps.register()
    assert len(bpy.app.handlers.exit_pre) == 1
    bpy.app.timers.register(deps._poll)
    monkeypatch.setattr(deps, 'activate', lambda **kwargs: pytest.fail('disabled add-on imported Warp'))
    if exiting:
        bpy.app.handlers.exit_pre[0]()
    else:
        deps.unregister()
        assert not bpy.app.handlers.exit_pre
    assert events[0] == 'terminate'
    assert ('kill' in events) == needs_kill
    assert log.closed and deps._PROCESS is None and deps._LOG is None
    assert not bpy.app.timers.is_registered(deps._poll)
    assert deps._poll() is None
    deps.status()
    deps.unregister()  # Repeated cleanup is safe.


def test_poll_survives_file_load_and_is_registered_once(dependency_state, monkeypatch):
    import subprocess
    calls = []
    timers = sys.modules['bpy'].app.timers
    timers.register = lambda fn, **kwargs: calls.append((fn, kwargs))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: types.SimpleNamespace(poll=lambda: None))
    try:
        deps.install()
        deps.install()
        assert len(calls) == 1
        assert calls[0][1]['persistent'] is True
    finally:
        if deps._LOG:
            deps._LOG.close()


def test_download_reads_bounded_chunks_and_reports_progress(tmp_path, download, monkeypatch, capsys):
    data = download(wheel({'warp/__init__.py': 'x' * (11 * 1024 * 1024)}))
    original = installer.urllib.request.urlopen
    sizes = []
    class Stream(WheelResponse):
        def read(self, size=-1):
            sizes.append(size)
            assert 0 < size <= 1024 * 1024
            return super().read(size)
    monkeypatch.setattr(installer.urllib.request, 'urlopen',
                        lambda url, **kwargs: original(url, **kwargs) if request_url(url).endswith('/json') else Stream(data))
    installer.main(tmp_path)
    assert len(sizes) > 10
    assert 'Downloaded 10 MiB' in capsys.readouterr().out


def test_addon_registration_integrates_dependency_cleanup(monkeypatch):
    import astro_collision_lab as addon
    events = []
    monkeypatch.setattr(addon, '_REGISTERED', False)
    runtime = types.SimpleNamespace(register_handlers=lambda: events.append('runtime register'),
                                    shutdown=lambda: events.append('runtime shutdown'),
                                    unregister_handlers=lambda: events.append('runtime unregister'))
    ui = types.SimpleNamespace(register=lambda: None, unregister=lambda: None)
    authoring = types.SimpleNamespace(register=lambda: None, unregister=lambda: None)
    for name, value in [('runtime', runtime), ('ui', ui), ('authoring', authoring)]:
        monkeypatch.setattr(addon, name, value, raising=False)
    monkeypatch.setattr(deps, 'register', lambda: events.append('dependencies register'))
    monkeypatch.setattr(deps, 'unregister', lambda: events.append('dependencies unregister'))
    addon.register()
    addon.register()
    addon.unregister()
    assert events == ['dependencies register', 'runtime register', 'dependencies unregister',
                      'runtime shutdown', 'runtime unregister']


def test_disabled_dependency_activation_cannot_import_warp(dependency_state, monkeypatch):
    deps.unregister()
    monkeypatch.setattr(deps, 'directory', lambda: pytest.fail('disabled activation touched dependencies'))
    assert not deps.activate()
    with pytest.raises(RuntimeError, match='Enable Astro Collision Lab'):
        deps.activate(required=True)


@pytest.fixture
def redraw_events(dependency_state):
    events = []
    def area(kind, label):
        return types.SimpleNamespace(type=kind, tag_redraw=lambda: events.append(label))
    windows = [types.SimpleNamespace(screen=types.SimpleNamespace(areas=[area('PREFERENCES', 'first'),
                                                                       area('VIEW_3D', 'viewport')])),
               types.SimpleNamespace(screen=types.SimpleNamespace(areas=[area('PREFERENCES', 'second')])),
               types.SimpleNamespace(screen=None)]
    sys.modules['bpy'].context = types.SimpleNamespace(window_manager=types.SimpleNamespace(windows=windows))
    return events


@pytest.mark.parametrize('result', [0, 1])
def test_completed_poll_redraws_preferences_in_all_windows(dependency_state, redraw_events, monkeypatch, result):
    monkeypatch.setattr(deps, 'activate', lambda **kwargs: True)
    deps._LOG = (dependency_state / 'install.log').open('w')
    deps._PROCESS = types.SimpleNamespace(poll=lambda: result, returncode=result)
    deps._poll()
    assert redraw_events == ['first', 'second']
    deps._poll()
    assert redraw_events == ['first', 'second']


def test_timeout_error_precedes_blender_wrapper_and_keeps_retry_visible(dependency_state):
    path = dependency_state / 'install.log'
    deps._LOG = path.open('w')
    deps._LOG.write('Traceback (most recent call last):\n  File "installer.py", line 1\n'
                    'TimeoutError: The read operation timed out\n'
                    'Error: script failed, file: "installer.py", exiting.\nBlender quit\n')
    deps._PROCESS = types.SimpleNamespace(poll=lambda: 1, returncode=1)
    deps._poll()
    status = deps.status()
    assert status.startswith('TimeoutError:')
    assert 'Retry' in status[:60]
    assert 'read operation timed out' in status
    assert 'script failed' not in status
    assert str(path) in status


def test_spawn_failure_redraws_preferences(dependency_state, redraw_events, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(OSError('launch failed')))
    with pytest.raises(OSError):
        deps.install()
    assert redraw_events == ['first', 'second']
    assert 'Retry' in deps.status()[:60]


@pytest.mark.parametrize('bpy', [None, types.SimpleNamespace(), types.SimpleNamespace(context=None)])
def test_preferences_redraw_tolerates_no_blender_context(monkeypatch, bpy):
    monkeypatch.setitem(sys.modules, 'bpy', bpy)
    deps._redraw_preferences()
