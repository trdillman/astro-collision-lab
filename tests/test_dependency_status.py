"""User-facing recovery progress and cleanup of only installer-owned profiles."""
import json
import sys
import types

import pytest
from astro_collision_lab import dependencies as deps


@pytest.fixture
def state(monkeypatch, tmp_path):
    # These cases model a fresh Blender profile, not the host's Warp runtime.
    monkeypatch.delitem(sys.modules, 'warp', raising=False)
    for name, value in (('_PROCESS', None), ('_LOG', None), ('_ERROR', ''),
                        ('_TEMP_ROOT', None), ('_LAST_PROGRESS', ''), ('_ENABLED', True)):
        monkeypatch.setattr(deps, name, value, raising=False)
    monkeypatch.setattr(deps, 'directory', lambda: tmp_path)
    monkeypatch.setitem(sys.modules, 'bpy', types.SimpleNamespace(
        app=types.SimpleNamespace(binary_path='blender', timers=types.SimpleNamespace(
            register=lambda *a, **k: None, is_registered=lambda *a: False)),
        context=types.SimpleNamespace(window_manager=types.SimpleNamespace(windows=[]))))
    yield tmp_path
    if deps._LOG is not None and not deps._LOG.closed:
        deps._LOG.close()


def write_progress(root, **updates):
    data = {'phase': 'downloading', 'downloaded': 50 * 1024**2,
            'total': 100 * 1024**2, 'attempt': 1, 'max_attempts': 4}
    data.update(updates)
    folder = root / '.downloads'
    folder.mkdir(exist_ok=True)
    (folder / 'progress.json').write_text(json.dumps(data))


def test_status_shows_download_and_reconnection_progress(state):
    deps._PROCESS = types.SimpleNamespace(poll=lambda: None)
    write_progress(state)
    text = deps.status()
    assert '50' in text and '100' in text and 'MiB' in text
    write_progress(state, phase='retrying', attempt=2)
    text = deps.status()
    assert '2' in text and '4' in text
    assert 'resum' in text.lower() or 'reconnect' in text.lower()


def test_corrupt_progress_is_tolerated(state):
    deps._PROCESS = types.SimpleNamespace(poll=lambda: None)
    folder = state / '.downloads'
    folder.mkdir()
    for content in ('{broken', '[]', '{"phase":"downloading","downloaded":"bad"}'):
        (folder / 'progress.json').write_text(content)
        assert 'Warp' in deps.status()


def test_running_poll_redraws_when_progress_changes(state, monkeypatch):
    deps._PROCESS = types.SimpleNamespace(poll=lambda: None)
    redraws = []
    monkeypatch.setattr(deps, '_redraw_preferences', lambda: redraws.append(True))
    write_progress(state)
    assert deps._poll() == .5
    assert len(redraws) == 1
    assert deps._poll() == .5
    assert len(redraws) == 1
    write_progress(state, downloaded=60 * 1024**2)
    assert deps._poll() == .5
    assert len(redraws) == 2


def test_completed_child_removes_profile_but_keeps_diagnostic_log(state, monkeypatch):
    owned = state / 'owned-child'
    owned.mkdir()
    (owned / 'temporary').write_text('profile')
    sentinel = state / 'unrelated'
    sentinel.write_text('preserve')
    deps._TEMP_ROOT = owned
    deps._LOG = (state / 'install.log').open('w')
    deps._LOG.write('complete')
    deps._PROCESS = types.SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(deps, 'activate', lambda **kwargs: True)
    assert deps._poll() is None
    assert not owned.exists()
    assert (state / 'install.log').read_text() == 'complete'
    assert sentinel.read_text() == 'preserve'


def test_spawn_failure_removes_profile_and_keeps_log(state, monkeypatch):
    import subprocess
    import tempfile
    owned = state / 'owned-child'
    owned.mkdir()
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kwargs: str(owned))
    def fail(*args, **kwargs):
        raise OSError('launch unavailable')
    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='launch unavailable'):
        deps.install()
    assert not owned.exists()
    assert (state / 'install.log').is_file()
    assert 'Retry' in deps.status()


def test_cancel_removes_only_owned_profile(state):
    owned = state / 'owned-child'
    owned.mkdir()
    (owned / 'temporary').write_text('profile')
    sentinel = state / 'unrelated'
    sentinel.write_text('preserve')
    deps._TEMP_ROOT = owned
    deps._LOG = (state / 'install.log').open('w')
    process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None,
                                    wait=lambda **k: 1, kill=lambda: None)
    deps._PROCESS = process
    deps._stop_installation()
    assert not owned.exists()
    assert sentinel.read_text() == 'preserve'
    assert (state / 'install.log').is_file()


def test_saved_progress_is_visible_after_blender_restart(state, monkeypatch):
    monkeypatch.setattr(deps, 'activate', lambda **kwargs: False)
    folder = state / '.downloads'
    folder.mkdir()
    (folder / (deps.WHEEL + '.part')).write_bytes(b'x' * 1024)
    text = deps.status()
    assert 'saved' in text.lower() and 'resume' in text.lower()


def test_failed_os_termination_preserves_live_child_and_its_profile(state):
    import subprocess
    owned = state / 'owned-child'
    owned.mkdir()
    deps._TEMP_ROOT = owned
    deps._LOG = (state / 'install.log').open('w')
    def timeout(**kwargs):
        raise subprocess.TimeoutExpired('owned', kwargs['timeout'])
    process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None,
                                    wait=timeout, kill=lambda: None)
    deps._PROCESS = process
    deps._stop_installation()
    assert owned.is_dir()
    assert deps._PROCESS is process
    assert 'Cannot stop' in deps._ERROR


def test_reenabled_retained_child_can_finish_and_retry(state, monkeypatch):
    import subprocess
    import tempfile
    owned = state / 'owned-child'
    owned.mkdir()
    deps._TEMP_ROOT = owned
    deps._LOG = (state / 'install.log').open('w')
    deps._LOG.write('RuntimeError: recovered child error\n')
    def timeout(**kwargs):
        raise subprocess.TimeoutExpired('owned', kwargs['timeout'])
    process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None,
                                    wait=timeout, kill=lambda: None, returncode=1)
    deps._PROCESS = process
    deps._stop_installation()
    armed = []
    bpy = sys.modules['bpy']
    bpy.app.handlers = types.SimpleNamespace(persistent=lambda fn: fn, exit_pre=[])
    bpy.app.timers.register = lambda fn, **k: armed.append(fn)
    deps.register()
    process.poll = lambda: 1
    assert deps._poll() is None
    assert deps._poll in armed
    assert not owned.exists() and deps._PROCESS is None
    assert 'recovered child error' in deps.status()
    second = state / 'second-owned'
    second.mkdir()
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **k: str(second))
    next_process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None,
                                         wait=lambda **k: 0, kill=lambda: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: next_process)
    deps.install()
    assert deps._PROCESS is next_process
    deps._stop_installation()


def test_actual_hash_failure_is_not_masked_by_blender_footer(state, monkeypatch):
    from astro_collision_lab import install_dependency as installer
    folder = state / '.downloads'
    folder.mkdir()
    (folder / (installer.NAME + '.part')).write_bytes(b'bad')
    monkeypatch.setattr(installer, '_metadata', lambda: ('https://files.pythonhosted.org/test', 3))
    with pytest.raises(RuntimeError) as caught:
        installer.main(state)
    path = state / 'install.log'
    deps._LOG = path.open('w')
    deps._LOG.write(type(caught.value).__name__ + ': ' + str(caught.value) +
                    '\nError: script failed, exiting.\nBlender quit\n')
    deps._PROCESS = types.SimpleNamespace(poll=lambda: 1, returncode=1)
    deps._poll()
    assert 'hash mismatch' in deps.status()
    assert 'Retry' in deps.status()
