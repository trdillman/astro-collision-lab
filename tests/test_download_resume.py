"""Real disk and loopback HTTP coverage for the pinned resumable installer."""
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import http.client
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from astro_collision_lab import install_dependency as installer


@pytest.fixture
def transfer(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('warp/__init__.py', '__version__="1.18.0"')
        archive.writestr('warp/bin/warp.dll', b'x' * (2 * 1024 * 1024 + 12345))
    data = buffer.getvalue()
    monkeypatch.setattr(installer, 'SHA', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    state = {'data': data, 'requests': [], 'actions': [], 'metadata_calls': 0}
    release = threading.Event()
    state['release'] = release

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state['requests'].append(dict(self.headers))
            offset = int(self.headers.get('Range', 'bytes=0-')[6:-1])
            action = state['actions'].pop(0) if state['actions'] else 'normal'
            if isinstance(action, int):
                self.send_error(action)
                return
            status = 206 if offset else 200
            body = state['data'][offset:]
            if action == 'ignore':
                status, offset, body = 200, 0, state['data']
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            if status == 206:
                start = offset + 1 if action == 'wrong_range' else offset
                total = len(data) + 1 if action == 'wrong_total' else len(data)
                end = len(data) - 2 if action == 'wrong_end' else len(data) - 1
                self.send_header('Content-Range', f'bytes {start}-{end}/{total}')
            if action == 'encoded':
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            if action in ('short', 'hold', 'timeout'):
                body = body[:1024 * 1024 + 17]
            try:
                self.wfile.write(body)
                self.wfile.flush()
                if action == 'hold':
                    release.wait(timeout=15)
                elif action == 'timeout':
                    release.wait(timeout=0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real_open = urllib.request.urlopen
    local_url = f'http://127.0.0.1:{server.server_port}/wheel'
    state['url'] = local_url

    def open_url(request, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        if url.endswith('/json'):
            state['metadata_calls'] += 1
            if state.get('metadata_failure'):
                raise OSError('metadata offline')
            metadata = {'urls': [{'filename': installer.NAME,
                                 'digests': {'sha256': installer.SHA},
                                 'size': state.get('size', len(data)),
                                 'url': 'https://files.pythonhosted.org/packages/' + installer.NAME}]}
            return io.BytesIO(json.dumps(metadata).encode())
        headers = dict(request.header_items()) if isinstance(request, urllib.request.Request) else {}
        if 'timeout' in state:
            kwargs['timeout'] = state['timeout']
        return real_open(urllib.request.Request(local_url, headers=headers), **kwargs)

    monkeypatch.setattr(installer.urllib.request, 'urlopen', open_url)
    yield state
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def part(destination):
    return destination / '.downloads' / (installer.NAME + '.part')


def seed(destination, data):
    path = part(destination)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    (path.parent / 'unrelated.txt').write_text('keep')
    return path


def test_interrupted_http_transfer_resumes_exact_persisted_offset(tmp_path, transfer, capsys):
    transfer['actions'] = ['short', 'normal']
    installer.main(tmp_path)
    assert len(transfer['requests']) == 2
    assert transfer['requests'][1]['Range'] == 'bytes=1048593-'
    assert all(request['Accept-Encoding'] == 'identity' for request in transfer['requests'])
    assert (tmp_path / 'warp-1.18.0/warp/bin/warp.dll').read_bytes() == b'x' * (2 * 1024 * 1024 + 12345)
    assert not part(tmp_path).exists()
    assert not (tmp_path / '.downloads/progress.json').exists()
    assert f'Accepted HTTP 206; starting byte 1048593 of {len(transfer["data"])}' in capsys.readouterr().out


def test_retry_budget_keeps_progress_and_next_invocation_resumes(tmp_path, transfer):
    transfer['actions'] = ['short', 503, 503, 503]
    seed(tmp_path, b'')
    with pytest.raises(RuntimeError, match='saved.*resume'):
        installer.main(tmp_path)
    saved = part(tmp_path).read_bytes()
    assert saved == transfer['data'][:1048593]
    assert len(transfer['requests']) == 4
    progress = json.loads((tmp_path / '.downloads/progress.json').read_text())
    assert progress == {'phase': 'failed', 'downloaded': len(saved), 'total': len(transfer['data']),
                        'attempt': 4, 'max_attempts': 4}
    installer.main(tmp_path)
    assert transfer['requests'][-1]['Range'] == f'bytes={len(saved)}-'
    assert (tmp_path / '.downloads/unrelated.txt').read_text() == 'keep'


def test_ignored_range_truncates_before_restart(tmp_path, transfer, capsys):
    seed(tmp_path, transfer['data'][:51])
    transfer['actions'] = ['ignore']
    installer.main(tmp_path)
    assert transfer['requests'][0]['Range'] == 'bytes=51-'
    assert (tmp_path / 'warp-1.18.0/.certified-sha256').read_text() == installer.SHA
    log = capsys.readouterr().out
    assert 'Server ignored Range; restarting at byte 0' in log
    assert f'Accepted HTTP 200; starting byte 0 of {len(transfer["data"])}' in log


@pytest.mark.parametrize('action', ['wrong_range', 'wrong_total', 'wrong_end', 'encoded', 400, 416])
def test_protocol_errors_are_not_retried_or_appended(tmp_path, transfer, action):
    original = transfer['data'][:51]
    seed(tmp_path, original)
    transfer['actions'] = [action]
    with pytest.raises(Exception):
        installer.main(tmp_path)
    assert len(transfer['requests']) == 1
    assert part(tmp_path).read_bytes() == original
    assert not (tmp_path / 'warp-1.18.0').exists()


@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
def test_transient_http_errors_resume_within_bounded_budget(tmp_path, transfer, status):
    seed(tmp_path, transfer['data'][:73])
    transfer['actions'] = [status, 'normal']
    installer.main(tmp_path)
    assert [r['Range'] for r in transfer['requests']] == ['bytes=73-', 'bytes=73-']


def test_complete_cache_installs_offline_and_is_removed_only_after_publish(tmp_path, transfer):
    seed(tmp_path, transfer['data'])
    transfer['metadata_failure'] = True
    installer.main(tmp_path)
    assert transfer['metadata_calls'] == 0 and not transfer['requests']
    assert not part(tmp_path).exists()
    assert (tmp_path / '.downloads/unrelated.txt').read_text() == 'keep'


def test_verified_cache_survives_extraction_failure(tmp_path, transfer, monkeypatch):
    seed(tmp_path, transfer['data'])
    def fail(*args, **kwargs):
        raise OSError('disk extraction failure')
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', fail)
    with pytest.raises(OSError, match='disk extraction failure'):
        installer.main(tmp_path)
    assert part(tmp_path).read_bytes() == transfer['data']
    assert not list(tmp_path.glob('.acl-warp-stage-*'))


def test_wrong_hash_discards_only_owned_partial_and_progress(tmp_path, transfer):
    seed(tmp_path, b'!' * len(transfer['data']))
    with pytest.raises(RuntimeError, match='hash mismatch'):
        installer.main(tmp_path)
    assert not transfer['requests']
    assert not part(tmp_path).exists()
    assert not (tmp_path / '.downloads/progress.json').exists()
    assert (tmp_path / '.downloads/unrelated.txt').read_text() == 'keep'


@pytest.mark.parametrize('size', [None, 0, -1, True, '123'])
def test_missing_or_invalid_metadata_size_preserves_partial(tmp_path, transfer, size):
    transfer['size'] = size
    seed(tmp_path, transfer['data'][:39])
    with pytest.raises(RuntimeError, match='size'):
        installer.main(tmp_path)
    assert not transfer['requests']
    assert part(tmp_path).read_bytes() == transfer['data'][:39]


def test_metadata_failure_preserves_partial_and_reports_failed_progress(tmp_path, transfer):
    seed(tmp_path, transfer['data'][:39])
    transfer['metadata_failure'] = True
    with pytest.raises(OSError, match='metadata offline'):
        installer.main(tmp_path)
    assert part(tmp_path).stat().st_size == 39
    assert json.loads((tmp_path / '.downloads/progress.json').read_text())['phase'] == 'failed'


def test_existing_certified_target_needs_no_network(tmp_path, transfer):
    target = tmp_path / 'warp-1.18.0'
    (target / 'warp').mkdir(parents=True)
    (target / 'warp/__init__.py').write_text('installed')
    (target / '.certified-sha256').write_text(installer.SHA)
    transfer['metadata_failure'] = True
    installer.main(tmp_path)
    assert not transfer['requests'] and transfer['metadata_calls'] == 0


def test_real_socket_timeout_preserves_completed_chunks_and_resumes(tmp_path, transfer):
    transfer['timeout'] = 0.05
    transfer['actions'] = ['timeout', 'normal']
    installer.main(tmp_path)
    assert len(transfer['requests']) == 2
    # http.client cannot expose the 17 bytes buffered by the timed-out read;
    # the next Range must follow bytes actually committed to the cache.
    assert transfer['requests'][1]['Range'] == 'bytes=1048576-'


def test_killed_installer_is_resumed_by_new_process_without_stale_lock(tmp_path, transfer):
    transfer['actions'] = ['hold', 'normal']
    first = transfer['requests']
    code = '''
import importlib.util, io, json, sys, urllib.request
from pathlib import Path
s = importlib.util.spec_from_file_location('installer', sys.argv[1])
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.SHA = sys.argv[3]
real_open = urllib.request.urlopen
def open_url(request, **kwargs):
    url = getattr(request, 'full_url', request)
    if url.endswith('/json'):
        return io.BytesIO(json.dumps({'urls': [{'filename': m.NAME, 'digests': {'sha256': m.SHA},
            'size': int(sys.argv[4]), 'url': 'https://files.pythonhosted.org/packages/' + m.NAME}]}).encode())
    return real_open(urllib.request.Request(sys.argv[5], headers=dict(request.header_items())), **kwargs)
m.urllib.request.urlopen = open_url
m.main(Path(sys.argv[2]))
'''
    arguments = [sys.executable, '-u', '-c', code, installer.__file__, str(tmp_path),
                 installer.SHA, str(len(transfer['data'])), transfer['url']]
    child = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if part(tmp_path).exists() and part(tmp_path).stat().st_size >= 1024 * 1024:
                break
            if child.poll() is not None:
                pytest.fail('installer exited before interruption: ' + child.communicate()[1])
            threading.Event().wait(0.01)
        assert part(tmp_path).stat().st_size >= 1024 * 1024
    finally:
        child.kill()
        child.communicate(timeout=10)
        transfer['release'].set()
    saved = part(tmp_path).read_bytes()
    assert saved == transfer['data'][:len(saved)]
    completed = subprocess.run(arguments, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert first[-1]['Range'] == f'bytes={len(saved)}-'
    assert not part(tmp_path).exists()
    assert (tmp_path / 'warp-1.18.0/.certified-sha256').read_text() == installer.SHA


def test_incomplete_read_exception_commits_its_partial_bytes(tmp_path, transfer, monkeypatch):
    original = installer.urllib.request.urlopen
    failed = False
    class Interrupted(io.BytesIO):
        headers = {'Content-Length': str(len(transfer['data']))}
        def getcode(self):
            return 200
        def read(self, size):
            raise http.client.IncompleteRead(transfer['data'][:87], len(transfer['data']) - 87)
    def open_url(request, **kwargs):
        nonlocal failed
        if not getattr(request, 'full_url', request).endswith('/json') and not failed:
            failed = True
            return Interrupted()
        return original(request, **kwargs)
    monkeypatch.setattr(installer.urllib.request, 'urlopen', open_url)
    installer.main(tmp_path)
    assert transfer['requests'][0]['Range'] == 'bytes=87-'


def test_corrupt_completed_download_is_not_retried_or_published(tmp_path, transfer):
    transfer['data'] = b'!' * len(transfer['data'])
    seed(tmp_path, b'')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        installer.main(tmp_path)
    assert len(transfer['requests']) == 1
    assert not part(tmp_path).exists()
    assert not (tmp_path / 'warp-1.18.0').exists()
    assert (tmp_path / '.downloads/unrelated.txt').read_text() == 'keep'


def test_short_disk_write_retries_from_bytes_actually_saved(tmp_path, transfer, monkeypatch):
    real_open = Path.open
    short_write = False
    class Writer:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def seek(self, offset):
            return self.stream.seek(offset)
        def write(self, chunk):
            nonlocal short_write
            if not short_write:
                short_write = True
                return self.stream.write(chunk[:len(chunk) // 2])
            return self.stream.write(chunk)
    def open_path(path, mode='r', *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        return Writer(stream) if path == part(tmp_path) and mode in ('wb', 'r+b') else stream
    monkeypatch.setattr(Path, 'open', open_path)
    installer.main(tmp_path)
    assert transfer['requests'][1]['Range'] == 'bytes=524288-'


def test_lock_is_exclusive_and_released_after_process_termination(tmp_path):
    cache = tmp_path / '.downloads'
    cache.mkdir()
    seed(tmp_path, b'persist through kill')
    code = ('import importlib.util,sys; from pathlib import Path; '
            's=importlib.util.spec_from_file_location("installer",sys.argv[1]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'lock=m._installer_lock(Path(sys.argv[2])); lock.__enter__(); '
            'print("locked",flush=True); sys.stdin.read()')
    child = subprocess.Popen([sys.executable, '-u', '-c', code, installer.__file__, str(cache)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'locked'
        with pytest.raises(RuntimeError, match='Another installer'):
            with installer._installer_lock(cache):
                pytest.fail('concurrent owner entered lock')
    finally:
        child.kill()
        child.communicate(timeout=10)
    with installer._installer_lock(cache):
        assert part(tmp_path).read_bytes() == b'persist through kill'
    assert (cache / 'install.lock').is_file()


@pytest.mark.skipif(os.name != 'nt', reason='Windows readers deny replacement without delete sharing')
def test_progress_tolerates_actual_windows_reader_and_cleans_temporary_file(tmp_path):
    installer._progress(tmp_path, 'downloading', 0, 1024, 1)
    progress = tmp_path / 'progress.json'
    with progress.open('rb') as reader:
        installer._progress(tmp_path, 'downloading', 512, 1024, 1)
        assert json.loads(reader.read())['downloaded'] == 0
        assert not (tmp_path / 'progress.json.tmp').exists()
    installer._progress(tmp_path, 'downloading', 1024, 1024, 1)
    assert json.loads(progress.read_text())['downloaded'] == 1024


def test_progress_preserves_non_sharing_permission_errors(tmp_path, monkeypatch):
    installer._progress(tmp_path, 'downloading', 0, 1024, 1)
    def denied(*args, **kwargs):
        raise PermissionError('ordinary permission failure')
    monkeypatch.setattr(Path, 'replace', denied)
    with pytest.raises(PermissionError, match='ordinary permission failure'):
        installer._progress(tmp_path, 'downloading', 512, 1024, 1)


def test_progress_creation_permission_errors_are_not_hidden(tmp_path, monkeypatch):
    installer._progress(tmp_path, 'downloading', 0, 1024, 1)
    def denied(*args, **kwargs):
        error = PermissionError('temporary file creation denied')
        error.winerror = 5
        raise error
    monkeypatch.setattr(Path, 'write_text', denied)
    with pytest.raises(PermissionError, match='temporary file creation denied'):
        installer._progress(tmp_path, 'downloading', 512, 1024, 1)


@pytest.mark.skipif(os.name != 'nt', reason='Windows readers deny deletion without delete sharing')
def test_download_cleanup_skips_only_locked_advisory_progress(tmp_path):
    partial = seed(tmp_path, b'owned payload')
    cache = partial.parent
    installer._progress(cache, 'installing', 13, 13, 1)
    temporary = cache / 'progress.json.tmp'
    temporary.write_text('abandoned advisory update')
    with (cache / 'progress.json').open('rb') as reader:
        installer._clear_download(cache, partial)
        assert not partial.exists()
        assert not temporary.exists()
        assert json.loads(reader.read())['phase'] == 'installing'
    installer._clear_download(cache, partial)
    assert not (cache / 'progress.json').exists()
    assert (cache / 'unrelated.txt').read_text() == 'keep'


@pytest.mark.skipif(os.name != 'nt', reason='Windows readers deny deletion without delete sharing')
def test_successful_install_survives_progress_reader_during_cleanup(tmp_path, transfer):
    partial = seed(tmp_path, transfer['data'])
    installer._progress(partial.parent, 'downloading', 0, len(transfer['data']), 1)
    with (partial.parent / 'progress.json').open('rb'):
        installer.main(tmp_path)
    assert (tmp_path / 'warp-1.18.0/.certified-sha256').read_text() == installer.SHA
    assert not partial.exists()
    assert not (partial.parent / 'progress.json.tmp').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows readers deny deletion without delete sharing')
def test_hash_failure_preserved_when_progress_cleanup_has_reader(tmp_path, transfer):
    partial = seed(tmp_path, b'!' * len(transfer['data']))
    installer._progress(partial.parent, 'downloading', len(transfer['data']), len(transfer['data']), 1)
    with (partial.parent / 'progress.json').open('rb'):
        with pytest.raises(installer._HashMismatchError, match='hash mismatch'):
            installer.main(tmp_path)
    assert not partial.exists()
    assert not (partial.parent / 'progress.json.tmp').exists()
    assert not (tmp_path / 'warp-1.18.0').exists()


def test_download_cleanup_does_not_hide_payload_permission_errors(tmp_path, monkeypatch):
    partial = seed(tmp_path, b'payload')
    real_unlink = Path.unlink
    def denied(path, *args, **kwargs):
        if path == partial:
            error = PermissionError('payload deletion denied')
            error.winerror = 32
            raise error
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', denied)
    with pytest.raises(PermissionError, match='payload deletion denied'):
        installer._clear_download(partial.parent, partial)
