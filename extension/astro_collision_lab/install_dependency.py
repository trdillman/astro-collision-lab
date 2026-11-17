"""Isolated dependency installer. Fetch only the hash-pinned official wheel."""
import contextlib
import hashlib
import http.client
import io
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
NAME='warp_lang-1.18.0-py3-none-win_amd64.whl'
SHA='1769e55746739f2e244a2b412123003c462ec3d977dfb5f420f924a8553d30aa'
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 4


class _HashMismatchError(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()

def verified_archive(data):
    if hashlib.sha256(data).hexdigest()!=SHA:raise RuntimeError('Certified Warp wheel hash mismatch')
    archive=zipfile.ZipFile(io.BytesIO(data))
    return _safe_archive(archive)


def _safe_archive(archive):
    for info in archive.infolist():
        name=info.orig_filename
        path=pathlib.PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or '\x00' in name
                or not path.parts or path.parts[0] not in ('warp','warp_lang-1.18.0.dist-info')
                or stat.S_ISLNK(info.external_attr >> 16)):
            archive.close()
            raise RuntimeError('Unsafe or unowned archive entry: '+name)
    if 'warp/__init__.py' not in archive.namelist():
        archive.close()
        raise RuntimeError('Certified wheel is missing the Warp package')
    return archive


@contextlib.contextmanager
def _installer_lock(cache):
    """The tiny lock file stays in place; the OS releases ownership on exit/kill."""
    lock_path = cache / 'install.lock'
    if lock_path.is_symlink():
        raise RuntimeError('Unsafe installer lock path: ' + str(lock_path))
    with lock_path.open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another installer is using this dependency destination; retry after it exits') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _progress(cache, phase, downloaded, total, attempt=0):
    temporary = cache / 'progress.json.tmp'
    path = cache / 'progress.json'
    if temporary.is_symlink() or path.is_symlink():
        raise RuntimeError('Unsafe installer progress path')
    temporary.write_text(json.dumps({'phase': phase, 'downloaded': downloaded,
                                    'total': total, 'attempt': attempt,
                                    'max_attempts': MAX_ATTEMPTS}), encoding='utf-8')
    for publication in range(3):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            # Windows UI readers briefly deny delete sharing. Progress is
            # advisory; the partial file's size remains authoritative.
            if not _windows_sharing_conflict(error, path, temporary):
                raise
            if publication < 2:
                time.sleep(0.01 * (publication + 1))
            else:
                _remove_advisory(temporary)


def _windows_sharing_conflict(error, *paths):
    return (os.name == 'nt' and getattr(error, 'winerror', None) in (5, 32)
            and all(not path.is_symlink() and path.is_file() for path in paths))


def _remove_advisory(path):
    for deletion in range(3):
        if path.is_symlink():
            raise RuntimeError('Unsafe installer progress path: ' + str(path))
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as error:
            if not _windows_sharing_conflict(error, path):
                raise
            if deletion < 2:
                time.sleep(0.01 * (deletion + 1))
            else:
                return  # A reader may keep the last advisory state until retry.


def _clear_download(cache, partial):
    # Only these exact extension-owned files are ours to delete.
    for path in (partial, cache / 'progress.json', cache / 'progress.json.tmp'):
        if path.is_symlink():
            raise RuntimeError('Unsafe installer cache path: ' + str(path))
        if path == partial:
            path.unlink(missing_ok=True)
        else:
            _remove_advisory(path)


def _metadata():
    print('Starting PyPI metadata request for certified Warp 1.18.0',flush=True)
    with urllib.request.urlopen('https://pypi.org/pypi/warp-lang/1.18.0/json',timeout=30) as response:meta=json.load(response)
    item=next((x for x in meta.get('urls',[]) if x.get('filename')==NAME),None)
    if item is None:raise RuntimeError('Certified Warp 1.18.0 Windows wheel is unavailable on PyPI; retry later')
    if item.get('digests',{}).get('sha256')!=SHA:raise RuntimeError('PyPI metadata differs from certified wheel')
    url=item.get('url','');origin=urllib.parse.urlsplit(url)
    if origin.scheme!='https' or origin.netloc!='files.pythonhosted.org':
        raise RuntimeError('Untrusted Warp download URL in PyPI metadata')
    size = item.get('size')
    if type(size) is not int or size <= 0:
        raise RuntimeError('PyPI metadata has no valid certified wheel size')
    return url, size


def _response_offset(response, offset, total):
    status = response.getcode()
    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise RuntimeError('Unexpected encoded Warp download response')
    if status == 206:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match or tuple(map(int, match.groups())) != (offset, total - 1, total):
            raise RuntimeError('Invalid Content-Range for certified Warp wheel')
    elif status == 200:
        # A server may legally ignore Range. Restart safely, never append a 200.
        offset = 0
    else:
        raise RuntimeError('Unexpected Warp download HTTP status: ' + str(status))
    length = response.headers.get('Content-Length')
    if length is not None and (not length.isdecimal() or int(length) != total - offset):
        raise RuntimeError('Invalid Content-Length for certified Warp wheel')
    return offset


def _download_wheel(url, total, partial, cache):
    print('Starting wheel download: ' + NAME, flush=True)
    reported = partial.stat().st_size if partial.exists() else 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        _progress(cache, 'downloading', offset, total, attempt)
        request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
        if offset:
            request.add_header('Range', f'bytes={offset}-')
            print(f'Resuming saved wheel download at byte {offset} of {total}', flush=True)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                requested_offset = offset
                offset = _response_offset(response, offset, total)
                if requested_offset and response.getcode() == 200:
                    print('Server ignored Range; restarting at byte 0', flush=True)
                print(f'Accepted HTTP {response.getcode()}; starting byte {offset} of {total}', flush=True)
                with partial.open('r+b' if offset else 'wb', buffering=0) as stream:
                    stream.seek(offset)
                    while True:
                        interrupted = None
                        try:
                            chunk = response.read(CHUNK_SIZE)
                        except http.client.IncompleteRead as error:
                            chunk, interrupted = error.partial, error
                        if len(chunk) > total - offset:
                            raise RuntimeError('Certified Warp wheel exceeds metadata size')
                        if chunk:
                            if stream.write(chunk) != len(chunk):
                                raise OSError('Short write to the saved Warp wheel; resuming from disk')
                            offset += len(chunk)
                            _progress(cache, 'downloading', offset, total, attempt)
                            if offset - reported >= 10 * 1024 * 1024:
                                print('Downloaded ' + str(offset // (1024 * 1024)) + ' MiB', flush=True)
                                reported = offset
                        if interrupted:
                            raise interrupted
                        if not chunk:
                            if offset != total:
                                raise http.client.IncompleteRead(b'', total - offset)
                            return attempt
        except urllib.error.HTTPError as error:
            error.close()
            if error.code not in (408, 429) and not 500 <= error.code <= 599:
                raise
            failure = error
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            failure = error
        saved = partial.stat().st_size if partial.exists() else 0
        if saved == total:
            return attempt  # Full bytes still require the pinned SHA before use.
        if attempt == MAX_ATTEMPTS:
            _progress(cache, 'failed', saved, total, attempt)
            raise RuntimeError(f'Warp download interrupted: {failure}; saved {saved} of {total} bytes. '
                               'Click Install again to resume the saved download.') from failure
        _progress(cache, 'retrying', saved, total, attempt)
        delay = 2 ** (attempt - 1)
        print(f'Download interrupted: {failure}; saved {saved} of {total} bytes; '
              f'retrying in {delay}s (attempt {attempt + 1}/{MAX_ATTEMPTS})', flush=True)
        time.sleep(delay)


def _installed(target):
    if not target.exists() and not target.is_symlink():
        return False
    marker = target / '.certified-sha256'
    if (not target.is_symlink() and marker.is_file() and marker.read_text() == SHA
            and (target / 'warp/__init__.py').is_file()):
        return True
    raise RuntimeError('Warp install destination already exists without certification: ' + str(target))


def main(destination):
    destination=pathlib.Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'warp-1.18.0'
    cache = destination / '.downloads'
    if cache.is_symlink():
        raise RuntimeError('Unsafe installer cache directory: ' + str(cache))
    cache.mkdir(exist_ok=True)
    partial = cache / (NAME + '.part')
    with _installer_lock(cache):
        if partial.is_symlink():
            raise RuntimeError('Unsafe installer partial path: ' + str(partial))
        if _installed(target):
            print('Certified Warp 1.18.0 already installed', flush=True)
            _clear_download(cache, partial)
            return
        total, attempt = 0, 0
        try:
            # The pinned digest alone certifies a completed cache, even offline.
            cached = partial.exists() and _sha256(partial) == SHA
            if cached:
                total = partial.stat().st_size
            else:
                url, total = _metadata()
                size = partial.stat().st_size if partial.exists() else 0
                if size > total:
                    _clear_download(cache, partial)
                    raise _HashMismatchError('Certified Warp wheel hash mismatch (cached size exceeds metadata)')
                if size < total:
                    attempt = _download_wheel(url, total, partial, cache)
            _progress(cache, 'verifying', partial.stat().st_size, total, attempt)
            if _sha256(partial) != SHA:
                _clear_download(cache, partial)
                raise _HashMismatchError('Certified Warp wheel hash mismatch; discarded only the owned cached download')
            with _safe_archive(zipfile.ZipFile(partial)) as archive:
                _progress(cache, 'installing', total, total, attempt)
                print('Verified wheel: ' + NAME + '; bytes: ' + str(total) +
                      '; SHA256: ' + SHA + '; destination: ' + str(target), flush=True)
                _publish(archive, destination, target)
            _clear_download(cache, partial)
        except BaseException as error:
            if not isinstance(error, _HashMismatchError):
                # Preserve a download attempt's exact count and trusted total.
                try:
                    progress = json.loads((cache / 'progress.json').read_text(encoding='utf-8'))
                    if isinstance(progress, dict):
                        attempt = progress.get('attempt', attempt)
                        total = progress.get('total', total)
                except (OSError, ValueError):
                    pass
                try:
                    _progress(cache, 'failed', partial.stat().st_size if partial.exists() else 0, total, attempt)
                except OSError:
                    pass  # Preserve the original transport/disk failure.
            raise
    print('Installed certified Warp 1.18.0 into extension-owned directory', flush=True)


def _publish(archive, destination, target):
    # Publish a complete immutable directory. Never rename/remove a working install;
    # killed installers leave only invisible staging directories, safe to retry.
    staging=pathlib.Path(tempfile.mkdtemp(prefix='.acl-warp-stage-',dir=destination))
    try:
        archive.extractall(staging)
        (staging/'.certified-sha256').write_text(SHA)
        staging.rename(target)
    finally:
        if staging.exists():shutil.rmtree(staging)

if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1])
