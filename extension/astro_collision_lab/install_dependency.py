"""Isolated dependency installer. Fetch only the hash-pinned official wheel."""
import hashlib,io,json,pathlib,sys,urllib.request,zipfile
NAME='warp_lang-1.15.0-py3-none-win_amd64.whl'
SHA='06ef42c3ce522749268056653ea253645eb4a96ca164af74ae5113f0d55a6970'

def verified_archive(data):
    if hashlib.sha256(data).hexdigest()!=SHA:raise RuntimeError('Certified Warp wheel hash mismatch')
    archive=zipfile.ZipFile(io.BytesIO(data))
    for name in archive.namelist():
        path=pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:raise RuntimeError('Unsafe archive entry')
    return archive


def main(destination):
    with urllib.request.urlopen('https://pypi.org/pypi/warp-lang/1.15.0/json',timeout=30) as response:meta=json.load(response)
    item=next(x for x in meta['urls'] if x['filename']==NAME)
    if item['digests']['sha256']!=SHA:raise RuntimeError('PyPI metadata differs from certified wheel')
    with urllib.request.urlopen(item['url'],timeout=120) as response:data=response.read()
    verified_archive(data).extractall(destination)
    print('Installed certified Warp 1.15.0 into extension-owned directory')

if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1])
