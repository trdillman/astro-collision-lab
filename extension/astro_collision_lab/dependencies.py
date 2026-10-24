"""Extension-owned, exact-version runtime installation; no wheels in releases."""
import sys
from pathlib import Path
VERSION='1.15.0'
WHEEL='warp_lang-1.15.0-py3-none-win_amd64.whl'
SHA256='06ef42c3ce522749268056653ea253645eb4a96ca164af74ae5113f0d55a6970'
_PROCESS=None
_LOG=None
_ERROR=''


def directory():
    import bpy
    try:return Path(bpy.utils.extension_path_user(__package__,path='dependencies',create=True))
    except ValueError:return Path(bpy.utils.user_resource('EXTENSIONS',path='astro_collision_lab/dependencies',create=True))


def activate(required=False):
    path=directory()
    if (path/'warp/__init__.py').exists() and str(path) not in sys.path:
        sys.path.insert(0,str(path))
    try:
        import warp as wp
    except ImportError:
        if required:raise RuntimeError('Install Warp 1.15.0 in Astro Collision Lab preferences')
        return False
    if wp.__version__!=VERSION:
        if required:raise RuntimeError('A different Warp version is already loaded. Use an isolated Blender profile and install Warp 1.15.0 in this add-on preferences.')
        return False
    return True


def install():
    global _PROCESS,_LOG,_ERROR
    import bpy,os,subprocess,tempfile
    if _PROCESS is not None and _PROCESS.poll() is None:return
    owned=Path(tempfile.mkdtemp(prefix='acl-install-'))
    env=os.environ.copy()
    for key in ('RESOURCES','CONFIG','EXTENSIONS','SCRIPTS','DATAFILES'):
        p=owned/key.lower();p.mkdir();env['BLENDER_USER_'+key]=str(p)
    _LOG=(owned/'install.log').open('w',encoding='utf-8');_ERROR=''
    _PROCESS=subprocess.Popen([bpy.app.binary_path,'--background','--factory-startup','--python-exit-code','1',
                               '--python',str(Path(__file__).with_name('install_dependency.py')),'--',str(directory())],
                              env=env,stdout=_LOG,stderr=subprocess.STDOUT)
    bpy.app.timers.register(_poll,first_interval=.5)


def _poll():
    global _PROCESS,_LOG,_ERROR
    if _PROCESS.poll() is None:return .5
    result=_PROCESS.returncode;_LOG.close();_PROCESS=None
    if result:_ERROR='Dependency installation failed; see '+str(_LOG.name)
    else:
        try:activate(required=True)
        except Exception as exc:_ERROR=str(exc)
    _LOG=None
    return None


def status():
    if _PROCESS is not None:return 'Installing certified Warp 1.15.0...'
    if _ERROR:return _ERROR
    return 'Warp 1.15.0 available' if activate() else 'Warp 1.15.0 required (download separately)'
