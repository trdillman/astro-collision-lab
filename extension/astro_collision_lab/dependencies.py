"""Extension-owned, exact-version runtime installation; no wheels in releases."""
import sys
from pathlib import Path

VERSION='1.18.0'
FLOOR=(1,18,0)


def compatible(version):
    """True for Warp 1.18.0 or later (the certified wheel is exactly VERSION)."""
    try:return tuple(int(x) for x in str(version).split('.')[:3])>=FLOOR
    except ValueError:return False

WHEEL='warp_lang-1.18.0-py3-none-win_amd64.whl'
SHA256='1769e55746739f2e244a2b412123003c462ec3d977dfb5f420f924a8553d30aa'
_PROCESS=None
_LOG=None
_ERROR=''
_ENABLED=False
_TEMP_ROOT=None
_LAST_PROGRESS=''


def register():
    global _ENABLED
    import bpy
    _ENABLED=True
    bpy.app.handlers.persistent(_on_exit)
    if _on_exit not in bpy.app.handlers.exit_pre:
        bpy.app.handlers.exit_pre.append(_on_exit)
    if _PROCESS is not None and not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll,first_interval=.5,persistent=True)


def unregister():
    import bpy
    _stop_installation()
    if _on_exit in bpy.app.handlers.exit_pre:
        bpy.app.handlers.exit_pre.remove(_on_exit)


def _on_exit(*args):
    _stop_installation()


def _stop_installation():
    global _ENABLED,_PROCESS,_LOG,_ERROR
    import subprocess

    import bpy
    _ENABLED=False
    process,log=_PROCESS,_LOG
    # Detach first: any late timer must not activate a module after disable/exit.
    _PROCESS=None;_LOG=None
    if bpy.app.timers.is_registered(_poll):bpy.app.timers.unregister(_poll)
    stopped=process is None
    try:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        stopped=True
    except (OSError,subprocess.TimeoutExpired) as exc:
        _PROCESS=process
        _LOG=log
        _ERROR='Cannot stop owned Warp installer: '+str(exc)
    finally:
        if log is not None:log.close()
        if stopped:_cleanup_temp()


def _cleanup_temp():
    global _TEMP_ROOT,_ERROR
    import shutil
    if _TEMP_ROOT is None:return
    owned=_TEMP_ROOT
    try:
        shutil.rmtree(owned)
    except FileNotFoundError:pass
    except OSError as exc:
        _ERROR+=' Cannot remove installer temporary profile: '+str(exc)
        return
    _TEMP_ROOT=None


def directory():
    import bpy
    try:return Path(bpy.utils.extension_path_user(__package__,path='dependencies',create=True))
    except ValueError:return Path(bpy.utils.user_resource('EXTENSIONS',path='astro_collision_lab/dependencies',create=True))


def activate(required=False):
    if not _ENABLED:
        if required:raise RuntimeError('Enable Astro Collision Lab before activating Warp')
        return False
    # A loaded native module cannot be safely replaced by changing sys.path.
    loaded=sys.modules.get('warp')
    if loaded is not None and not compatible(getattr(loaded,'__version__',None)):
        if required:raise RuntimeError(_version_error(loaded))
        return False
    path=directory()
    certified=path/'warp-1.18.0'
    try:
        if (certified/'.certified-sha256').read_text()==SHA256 and (certified/'warp/__init__.py').is_file():
            path=certified
    except OSError:pass
    if (path/'warp/__init__.py').exists() and str(path) not in sys.path:
        sys.path.insert(0,str(path))
    try:
        import warp as wp
    except ImportError as error:
        if required:raise RuntimeError('Install Warp 1.18.0 in Astro Collision Lab preferences') from error
        return False
    if not compatible(wp.__version__):
        if required:raise RuntimeError(_version_error(wp))
        return False
    return True


def _version_error(module):
    return ('Warp '+str(getattr(module,'__version__','unknown'))+' is already loaded; Warp '+VERSION+
            ' or later is required. Install the certified version in Astro Collision Lab preferences, then Restart Blender using an isolated profile.')


def _redraw_preferences():
    try:
        import bpy
        manager=getattr(getattr(bpy,'context',None),'window_manager',None)
        for window in getattr(manager,'windows',()):
            for area in getattr(getattr(window,'screen',None),'areas',()):
                if area.type=='PREFERENCES':area.tag_redraw()
    except (ImportError,ReferenceError,RuntimeError):
        # Headless callers and windows closing during shutdown have no UI.
        pass


def install():
    global _PROCESS,_LOG,_ERROR,_TEMP_ROOT,_LAST_PROGRESS
    import os
    import subprocess
    import tempfile

    import bpy
    if not _ENABLED:raise RuntimeError('Enable Astro Collision Lab before installing Warp')
    if _PROCESS is not None:
        if _PROCESS.poll() is None:return
        _poll()
    if _TEMP_ROOT is not None:
        _cleanup_temp()
        if _TEMP_ROOT is not None:raise RuntimeError(_ERROR)
    owned=Path(tempfile.mkdtemp(prefix='acl-install-'))
    _TEMP_ROOT=owned;_ERROR='';_LAST_PROGRESS=''
    log_path=None
    try:
        env=os.environ.copy()
        for key in ('RESOURCES','CONFIG','EXTENSIONS','SCRIPTS','DATAFILES'):
            p=owned/key.lower();p.mkdir();env['BLENDER_USER_'+key]=str(p)
        log_path=directory()/'install.log'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        _LOG=log_path.open('w',encoding='utf-8')
        _PROCESS=subprocess.Popen([bpy.app.binary_path,'--background','--factory-startup','--python-exit-code','1',
                               '--python',str(Path(__file__).with_name('install_dependency.py')),'--',str(directory())],
                              env=env,stdout=_LOG,stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0)
    except Exception as exc:
        _ERROR=(type(exc).__name__+': Retry installation. Cannot start Warp installer: '+
                str(exc)[:200]+'; log: '+str(log_path))
        if _LOG is not None:_LOG.close()
        _LOG=None
        _cleanup_temp()
        _redraw_preferences()
        raise
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll,first_interval=.5,persistent=True)


def _poll():
    global _PROCESS,_LOG,_ERROR,_LAST_PROGRESS
    if not _ENABLED or _PROCESS is None:return None
    if _PROCESS.poll() is None:
        progress=_download_status()
        if progress!=_LAST_PROGRESS:
            _LAST_PROGRESS=progress
            _redraw_preferences()
        return .5
    result=_PROCESS.returncode
    log_path=Path(_LOG.name) if _LOG is not None else directory()/'install.log'
    if _LOG is not None:_LOG.close()
    _PROCESS=None;_ERROR=''
    if result:
        import re
        try:
            with log_path.open('rb') as log:
                log.seek(max(0,log_path.stat().st_size-4096))
                lines=[line.strip() for line in log.read().decode('utf-8',errors='replace').splitlines() if line.strip()]
            # Blender appends a generic "Error: script failed" footer after the
            # traceback. Prefer a Python exception class over that wrapper.
            exception=next((match for line in reversed(lines)
                            if (match:=re.match(r'^([\w.]+(?:Error|Exception)|Exception|KeyboardInterrupt|SystemExit):\s*(.*)$',line))),None)
            if exception:
                detail=exception[1]+': Retry installation. '+exception[2][:200]
            else:
                detail='Warp install failed: Retry installation. '+(lines[-1][:200] if lines else 'No installer output')
        except OSError:detail='Warp install failed: Retry installation. Cannot read installer log'
        _ERROR=detail+' (exit '+str(result)+'); log: '+str(log_path)
    else:
        try:activate(required=True)
        except Exception as exc:_ERROR=str(exc)  # noqa: BLE001 - record any activation failure for the preferences UI
    _LOG=None
    _cleanup_temp()
    _LAST_PROGRESS=''
    _redraw_preferences()
    return None


def _download_status():
    import json
    try:
        state=json.loads((directory()/'.downloads/progress.json').read_text(encoding='utf-8'))
        if not isinstance(state,dict):raise ValueError('Invalid progress')
        phase=state.get('phase','')
        if phase=='verifying':return 'Verifying certified Warp wheel...'
        if phase=='installing':return 'Installing verified Warp 1.18.0...'
        size=int(state.get('downloaded',0));total=int(state.get('total',0))
        if size<0 or total<=0 or size>total:raise ValueError('Invalid progress sizes')
        amount=f'{size/(1024*1024):.1f} / {total/(1024*1024):.1f} MiB'
        if phase=='retrying':
            return 'Reconnecting to resume Warp: '+amount+f" (attempt {int(state.get('attempt',1))}/{int(state.get('max_attempts',4))})"
        if phase=='failed':return 'Resuming saved Warp download: '+amount
        return 'Downloading Warp 1.18.0: '+amount
    except (OSError,ValueError,TypeError):
        return 'Installing certified Warp 1.18.0...'


def status():
    if not _ENABLED:return 'Enable Astro Collision Lab to install Warp 1.18.0'
    if _ERROR:return _ERROR
    if _PROCESS is not None:return _download_status()
    loaded=sys.modules.get('warp')
    if loaded is not None and not compatible(getattr(loaded,'__version__',None)):return _version_error(loaded)
    if activate():return 'Warp 1.18.0 available'
    partial=directory()/'.downloads'/(WHEEL+'.part')
    try:
        size=partial.stat().st_size
        if size:return f'Warp download saved: {size/(1024*1024):.1f} MiB. Click Install to resume.'
    except OSError:pass
    return 'Warp 1.18.0 required (download separately)'
