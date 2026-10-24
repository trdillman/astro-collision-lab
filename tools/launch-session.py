"""Launch the project UI with independent handles and verified per-process readiness."""
import argparse,ctypes,json,os,pathlib,subprocess,time,uuid


def ready_matches(d,pid,token):
    return bool(d.get('pid')==pid and d.get('launch_id')==token and d.get('phase')=='READY'
                and d.get('ticks',0)>=5 and d.get('draws',0)>=3
                and d.get('authored_count',0)>0 and d.get('preview_count')==d.get('authored_count'))


def await_ready(read,ping,alive,pid,token,timeout,clock=time.monotonic,sleep=time.sleep):
    deadline=clock()+timeout
    while clock()<deadline and alive():
        d=read()
        if d.get('phase')=='ERROR' and d.get('launch_id')==token:return False
        if ready_matches(d,pid,token) and ping():return True
        sleep(.2)
    return False


def spawn(args,env,profile,popen=subprocess.Popen):
    # Real file handles remain usable after PowerShell/the controller exits.
    # No descendant retains a controller PIPE or stdin handle.
    with (profile/'stdout.log').open('w') as out,(profile/'stderr.log').open('w') as err:
        return popen(args,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,
                     creationflags=(getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)))


def process_identity(pid):
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.restype=w.HANDLE;k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD]
    h=k.OpenProcess(0x1000,False,pid)
    if not h:return None
    try:
        created=w.FILETIME();ended=w.FILETIME();kernel=w.FILETIME();user=w.FILETIME()
        if not k.GetProcessTimes(w.HANDLE(h),ctypes.byref(created),ctypes.byref(ended),ctypes.byref(kernel),ctypes.byref(user)):return None
        exit_code=w.DWORD();k.GetExitCodeProcess(w.HANDLE(h),ctypes.byref(exit_code))
        if exit_code.value!=259:return None
        name=ctypes.create_unicode_buffer(32768);size=w.DWORD(len(name))
        if not k.QueryFullProcessImageNameW(w.HANDLE(h),0,name,ctypes.byref(size)):return None
        return {'created':(created.dwHighDateTime<<32)|created.dwLowDateTime,'exe':name.value.casefold()}
    finally:k.CloseHandle(w.HANDLE(h))


def window_state(pid,focus=False):
    from ctypes import wintypes as w
    u=ctypes.WinDLL('user32',use_last_error=True);rows=[]
    callback=ctypes.WINFUNCTYPE(w.BOOL,w.HWND,w.LPARAM)
    u.GetWindowThreadProcessId.argtypes=[w.HWND,ctypes.POINTER(w.DWORD)]
    u.IsWindowVisible.argtypes=[w.HWND];u.ShowWindow.argtypes=[w.HWND,ctypes.c_int];u.SetForegroundWindow.argtypes=[w.HWND]
    u.EnumWindows.argtypes=[callback,w.LPARAM]
    u.SendMessageTimeoutW.argtypes=[w.HWND,w.UINT,w.WPARAM,w.LPARAM,w.UINT,w.UINT,ctypes.POINTER(ctypes.c_size_t)]
    @callback
    def each(hwnd,_):
        owner=w.DWORD();u.GetWindowThreadProcessId(hwnd,ctypes.byref(owner))
        if owner.value==pid and u.IsWindowVisible(hwnd):
            result=ctypes.c_size_t();ok=bool(u.SendMessageTimeoutW(hwnd,0,0,0,2,500,ctypes.byref(result)))
            rows.append(ok)
            if focus and ok:
                u.ShowWindow(hwnd,9)
                u.SetForegroundWindow(hwnd)
        return True
    u.EnumWindows(each,0)
    return bool(rows) and all(rows)


def read_state(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    p=argparse.ArgumentParser();p.add_argument('--blender',required=True);p.add_argument('--profile',required=True);p.add_argument('--bootstrap',required=True);p.add_argument('--timeout',type=float,default=30.)
    a=p.parse_args();profile=pathlib.Path(a.profile);record=profile/'session.json';old=read_state(record)
    if old:
        ident=process_identity(old.get('pid',0))
        if ident==old.get('identity') and ident is not None:
            state=read_state(profile/'config'/('session-'+str(old['pid'])+'.json'))
            if ready_matches(state,old['pid'],old['launch_id']) and window_state(old['pid'],focus=True):
                print('Existing project Blender is responsive; brought it forward.');return
            raise RuntimeError('The existing project Blender is not responding. No duplicate window was launched.')
    token=uuid.uuid4().hex;env=os.environ.copy();env.update(ACL_LAUNCH_ID=token,PYTHONUNBUFFERED='1')
    for k in ('RESOURCES','CONFIG','EXTENSIONS','SCRIPTS','DATAFILES'):env['BLENDER_USER_'+k]=str(profile/k.lower())
    child=spawn([a.blender,'--gpu-backend','opengl','--python',a.bootstrap],env,profile)
    identity=process_identity(child.pid)
    record.write_text(json.dumps({'pid':child.pid,'identity':identity,'launch_id':token},indent=2))
    (profile/'session.pid').write_text(str(child.pid))
    state_path=profile/'config'/('session-'+str(child.pid)+'.json')
    ok=await_ready(lambda:read_state(state_path),lambda:window_state(child.pid),lambda:child.poll() is None,child.pid,token,a.timeout)
    state=read_state(state_path);state.update(window_responding=ok)
    (profile/'startup-result.json').write_text(json.dumps(state,indent=2))
    if not ok:
        # Only the new, unverified launch owned by this call may be closed.
        if child.poll() is None:child.kill();child.wait(timeout=10)
        raise RuntimeError('Blender startup did not become responsive; the failed launch was closed. See '+str(profile/'stderr.log'))
    window_state(child.pid,focus=True)
    print('Blender is responding: '+str(state['authored_count'])+' preview particles, '+str(state['draws'])+' completed viewport draws.')
if __name__=='__main__':main()
