"""Create the authoring scene and report fresh, rendered UI readiness."""
import bpy,importlib,json,os,pathlib,time,traceback,faulthandler
_PID=os.getpid();_TOKEN=os.environ.get('ACL_LAUNCH_ID','legacy')
_CONFIG=pathlib.Path(os.environ['BLENDER_USER_CONFIG'])
_STATE={'pid':_PID,'launch_id':_TOKEN,'phase':'BOOTSTRAP','ticks':0,'draws':0,'started':time.time()}
_PATH=_CONFIG/('session-'+str(_PID)+'.json')
_TRACE=(_CONFIG/('startup-'+str(_PID)+'.log')).open('w',buffering=1)
faulthandler.enable(file=_TRACE);faulthandler.dump_traceback_later(15,repeat=True,file=_TRACE)
_HANDLE=None

def publish():
    _STATE['time']=time.time();tmp=_PATH.with_suffix('.tmp');tmp.write_text(json.dumps(_STATE,indent=2));tmp.replace(_PATH)

def drawn():_STATE['draws']+=1

def preview_count(entries):
    """Inspect the full-count local preview entries; this is not a live FPS gate."""
    return sum(len(entry['points']) for entry in (entries or {}).values())

def ready_tick():
    global _HANDLE
    try:
        _STATE['ticks']+=1
        from bl_ext.aclv.astro_collision_lab import authoring
        entries=authoring._PREVIEWS.get(bpy.context.scene.as_pointer())
        _STATE['preview_count']=preview_count(entries)
        if _STATE['ticks']>=5 and _STATE['draws']>=3 and _STATE['preview_count']==_STATE['authored_count']:
            _STATE['phase']='READY';publish();faulthandler.cancel_dump_traceback_later()
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLE,'WINDOW');_HANDLE=None
            return None
        publish()
        for w in bpy.context.window_manager.windows:
            for area in w.screen.areas:
                if area.type=='VIEW_3D':area.tag_redraw()
        return .2
    except Exception:
        _STATE.update(phase='ERROR',error=traceback.format_exc());publish();faulthandler.cancel_dump_traceback_later();return None

def setup():
    global _HANDLE
    try:
        _STATE['phase']='SETUP';publish()
        module=importlib.import_module('bl_ext.aclv.astro_collision_lab');module.register()
        window=bpy.context.window_manager.windows[0]
        area=next(a for a in window.screen.areas if a.type=='VIEW_3D')
        region=next(r for r in area.regions if r.type=='WINDOW')
        with bpy.context.temp_override(window=window,screen=window.screen,area=area,region=region):
            if bpy.context.scene.acl_settings.source_collection is None:bpy.ops.acl.galaxy_preset()
            area.spaces.active.show_region_ui=True
            area.spaces.active.region_3d.view_location=(0.,0.,0.)
            area.spaces.active.region_3d.view_distance=35.
        config,bodies,objects=importlib.import_module('bl_ext.aclv.astro_collision_lab.runtime').authored_scene(bpy.context.scene)
        _STATE.update(phase='DRAWING',authored_count=config.particle_count,mode=config.mode,sources=len(objects))
        _HANDLE=bpy.types.SpaceView3D.draw_handler_add(drawn,(),'WINDOW','POST_PIXEL')
        publish();bpy.app.timers.register(ready_tick,first_interval=.2)
    except Exception:
        _STATE.update(phase='ERROR',error=traceback.format_exc());publish();faulthandler.cancel_dump_traceback_later()
    return None
publish();bpy.app.timers.register(setup,first_interval=.2)
