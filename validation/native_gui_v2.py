"""Installed real-window authoring, undo and pixel-effect controls. No live FPS claim."""
import argparse,ctypes,hashlib,importlib,json,pathlib,sys,time,traceback
import bpy,gpu
parser=argparse.ArgumentParser();parser.add_argument('--module',required=True);parser.add_argument('--report',required=True)
args,_=parser.parse_known_args(sys.argv[sys.argv.index('--')+1:]);report=pathlib.Path(args.report)
state={'schema':'astro-collision-lab-native-gui-v2','purpose':'Native undo and preview pixel controls; no live solver FPS claim','pass':False,'checks':{},'stage':0,'started':time.monotonic()};handle=None
normal_entries=None;normal_colors={};control_scene=None


def preview_count(entries):
    """Count the real full-count local preview cache, without sampling it down."""
    return sum(len(entry['points']) for entry in (entries or {}).values())


def point_control_entries(entries, world_point):
    """Make disposable float64 payloads at one world point, retaining transforms.

    The production renderer applies entry rotation and translation. Convert the
    control to each source's local coordinates and invalidate its GPU batch so
    the next actual draw must exercise the installed vertex packing expression.
    Original cached arrays and batches are untouched; every source keeps its
    full authored count, including the projected-position negative control.
    """
    import numpy as np
    world_point=np.asarray(world_point,dtype=np.float64)
    if world_point.shape!=(3,) or not np.isfinite(world_point).all():
        raise ValueError('Control world point must be a finite three-vector')
    controls={}
    for key,entry in entries.items():
        rotation=np.asarray(entry['rotation'],dtype=np.float64)
        position=np.asarray(entry['position'],dtype=np.float64)
        local=np.linalg.solve(rotation,world_point-position)
        points=np.tile(local,(len(entry['points']),1)).astype(np.float64,copy=False)
        controls[key]={**entry,'points':points,'batch':None}
    return controls


def control_points_match(entries, world_point):
    """Verify the injected local payload maps to the intended world coordinates."""
    import numpy as np
    return bool(entries) and all(
        len(entry['points'])>0 and np.allclose(
            entry['points']@np.asarray(entry['rotation']).T+entry['position'],
            world_point,rtol=0.,atol=1e-6)
        for entry in entries.values())


def restore_preview():
    global normal_colors
    if normal_entries is not None and control_scene is not None:
        editor._PREVIEWS[control_scene.as_pointer()]=normal_entries
    for obj,color in normal_colors.values():
        obj.color=color
    normal_colors={}


def check(name,result):
    state['checks'][name]=bool(result)
    if not result:raise AssertionError(name)


def flush():
    bpy.context.view_layer.update()
    for key,(scene,deadline) in list(editor._PENDING.items()):editor._PENDING[key]=(scene,0.)
    editor.flush();bpy.context.view_layer.update()


def finish(error=None):
    global handle
    if error:state['error']=error
    if 'editor' in globals():
        try:restore_preview()
        except Exception:state['restore_error']=traceback.format_exc()
    state['pass']=bool(not error and not state.get('restore_error') and state['checks'] and all(state['checks'].values()))
    report.write_text(json.dumps(state,indent=2,default=str))
    if handle is not None:bpy.types.SpaceView3D.draw_handler_remove(handle,'WINDOW');handle=None
    if 'editor' in globals():editor._HANDLE=None;editor._BUSY=False
    print('NATIVE_GUI_RESULT='+json.dumps(state,default=str),flush=True)
    bpy.ops.wm.quit_blender()


def pixels():
    """Blender 5.1 Buffer's multidimensional PEP-3118 strides are not contiguous.

    Flatten the native buffer and iterate its scalar sequence before NumPy. The
    previous direct np.asarray(Buffer) view skipped channels and misread RGB.
    This is diagnostic pixel readback, never live particle-array transfer.
    """
    import numpy as np
    ctypes.windll.opengl32.glFinish()
    x,y,w,h=gpu.state.viewport_get()
    buffer=gpu.state.active_framebuffer_get().read_color(x,y,w,h,4,0,'FLOAT')
    state['raw_gpu_buffer_protocol']={'format':memoryview(buffer).format,'shape':memoryview(buffer).shape,'strides':memoryview(buffer).strides}
    buffer.dimensions=w*h*4
    array=np.asarray(list(buffer),dtype=np.float32).reshape(h,w,4)
    return hashlib.sha256(array.tobytes()).hexdigest(),array,(w,h)


def draw():
    global normal_colors
    try:
        scene=bpy.context.scene;entries=editor._PREVIEWS.get(scene.as_pointer())
        if entries is None:return
        if state['stage']==1:
            before,_,_=pixels();original_draw();after,buffer,size=pixels()
            check('100K preview vertices',preview_count(entries)==100000)
            check('Preview has actual visible pixel effect',before!=after)
            # Retain a clean authoring-only picture for internal visual QA.
            image=bpy.data.images.new('ACL authoring proof',width=size[0],height=size[1],alpha=True)
            image.pixels.foreach_set(buffer.ravel())
            image.filepath_raw=str(report.with_suffix('.png'));image.file_format='PNG';image.save();bpy.data.images.remove(image)
            editor._PREVIEWS[scene.as_pointer()]=point_control_entries(entries,(1e9,1e9,1e9))
            state['stage']=2
        elif state['stage']==2:
            before,_,_=pixels();original_draw();after,_,_=pixels()
            check('Offscreen control retains 100K vertices',preview_count(entries)==100000)
            check('Negative control reached offscreen positions',control_points_match(entries,(1e9,1e9,1e9)))
            check('Offscreen control has no particle pixel effect',before==after)
            restore_preview()
            state['stage']=3
        elif state['stage']==3:
            before,_,_=pixels();original_draw();after,_,_=pixels()
            check('Restored preview pixels return',before!=after)
            import numpy as np
            from mathutils import Vector
            point=np.array(bpy.context.region_data.view_matrix.inverted() @ Vector((.8,.5,-10.)),dtype=np.float64)
            state['control_point']=point.tolist()
            normal_colors={key:(entry['obj'],tuple(entry['obj'].color)) for key,entry in entries.items()}
            for obj,_color in normal_colors.values():obj.color=(1.,0.,0.,1.)
            editor._PREVIEWS[scene.as_pointer()]=point_control_entries(entries,point)
            state['stage']=5
        elif state['stage']==5:
            import numpy as np
            check('Known point control retains 100K vertices',preview_count(entries)==100000)
            check('Known control payload is float64',all(entry['points'].dtype==np.dtype('float64') for entry in entries.values()))
            check('Known control payload reached renderer',control_points_match(entries,state['control_point']))
            original_draw();after,image,size=pixels()
            from bpy_extras.view3d_utils import location_3d_to_region_2d
            xy=location_3d_to_region_2d(bpy.context.region,bpy.context.region_data,state['control_point'])
            check('Known world point projects inside viewport',xy is not None and 3<xy.x<size[0]-3 and 3<xy.y<size[1]-3)
            x,y=int(round(xy.x)),int(round(xy.y));roi=image[y-3:y+4,x-3:x+4]
            state['projected_control_pixel']=[x,y];state['projected_pixel_rgb']=image[y,x,:3].tolist()
            expected_red=bool(np.any((roi[...,0]-roi[...,1]>.5)&(roi[...,0]-roi[...,2]>.5)))
            check('Float64 authored points render at expected projected position',expected_red)
            restore_preview()
            state['stage']=4
        else:original_draw()
    except BaseException:
        state['draw_error']=traceback.format_exc();state['stage']=99


def setup():
    global editor,original_draw,handle,normal_entries,control_scene
    module=importlib.import_module(args.module);module.register()
    editor=importlib.import_module(args.module+'.authoring')
    window=bpy.context.window_manager.windows[0];area=next(a for a in window.screen.areas if a.type=='VIEW_3D');region=next(r for r in area.regions if r.type=='WINDOW')
    area.spaces.active.overlay.show_overlays=False;area.spaces.active.show_region_ui=True
    area.spaces.active.region_3d.view_distance=18.
    with bpy.context.temp_override(window=window,screen=window.screen,area=area,region=region):
        bpy.context.preferences.edit.use_global_undo=True
        bpy.ops.ed.undo_push(message='ACL catalogue baseline')
        bpy.ops.acl.add_source(kind='SPHERE');name=bpy.context.object.name
        bpy.ops.ed.undo_push(message='ACL added source')
        check('Undo available in real window',bpy.ops.ed.undo.poll())
        bpy.ops.ed.undo();check('Undo removes added primitive',bpy.data.objects.get(name) is None)
        bpy.ops.ed.redo();restored=bpy.data.objects.get(name)
        check('Redo restores persistent source',restored is not None and restored.acl_body.enabled)
        bpy.data.objects.remove(restored,do_unlink=True)
        bpy.ops.acl.add_source(kind='GALAXY');root=bpy.context.object;root.acl_body.particle_count=100000;flush()
        original_radius=root.acl_body.radius
        clone=root.copy();root.users_collection[0].objects.link(clone);clone.location.x=20;flush()
        check('Duplicate root creates independent components',len(clone.children)==2 and all(c.parent==clone for c in clone.children))
        clone.acl_body.radius=7.;flush()
        check('Duplicate edits preserve original geometry',root.acl_body.radius==original_radius)
        for obj in list(clone.children)+[clone]:bpy.data.objects.remove(obj,do_unlink=True)
        root.location=(1.25,-.75,.5);root.rotation_euler=(.2,-.35,.4)
        flush()
        check('Deletion removes clone from source budget',sum(o.acl_body.particle_count for o in importlib.import_module(args.module+'.ui').source_objects(bpy.context.scene))==100000)
        # Pixel controls own this disposable authoring snapshot. A later
        # depsgraph notification must not replace their synthetic payload with
        # the unchanged authored revision before the next actual draw.
        editor._PENDING.clear()
        if bpy.app.timers.is_registered(editor.flush):bpy.app.timers.unregister(editor.flush)
        editor._BUSY=True
        control_scene=bpy.context.scene
        normal_entries=editor._PREVIEWS[control_scene.as_pointer()]
        original_draw=editor.draw
        if editor._HANDLE is not None:bpy.types.SpaceView3D.draw_handler_remove(editor._HANDLE,'WINDOW');editor._HANDLE=None
        handle=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW','POST_VIEW')
        editor._HANDLE=handle
    state['stage']=1;area.tag_redraw()


def tick():
    try:
        if state['stage']==0:setup()
        if state['stage']==4:finish();return None
        if state['stage']==99:finish(state['draw_error']);return None
        if time.monotonic()-state['started']>120:finish('native GUI deadline');return None
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type=='VIEW_3D':a.tag_redraw()
    except BaseException:finish(traceback.format_exc());return None
    return .1

bpy.app.timers.register(tick,first_interval=2.)
