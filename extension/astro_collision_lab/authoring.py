"""Native Add primitives, compound membership, and debounced authoring previews.

Only initial-condition previews upload CPU vertices. Live simulation continues
through viewport.py's CUDA/GL path. No handler mutates the user's active context.
"""
import time
import threading
from types import SimpleNamespace
import bpy
from . import ui
from .preset_specs import BodySpec as Recipe
from .interaction import same_sources

_PENDING={}
_SIGNATURES={}
_PREVIEWS={}
_RESTARTS={}
_HANDLE=None
_BUSY=False
_PREVIEW_CACHE=None
_THREADED_PHYSICS = threading.Event()
_THREADED_APPEARANCE = threading.Event()


def appearance_changed():
    if threading.current_thread() is not threading.main_thread():
        _THREADED_APPEARANCE.set()
        return
    from .viewport import request_redraw
    request_redraw()


def drain_threaded_updates():
    # RNA callbacks can run on workers. Resolve Blender data only on this timer.
    if _THREADED_PHYSICS.is_set():
        _THREADED_PHYSICS.clear()
        reload()
    if _THREADED_APPEARANCE.is_set():
        _THREADED_APPEARANCE.clear()
        appearance_changed()
    return .03


def changed(settings, context):
    if threading.current_thread() is not threading.main_thread():
        _THREADED_PHYSICS.set()
        return
    if _BUSY:
        return
    owner = getattr(settings, 'id_data', None)
    # Property updates can target a scene other than the active UI scene.
    if isinstance(owner, bpy.types.Scene):
        scenes = (owner,)
    elif isinstance(owner, bpy.types.Object):
        scenes = tuple(scene for scene in bpy.data.scenes if owner.name in scene.objects)
    else:
        scene = getattr(context, 'scene', None)
        scenes = () if scene is None else (scene,)
    for scene in scenes:
        pointer = scene.as_pointer()
        deadline = _PENDING[pointer][1] if pointer in _PENDING else time.monotonic() + .02
        _PENDING[pointer] = (scene, deadline)
    if scenes and not bpy.app.timers.is_registered(flush):
        bpy.app.timers.register(flush, first_interval=.03)


def is_dirty(scene):
    return _THREADED_PHYSICS.is_set() or scene.as_pointer() in _PENDING


def _collection(context):
    c=context.scene.acl_settings.source_collection
    if c is None or not any(x==c for x in ui._scene_collections(context.scene)):
        c=ui._new_source_collection(context.scene,'Objects')
    return c


def add_component(context,kind,*,collection=None,name=None):
    c=collection or _collection(context)
    spec=Recipe(name=name or {'SPHERE':'Planet','DISK':'Stellar Disk','BULGE':'Stellar Bulge','HALO':'Dark Matter Halo'}[kind],
                kind=kind,count=25000,mass=1.,radius=1.,location=tuple(context.scene.cursor.location),
                velocity=(0.,0.,0.),material='ROCK')
    obj=ui._create_body(context,c,spec)
    obj.acl_body.native_primitive=True
    obj.acl_body.radius=1.
    obj.acl_body.scale_length=.2 if kind=='DISK' else .1
    obj.acl_body.thickness=.14
    obj.acl_body.rotation_fraction=1. if kind=='DISK' else 0.
    obj['acl_guide_radius']=1.;obj['acl_guide_thickness']=.14
    changed(obj.acl_body,context)
    return obj


def add_galaxy(context,kind='DISK',*,collection=None):
    c=collection or _collection(context)
    root=bpy.data.objects.new('Galaxy',None);c.objects.link(root)
    root.empty_display_type='PLAIN_AXES';root.location=context.scene.cursor.location
    p=root.acl_body;p.is_galaxy=True;p.galaxy_type=kind
    p.mass=1.;p.particle_count=50000;p.radius=5.;p.scale_length=1.
    p.disk_fraction=.2;p.bulge_fraction=0.;p.thickness=.4
    _sync_galaxy(root,context.scene)
    ui._set_active_sources(context,[root])
    changed(p,context)
    return root


def _managed_components(root):
    """Return one mesh per component; never silently absorb user duplicates."""
    components = {}
    for obj in root.children:
        kind = obj.get('acl_component')
        if not kind:
            continue
        if kind not in {'DISK', 'BULGE', 'HALO'} or obj.type != 'MESH':
            raise ValueError(f'{obj.name}: invalid managed Galaxy component; undo this edit')
        if kind in components:
            raise ValueError(
                f'{root.name}: duplicate {kind} component. Undo the duplicate or remove '
                'the extra component; add a standalone primitive outside the Galaxy parent.')
        components[kind] = obj
    return components


def _validate_group_membership(collection):
    """A compound Galaxy and all its generated sources share one active group."""
    objects = tuple(collection.all_objects)
    included = {obj.as_pointer() for obj in objects}
    for obj in objects:
        parent = obj.parent
        if obj.get('acl_component'):
            if parent is None or not parent.acl_body.is_galaxy:
                raise ValueError(
                    f'{obj.name}: managed Galaxy component has no Galaxy parent. '
                    'Undo the reparenting or add a new standalone primitive.')
            if parent.as_pointer() not in included:
                raise ValueError(
                    f'{obj.name}: its Galaxy parent must be in the active source collection.')
        elif parent is not None and parent.acl_body.is_galaxy and obj.acl_body.enabled:
            raise ValueError(
                f'{obj.name}: an extra source inside a Galaxy would change its component budget. '
                'Unparent this standalone source or edit the Galaxy root.')
        if obj.acl_body.is_galaxy and not obj.get('acl_component'):
            for child in _managed_components(obj).values():
                if child.as_pointer() not in included:
                    raise ValueError(
                        f'{child.name}: link the managed component into its Galaxy\'s active source collection.')


def _sync_galaxy(root,scene):
    existing = _managed_components(root)
    p=root.acl_body
    if root.get('acl_galaxy_type')!=p.galaxy_type:
        if p.galaxy_type=='ELLIPTICAL':p.disk_fraction=0.;p.bulge_fraction=.8
        elif p.galaxy_type=='LENTICULAR':p.disk_fraction=.15;p.bulge_fraction=.35;p.thickness=.6
        elif p.galaxy_type=='DWARF':p.disk_fraction=.05;p.bulge_fraction=.0;p.thickness=.5
        else:p.disk_fraction=.2;p.bulge_fraction=0.;p.thickness=.4
        root['acl_galaxy_type']=p.galaxy_type
    disk=p.disk_fraction;bulge=p.bulge_fraction
    if disk+bulge>1.:raise ValueError(root.name+': disk + bulge mass fractions must be <=1')
    parts=[('DISK',disk, p.radius,p.scale_length),('BULGE',bulge,p.radius*2,p.scale_length),('HALO',1-disk-bulge,p.radius*2,p.scale_length)]
    active=[s for s in parts if s[1]>0]
    if p.particle_count<len(active):raise ValueError('Particle budget smaller than component count')
    quotas=[p.particle_count*s[1] for s in active];counts=[max(1,int(q)) for q in quotas]
    while sum(counts)>p.particle_count:
        i=max((i for i,c in enumerate(counts) if c>1),key=lambda i:counts[i]-quotas[i]);counts[i]-=1
    for i in sorted(range(len(counts)),key=lambda i:quotas[i]-counts[i],reverse=True)[:p.particle_count-sum(counts)]:counts[i]+=1
    # New managed children follow the active group, even when the root is linked
    # to more than one collection. No existing user object is moved or removed.
    collection = scene.acl_settings.source_collection
    if collection is None or not any(obj == root for obj in collection.all_objects):
        collection = root.users_collection[0]
    for (kind,fraction,radius,a),count in zip(active,counts):
        obj=existing.pop(kind,None)
        created=obj is None
        if created:
            mesh=bpy.data.meshes.new('Galaxy component guide')
            obj=bpy.data.objects.new(root.name+' '+kind.title(),mesh)
            collection.objects.link(obj);obj.parent=root;obj['acl_component']=kind
            obj.display_type='WIRE'
        b=obj.acl_body;b.enabled=True;b.kind=kind;b.native_primitive=True;b.is_galaxy=False
        b.mass=p.mass*fraction;b.particle_count=count;b.radius=radius;b.scale_length=a
        b.thickness=p.thickness;b.toomre_q=p.toomre_q;b.rotation_fraction=p.rotation_fraction if kind=='DISK' else 0.
        b.seed=p.seed;b.velocity=p.velocity;b.spin=p.spin
        if created:obj.color=(.25,.6,1.,1.) if kind=='DISK' else ((1.,.65,.25,1.) if kind=='BULGE' else (.08,.12,.2,.12))
        if created or obj.get('acl_guide_radius') != b.radius or obj.get('acl_guide_thickness') != b.thickness or obj.get('acl_guide_kind') != b.kind:
            _guide(obj)
    for obj in existing.values():
        mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
        if mesh.users==0:bpy.data.meshes.remove(mesh)


def _guide(obj):
    import math
    b=obj.acl_body;r=b.radius;z=b.thickness/2 if b.kind=='DISK' else r
    # Wire guide with exact bounds, not the sampled payload.
    if b.kind=='DISK':
        verts=[(r*math.cos(i*math.pi/16),r*math.sin(i*math.pi/16),h) for h in (-z,z) for i in range(32)]
        edges=[(k*32+i,k*32+(i+1)%32) for k in (0,1) for i in range(32)]+[(i,i+32) for i in range(0,32,8)]
    else:
        verts=[]
        for plane in range(3):
            for i in range(32):
                a=r*math.cos(i*math.pi/16);c=r*math.sin(i*math.pi/16)
                verts.append((a,c,0) if plane==0 else ((a,0,c) if plane==1 else (0,a,c)))
        edges=[(k*32+i,k*32+(i+1)%32) for k in range(3) for i in range(32)]
    if obj.data.users>1:obj.data=obj.data.copy()
    obj.data.clear_geometry();obj.data.from_pydata(verts,edges,[]);obj.data.update()
    obj['acl_guide_radius']=r;obj['acl_guide_thickness']=b.thickness;obj['acl_guide_kind']=b.kind


def _root_key(root):
    p=root.acl_body
    inherited = ('enabled', 'is_galaxy', 'kind', 'native_primitive', 'mass', 'particle_count',
                 'radius', 'scale_length', 'thickness', 'toomre_q', 'rotation_fraction',
                 'seed', 'velocity', 'spin')
    children = tuple(sorted(
        (obj.as_pointer(), obj.get('acl_component'),
         tuple(tuple(getattr(obj.acl_body, key)) if key in ('velocity', 'spin')
               else getattr(obj.acl_body, key) for key in inherited))
        for obj in root.children if obj.get('acl_component')))
    return tuple(getattr(p,k) if k not in ('velocity','spin') else tuple(getattr(p,k)) for k in
                 ('galaxy_type','mass','particle_count','radius','scale_length','thickness','disk_fraction','bulge_fraction','toomre_q','rotation_fraction','seed','velocity','spin')) + children


def synchronize(scene):
    """Reconcile root-owned primitives in the active source collection only."""
    global _BUSY
    was_busy, _BUSY = _BUSY, True
    try:
        collection = scene.acl_settings.source_collection
        if collection is None or collection not in tuple(ui._scene_collections(scene)):
            return
        _validate_group_membership(collection)
        # Update roots before capturing new children and their evaluated matrices.
        for obj in tuple(collection.all_objects):
            if obj.acl_body.is_galaxy and not obj.get('acl_component'):
                key = _root_key(obj)
                if _SIGNATURES.get(('root', obj.as_pointer())) != key:
                    _sync_galaxy(obj, scene)
                    _SIGNATURES[('root', obj.as_pointer())] = _root_key(obj)
        for obj in tuple(collection.all_objects):
            p = obj.acl_body
            if p.native_primitive and p.enabled and not p.is_galaxy:
                if obj.get('acl_guide_radius') != p.radius or obj.get('acl_guide_thickness') != p.thickness or obj.get('acl_guide_kind') != p.kind:
                    _guide(obj)
        # Native transform matrices need an explicit evaluation when Start/Step
        # immediately follows a scripted/property edit before the next draw.
        for layer in scene.view_layers:
            layer.update()
    finally:
        _BUSY = was_busy


def accept_snapshot(scene, config, bodies, objects, *, preserve_restart=False):
    """Retain local preview geometry; native transforms change draw matrices."""
    global _HANDLE, _PREVIEW_CACHE
    from .runtime import _signature
    from .preview_cache import PreviewCache
    if _PREVIEW_CACHE is None:
        _PREVIEW_CACHE = PreviewCache()
    pointer = scene.as_pointer()
    previous = _PREVIEWS.get(pointer, {})
    entries = {}
    for obj, body in zip(objects, bodies):
        key, points = _PREVIEW_CACHE.sample(body, config)
        old = previous.get(obj.as_pointer())
        batch = old['batch'] if old is not None and old['key'] == key else None
        entries[obj.as_pointer()] = dict(obj=obj, key=key, points=points, batch=batch,
                                        rotation=body.rotation, position=body.position)
    _PREVIEWS[pointer] = entries
    _SIGNATURES[pointer] = _signature(config, bodies)
    _PENDING.pop(pointer, None)
    if not preserve_restart:
        _RESTARTS.pop(pointer, None)
    if _HANDLE is None and not bpy.app.background:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')


def flush():
    global _BUSY
    if _BUSY:
        return .03
    _BUSY = True
    try:
        from .runtime import authored_scene, _signature, controller
        for pointer, (scene, deadline) in list(_PENDING.items()):
            if time.monotonic() < deadline:
                continue
            del _PENDING[pointer]
            try:
                synchronize(scene)
                if not ui.source_objects(scene):
                    if controller.scene == scene:
                        controller.reset(preserve_intent=True)
                    _PREVIEWS.pop(pointer, None)
                    _SIGNATURES.pop(pointer, None)
                    _RESTARTS.pop(pointer, None)
                    scene.acl_settings.authoring_status = 'Preview; add primitives or choose a source collection'
                    continue
                config, bodies, objects = authored_scene(scene)
                signature = _signature(config, bodies)
                owned = controller.scene == scene
                replaced = (owned and controller.simulation is not None
                            and not same_sources(controller.sources, objects))
                if _SIGNATURES.get(pointer) == signature and not replaced:
                    accept_snapshot(scene, config, bodies, objects, preserve_restart=True)
                    if controller.scene == scene:
                        controller.sources = tuple(objects)
                    continue
                if owned and (controller.signature != signature or replaced):
                    controller.reset(preserve_intent=True)
                accept_snapshot(scene, config, bodies, objects)
                scene.acl_settings.authoring_status = 'Preview - equilibrium not validated' if config.mode == 'GALAXY' else 'Preview'
                if owned and controller.intent != 'STOPPED':
                    _RESTARTS[pointer] = (scene, time.monotonic() + .35)
                    controller.message = 'Preview; waiting for edits to settle'
            except Exception as exc:
                try:
                    scene.acl_settings.authoring_status = 'Invalid settings: ' + str(exc)
                except ReferenceError:
                    pass
                if controller.scene == scene:
                    controller.reset(preserve_intent=True)
                    controller.error = str(exc)
                    controller.message = 'Invalid settings: ' + str(exc)
                _PREVIEWS.pop(pointer, None)
                _SIGNATURES.pop(pointer, None)
                _RESTARTS.pop(pointer, None)
        for pointer, (scene, deadline) in list(_RESTARTS.items()):
            if time.monotonic() < deadline or pointer in _PENDING:
                continue
            del _RESTARTS[pointer]
            if controller.scene != scene or controller.intent == 'STOPPED':
                continue
            try:
                controller.begin(SimpleNamespace(scene=scene), play=controller.intent == 'PLAY')
            except Exception as exc:
                controller.reset(preserve_intent=True)
                controller.error = str(exc)
                controller.message = 'Invalid settings: ' + str(exc)
                scene.acl_settings.authoring_status = controller.message
        from .viewport import request_redraw
        request_redraw()
    finally:
        _BUSY = False
    return .03 if _PENDING or _RESTARTS else None


def cancel_restart(scene):
    if scene is None:
        _RESTARTS.clear()
    else:
        _RESTARTS.pop(scene.as_pointer(), None)


def pause_restart(scene):
    # Playback intent is owned by the controller, including a pending rebuild.
    from .runtime import controller
    if scene is None or controller.scene == scene:
        controller.intent = 'PAUSE'


def draw():
    from .runtime import controller
    scene = bpy.context.scene
    if controller.scene == scene and controller.simulation is not None:
        return
    entries = _PREVIEWS.get(scene.as_pointer())
    if entries is None:
        return
    import gpu
    import numpy as np
    from mathutils import Matrix
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    old_blend = gpu.state.blend_get()
    old_depth = gpu.state.depth_test_get()
    old_mask = gpu.state.depth_mask_get()
    try:
        gpu.state.blend_set({'OPAQUE': 'NONE', 'ALPHA': 'ALPHA', 'ADDITIVE': 'ADDITIVE'}[scene.acl_settings.blend_mode])
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        gpu.state.point_size_set(scene.acl_settings.particle_size_px)
        for entry in entries.values():
            obj = entry['obj']
            if obj.name not in scene.objects or not obj.visible_get():
                continue
            if entry['batch'] is None:
                entry['batch'] = batch_for_shader(shader, 'POINTS', {'pos': np.ascontiguousarray(entry['points'], dtype=np.float32)})
            transform = Matrix(entry['rotation']).to_4x4()
            transform.translation = entry['position']
            with gpu.matrix.push_pop():
                gpu.matrix.multiply_matrix(transform)
                shader.bind()
                shader.uniform_float('color', tuple(obj.color))
                entry['batch'].draw(shader)
    except ReferenceError:
        pass
    finally:
        gpu.state.blend_set(old_blend)
        gpu.state.depth_test_set(old_depth)
        gpu.state.depth_mask_set(old_mask)
        gpu.state.point_size_set(1.)


def depsgraph(scene,graph):
    if _BUSY:return
    if any(u.is_updated_transform or u.is_updated_geometry or isinstance(u.id,(bpy.types.Object,bpy.types.Scene,bpy.types.Collection)) for u in graph.updates):
        changed(None,SimpleNamespace(scene=scene))


def clear(_unused=None):
    global _HANDLE
    _THREADED_PHYSICS.clear();_THREADED_APPEARANCE.clear()
    _PENDING.clear();_SIGNATURES.clear();_PREVIEWS.clear();_RESTARTS.clear()
    if _PREVIEW_CACHE is not None:
        _PREVIEW_CACHE.clear()
    if bpy.app.timers.is_registered(flush):bpy.app.timers.unregister(flush)
    if _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE,'WINDOW');_HANDLE=None


def reload(_unused=None):
    for scene in bpy.data.scenes:
        changed(None,SimpleNamespace(scene=scene))


def register():
    if not bpy.app.timers.is_registered(drain_threaded_updates):
        bpy.app.timers.register(drain_threaded_updates, first_interval=.03, persistent=True)
    for name,fn in (('depsgraph_update_post',depsgraph),('load_pre',clear),('undo_pre',clear),('redo_pre',clear),('load_post',reload),('undo_post',reload),('redo_post',reload)):
        bpy.app.handlers.persistent(fn)
        if fn not in getattr(bpy.app.handlers,name):getattr(bpy.app.handlers,name).append(fn)
    if not bpy.app.timers.is_registered(reload):bpy.app.timers.register(reload,first_interval=.1)


def unregister():
    if bpy.app.timers.is_registered(drain_threaded_updates):
        bpy.app.timers.unregister(drain_threaded_updates)
    if bpy.app.timers.is_registered(reload):bpy.app.timers.unregister(reload)
    clear()
    for name,fn in (('depsgraph_update_post',depsgraph),('load_pre',clear),('undo_pre',clear),('redo_pre',clear),('load_post',reload),('undo_post',reload),('redo_post',reload)):
        if fn in getattr(bpy.app.handlers,name):getattr(bpy.app.handlers,name).remove(fn)
