"""Native RNA acceptance for new object catalogue. No physics/performance claim."""
import sys,json,time,traceback
from pathlib import Path
import bpy
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'extension'))
import astro_collision_lab as addon
from astro_collision_lab import ui,authoring,runtime
report={'pass':False,'checks':[]}
def check(name,value):
    assert value,name
    report['checks'].append(name)
def flush():
    for key,(scene,deadline) in list(authoring._PENDING.items()):authoring._PENDING[key]=(scene,0.)
    authoring.flush();bpy.context.view_layer.update()
try:
    addon.register()
    scene=bpy.context.scene;scene.cursor.location=(3,4,5)
    for kind in ('SPHERE','DISK','BULGE','HALO'):
        bpy.ops.acl.add_source(kind=kind);obj=bpy.context.object
        check('Add '+kind,obj.acl_body.kind==kind and tuple(obj.location)==(3.,4.,5.))
        obj.acl_body.radius=2.;flush()
        check('Radius updates guide '+kind,abs(obj.dimensions.x-4)<1e-4)
    bpy.ops.acl.add_source(kind='GALAXY');gal=bpy.context.object;flush()
    check('Galaxy root',gal.acl_body.is_galaxy and len(gal.children)==2)
    check('Exact 50K budget',sum(o.acl_body.particle_count for o in gal.children)==50000)
    scene.acl_settings.source_collection=gal.users_collection[0]
    for kind in ('ELLIPTICAL','LENTICULAR','DWARF','DISK'):
        gal.acl_body.galaxy_type=kind;flush()
        check('Family '+kind,sum(o.acl_body.particle_count for o in gal.children)==50000)
    gal.location.x+=2.;bpy.context.view_layer.update();flush()
    cfg,bodies,objects=runtime.authored_scene(scene)
    linked=[b for b in bodies if b.galaxy_id==gal.name]
    check('Explicit membership',len(linked)==2 and all(abs(b.position[0]-5)<1e-4 for b in linked))
    check('Preview exists',scene.as_pointer() in authoring._PREVIEWS)
    check('No unearned validation','Validated' not in scene.acl_settings.authoring_status)
    gal.acl_body.mass=2.;flush()
    linked=[b for b in runtime.authored_scene(scene)[1] if b.galaxy_id==gal.name]
    check('Mass mapped',abs(sum(b.mass for b in linked)-2)<1e-5)
    bpy.ops.acl.galaxy_preset();flush()
    check('Demo exactly 100K',sum(o.acl_body.particle_count for o in ui.source_objects(scene))==100000)
    roots=[o for o in scene.objects if o.acl_body.is_galaxy and o.name in ('Galaxy A','Galaxy B')]
    check('Demo compound sources',len(roots)==2 and all(len(o.children)==2 for o in roots))
    cfg,bodies,objects=runtime.authored_scene(scene)
    check('Demo 2 groups',len(set(b.galaxy_id for b in bodies))==2)
    import dataclasses
    (root/'status'/'native-galaxy-snapshot-v2.json').write_text(json.dumps({'config':dataclasses.asdict(cfg),'bodies':[dataclasses.asdict(b) for b in bodies]}))
    bpy.ops.acl.moon_preset();flush()
    check('Impact still 100K',sum(o.acl_body.particle_count for o in ui.source_objects(scene))==100000)
    cfg,bodies,objects=runtime.authored_scene(scene)
    check('Synthetic controls mapped',cfg.bulk_modulus==scene.acl_settings.bulk_modulus)
    blend=root/'status'/'native-v2.blend';bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.open_mainfile(filepath=str(blend));flush()
    check('Save reopen sources',sum(o.acl_body.particle_count for o in ui.source_objects(bpy.context.scene))==100000)
    addon.unregister();check('Clean unregister',not hasattr(bpy.types.Object,'acl_body'))
    report['pass']=True
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'status'/'native-authoring-v2.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if not report['pass']:raise RuntimeError(report.get('error'))
