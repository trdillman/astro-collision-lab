import bpy,importlib,json,pathlib,sys,traceback
args=sys.argv[sys.argv.index("--")+1:]
module_name,mode,phase,blend,report=args
mod=importlib.import_module(module_name)
mod.register()
from bl_ext.acla.astro_collision_lab import runtime,ui
bpy.ops.object.select_all(action="SELECT")
if phase=="create":
    if bpy.data.objects:bpy.ops.object.delete()
    op=bpy.ops.acl.moon_preset if mode=="IMPACT" else bpy.ops.acl.galaxy_preset
    assert op()=={"FINISHED"}
    config,bodies,objects=runtime.authored_scene(bpy.context.scene)
    assert config.mode==mode and config.particle_count==100000
    assert len(bodies)==(2 if mode=="IMPACT" else 4)
    first=objects[0]
    first.location.x+=0.375
    bpy.context.view_layer.update()
    config,bodies,objects=runtime.authored_scene(bpy.context.scene)
    assert bodies[0].position[0]!=0
    if mode=="IMPACT":
        assert bpy.ops.acl.balance_sampling()=={"FINISHED"}
        config,bodies,objects=runtime.authored_scene(bpy.context.scene)
        assert sum(b.particle_count for b in bodies)==100000
    bpy.ops.wm.save_as_mainfile(filepath=blend)
else:
    bpy.ops.wm.open_mainfile(filepath=blend)
    config,bodies,objects=runtime.authored_scene(bpy.context.scene)
    assert config.mode==mode and config.particle_count==100000
    assert len(bodies)==(2 if mode=="IMPACT" else 4)
    assert objects[0].location.x!=0
    if mode=="IMPACT":
        assert sum(b.particle_count for b in bodies)==100000
result={"mode":mode,"phase":phase,"pass":True,"active_count":config.particle_count,"bodies":len(bodies),
        "saved_blend":blend,"native_blender":bpy.app.version_string}
pathlib.Path(report).write_text(json.dumps(result,indent=2)+"\n")
print("WINDOWS_AUTHORING_PHASE_PASS",json.dumps(result))
