"""Real installed Blender GUI / CUDA-OpenGL / 100K acceptance diagnostic.

Run only through the isolated Windows harness. Actual drawing and glFinish inside
the add-on's POST_VIEW callback must complete every measured revision.
"""
import argparse
import importlib
import json
import math
import os
import pathlib
import sys
import time
import traceback
import bpy

parser=argparse.ArgumentParser()
parser.add_argument("--edit-probe",action="store_true")
parser.add_argument("--mode",choices=["IMPACT","GALAXY"],required=True)
parser.add_argument("--module",required=True)
parser.add_argument("--report",required=True)
parser.add_argument("--frames",type=int,default=120)
parser.add_argument("--warmup",type=int,default=15)
parser.add_argument("--timeout",type=float,default=240.0)
args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
REPORT=pathlib.Path(args.report)
REPORT.parent.mkdir(parents=True,exist_ok=True)
STATE={"started":time.monotonic(),"initialized":False,"last_progress":time.monotonic(),
       "last_revision":-1,"mode":args.mode,"module":args.module,"frames":args.frames,
       "warmup":args.warmup}
MODULE=None
RUNTIME=None

def p95(values):
    a=sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not a:return None
    index=(len(a)-1)*0.95
    lo=int(index);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(index-lo)

def finish(failure=None):
    owner=RUNTIME.controller if RUNTIME is not None else None
    rows=list(owner.measurements) if owner is not None else []
    usable=[r for r in rows if r.get("revision",0)>args.warmup]
    usable=usable[:args.frames]
    intervals=[r.get("frame_interval_ms") for r in usable if r.get("frame_interval_ms") is not None]
    e2e=[r.get("end_to_end_ms") for r in usable if r.get("end_to_end_ms") is not None]
    elapsed=sum(intervals)/1000 if intervals else 0
    fps=len(intervals)/elapsed if elapsed else 0.0
    sizes=[(r.get("active_count"),r.get("displayed_count")) for r in usable]
    unique=len({r.get("revision") for r in usable})
    viewport={} if owner is None else owner.status().get("viewport",{})
    batches=viewport.get("batches",[])
    strict=bool(batches) and all(b.get("strict_interop") and
        not b.get("fallback_to_copy",True) and b.get("host_particle_copies")==0 for b in batches)
    limit=1000.0/24.0
    checks={
      "exact_active_and_displayed_100k":len(sizes)==args.frames and all(pair==(100000,100000) for pair in sizes),
      "new_distinct_states":unique==args.frames,
      "cuda_opengl_strict":strict,
      "complete_fps_at_least_24":fps>=24.0,
      "p95_display_interval_under_budget":p95(intervals) is not None and p95(intervals)<=limit,
      "p95_step_to_glfinish_under_budget":p95(e2e) is not None and p95(e2e)<=limit,
      "actual_physical_progress":len(usable)>=2 and usable[-1]["physical_time"]>usable[0]["physical_time"],
      "no_runtime_error":not failure and not (owner and owner.error),
      "edit_workflow":not args.edit_probe or STATE.get("edit_stage")==3,
    }
    out={
      "schema":"astro-collision-lab-windows-installed-live-100k-v1",
      "mode":args.mode,"timestamp_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
      "blender":bpy.app.version_string,"background":bpy.app.background,
      "gpu_backend":"OpenGL-required-and-reported-by-cuda_gl",
      "requested_count":100000,"warmup":args.warmup,"measured_frames":args.frames,
      "observed_rows":len(rows),"usable_rows":len(usable),"unique_frames":unique,
      "end_to_end_complete_fps":fps,
      "p95_draw_interval_ms":p95(intervals),"p95_step_to_glfinish_ms":p95(e2e),
      "mean_step_to_glfinish_ms":sum(e2e)/len(e2e) if e2e else None,
      "max_step_to_glfinish_ms":max(e2e) if e2e else None,
      "frame_budget_ms":limit,"checklist":checks,
      "source_presented_counts":sizes[:4],
      "first_last_physical_time":[usable[0]["physical_time"],usable[-1]["physical_time"]] if usable else [],
      "solver_status":None if owner is None else owner.status().get("solver"),
      "viewport_status":viewport,
      "edit_checks":STATE.get("edit_checks",{}),
      "controller_message":None if owner is None else owner.message,
      "controller_error":None if owner is None else owner.error,
      "error":failure,"pass":all(checks.values())
    }
    REPORT.write_text(json.dumps(out,indent=2,default=str)+"\n",encoding="utf8")
    print("LIVE_100K_REPORT="+str(REPORT))
    print("LIVE_100K_SUMMARY="+json.dumps({k:out[k] for k in
        ("mode","usable_rows","end_to_end_complete_fps","p95_step_to_glfinish_ms","pass","error")}))
    try:
        if owner is not None: owner.reset()
    except Exception as exc:print("RESET_WARNING",repr(exc))
    bpy.ops.wm.quit_blender()

def setup():
    global MODULE,RUNTIME
    MODULE=importlib.import_module(args.module)
    MODULE.register()
    RUNTIME=importlib.import_module(args.module+".runtime")
    _read_scene = RUNTIME.authored_scene
    def _traced_authored_scene(scene):
        cfg, bodies, objs = _read_scene(scene)
        print("TRACE_AUTHORING_READ", json.dumps({
            "mode": cfg.mode, "scene_id": scene.as_pointer(),
            "source_count": len(objs),
            "source_kinds": [b.kind for b in bodies],
        }), flush=True)
        return cfg, bodies, objs
    RUNTIME.authored_scene = _traced_authored_scene
    window=bpy.context.window_manager.windows[0]
    area=next(a for a in window.screen.areas if a.type=="VIEW_3D")
    region=next(r for r in area.regions if r.type=="WINDOW")
    area.spaces.active.shading.type="SOLID"
    area.spaces.active.overlay.show_overlays=False
    area.spaces.active.show_region_ui=False
    area.spaces.active.show_region_toolbar=False
    bpy.context.scene.render.fps=24
    with bpy.context.temp_override(window=window,screen=window.screen,area=area,region=region):
        op=bpy.ops.acl.moon_preset if args.mode=="IMPACT" else bpy.ops.acl.galaxy_preset
        assert op()=={"FINISHED"},"preset authoring failed"
        cfg,bodies,objects=RUNTIME.authored_scene(bpy.context.scene)
        assert cfg.mode==args.mode and cfg.particle_count==100000
        print("TRACE_PRE_START_MODE", bpy.context.scene.acl_settings.mode, flush=True)
        RUNTIME.controller.measurement_sync=True
        assert bpy.ops.acl.start()=={"FINISHED"},"Start failed"
        print("TRACE_POST_START_MODE", bpy.context.scene.acl_settings.mode, flush=True)
        if RUNTIME.controller.pending is not None:
            print("TRACE_PENDING_CONFIG_MODE", RUNTIME.controller.pending[1].mode, flush=True)
    # Frame the complete larger galaxy definitions instead of measuring an offscreen cloud.
    positions=[b.position for b in bodies]
    import mathutils
    centre=tuple(sum(p[i] for p in positions)/len(positions) for i in range(3))
    area.spaces.active.region_3d.view_location=centre
    area.spaces.active.region_3d.view_distance=35. if args.mode=="GALAXY" else 5.
    STATE["initialized"]=True
    STATE["last_progress"]=time.monotonic()
    print("GUI_PRESET_START="+args.mode)
    
def tick():
    try:
        if not STATE["initialized"]:setup()
        owner=RUNTIME.controller
        if owner.revision>STATE["last_revision"]:
            STATE["last_revision"]=owner.revision
            STATE["last_progress"]=time.monotonic()
        if args.edit_probe and owner.simulation is not None and owner.playing:
            stage=STATE.get("edit_stage",0)
            if stage==0 and owner.revision>=30:
                STATE["edit_run"]=owner.run_id
                STATE["edit_checks"]={}
                owner.scene.acl_settings.particle_size_px=3.
                obj=next(o for o in owner.scene.acl_settings.source_collection.all_objects if o.acl_body.enabled)
                obj.color=(.95,.3,.12,1.)
                STATE["edit_stage"]=1
            elif stage==1 and owner.revision>=50:
                assert owner.run_id==STATE["edit_run"],"appearance edit restarted physics"
                STATE["edit_checks"]["appearance_preserved_run"]=True
                STATE["new_dt"]=owner.scene.acl_settings.seconds_per_frame*1.01
                owner.scene.acl_settings.seconds_per_frame=STATE["new_dt"]
                STATE["edit_stage"]=2
            elif stage==2 and owner.run_id!=STATE["edit_run"] and owner.revision>=30:
                assert abs(owner.simulation.config.dt-STATE["new_dt"])<1e-6,"rebuild used stale settings"
                assert owner.simulation.count==100000,"edit changed requested particle budget"
                STATE["edit_checks"]["physical_edit_rebuilt_and_resumed"]=True
                STATE["edit_checks"]["new_dt"]=owner.simulation.config.dt
                STATE["edit_stage"]=3
        rows=[r for r in owner.measurements if r.get("revision",0)>args.warmup]
        if len(rows)>=args.frames:
            finish()
            return None
        if owner.error:
            finish("controller_error: "+str(owner.error))
            return None
        if time.monotonic()-STATE["started"]>args.timeout:
            finish("timeout")
            return None
        if (not owner.warming and not owner.playing and owner.simulation is not None and
            time.monotonic()-STATE["last_progress"]>3):
            finish("playback stopped before required frames")
            return None
    except BaseException:
        finish(traceback.format_exc())
        return None
    return 0.05

bpy.app.timers.register(tick,first_interval=2.0)
