"""Real installed Blender GUI / CUDA-OpenGL / 100K acceptance diagnostic.

Run only through the isolated Windows harness. Actual drawing and glFinish inside
the add-on's POST_VIEW callback must complete every measured revision.
"""
import argparse
import importlib
import json
import math
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
# Counts other than the presets' 100,000 are reached the way a user does it: edit
# each Galaxy root's particle budget (children re-sync), optionally pick storage.
parser.add_argument("--count",type=int,default=100000)
parser.add_argument("--tree-storage",choices=["STRIDED","COMPACT"],default=None)
parser.add_argument("--tree-depth",type=int,default=None)
parser.add_argument("--trace-main",action="store_true",help="diagnostic: sample the main thread stack (perturbs timing; not a gate run)")
args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
if args.count!=100000 and args.mode!="GALAXY":
    parser.error("--count other than 100000 is only defined for GALAXY")
COUNT=args.count
REPORT=pathlib.Path(args.report)
REPORT.parent.mkdir(parents=True,exist_ok=True)
STATE={"started":time.monotonic(),"initialized":False,"last_progress":time.monotonic(),
       "last_revision":-1,"mode":args.mode,"module":args.module,"frames":args.frames,
       "warmup":args.warmup}
MODULE=None
RUNTIME=None
# Diagnostics only (no gate effect): whether Blender owns the foreground window and
# the process timer resolution, sampled every probe tick, to explain cadence stalls.
ENV_SAMPLES=[]
TRACE={"samples":[],"events":[]}

def start_main_trace(controller):
    import threading
    main=threading.main_thread().ident
    cls=type(controller)
    real_advance,real_presented=cls.advance,cls.on_presented
    def advance(self):
        TRACE["events"].append(("step",self.revision+1,time.perf_counter()))
        return real_advance(self)
    def on_presented(self,revision,count,draw_start,packed,finished):
        TRACE["events"].append(("present",revision,finished))
        return real_presented(self,revision,count,draw_start,packed,finished)
    cls.advance,cls.on_presented=advance,on_presented
    def sampler():
        while not STATE.get("finished"):
            frame=sys._current_frames().get(main)
            stack=[]
            while frame is not None and len(stack)<4:
                stack.append(f"{pathlib.Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}")
                frame=frame.f_back
            TRACE["samples"].append((time.perf_counter(),"|".join(stack) or "<C>"))
            time.sleep(.001)
    threading.Thread(target=sampler,daemon=True).start()

def sample_environment(revision):
    if sys.platform!="win32":return
    import ctypes
    import os
    user32=ctypes.windll.user32
    hwnd=user32.GetForegroundWindow()
    pid=ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
    lo,hi,cur=ctypes.c_ulong(),ctypes.c_ulong(),ctypes.c_ulong()
    ctypes.windll.ntdll.NtQueryTimerResolution(ctypes.byref(lo),ctypes.byref(hi),ctypes.byref(cur))
    ENV_SAMPLES.append({"t":round(time.monotonic()-STATE["started"],3),"revision":revision,
                        "foreground_is_blender":pid.value==os.getpid(),"foreground_pid":pid.value,
                        "timer_resolution_ms":cur.value/10000})

def p95(values):
    a=sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not a:return None
    index=(len(a)-1)*0.95
    lo=int(index);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(index-lo)

def finish(failure=None):
    STATE["finished"]=True
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
      "exact_active_and_displayed_count":len(sizes)==args.frames and all(pair==(COUNT,COUNT) for pair in sizes),
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
      "requested_count":COUNT,"tree_storage":args.tree_storage,"tree_depth":args.tree_depth,"warmup":args.warmup,"measured_frames":args.frames,
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
      "error":failure,"pass":all(checks.values()),
      "measured_rows":usable,"environment_samples":ENV_SAMPLES,"main_trace":TRACE if args.trace_main else None,
    }
    REPORT.write_text(json.dumps(out,indent=2,default=str)+"\n",encoding="utf8")
    print("LIVE_100K_REPORT="+str(REPORT))
    print("LIVE_100K_SUMMARY="+json.dumps({k:out[k] for k in
        ("mode","usable_rows","end_to_end_complete_fps","p95_step_to_glfinish_ms","pass","error")}))
    try:
        if owner is not None: owner.reset()
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup after the report is written
        print("RESET_WARNING",repr(exc))
    bpy.ops.wm.quit_blender()

def setup():
    global MODULE,RUNTIME
    MODULE=importlib.import_module(args.module)
    MODULE.register()
    RUNTIME=importlib.import_module(args.module+".runtime")
    _read_scene = RUNTIME.authored_scene
    def _traced_authored_scene(scene):
        cfg, bodies, objs = _read_scene(scene)
        # The runtime reads the authored scene on every playback tick. Printing (with
        # flush) each time put probe stdout I/O inside the measured loop and stalled
        # the main thread whenever the pipe backed up (2026-11-18 traces), so log
        # only when what was read changes.
        line = json.dumps({
            "mode": cfg.mode, "scene_id": scene.as_pointer(),
            "source_count": len(objs),
            "source_kinds": [b.kind for b in bodies],
        })
        if line != STATE.get("last_authoring_read"):
            STATE["last_authoring_read"] = line
            print("TRACE_AUTHORING_READ", line, flush=True)
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
        settings=bpy.context.scene.acl_settings
        if args.tree_storage is not None: settings.tree_storage=args.tree_storage
        if args.tree_depth is not None: settings.tree_max_depth=args.tree_depth
        if COUNT!=100000:
            roots=[o for o in settings.source_collection.all_objects if o.acl_body.is_galaxy and o.parent is None]
            assert roots,"no Galaxy roots authored"
            share,extra=divmod(COUNT,len(roots))
            for index,root in enumerate(roots):
                root.acl_body.particle_count=share+(index<extra)
            # Root edits re-sync children through the debounced authoring queue;
            # settle it now, as the native authoring suite does.
            authoring=importlib.import_module(args.module+".authoring")
            for key,(queued_scene,_deadline) in list(authoring._PENDING.items()):
                authoring._PENDING[key]=(queued_scene,0.)
            authoring.flush()
            bpy.context.view_layer.update()
        cfg,bodies,_objects=RUNTIME.authored_scene(bpy.context.scene)
        assert cfg.mode==args.mode and cfg.particle_count==COUNT,(cfg.mode,cfg.particle_count)
        assert args.tree_storage is None or cfg.tree_storage==args.tree_storage,cfg.tree_storage
        print("TRACE_AUTHORED", json.dumps({"count":cfg.particle_count,"storage":cfg.tree_storage,
                                            "depth":cfg.tree_max_depth}), flush=True)
        print("TRACE_PRE_START_MODE", bpy.context.scene.acl_settings.mode, flush=True)
        RUNTIME.controller.measurement_sync=True
        if args.trace_main: start_main_trace(RUNTIME.controller)
        assert bpy.ops.acl.start()=={"FINISHED"},"Start failed"
        print("TRACE_POST_START_MODE", bpy.context.scene.acl_settings.mode, flush=True)
        if RUNTIME.controller.pending is not None:
            print("TRACE_PENDING_CONFIG_MODE", RUNTIME.controller.pending[1].mode, flush=True)
    # Frame the complete larger galaxy definitions instead of measuring an offscreen cloud.
    positions=[b.position for b in bodies]
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
        sample_environment(owner.revision)
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
                assert owner.simulation.count==COUNT,"edit changed requested particle budget"
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
    except BaseException:  # noqa: BLE001 - probe timer records any failure in the report
        finish(traceback.format_exc())
        return None
    return 0.05

bpy.app.timers.register(tick,first_interval=2.0)
