"""Completed GPU solver diagnostic, NOT live viewport acceptance.

Example: python validation/solver_probe.py --mode IMPACT --count 100000 --frames 120
Full particle readbacks occur only before/after measured stepping for validation.
"""
from __future__ import annotations
import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "extension"))
sys.path.insert(0,str(ROOT))

def _presets():
    path=ROOT/'preset_specs.py'
    if not path.exists():path=ROOT/'extension'/'astro_collision_lab'/'preset_specs.py'
    spec=importlib.util.spec_from_file_location('acl_probe_presets',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def _rotation(euler):
    import numpy as np
    x,y,z=euler;c=np.cos; s=np.sin
    rx=np.array(((1,0,0),(0,c(x),-s(x)),(0,s(x),c(x))))
    ry=np.array(((c(y),0,s(y)),(0,1,0),(-s(y),0,c(y))))
    rz=np.array(((c(z),-s(z),0),(s(z),c(z),0),(0,0,1)))
    return tuple(tuple(float(v) for v in row) for row in rz@ry@rx)

def probe_scene(mode,count,dt=None):
    """Same authored presets, largest-remainder count scaling with no idle slots."""
    from astro_collision_lab.simulation import BodySpec,SimulationConfig
    mode=mode.upper();presets=_presets()
    recipe=presets.galaxy_collision() if mode=='GALAXY' else presets.moon_collision()
    if count<len(recipe.bodies):raise ValueError('count must allocate at least one particle per source')
    total=sum(b.count for b in recipe.bodies)
    ideal=[count*b.count/total for b in recipe.bodies]
    counts=[max(1,math.floor(v)) for v in ideal]
    while sum(counts)>count:
        i=max((i for i,n in enumerate(counts) if n>1),key=lambda i:counts[i]-ideal[i]);counts[i]-=1
    order=sorted(range(len(counts)),key=lambda i:ideal[i]-counts[i],reverse=True)
    for i in range(count-sum(counts)):counts[order[i%len(order)]]+=1
    bodies=[BodySpec(name=b.name,kind=b.kind,particle_count=n,mass=b.mass,radius=b.radius,
        position=b.location,velocity=b.velocity,spin=b.spin,rotation=_rotation(b.rotation),
        material='TILLOTSON' if mode=='COMPRESSIBLE' else b.material)
        for b,n in zip(recipe.bodies,counts)]
    config=SimulationConfig(mode=mode,particle_count=count,dt=recipe.seconds_per_frame if dt is None else dt)
    config.validate()
    return config,bodies

def _substeps(sim):
    """Host scalars only; unknown actual adaptive counts remain explicitly unknown."""
    runtime=sim.runtime;mode=sim.config.mode.upper() if hasattr(sim,'config') else 'TEST'
    if mode=='GALAXY':return {'actual_substeps':sim.config.substeps,'inner_dt':sim.config.dt/sim.config.substeps}
    if mode=='COMPRESSIBLE':return {'actual_substeps':int(runtime.last_substeps),
        'last_committed_inner_dt':float(runtime.hydro.committed_timestep())}
    if mode=='IMPACT':
        solver=runtime._require_solver()
        # One small adaptive tally read, explicitly outside each frame timing.
        outer=int(solver.active_outer_steps)
        return {'planned_outer_steps':int(solver.outer_steps),'actual_outer_steps':outer,
            'inner_steps_per_outer':int(solver.inner_steps),'actual_substeps':outer*int(solver.inner_steps),
            'planned_inner_dt':float(solver.dt_inner),'adaptive':bool(solver._adaptive),
            'inner_dt_note':'adaptive individual dt varies; average is physical frame advance / actual_substeps'}
    return {}

def final_guardrails(sim):
    """Untimed scalar diagnostics. Passing means no guard bound/truncation, not valid science."""
    mode=sim.config.mode.upper() if hasattr(sim,'config') else 'TEST'
    counters={}; failures=[]
    if mode=='IMPACT':
        solver=sim.runtime._require_solver()
        counters={
            'initial_neighbor_overflow':int(solver.neighbor_overflow),
            'neighbor_overflow_total':int(solver.neighbor_overflow_total()),
            'speed_clamps':int(solver.speed_clamps()),
            'budget_bound_steps':int(solver.budget_bound_steps),
            'pressure_sweeps':int(solver.pressure_sweeps),
            'last_outer_steps':int(solver.active_outer_steps),
            'max_outer_steps_used':int(solver.max_outer_steps_used),
            'initial_bonds_dropped_to_degree_limit':int(solver.dropped_bonds),
        }
        critical=('initial_neighbor_overflow','neighbor_overflow_total','speed_clamps','budget_bound_steps')
        failures=[key for key in critical if counters[key]!=0]
    elif mode=='GALAXY':
        diag=sim.runtime._arrays['diag'].numpy()
        counters={'tree_walk_overflow':int(diag[3]),'tree_node_capacity_overflow':int(diag[7])}
        failures=[key for key,value in counters.items() if value!=0]
    elif mode=='COMPRESSIBLE':
        hydro=sim.runtime.hydro;neighbors=hydro.neighbors.diagnostics()
        counters={'neighbor_list_overflow':int(neighbors.list_overflow),
            'unconverged_neighbors':int(neighbors.unconverged),
            'energy_floor_hits':int(hydro.energy_floor_hits()),'reach_clamps':int(hydro.reach_clamps())}
        if hydro.gravity is not None:hydro.gravity._raise_tree_overflow()
        failures=[key for key,value in counters.items() if value!=0]
    else:
        return {'solver_guardrails_pass':None,'counters':{},'failure_reasons':['unsupported mode'],
            'scientific_validation':False,'scope':'not measured'}
    return {'solver_guardrails_pass':not failures,'counters':counters,'failure_reasons':failures,
        'scientific_validation':False,
        'scope':'cumulative solver guards through initialization/warmup/measured frames; final scalar reads outside timing',
        'policy':'any neighbor truncation, velocity clamp, budget-bound step, unconverged neighbor or energy floor hit rejects this diagnostic gate; does not validate a scientific outcome'}

def measure_simulation(sim,bodies,*,frames=120,warmup=5,synchronize):
    """Initialize/warm, time completed frames, then validate host snapshots outside timing."""
    import numpy as np
    if frames<1 or warmup<0:raise ValueError('frames positive and warmup nonnegative required')
    started=time.perf_counter();sim.initialize(bodies);synchronize()
    init_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    for _ in range(warmup):sim.step();synchronize()
    warmup_ms=(time.perf_counter()-started)*1000
    before=sim.display_channels()['positions'].numpy().copy()
    physical_start=float(sim.physical_time)
    frame_ms=[];physical_deltas=[];substeps=[];counter_ms=[]
    for _ in range(frames):
        prior=float(sim.physical_time)
        started=time.perf_counter();sim.step();synchronize()
        frame_ms.append((time.perf_counter()-started)*1000)
        physical_deltas.append(float(sim.physical_time)-prior)
        counter_start=time.perf_counter()
        info=_substeps(sim)
        if info.get('actual_substeps',0)>0:
            info['average_inner_dt']=physical_deltas[-1]/info['actual_substeps']
        substeps.append(info)
        counter_ms.append((time.perf_counter()-counter_start)*1000)
    physical_end=float(sim.physical_time)
    channels=sim.display_channels()
    after=channels['positions'].numpy().copy()
    finite=bool(np.isfinite(before).all() and np.isfinite(after).all())
    checked={'positions':finite}
    for name in ('velocities','masses','radii','densities','internal_energies'):
        if name in channels:
            checked[name]=bool(np.isfinite(channels[name].numpy()).all())
            finite=finite and checked[name]
    active=int(np.count_nonzero(channels['active'].numpy())) if 'active' in channels else None
    delta=np.linalg.norm(after.astype(np.float64)-before.astype(np.float64),axis=1)
    motion={'moving_particles':int(np.count_nonzero(delta>0)),
            'max_displacement':float(np.max(delta)),'rms_displacement':float(np.sqrt(np.mean(delta*delta)))}
    a=np.asarray(frame_ms)
    guardrails=final_guardrails(sim)
    return {'schema':2,'solver_guardrails_pass':guardrails['solver_guardrails_pass'],'guardrails':guardrails,
        'health_status':'guardrails_pass' if guardrails['solver_guardrails_pass'] is True else 'guardrails_failed' if guardrails['solver_guardrails_pass'] is False else 'not_measured','diagnostic':'synchronized GPU solver + device radius update; no viewport drawing',
        'live_viewport_acceptance':False,'measured_frames':frames,'warmup_frames':warmup,
        'count':int(sim.count),'active_count':active,'initialization_ms':init_ms,'warmup_ms':warmup_ms,
        'frame_ms':frame_ms,'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),
        'p95_ms':float(np.percentile(a,95)),'max_ms':float(a.max()),'completed_solver_fps':1000/float(a.mean()),
        'physical_time_start':physical_start,'physical_time_end':physical_end,
        'physical_time_advanced':physical_end-physical_start,'physical_dt_per_frame':physical_deltas,
        'substeps_per_frame':substeps,'excluded_counter_ms':counter_ms,'finite':finite,'finite_channels':checked,'motion':motion,
        'stats':dict(sim.stats),'readbacks':'particle validation before/after timing; additional small substep counters between timed frames, separately reported; engine internal scalar counters remain in timed step'}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('GALAXY','IMPACT','COMPRESSIBLE'),default='GALAXY')
    parser.add_argument('--count',type=int,default=100000)
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--dt',type=float,default=None)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args(argv)
    output=args.output or ROOT/'validation'/'reports'/f'solver_{args.mode.lower()}_{args.count}.json'
    report={'schema':2,'mode':args.mode,'requested_count':args.count,'live_viewport_acceptance':False}
    sim=None
    try:
        import warp as wp
        wp.config.quiet=True
        wp.init()
        if not wp.is_cuda_available():raise RuntimeError('CUDA GPU required; refusing CPU fallback')
        from astro_collision_lab.simulation import AstroCollisionSimulation
        config,bodies=probe_scene(args.mode,args.count,args.dt)
        sim=AstroCollisionSimulation(config)
        report.update(config=dataclasses.asdict(config),bodies=[dataclasses.asdict(b) for b in bodies],
            warp_version=wp.__version__)
        # Context is created by initialize; resolve its device when sync is invoked.
        report.update(measure_simulation(sim,bodies,frames=args.frames,warmup=args.warmup,
            synchronize=lambda:wp.synchronize_device(sim.device_context.device)))
        report['device']=str(sim.device_context.device_name)
        report['success']=bool(report['finite'] and report['solver_guardrails_pass'] is True and report['active_count']==args.count and
            report['motion']['moving_particles']>0 and all(dt>0 for dt in report['physical_dt_per_frame']))
    except Exception as exc:
        import traceback
        report.update(success=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if sim is not None:
            try:sim.close()
            except Exception as exc:report.update(success=False,close_error=str(exc))
        output.parent.mkdir(parents=True,exist_ok=True)
        def json_finite(value):
            if isinstance(value,float) and not math.isfinite(value):return None
            if isinstance(value,dict):return {k:json_finite(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [json_finite(v) for v in value]
            return value
        report=json_finite(report)
        output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'report':str(output),'success':report['success'],
        'mean_ms':report.get('mean_ms'),'physical_time_advanced':report.get('physical_time_advanced')}))
    return 0 if report['success'] else 1

if __name__=='__main__':raise SystemExit(main())
