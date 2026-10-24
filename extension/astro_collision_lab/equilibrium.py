"""Owned preparation-only isolation validation; never invoked in live stepping."""
import dataclasses
import json
import math
from pathlib import Path
from .simulation import AstroCollisionSimulation
from . import galaxy_models as gm


def groups(bodies):
    result={}
    for b in bodies:result.setdefault(b.galaxy_id or b.name,[]).append(b)
    return result


def qualify(config,bodies,output):
    import numpy as np
    report={'pass':False,'groups':{},'purpose':'isolated evolution, not viewport performance','model':'plummer-eddington-axisymmetric-v2'}
    output=Path(output)
    for name,group in groups(bodies).items():
        total=sum(b.mass for b in group)
        centre=sum(np.array(b.position)*b.mass for b in group)/total
        bulk=sum(np.array(b.velocity)*b.mass for b in group)/total
        local=tuple(dataclasses.replace(b,position=tuple(np.array(b.position)-centre),velocity=tuple(np.array(b.velocity)-bulk)) for b in group)
        cfg=dataclasses.replace(config,particle_count=sum(b.particle_count for b in group))
        sim=AstroCollisionSimulation(cfg)
        entry={'pass':False,'active_count':cfg.particle_count}
        report['groups'][name]=entry
        try:
            sim.initialize(local)
            channels=sim.display_channels()
            p=channels['positions'].numpy();v=channels['velocities'].numpy()
            m=np.concatenate([np.full(b.particle_count,b.mass/b.particle_count) for b in group])
            baseline=gm.diagnostics(p,v,m,cfg.gravitational_constant,cfg.softening)
            if not baseline['finite']:raise ValueError('Nonfinite initial galaxy state')
            rhalf=baseline['radii'][1]
            period=2*math.pi*math.sqrt(rhalf**3/(cfg.gravitational_constant*total*.5))
            steps=math.ceil(5*period/cfg.dt)
            if steps>100000:raise ValueError('Isolation validation would exceed 100000 updates; increase dt or reduce model scale')
            base_components=[];start=0
            for b in group:
                sl=slice(start,start+b.particle_count);start+=b.particle_count
                base_components.append((sl,gm.diagnostics(p[sl]@np.asarray(b.rotation),v[sl]@np.asarray(b.rotation),m[sl],cfg.gravitational_constant,cfg.softening)))
            samples=[];component_samples=[[] for b in group]
            checkpoints={math.ceil(steps*f/10) for f in range(1,11)}
            for k in range(1,steps+1):
                sim.step()
                if k in checkpoints:
                    channels=sim.display_channels();p=channels['positions'].numpy();v=channels['velocities'].numpy()
                    diag=sim.runtime._arrays['diag'].numpy()
                    if int(diag[3]) or int(diag[7]):raise RuntimeError('Tree capacity/walk overflow in isolation preparation')
                    samples.append(gm.diagnostics(p,v,m,cfg.gravitational_constant,cfg.softening))
                    for b,c,(sl,base) in zip(group,component_samples,base_components):
                        c.append(gm.diagnostics(p[sl]@np.asarray(b.rotation),v[sl]@np.asarray(b.rotation),m[sl],cfg.gravitational_constant,cfg.softening))
            result=gm.assess(baseline,samples,sim.physical_time/period)
            component_results={}
            for b,(sl,base),snapshots in zip(group,base_components,component_samples):
                drift=max(float(np.max(np.abs(np.array(s['radii'])[1:]/np.array(base['radii'])[1:]-1))) for s in snapshots)
                thick=max(abs(s['thickness']/base['thickness']-1) for s in snapshots)
                ok=drift<=.05 and (b.kind!='DISK' or thick<=.10)
                component_results[b.name]={'pass':bool(ok),'radius_drift':drift,'thickness_drift':thick,'radius_factors':[s['radii'] for s in snapshots],'initial_radii':base['radii'],'thresholds':{'R50_R90_drift':.05,'disk_height_drift':.10},'thickness_factors':[s['thickness']/base['thickness'] for s in snapshots]}
            entry.update(result,components=component_results,baseline=baseline,samples=samples,steps=steps,reference_period=period)
            entry['pass']=bool(result['pass'] and all(c['pass'] for c in component_results.values()))
        except Exception as exc:
            entry['error']=f'{type(exc).__name__}: {exc}'
        finally:sim.close()
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['pass']=all(g['pass'] for g in report['groups'].values())
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['pass']:raise RuntimeError('Galaxy isolation gate failed; inspect '+str(output))
    return report
