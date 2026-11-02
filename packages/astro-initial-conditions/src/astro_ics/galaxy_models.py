"""Deterministic collisionless initial conditions and isolated-evolution gates.

Plummer spheres use the analytic isotropic DF when all components are homologous
spheres. Disk compounds use numerical axisymmetric circular support and spherical
Jeans support for spheroids. The latter are candidates, never certified by formula.
"""
from dataclasses import replace
import math
import zlib
_CURVE_CACHE={}


def members(spec, specs):
    # Unassociated sources are independent even when their centres coincide.
    return tuple(s for s in specs if s is spec or
                 (spec.galaxy_id and s.galaxy_id == spec.galaxy_id))


def enclosed(spec, r, np):
    a = spec.scale_length if spec.scale_length > 0 else spec.radius / (5 if spec.kind == 'DISK' else 10)
    x = np.asarray(r)/a
    if spec.kind == 'DISK':
        f = lambda t: 1-(1+t)*np.exp(-t)
        return spec.mass * np.minimum(f(x)/f(spec.radius/a), 1.)
    f = lambda t: t**3/(1+t*t)**1.5
    return spec.mass * np.minimum(f(x)/f(spec.radius/a), 1.)


def circular_curve(spec, group, G, eps, np):
    key=(spec.radius,G,eps,tuple((s.kind,s.mass,s.radius,s.scale_length,s.thickness) for s in group))
    if key in _CURVE_CACHE:return _CURVE_CACHE[key]
    grid=np.geomspace(max(spec.radius*1e-5,eps*.02),spec.radius,300)
    force=np.zeros_like(grid)
    for s in group:
        a=s.scale_length if s.scale_length>0 else s.radius/(5 if s.kind=='DISK' else 10)
        if s.kind != 'DISK':
            norm=(s.radius/a)**3/(1+(s.radius/a)**2)**1.5
            force+=G*s.mass/norm*grid/(grid*grid+a*a+eps*eps)**1.5
        else:
            # Axisymmetric annular quadrature; finite thickness softens ring force.
            edges=np.linspace(0,s.radius,161); rr=(edges[1:]+edges[:-1])*.5
            mass=np.diff(enclosed(s,edges,np)); phi=(np.arange(128)+.5)*2*np.pi/128
            x=rr[:,None]*np.cos(phi);y=rr[:,None]*np.sin(phi)
            for i,r in enumerate(grid):
                dx=r-x
                force[i]+=G*np.sum(mass[:,None]*dx/(dx*dx+y*y+eps*eps+(s.thickness*.35)**2)**1.5)/128
    curve=(grid,np.maximum(force*grid,0.))
    if len(_CURVE_CACHE)>=64:_CURVE_CACHE.pop(next(iter(_CURVE_CACHE)))
    _CURVE_CACHE[key]=curve
    return curve


def vertical_dispersion(spec,group,r,z,grid,surface,G,eps,np):
    """Vertical Jeans support of the declared sech-squared density profile.

    Includes the softened local disk sheet and the compound spheroid field;
    the unsoftened isolated-sheet dispersion omits that external confinement.
    """
    z0=spec.thickness/2
    u=np.linspace(0,12,512);nodes=np.linspace(-12,12,1024)
    density=1/np.cosh(u)**2;weights=.5/np.cosh(nodes)**2
    d=u[:,None]-nodes[None,:]
    sheet=np.trapezoid(weights[None,:]*d/np.sqrt(d*d+(eps/z0)**2),nodes,axis=1)
    height=u*z0
    force=2*np.pi*G*surface[:,None]*sheet[None,:]
    for o in group:
        if o.kind=='DISK':continue
        a=o.scale_length or o.radius/10
        norm=(o.radius/a)**3/(1+(o.radius/a)**2)**1.5
        force+=G*o.mass/norm*height[None,:]/(grid[:,None]**2+height[None,:]**2+a*a+eps*eps)**1.5
    f=density[None,:]*force
    integral=np.zeros_like(f)
    integral[:,:-1]=np.cumsum(((f[:,1:]+f[:,:-1])*.5*np.diff(height)[None,:])[:,::-1],axis=1)[:,::-1]
    sigma2=integral/density[None,:]
    bins=np.clip(np.searchsorted(grid,r),0,len(grid)-1);out=np.empty(len(r))
    for k in np.unique(bins):
        ix=np.flatnonzero(bins==k);out[ix]=np.interp(abs(z[ix])/z0,u,sigma2[k])
    return np.sqrt(np.maximum(out,0))


def eddington_speeds(spec,group,r,rng,G,eps,np):
    """Isotropic DF inversion in the spherical average of the compound field.

    Eddington inversion determines the complete velocity distribution, rather
    than assigning Gaussian velocities with only a Jeans second moment.
    """
    a=spec.scale_length or spec.radius/10
    grid=np.geomspace(a*1e-3,max(o.radius for o in group)*1000,4096)
    mass=np.zeros_like(grid)
    for o in group:
        part=enclosed(o,grid,np)
        if o.kind=='DISK':part*=grid/np.sqrt(grid*grid+(o.thickness/2)**2)
        mass+=part
    force=G*mass/(grid*grid+eps*eps)
    psi=np.zeros_like(grid);psi[-1]=G*sum(o.mass for o in group)/grid[-1]
    psi[:-1]=psi[-1]+np.cumsum(((force[1:]+force[:-1])*.5*np.diff(grid))[::-1])[::-1]
    rho=(1+(grid/a)**2)**-2.5
    xp,indices=np.unique(psi[::-1],return_index=True);yp=rho[::-1][indices]
    first=np.gradient(yp,xp);second=np.gradient(first,xp)
    energy=np.linspace(0,psi[0],2049)
    u=np.linspace(0,1,512)
    # psi=E*(1-u^2) removes the Abel-integral square-root singularity.
    d2=np.interp(energy[:,None]*(1-u*u),xp,second,left=0,right=second[-1])
    df=2*np.sqrt(energy)*np.trapezoid(d2,u,axis=1)
    # Reject a substantially negative distribution instead of disguising an
    # incompatible density/potential pair as an equilibrium candidate.
    if np.min(df)<-.01*np.max(df):raise ValueError('Density profiles do not admit a positive isotropic DF; adjust component scales/masses')
    df=np.maximum(df,0)
    rb=np.geomspace(max(min(r)*.99,a*1e-5),max(r)*1.01,256)
    pp=np.interp(rb,grid,psi);q=np.linspace(0,1,512)
    prob=q[None,:]**2*np.interp(pp[:,None]*(1-q*q),energy,df)
    cdf=np.cumsum((prob[:,1:]+prob[:,:-1])*.5*np.diff(q),axis=1)
    cdf=np.column_stack((np.zeros(len(rb)),cdf));cdf/=cdf[:,-1,None]
    bins=np.clip(np.searchsorted(rb,r)-1,0,len(rb)-1)
    out=np.empty(len(r))
    for k in np.unique(bins):
        ix=np.flatnonzero(bins==k);out[ix]=np.interp(rng.random(len(ix)),cdf[k],q)
    return out*np.sqrt(2*np.interp(r,grid,psi))


def sample(specs, config, np):
    rows=[]
    for i,s in enumerate(specs):
        rng=np.random.default_rng([config.seed,s.seed,zlib.crc32(s.kind.encode())]);n=s.particle_count
        a=s.scale_length if s.scale_length>0 else s.radius/(5 if s.kind=='DISK' else 10)
        group=members(s,specs); G=config.gravitational_constant
        if s.kind=='DISK':
            rg=np.linspace(0,s.radius,4097); cdf=enclosed(s,rg,np)/s.mass
            r=np.interp(rng.random(n),cdf,rg);phi=rng.random(n)*2*np.pi
            z0=max(s.thickness/2,1e-6)
            z=z0*np.arctanh(np.clip(rng.uniform(-1,1,n),-.999999,.999999))
            local=np.column_stack((r*np.cos(phi),r*np.sin(phi),z))
            grid,vc2=circular_curve(s,group,G,config.softening,np)
            omega2=vc2/grid**2
            kappa2=np.maximum(grid*np.gradient(omega2,grid)+4*omega2,omega2*.1)
            sigma_surface=s.mass/(2*np.pi*a*a*(1-(1+s.radius/a)*np.exp(-s.radius/a)))*np.exp(-grid/a)
            sr=np.minimum(s.toomre_q*3.36*G*sigma_surface/np.sqrt(kappa2),np.sqrt(vc2)*.4)
            # Radial Jeans asymmetric drift, using the actual radial dispersion profile.
            drift=sr*sr*(grid/a-1-grid*np.gradient(np.log(np.maximum(sr*sr,1e-20)),grid)+kappa2/(4*np.maximum(omega2,1e-20)))
            mean=np.sqrt(np.maximum(vc2-drift,0))
            vr=rng.normal(size=n)*np.interp(r,grid,sr)
            vt=s.rotation_fraction*np.interp(r,grid,mean)+rng.normal(size=n)*np.interp(r,grid,sr*np.sqrt(kappa2/(4*np.maximum(omega2,1e-20))))
            vz=rng.normal(size=n)*vertical_dispersion(s,group,r,z,grid,sigma_surface,G,config.softening,np)
            motion=np.column_stack((vr*np.cos(phi)-vt*np.sin(phi),vr*np.sin(phi)+vt*np.cos(phi),vz))
        else:
            cutoff=(s.radius/a)**3/(1+(s.radius/a)**2)**1.5
            u=np.maximum(rng.random(n)*cutoff,1e-12)
            r=a/np.sqrt(u**(-2/3)-1)
            direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
            local=direction*r[:,None]
            homologous=all(o.kind!='DISK' and math.isclose(o.scale_length or o.radius/10,a,rel_tol=1e-6) for o in group)
            if homologous:
                # f(E) proportional to (-E)^(7/2); rejection samples v/v_escape.
                q=np.empty(n);todo=np.arange(n)
                while len(todo):
                    trial=rng.random(len(todo)); accept=rng.random(len(todo))*.1 < trial**2*(1-trial**2)**3.5
                    q[todo[accept]]=trial[accept];todo=todo[~accept]
                esc=np.sqrt(2*G*sum(o.mass for o in group)/np.sqrt(r*r+a*a))
                speed=q*esc
            else:
                speed=eddington_speeds(s,group,r,rng,G,config.softening,np)
            vel_dir=rng.normal(size=(n,3));vel_dir/=np.linalg.norm(vel_dir,axis=1)[:,None]
            motion=vel_dir*speed[:,None]
            # Streaming changes only azimuthal signs, preserving kinetic support.
            if s.rotation_fraction:
                cross=local[:,0]*motion[:,1]-local[:,1]*motion[:,0]
                flip=(cross*np.sign(s.rotation_fraction)<0)&(rng.random(n)<abs(s.rotation_fraction))
                motion[flip,:2]*=-1
        matrix=np.asarray(s.rotation);offsets=local@matrix.T;motion=motion@matrix.T
        motion+=np.cross(np.asarray(s.spin),offsets)
        # Balanced finite sampling removes only bulk drift, no relaxation/damping.
        offsets-=offsets.mean(0);motion-=motion.mean(0)
        rows.append((offsets+s.position,motion+s.velocity,np.full(n,s.mass/n),
                     np.full(n,.5*(4*np.pi*s.radius**3/(3*n))**(1/3))))
    return rows


def preview(specs,config,limit=None):
    import numpy as np
    reduced=tuple(replace(s,particle_count=s.particle_count if limit is None else min(limit,s.particle_count)) for s in specs)
    cfg=replace(config,particle_count=sum(s.particle_count for s in reduced))
    if cfg.mode.upper()=='GALAXY':
        # Editing only needs positions. Never solve force curves, distribution
        # functions or velocity support on the interactive redraw path.
        rows=[]
        for s in reduced:
            rng=np.random.default_rng([cfg.seed,s.seed,zlib.crc32(s.kind.encode())]);n=s.particle_count
            a=s.scale_length if s.scale_length>0 else s.radius/(5 if s.kind=='DISK' else 10)
            if s.kind=='DISK':
                rg=np.linspace(0,s.radius,4097);cdf=enclosed(s,rg,np)/s.mass
                r=np.interp(rng.random(n),cdf,rg);phi=rng.random(n)*2*np.pi
                z0=max(s.thickness/2,1e-6)
                z=z0*np.arctanh(np.clip(rng.uniform(-1,1,n),-.999999,.999999))
                local=np.column_stack((r*np.cos(phi),r*np.sin(phi),z))
            else:
                cutoff=(s.radius/a)**3/(1+(s.radius/a)**2)**1.5
                u=np.maximum(rng.random(n)*cutoff,1e-12)
                r=a/np.sqrt(u**(-2/3)-1)
                direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
                local=direction*r[:,None]
            offsets=local@np.asarray(s.rotation).T;offsets-=offsets.mean(0)
            rows.append(offsets+s.position)
        return rows
    rows=[]
    for i,s in enumerate(reduced):
        rng=np.random.default_rng([config.seed,i]);d=rng.normal(size=(s.particle_count,3));d/=np.linalg.norm(d,axis=1)[:,None]
        rows.append((d*(rng.random(s.particle_count)**(1/3)*s.radius)[:,None])@np.asarray(s.rotation).T+s.position)
    return rows


def diagnostics(p,v,m,G,eps):
    import numpy as np
    finite=bool(np.isfinite(p).all() and np.isfinite(v).all())
    if not finite:return {'finite':False}
    centre=np.average(p,axis=0,weights=m);bulk=np.average(v,axis=0,weights=m)
    x=p-centre;w=v-bulk;r=np.linalg.norm(x,axis=1)
    order=np.argsort(r);cum=np.cumsum(m[order])/sum(m)
    radii=np.interp([.1,.5,.9],cum,r[order])
    # Fixed Monte Carlo particle pairs estimate softened potential consistently.
    rng=np.random.default_rng(12345);i=rng.integers(len(p),size=500000);j=rng.integers(len(p),size=500000)
    keep=i!=j;i=i[keep];j=j[keep]
    weights=m[i]*m[j]*len(p)**2
    U=-.5*G*np.mean(weights/np.sqrt(np.sum((p[i]-p[j])**2,axis=1)+eps*eps))
    safe_r=np.maximum(r[order],eps)
    psi=G*(np.cumsum(m[order])/safe_r+np.cumsum((m[order]/safe_r)[::-1])[::-1]-m[order]/safe_r)
    bound=float(np.sum(m[order][.5*np.sum(w[order]**2,axis=1)<psi])/sum(m))
    K=.5*np.sum(m[:,None]*w*w);L=np.sum(m[:,None]*np.cross(x,w),axis=0)
    return dict(finite=True,radii=radii.tolist(),thickness=float(np.sqrt(np.average(x[:,2]**2,weights=m))),
                energy=float(K+U),kinetic=float(K),bound_fraction=bound,velocity_rms=float(np.sqrt(2*K/sum(m))),angular_momentum=L.tolist(),centre=centre.tolist(),bulk=bulk.tolist())


def assess(base,samples,periods):
    import numpy as np
    if not samples or not base['finite'] or any(not s['finite'] for s in samples):return {'pass':False,'reason':'nonfinite or missing samples'}
    radius=max(float(np.max(np.abs(np.array(s['radii'])[1:]/np.array(base['radii'])[1:]-1))) for s in samples)
    energy=max(abs((s['energy']-base['energy'])/base['energy']) for s in samples)
    L=np.array(base['angular_momentum']);den=max(np.linalg.norm(L),math.sqrt(2*base['kinetic'])*base['radii'][1]*.05)
    angular=max(float(np.linalg.norm(np.array(s['angular_momentum'])-L)/den) for s in samples)
    return {'pass':bool(periods>=5 and radius<=.05 and energy<=.01 and angular<=.01 and min(s.get('bound_fraction',1.) for s in samples)>=.99),
            'periods':float(periods),'max_radius_drift':radius,'max_energy_drift':energy,'max_angular_drift':angular,
            'thresholds':{'periods':5,'radius':.05,'energy':.01,'angular':.01}}
