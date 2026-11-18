"""Deterministic collisionless initial conditions and isolated-evolution gates.

Plummer spheres use the analytic isotropic DF when all components are homologous
spheres. Disk compounds use numerical axisymmetric circular support and spherical
Jeans support for spheroids. The latter are candidates, never certified by formula.
"""
import math
import zlib
from dataclasses import replace

_CURVE_CACHE={}
# 'auto' runs ring-field, relaxation and gate-potential sums as CUDA Warp kernels when
# available; 'numpy' forces the reference path; 'warp' requires the kernels.
KERNEL_BACKEND='auto'


def _kernels():
    if KERNEL_BACKEND=='numpy':return None
    try:
        from . import _galaxy_kernels as kernels
        dev=kernels.device()
    except ImportError:
        if KERNEL_BACKEND=='warp':raise
        return None
    if dev is None:
        if KERNEL_BACKEND=='warp':raise RuntimeError('CUDA Warp galaxy kernels are unavailable')
        return None
    return kernels,dev


def members(spec, specs):
    # Unassociated sources are independent even when their centres coincide.
    return tuple(s for s in specs if s is spec or
                 (spec.galaxy_id and s.galaxy_id == spec.galaxy_id))


def enclosed(spec, r, np):
    a = spec.scale_length if spec.scale_length > 0 else spec.radius / (5 if spec.kind == 'DISK' else 10)
    x = np.asarray(r)/a
    if spec.kind == 'DISK':
        def f(t):
            return 1-(1+t)*np.exp(-t)
        return spec.mass * np.minimum(f(x)/f(spec.radius/a), 1.)
    def f(t):
        return t**3/(1+t*t)**1.5
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


def _ellip_ke(m, np):
    """Complete elliptic integrals K(m), E(m) (parameter m=k^2) by the AGM."""
    a=np.ones_like(m);b=np.sqrt(np.maximum(1-m,0));c2=m.copy();total=.5*c2;power=.5
    for _ in range(12):
        c=.5*(a-b);a,b=.5*(a+b),np.sqrt(a*b);power*=2;total+=power*c*c
    K=np.pi/(2*a)
    return K,K*(1-total)


def ring_field(R,z,ring_r,ring_z,ring_m,G,eps,np,chunk=1<<21):
    """Exact Plummer-softened field (F_R, F_z) of coaxial rings at (R, z).

    A softened point mass averaged over a ring gives Phi=-2Gm K(k)/(pi sqrt(D)),
    D=(R+r')^2+(z-z')^2+eps^2, k^2=4Rr'/D; forces are its analytic derivatives.
    """
    R=np.asarray(R,float).ravel();z=np.asarray(z,float).ravel()
    fast=_kernels()
    if fast:return fast[0].ring_field(R,z,ring_r,ring_z,ring_m,G,eps,fast[1])
    FR=np.zeros(len(R));FZ=np.zeros(len(R))
    step=max(1,chunk//max(1,len(ring_r)))
    for lo in range(0,len(R),step):
        r=R[lo:lo+step,None];h=z[lo:lo+step,None]-ring_z[None,:]
        base=h*h+eps*eps;D=(r+ring_r[None,:])**2+base;near=(r-ring_r[None,:])**2+base
        K,E=_ellip_ke(np.minimum(4*r*ring_r[None,:]/D,1-1e-15),np)
        c=2*G*ring_m[None,:]/np.pi/np.sqrt(D)
        FZ[lo:lo+step]=-np.sum(c*h*E/near,axis=1)
        safe=np.maximum(r,1e-300)
        dphi=-c*(E*D/near*(1/(2*safe)-(r+ring_r[None,:])/D)-K/(2*safe))
        FR[lo:lo+step]=np.where(r[:,0]>0,-np.sum(dphi,axis=1),0.)
    return FR,FZ


def disk_field(spec,group,G,eps,np):
    """Axisymmetric field of the compound galaxy on a cylindrical (R, z>=0) grid.

    Every disk is its exponential x sech^2 density (rings in R and z); every
    spheroid is the truncated-Plummer field used by the circular curve.
    """
    key=('field',spec.radius,spec.thickness,G,eps,tuple((s.kind,s.mass,s.radius,s.scale_length,s.thickness) for s in group))
    if key in _CURVE_CACHE:return _CURVE_CACHE[key]
    z0=max(spec.thickness/2,1e-6);rmax=max(s.radius for s in group if s.kind=='DISK')
    Rg=np.concatenate(([0.],np.geomspace(rmax*1e-3,rmax*1.02,95)))
    zg=np.concatenate(([0.],np.geomspace(z0*1e-3,z0*7.5,63)))
    RR,ZZ=np.meshgrid(Rg,zg,indexing='ij')
    FR=np.zeros(RR.size);FZ=np.zeros(RR.size)
    for s in group:
        a=s.scale_length if s.scale_length>0 else s.radius/(5 if s.kind=='DISK' else 10)
        if s.kind=='DISK':
            edges=np.linspace(0,s.radius,161);rr=(edges[1:]+edges[:-1])*.5;mass=np.diff(enclosed(s,edges,np))
            u=np.linspace(-7.5,7.5,81);uc=(u[1:]+u[:-1])*.5;w=np.diff(np.tanh(u));w/=w.sum()
            sz=max(s.thickness/2,1e-6)
            ring_r=np.repeat(rr,len(uc));ring_z=np.tile(uc*sz,len(rr));ring_m=np.repeat(mass,len(uc))*np.tile(w,len(rr))
            fr,fz=ring_field(RR.ravel(),ZZ.ravel(),ring_r,ring_z,ring_m,G,eps,np)
            FR+=fr;FZ+=fz
        else:
            norm=(s.radius/a)**3/(1+(s.radius/a)**2)**1.5
            d3=(RR.ravel()**2+ZZ.ravel()**2+a*a+eps*eps)**1.5
            FR-=G*s.mass/norm*RR.ravel()/d3;FZ-=G*s.mass/norm*ZZ.ravel()/d3
    field=(Rg,zg,FR.reshape(RR.shape),FZ.reshape(RR.shape))
    if len(_CURVE_CACHE)>=64:_CURVE_CACHE.pop(next(iter(_CURVE_CACHE)))
    _CURVE_CACHE[key]=field
    return field


def _bilinear(xg,yg,table,x,y,np):
    i=np.clip(np.searchsorted(xg,x)-1,0,len(xg)-2);j=np.clip(np.searchsorted(yg,y)-1,0,len(yg)-2)
    tx=np.clip((x-xg[i])/(xg[i+1]-xg[i]),0,1);ty=np.clip((y-yg[j])/(yg[j+1]-yg[j]),0,1)
    return (table[i,j]*(1-tx)*(1-ty)+table[i+1,j]*tx*(1-ty)+table[i,j+1]*(1-tx)*ty+table[i+1,j+1]*tx*ty)


def disk_moments(spec,group,G,eps,np):
    """Height-dependent Jeans moments of a disk in the full compound field.

    sigma_z^2 integrates the vertical Jeans equation of the sech^2 tracer in the
    actual field. sigma_R is the Toomre profile, raised to sigma_z where the
    disk is vertically hotter (the f(E, Lz) limit, whose meridional dispersion
    is isotropic), so thick disks do not trade vertical for radial energy.
    The mean rotation then solves the radial Jeans equation at every height
    instead of reusing the midplane circular speed.
    Returns grids and tables (sigma_R, sigma_phi, sigma_z, mean v_phi).
    """
    a=spec.scale_length if spec.scale_length>0 else spec.radius/5
    z0=max(spec.thickness/2,1e-6)
    Rg,zg,FR,FZ=disk_field(spec,group,G,eps,np)
    grid,vc2=circular_curve(spec,group,G,eps,np)
    omega2=vc2/grid**2
    kappa2=np.maximum(grid*np.gradient(omega2,grid)+4*omega2,omega2*.1)
    surface=spec.mass/(2*np.pi*a*a*(1-(1+spec.radius/a)*np.exp(-spec.radius/a)))*np.exp(-grid/a)
    sq=np.minimum(spec.toomre_q*3.36*G*surface/np.sqrt(kappa2),np.sqrt(vc2)*.4)
    rho=1/np.cosh(zg/z0)**2;f=rho[None,:]*np.maximum(-FZ,0)
    integral=np.zeros_like(f)
    integral[:,:-1]=np.cumsum(((f[:,1:]+f[:,:-1])*.5*np.diff(zg)[None,:])[:,::-1],axis=1)[:,::-1]
    sz2=integral/rho[None,:]
    sr2=np.maximum(np.interp(Rg,grid,sq*sq)[:,None],sz2)
    ratio=np.interp(Rg,grid,kappa2/(4*np.maximum(omega2,1e-20)))[:,None]
    vphi2=Rg[:,None]*np.maximum(-FR,0)-sr2*(Rg[:,None]/a-1)+Rg[:,None]*np.gradient(sr2,Rg,axis=0)
    sp2=sr2*ratio
    mean=np.sqrt(np.maximum(vphi2-sp2,0))
    return Rg,zg,np.sqrt(sr2),np.sqrt(sp2),np.sqrt(np.maximum(sz2,0)),mean


def vertical_df(spec,group,G,eps,np):
    """Vertical distribution function f(E_z) of the sech^2 profile at every grid radius.

    Gaussian vertical velocities with the Jeans dispersion have tails that the
    compound field cannot hold near the centre (the spheroid's vertical well
    flattens), so a few particles climb far above the declared profile. Abel
    inversion of rho(psi) in the actual vertical potential gives the unique
    f(E_z) whose density is exactly the declared (truncated) sech^2 profile.
    f is piecewise constant in E_z between grid energies, so v_z is uniform in
    every energy shell and sampling is exact.
    Returns (Rg, zg, psi, f) with psi[i, k] = Phi(R_i, z_k) - Phi(R_i, 0).
    """
    key=('vdf',spec.radius,spec.thickness,G,eps,tuple((s.kind,s.mass,s.radius,s.scale_length,s.thickness) for s in group))
    if key in _CURVE_CACHE:return _CURVE_CACHE[key]
    z0=max(spec.thickness/2,1e-6)
    Rg,zg,_FR,FZ=disk_field(spec,group,G,eps,np)
    force=np.maximum(-FZ,0)
    psi=np.zeros_like(force)
    psi[:,1:]=np.cumsum((force[:,1:]+force[:,:-1])*.5*np.diff(zg)[None,:],axis=1)
    # A strictly increasing potential keeps every energy shell non-empty.
    psi=np.maximum.accumulate(psi+1e-12*np.arange(len(zg))[None,:]*max(float(psi[:,-1].max()),1e-30),axis=1)
    rho=1/np.cosh(zg/z0)**2;rho=rho-rho[-1]
    K=len(zg);f=np.zeros((len(Rg),K-1))
    for j in range(K-2,-1,-1):
        def width(k,j=j):return np.sqrt(np.maximum(psi[:,k+1]-psi[:,j],0))-np.sqrt(np.maximum(psi[:,k]-psi[:,j],0))
        known=sum(f[:,k]*width(k) for k in range(j+1,K-1)) if j<K-2 else 0.
        f[:,j]=(rho[j]-known)/width(j)
    if np.min(f)<-1e-3*np.max(f):raise ValueError('Disk thickness does not admit a positive vertical distribution in this galaxy')
    out=(Rg,zg,psi,np.maximum(f,0))
    if len(_CURVE_CACHE)>=64:_CURVE_CACHE.pop(next(iter(_CURVE_CACHE)))
    _CURVE_CACHE[key]=out
    return out


def vertical_speeds(spec,group,r,az,rng,G,eps,np):
    """Draw v_z from f(E_z) at each particle's (R, |z|); radius rows are chosen stochastically."""
    Rg,zg,psi,f=vertical_df(spec,group,G,eps,np)
    i=np.clip(np.searchsorted(Rg,r)-1,0,len(Rg)-2)
    t=np.clip((r-Rg[i])/(Rg[i+1]-Rg[i]),0,1)
    row=i+(rng.random(len(r))<t)
    out=np.zeros(len(r));u=rng.random(len(r));v=rng.random(len(r));sign=np.where(rng.random(len(r))<.5,-1.,1.)
    for k in np.unique(row):
        ix=np.flatnonzero(row==k)
        p=np.interp(az[ix],zg,psi[k])
        lo=np.sqrt(np.maximum(psi[k][None,:-1]-p[:,None],0));hi=np.sqrt(np.maximum(psi[k][None,1:]-p[:,None],0))
        w=f[k][None,:]*(hi-lo);c=np.cumsum(w,axis=1);total=c[:,-1]
        shell=np.minimum((c<(u[ix]*total)[:,None]).sum(1),len(zg)-2)
        s=lo[np.arange(len(ix)),shell]+v[ix]*(hi[np.arange(len(ix)),shell]-lo[np.arange(len(ix)),shell])
        out[ix]=np.where(total>0,np.sqrt(2)*s,0.)
    return out*sign


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


def _disk_positions(s,n,rng,np):
    rg=np.linspace(0,s.radius,4097); cdf=enclosed(s,rg,np)/s.mass
    r=np.interp(rng.random(n),cdf,rg);phi=rng.random(n)*2*np.pi
    z0=max(s.thickness/2,1e-6)
    z=z0*np.arctanh(np.clip(rng.uniform(-1,1,n),-.999999,.999999))
    return r,phi,z


def _disk_moment_velocities(s,group,r,z,rng,G,eps,np):
    """Cylindrical (v_R, v_phi, v_z) from Jeans moments and the vertical DF."""
    Rg,zg,sr,sp,_unused,mean=disk_moments(s,group,G,eps,np)
    az=np.abs(z);n=len(r)
    vr=rng.normal(size=n)*_bilinear(Rg,zg,sr,r,az,np)
    vt=s.rotation_fraction*_bilinear(Rg,zg,mean,r,az,np)+rng.normal(size=n)*_bilinear(Rg,zg,sp,r,az,np)
    vz=vertical_speeds(s,group,r,az,rng,G,eps,np)
    return np.column_stack((vr,vt,vz))


def _relax_field(group,G,eps,np):
    """Frozen axisymmetric field of every disk in the group on an extended (R, |z|) grid."""
    disks=tuple(s for s in group if s.kind=='DISK')
    key=('relaxfield',G,eps,tuple((s.kind,s.mass,s.radius,s.scale_length,s.thickness) for s in group))
    if key in _CURVE_CACHE:return _CURVE_CACHE[key]
    extent=1.2*max(s.radius for s in group);zmax=max(extent,8*max(max(s.thickness/2,1e-6) for s in disks))
    small=min(max(s.thickness/2,1e-6) for s in disks)
    Rg=np.concatenate(([0.],np.geomspace(extent*4e-4,extent,79)))
    zg=np.concatenate(([0.],np.geomspace(small*1e-3,zmax,47)))
    RR,ZZ=np.meshgrid(Rg,zg,indexing='ij');FR=np.zeros(RR.size);FZ=np.zeros(RR.size)
    for s in disks:
        edges=np.linspace(0,s.radius,81);rr=(edges[1:]+edges[:-1])*.5;mass=np.diff(enclosed(s,edges,np))
        u=np.linspace(-7.5,7.5,41);uc=(u[1:]+u[:-1])*.5;w=np.diff(np.tanh(u));w/=w.sum()
        sz=max(s.thickness/2,1e-6)
        fr,fz=ring_field(RR.ravel(),ZZ.ravel(),np.repeat(rr,len(uc)),np.tile(uc*sz,len(rr)),
                         np.repeat(mass,len(uc))*np.tile(w,len(rr)),G,eps,np)
        FR+=fr;FZ+=fz
    out=(Rg,zg,FR.reshape(RR.shape),FZ.reshape(RR.shape))
    if len(_CURVE_CACHE)>=64:_CURVE_CACHE.pop(next(iter(_CURVE_CACHE)))
    _CURVE_CACHE[key]=out
    return out


def _frozen_accel(group,field,G,eps,np):
    Rg,zg,FRd,FZd=field
    disk_mass=sum(s.mass for s in group if s.kind=='DISK')
    spheroids=[(s.mass,s.radius,s.scale_length if s.scale_length>0 else s.radius/10) for s in group if s.kind!='DISK']
    def accel(p):
        R=np.hypot(p[:,0],p[:,1]);az=np.abs(p[:,2])
        i=np.clip(np.searchsorted(Rg,R)-1,0,len(Rg)-2);j=np.clip(np.searchsorted(zg,az)-1,0,len(zg)-2)
        tx=np.clip((R-Rg[i])/(Rg[i+1]-Rg[i]),0,1);ty=np.clip((az-zg[j])/(zg[j+1]-zg[j]),0,1)
        w00=(1-tx)*(1-ty);w10=tx*(1-ty);w01=(1-tx)*ty;w11=tx*ty
        fr=FRd[i,j]*w00+FRd[i+1,j]*w10+FRd[i,j+1]*w01+FRd[i+1,j+1]*w11
        fz=FZd[i,j]*w00+FZd[i+1,j]*w10+FZd[i,j+1]*w01+FZd[i+1,j+1]*w11
        r2=R*R+az*az;out=(R>Rg[-1])|(az>zg[-1])
        if out.any():
            d3=(r2[out]+eps*eps)**1.5;fr[out]=-G*disk_mass*R[out]/d3;fz[out]=-G*disk_mass*az[out]/d3
        r=np.sqrt(r2)
        for mass,radius,a in spheroids:
            # Truncated Plummer: enclosed mass stops growing at the authored radius.
            norm=(radius/a)**3/(1+(radius/a)**2)**1.5
            menc=np.where(r<radius,mass/norm*r**3/(r2+a*a)**1.5,mass)
            g=G*menc/np.maximum(r2+eps*eps,1e-30)/np.maximum(r,1e-12)
            fr-=g*R;fz-=g*az
        inv=np.where(R>0,1/np.maximum(R,1e-12),0.)
        return np.column_stack((fr*p[:,0]*inv,fr*p[:,1]*inv,fz*np.sign(p[:,2])))
    return accel


RELAX_TRACERS=16000
RELAX_ITERATIONS=8
RELAX_STEPS=400
RELAX_WINDOW=.6
RELAX_CELLS=(32,16)


def _group_period(group,G,np):
    # Half-mass orbital period of the group (spherical enclosed-mass estimate).
    grid=np.geomspace(1e-4,max(s.radius for s in group),2048)
    mass=sum(enclosed(s,grid,np) for s in group);total=sum(s.mass for s in group)
    rhalf=float(np.interp(.5*total,mass,grid))
    return 2*np.pi*math.sqrt(rhalf**3/(G*total*.5))


def relaxed_disk_library(spec,group,G,eps,np):
    """Stationary velocity library of a disk by iterative constrained evolution.

    Moment-based velocities (Jeans + vertical DF) are not a joint equilibrium of
    a thick disk in the compound field: radial and vertical motions couple, the
    outer disk flares and the inner disk contracts over a few orbits. Following
    Rodionov, Athanassoula & Sotnikova (2009), tracers at fixed positions drawn
    from the authored density are evolved in the frozen field and their
    velocities are replaced by those of evolved tracers that arrive in the same
    (R, |z|) cell. Positions never change, so the authored density is kept; the
    velocity field converges to one that leaves that density stationary.
    Returns (R edges, |z| edges, cell id per tracer, cylindrical velocities).
    """
    key=('relaxed',G,eps,spec.mass,spec.radius,spec.scale_length,spec.thickness,spec.toomre_q,spec.rotation_fraction,
         tuple((s.kind,s.mass,s.radius,s.scale_length,s.thickness) for s in group))
    if key in _CURVE_CACHE:return _CURVE_CACHE[key]
    rng=np.random.default_rng([2009,zlib.crc32(b'relax')])
    r,phi,z=_disk_positions(spec,RELAX_TRACERS,rng,np)
    w=_disk_moment_velocities(spec,group,r,z,rng,G,eps,np)
    p0=np.column_stack((r*np.cos(phi),r*np.sin(phi),z))
    field=_relax_field(group,G,eps,np);accel=_frozen_accel(group,field,G,eps,np);fast=_kernels()
    disk_mass=sum(s.mass for s in group if s.kind=='DISK')
    spheroids=[(s.mass,s.radius,s.scale_length if s.scale_length>0 else s.radius/10) for s in group if s.kind!='DISK']
    az=np.abs(z)
    redges=np.quantile(r,np.linspace(0,1,RELAX_CELLS[0]+1));redges[0]=0;redges[-1]=np.inf
    zedges=np.quantile(az,np.linspace(0,1,RELAX_CELLS[1]+1));zedges[0]=0;zedges[-1]=np.inf
    def cells(R,Z):
        i=np.clip(np.searchsorted(redges,R)-1,0,len(redges)-2);j=np.clip(np.searchsorted(zedges,Z)-1,0,len(zedges)-2)
        return i*(len(zedges)-1)+j
    target=cells(r,az);c,s_=np.cos(phi),np.sin(phi)
    dt=RELAX_WINDOW*_group_period(group,G,np)/RELAX_STEPS
    for _ in range(RELAX_ITERATIONS):
        v=np.column_stack((w[:,0]*c-w[:,1]*s_,w[:,0]*s_+w[:,1]*c,w[:,2]*np.sign(z)))
        if fast:
            p,v=fast[0].evolve(p0,v,RELAX_STEPS,dt,field,disk_mass,spheroids,G,eps,fast[1])
        else:
            p=p0.copy();a=accel(p)
            for _unused in range(RELAX_STEPS):
                v+=.5*dt*a;p+=dt*v;a=accel(p);v+=.5*dt*a
        R=np.maximum(np.hypot(p[:,0],p[:,1]),1e-12);cr,sr=p[:,0]/R,p[:,1]/R
        moved=np.column_stack((v[:,0]*cr+v[:,1]*sr,-v[:,0]*sr+v[:,1]*cr,v[:,2]*np.sign(p[:,2])))
        arrived=cells(R,np.abs(p[:,2]));order=np.argsort(arrived,kind='stable');sc=arrived[order]
        lo=np.searchsorted(sc,target,'left');hi=np.searchsorted(sc,target,'right')
        keep=hi>lo;pick=lo+(rng.random(len(target))*np.maximum(hi-lo,1)).astype(int)
        w[keep]=moved[order[pick[keep]]]
    out=(redges,zedges,target,w)
    if len(_CURVE_CACHE)>=64:_CURVE_CACHE.pop(next(iter(_CURVE_CACHE)))
    _CURVE_CACHE[key]=out
    return out


def relaxed_velocities(spec,group,r,z,rng,G,eps,np):
    """Cylindrical velocities drawn from relaxed tracers in each particle's (R, |z|) cell."""
    redges,zedges,cell,w=relaxed_disk_library(spec,group,G,eps,np)
    i=np.clip(np.searchsorted(redges,r)-1,0,len(redges)-2);j=np.clip(np.searchsorted(zedges,np.abs(z))-1,0,len(zedges)-2)
    want=i*(len(zedges)-1)+j
    order=np.argsort(cell,kind='stable');sc=cell[order]
    lo=np.searchsorted(sc,want,'left');hi=np.searchsorted(sc,want,'right')
    found=hi>lo
    pick=order[np.minimum(lo+(rng.random(len(r))*(hi-lo)).astype(int),len(order)-1)]
    out=w[pick].copy()
    if not found.all():
        out[~found]=_disk_moment_velocities(spec,group,r[~found],z[~found],rng,G,eps,np)
    out[:,2]*=np.where(z<0,-1.,1.)
    return out


def sample(specs, config, np):
    rows=[]
    for _i,s in enumerate(specs):
        rng=np.random.default_rng([config.seed,s.seed,zlib.crc32(s.kind.encode())]);n=s.particle_count
        a=s.scale_length if s.scale_length>0 else s.radius/(5 if s.kind=='DISK' else 10)
        group=members(s,specs); G=config.gravitational_constant
        if s.kind=='DISK':
            r,phi,z=_disk_positions(s,n,rng,np)
            local=np.column_stack((r*np.cos(phi),r*np.sin(phi),z))
            w=relaxed_velocities(s,group,r,z,rng,G,config.softening,np)
            vr,vt,vz=w[:,0],w[:,1],w[:,2]
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


def softened_potential(p,m,G,eps,np):
    """Exact Plummer-softened pair potential energy.

    The gate's energy drift is measured on this exact sum. A sampled-pair
    estimate re-evaluated on moved particles has ~0.2-0.8% scatter, most of
    the 1% gate, so it reported spurious drift (D-016).
    """
    p=np.asarray(p,float);m=np.asarray(m,float);fast=_kernels() if len(p)>4096 else None
    if fast:return fast[0].softened_potential(p,m,G,eps,fast[1])
    U=0.;step=max(1,(1<<22)//max(len(p),1))
    for lo in range(0,len(p),step):
        d=np.sqrt(((p[lo:lo+step,None,:]-p[None,:,:])**2).sum(-1)+eps*eps)
        U-=.5*G*float(np.sum(m[lo:lo+step,None]*m[None,:]/d))
    return U+.5*G*float(np.sum(m*m))/eps


def diagnostics(p,v,m,G,eps):
    import numpy as np
    finite=bool(np.isfinite(p).all() and np.isfinite(v).all())
    if not finite:return {'finite':False}
    centre=np.average(p,axis=0,weights=m);bulk=np.average(v,axis=0,weights=m)
    x=p-centre;w=v-bulk;r=np.linalg.norm(x,axis=1)
    order=np.argsort(r);cum=np.cumsum(m[order])/sum(m)
    radii=np.interp([.1,.5,.9],cum,r[order])
    U=softened_potential(p,m,G,eps,np)
    safe_r=np.maximum(r[order],eps)
    psi=G*(np.cumsum(m[order])/safe_r+np.cumsum((m[order]/safe_r)[::-1])[::-1]-m[order]/safe_r)
    bound=float(np.sum(m[order][.5*np.sum(w[order]**2,axis=1)<psi])/sum(m))
    K=.5*np.sum(m[:,None]*w*w);L=np.sum(m[:,None]*np.cross(x,w),axis=0)
    return {'finite': True,'radii': radii.tolist(),'thickness': float(np.sqrt(np.average(x[:,2]**2,weights=m))),
                'energy': float(K+U),'kinetic': float(K),'bound_fraction': bound,'velocity_rms': float(np.sqrt(2*K/sum(m))),'angular_momentum': L.tolist(),'centre': centre.tolist(),'bulk': bulk.tolist()}


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
