from dataclasses import replace

import numpy as np
import pytest
from astro_collision_lab import galaxy_models as gm
from astro_collision_lab.simulation import BodySpec, SimulationConfig


def test_plummer_has_concentrated_profile_and_bound_distribution():
    body = BodySpec('halo', kind='HALO', particle_count=30000, radius=10., scale_length=1., galaxy_id='one')
    p,v,_m,_h = gm.sample((body,), SimulationConfig(particle_count=30000), np)[0]
    r=np.linalg.norm(p,axis=1)
    assert 1.15 < np.median(r) < 1.45
    assert np.max(r) <= 10.
    escape2=2/np.sqrt(r*r+1)
    assert np.all(np.sum(v*v,axis=1) < escape2*1.02)
    assert np.linalg.norm(np.mean(v,axis=0)) < 1e-12


def test_membership_is_explicit_not_coincident_position():
    disk=BodySpec('disk',kind='DISK',particle_count=2000,mass=.2,radius=4.,scale_length=1.,galaxy_id='a')
    halo=BodySpec('halo',kind='HALO',particle_count=2000,mass=1.,radius=10.,scale_length=1.,galaxy_id='b')
    cfg=SimulationConfig(particle_count=4000)
    a=gm.sample((disk,halo),cfg,np)[0][1]
    b=gm.sample((disk,replace(halo,position=(50.,0.,0.))),cfg,np)[0][1]
    assert np.array_equal(a,b)
    c=gm.sample((disk,replace(halo,galaxy_id='a')),cfg,np)[0][1]
    assert np.mean(np.linalg.norm(c,axis=1)) > np.mean(np.linalg.norm(a,axis=1))


def test_preview_does_not_mutate_spec_or_particle_budget():
    b=BodySpec('disk',kind='DISK',particle_count=100000,radius=4.,scale_length=1.)
    cfg=SimulationConfig(particle_count=100000)
    p=gm.preview((b,),cfg,limit=1000)
    assert len(p)==1 and p[0].shape==(1000,3)
    assert b.particle_count==100000


def test_stability_rejects_expanding_cloud_and_nonfinite():
    rng=np.random.default_rng(5); p=rng.normal(size=(1000,3)); v=rng.normal(size=p.shape)
    m=np.ones(1000)/1000
    base=gm.diagnostics(p,v,m,1.,.03)
    expanded=gm.diagnostics(p*1.5,v,m,1.,.03)
    result=gm.assess(base,[expanded],periods=5.)
    assert not result['pass'] and result['max_radius_drift']>.4
    bad=p.copy();bad[0,0]=np.nan
    assert not gm.diagnostics(bad,v,m,1.,.03)['finite']


def test_compound_distribution_is_finite_and_deterministic():
    specs=(BodySpec('disk',kind='DISK',particle_count=1500,mass=.15,radius=5.,scale_length=1.,galaxy_id='one'),
           BodySpec('bulge',kind='BULGE',particle_count=500,mass=.05,radius=5.,scale_length=.5,galaxy_id='one',rotation_fraction=0.),
           BodySpec('halo',kind='HALO',particle_count=8000,mass=.8,radius=10.,scale_length=1.,galaxy_id='one',rotation_fraction=0.))
    cfg=SimulationConfig(particle_count=10000)
    rows=gm.sample(specs,cfg,np)
    for row in rows:
        assert all(np.isfinite(x).all() for x in row)
    again=gm.sample(specs,cfg,np)
    assert all(np.array_equal(a[1],b[1]) for a,b in zip(rows,again))


def _thick_disk_drift(monkeypatch=None):
    disk=BodySpec('disk',kind='DISK',particle_count=20000,mass=.2,radius=5.,scale_length=1.,thickness=1.2,galaxy_id='g')
    halo=BodySpec('halo',kind='HALO',particle_count=2000,mass=.8,radius=10.,scale_length=1.,galaxy_id='g',rotation_fraction=0.)
    group=(disk,halo);cfg=SimulationConfig(particle_count=22000);eps=cfg.softening
    p,v=gm.sample(group,cfg,np)[0][:2]
    accel=gm._frozen_accel(group,gm._relax_field(group,1.,eps,np),1.,eps,np)
    dt=gm._group_period(group,1.,np)/200
    z0=np.sqrt(np.mean(p[:,2]**2));r0=np.median(np.hypot(p[:,0],p[:,1]));dz=dr=0.;a=accel(p)
    for k in range(1,601):
        v+=.5*dt*a;p+=dt*v;a=accel(p);v+=.5*dt*a
        if k%60==0:
            dz=max(dz,abs(np.sqrt(np.mean(p[:,2]**2))/z0-1));dr=max(dr,abs(np.median(np.hypot(p[:,0],p[:,1]))/r0-1))
    return dz,dr


def test_thick_disk_profile_stays_stationary_in_its_own_field():
    # Moment-based (Jeans + vertical DF) velocities are not a joint equilibrium of
    # a thick disk: it flares and contracts by ~15% within three half-mass periods
    # and fails the isolation thickness gate (defect 2026-11-16).
    dz,dr=_thick_disk_drift()
    assert dz < .08 and dr < .02


def test_moment_velocities_reproduce_the_thick_disk_defect(monkeypatch):
    # Guards the stationarity test above: the superseded sampler must still fail it.
    monkeypatch.setattr(gm,'relaxed_velocities',gm._disk_moment_velocities)
    dz,_dr=_thick_disk_drift()
    assert dz > .1


def test_warp_sampler_kernels_match_numpy_reference(monkeypatch):
    kernels=pytest.importorskip('astro_collision_lab._galaxy_kernels')
    if kernels.device() is None:pytest.skip('CUDA unavailable: the numpy reference path is the sampler')
    rng=np.random.default_rng(1);R=rng.uniform(0,6,500);R[:3]=0;z=rng.uniform(0,3,500)
    rr=rng.uniform(.01,5,700);rz=rng.normal(0,.5,700);rm=rng.random(700)/700
    monkeypatch.setattr(gm,'KERNEL_BACKEND','numpy');ref=gm.ring_field(R,z,rr,rz,rm,1.,.03,np)
    monkeypatch.setattr(gm,'KERNEL_BACKEND','warp');got=gm.ring_field(R,z,rr,rz,rm,1.,.03,np)
    for a,b in zip(ref,got):assert np.max(np.abs(a-b)) <= 1e-12*np.max(np.abs(a))
    disk=BodySpec('disk',kind='DISK',particle_count=1000,mass=.2,radius=5.,scale_length=1.,thickness=.8,galaxy_id='g')
    halo=BodySpec('halo',kind='HALO',particle_count=1000,mass=.8,radius=10.,scale_length=1.,galaxy_id='g')
    group=(disk,halo);field=gm._relax_field(group,1.,.03,np);accel=gm._frozen_accel(group,field,1.,.03,np)
    p=np.column_stack((rng.uniform(-7,7,(300,2)),rng.uniform(-2,2,300)));p[0]=(13.,0.,0.);v=rng.normal(0,.2,(300,3))
    q,w=kernels.evolve(p,v,20,.01,field,.2,[(.8,10.,1.)],1.,.03,kernels.device())
    a=accel(p);pp,vv=p.copy(),v.copy()
    for _ in range(20):vv+=.005*a;pp+=.01*vv;a=accel(pp);vv+=.005*a
    assert np.max(np.abs(q-pp)) < 1e-9 and np.max(np.abs(w-vv)) < 1e-9


def test_object_renames_do_not_change_particle_realization():
    b=BodySpec('a',kind='HALO',particle_count=2000,radius=10.,scale_length=1.)
    cfg=SimulationConfig(particle_count=2000)
    a=gm.sample((b,),cfg,np)[0]
    c=gm.sample((replace(b,name='renamed'),),cfg,np)[0]
    assert np.array_equal(a[0],c[0]) and np.array_equal(a[1],c[1])


def test_fast_preview_matches_full_initializer_without_solving_forces(monkeypatch):
    specs=(BodySpec('disk',kind='DISK',particle_count=1000,mass=.2,radius=5.,scale_length=1.,galaxy_id='x'),
           BodySpec('halo',kind='HALO',particle_count=4000,mass=.8,radius=10.,scale_length=1.,galaxy_id='x'))
    cfg=SimulationConfig(particle_count=5000)
    expected=gm.sample(specs,cfg,np)
    def forbidden(*args,**kwargs):raise AssertionError('force solve on preview path')
    monkeypatch.setattr(gm,'circular_curve',forbidden)
    actual=gm.preview(specs,cfg)
    assert all(np.array_equal(p,row[0]) for p,row in zip(actual,expected))


def test_gate_energy_is_exact_not_a_sampled_pair_estimate(monkeypatch):
    # Sampled-pair potentials scatter by ~0.2-0.8% between checkpoints and used most
    # of the 1% energy gate on a disk whose exact energy drifted 1e-5 (defect 2026-11-17).
    rng=np.random.default_rng(3);p=rng.normal(size=(6000,3));v=rng.normal(size=p.shape)*.3;m=np.full(6000,1/6000)
    perm=rng.permutation(6000)
    a=gm.diagnostics(p,v,m,1.,.03)['energy'];b=gm.diagnostics(p[perm],v[perm],m[perm],1.,.03)['energy']
    assert abs(a/b-1) < 1e-9
    d=np.sqrt(((p[:,None,:]-p[None,:,:])**2).sum(-1)+.03**2);np.fill_diagonal(d,np.inf)
    exact=-.5*np.sum(m[:,None]*m[None,:]/d)+.5*np.sum(m[:,None]*(v-v.mean(0))**2)
    assert abs(a/exact-1) < 1e-6
    monkeypatch.setattr(gm,'KERNEL_BACKEND','numpy')
    assert abs(gm.softened_potential(p,m,1.,.03,np)/(exact-.5*np.sum(m[:,None]*(v-v.mean(0))**2))-1) < 1e-12
