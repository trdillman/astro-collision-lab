from dataclasses import replace
import numpy as np
from astro_collision_lab.simulation import BodySpec, SimulationConfig
from astro_collision_lab import galaxy_models as gm


def test_grouped_sampling_preserves_rng_assignment_across_empty_bins():
    bins = np.array([5, 2, 5, 0, 2, 5, 0])
    groups = [(k, indices.tolist()) for k, indices in gm._bin_groups(bins, 9, np)]
    assert groups == [(0, [3, 6]), (2, [1, 4]), (5, [0, 2, 5])]


def test_reflected_disk_force_matches_full_azimuth_quadrature():
    disk = BodySpec('disk', kind='DISK', radius=3., scale_length=.7, thickness=.04)
    grid, curve = gm.circular_curve(disk, (disk,), 1., .012, np)
    edges = np.linspace(0, disk.radius, 161)
    ring = ((edges[1:] + edges[:-1]) * .5)[:, None]
    mass = np.diff(gm.enclosed(disk, edges, np))[:, None]
    phi = (np.arange(128) + .5) * 2 * np.pi / 128
    for index in (0, 75, 150, 225, 299):
        radius = grid[index]
        dx = radius - ring * np.cos(phi)
        d2 = dx**2 + (ring * np.sin(phi))**2 + .012**2 + (.35 * disk.thickness)**2
        expected = max(0., radius * np.sum(mass * dx / d2**1.5) / 128)
        np.testing.assert_allclose(curve[index], expected, rtol=1e-12, atol=1e-14)


def test_plummer_has_concentrated_profile_and_bound_distribution():
    body = BodySpec('halo', kind='HALO', particle_count=30000, radius=10., scale_length=1., galaxy_id='one')
    p,v,m,h = gm.sample((body,), SimulationConfig(particle_count=30000), np)[0]
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
