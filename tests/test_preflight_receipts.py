"""Strict preflight and content-bound receipt regressions, independent of CUDA."""
from dataclasses import asdict, replace
import json
import pytest
from astro_collision_lab import interaction as contract, preset_specs
from astro_collision_lab.simulation import BodySpec, SimulationConfig


def sources():
    return (BodySpec('Disk',kind='DISK',mass=.2,galaxy_id='g-a'),
            BodySpec('Halo',kind='HALO',mass=.8,galaxy_id='g-a'))


def identity(**changes):
    args=dict(signature=contract.physical_signature(SimulationConfig(),sources()),
              code_fingerprint='actual-hash',runtime=('Blender5.1','Warp1.15','CUDA-device0'))
    args.update(changes)
    return contract.warmup_identity(**args)


def test_strict_preflight_recovers_only_tagged_associated_components():
    assert contract.resolve_mode('IMPACT',sources(),True)=='GALAXY'
    for bodies,tag in ((sources(),False),(tuple(replace(b,galaxy_id='') for b in sources()),True),
                       ((sources()[0],BodySpec('Sphere')),True)):
        with pytest.raises(ValueError):
            contract.resolve_mode('IMPACT',bodies,tag)
    with pytest.raises(ValueError):
        contract.resolve_mode('COMPRESSIBLE',sources(),True)
    with pytest.raises(ValueError):
        contract.resolve_mode('UNKNOWN',sources(),True)


def test_valid_explicit_sphere_modes_are_preserved():
    bodies=(BodySpec('Planet'),)
    assert contract.resolve_mode('IMPACT',bodies,True)=='IMPACT'
    assert contract.resolve_mode('COMPRESSIBLE',bodies,True)=='COMPRESSIBLE'


def test_receipt_survives_json_roundtrip_and_binds_all_inputs(tmp_path):
    expected=identity();marker=contract.ready_marker(expected)
    path=tmp_path/'ready.json';path.write_text(json.dumps(marker))
    assert contract.marker_matches(json.loads(path.read_text()),expected)
    assert contract.marker_valid(path,expected)
    for changed in (identity(signature='other-state'),identity(code_fingerprint='edited-code'),
                    identity(runtime=('Blender5.2','Warp1.15','CUDA-device0'))):
        assert not contract.marker_valid(path,changed)
    for invalid in ({'schema':1,'key':'old','pass':True},dict(marker,**{'pass':1}),
                    dict(marker,**{'pass':False}),dict(marker,schema=999),
                    dict(marker,request_id='forged'),dict(marker,signature='edited-no-rehash')):
        assert not contract.marker_matches(invalid,expected)
    assert not contract.marker_valid(path,'legacy-key')


def test_receipt_rejects_invalid_request_identity():
    with pytest.raises(ValueError):
        contract.ready_marker(dict(identity(),request_id='forged'))
    for changes in ({'signature':''},{'code_fingerprint':''},{'runtime':()},
                    {'runtime':('cuda',float('nan'))}):
        with pytest.raises(ValueError):
            identity(**changes)


def test_complete_mapping_fields_participate_in_signature():
    config={'mode':'GALAXY','dt':.02};bodies=[asdict(b) for b in sources()]
    original=contract.physical_signature(config,bodies)
    assert original==contract.physical_signature(config,[dict(b,name='Renamed') for b in bodies])
    changed=[dict(bodies[0],future_physics={'new':1}),bodies[1]]
    assert original!=contract.physical_signature(config,changed)
    with pytest.raises(ValueError):
        contract.physical_signature(dict(config,dt=float('nan')),bodies)


def test_galaxy_recipe_matches_native_compound_parameters():
    scene=preset_specs.galaxy_collision()
    assert sum(b.count for b in scene.bodies)==100000
    for seed,x,y,vx in ((0,-8.,-.2,.12),(1,8.,.2,-.12)):
        group=scene.bodies[seed*2:seed*2+2]
        assert all(b.location==(x,y,0.) and b.velocity==(vx,0.,0.) for b in group)
        assert [(b.kind,b.count,b.mass,b.radius) for b in group]==[
            ('DISK',10000,.2,5.),('HALO',40000,.8,10.)]
        assert all(b.scale_length==1. and b.thickness==.4 and b.seed==seed for b in group)
        assert len({b.galaxy_id for b in group})==1 and group[0].galaxy_id
        assert [b.rotation_fraction for b in group]==[1.,0.]
    assert scene.bodies[0].galaxy_id!=scene.bodies[2].galaxy_id
