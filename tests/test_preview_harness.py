"""Host checks for preview-harness payloads; native pixel gates remain unrun."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT=Path(__file__).resolve().parents[1]


def functions_from(path, names):
    """Load pure helpers without starting the script's Blender/window lifecycle."""
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[node for node in tree.body
              if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected}==set(names)
    namespace={}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**{name:namespace[name] for name in names})


@pytest.fixture
def helpers():
    return functions_from(ROOT/'validation/native_gui_v2.py',
                          ('preview_count','point_control_entries','control_points_match'))


@pytest.fixture
def entries():
    rotation=((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.))
    result={}
    for key,count,position in ((11,25000,(2.,-3.,1.)),(22,75000,(-4.,5.,-2.))):
        points=np.zeros((count,3),dtype=np.float64)
        points.setflags(write=False)
        result[key]=dict(obj=object(),key=f'original-{key}',points=points,batch=object(),
                         rotation=rotation,position=position)
    return result


def test_startup_and_pixel_harness_count_dictionary_entries_at_full_resolution(entries,helpers):
    startup=functions_from(ROOT/'tools/start-user-session.py',('preview_count',))
    for count in (startup.preview_count,helpers.preview_count):
        assert count(entries)==100000
        assert count({})==0
        assert count(None)==0


@pytest.mark.parametrize('world_point',[(1e9,1e9,1e9),(.8,.5,-10.)])
def test_controls_preserve_all_vertices_and_use_existing_object_transforms(entries,helpers,world_point):
    controls=helpers.point_control_entries(entries,world_point)
    assert helpers.preview_count(controls)==100000
    assert helpers.control_points_match(controls,world_point)
    assert not helpers.control_points_match(controls,np.asarray(world_point)+(1.,0.,0.))
    for key,entry in controls.items():
        original=entries[key]
        assert entry['obj'] is original['obj']
        assert entry['rotation'] is original['rotation']
        assert entry['position'] is original['position']
        assert entry['points'].shape==original['points'].shape
        assert entry['points'].dtype==np.dtype('float64')
        assert entry['batch'] is None
        assert original['batch'] is not None
        assert not original['points'].flags.writeable
        assert not np.shares_memory(entry['points'],original['points'])
        np.testing.assert_array_equal(original['points'],0.)
        world=entry['points'][0]@np.asarray(original['rotation']).T+original['position']
        np.testing.assert_allclose(world,world_point,rtol=0.,atol=1e-6)


def test_controls_reject_invalid_world_points(entries,helpers):
    for invalid in ((1.,2.),(1.,2.,np.nan),(1.,2.,np.inf)):
        with pytest.raises(ValueError,match='finite three-vector'):
            helpers.point_control_entries(entries,invalid)
    assert not helpers.control_points_match({},(0.,0.,0.))


@pytest.fixture
def negative():
    path=ROOT/'validation/preview_format_negative_control.py'
    spec=importlib.util.spec_from_file_location('acl_preview_negative_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def position_expression(source):
    tree=ast.parse(source)
    draw=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='draw')
    positions=[]
    for node in ast.walk(draw):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='batch_for_shader':
            for argument in node.args:
                if isinstance(argument,ast.Dict):
                    positions.extend(value for key,value in zip(argument.keys,argument.values)
                                     if isinstance(key,ast.Constant) and key.value=='pos')
    assert len(positions)==1
    return positions[0]


def test_mutation_reaches_current_actual_draw_packing_and_keeps_float64(negative):
    path=ROOT/'extension/astro_collision_lab/authoring.py'
    source=path.read_text(encoding='utf-8')
    mutated,removed,replacement=negative.mutate_preview_packing(source)
    assert mutated==source.replace(removed,replacement,1)
    compile(mutated,str(path),'exec')
    points=np.asarray([[.8,.5,-10.]],dtype=np.float64)
    original=eval(compile(ast.Expression(position_expression(source)),str(path),'eval'),
                  {'np':np,'entry':{'points':points}})
    broken=eval(compile(ast.Expression(position_expression(mutated)),str(path),'eval'),
                {'np':np,'entry':{'points':points}})
    assert original.dtype==np.dtype('float32')
    assert original.flags.c_contiguous
    assert broken is points
    assert broken.dtype==np.dtype('float64')
    with pytest.raises(RuntimeError,match='exactly once'):
        negative.mutate_preview_packing(mutated)


def test_mutation_is_scoped_to_draw_and_handles_utf8_offsets(negative):
    packing="np.ascontiguousarray(entry['points'], dtype=np.float32)"
    source=(f"outside = {packing}\n"
            "def draw():\n"
            f"    label='π'; batch_for_shader(shader, 'POINTS', {{'pos': {packing}}})\n")
    mutated,removed,replacement=negative.mutate_preview_packing(source)
    assert mutated.splitlines()[0]==source.splitlines()[0]
    assert "label='π'" in mutated
    assert removed==packing and replacement=="entry['points']"
    assert mutated.count(packing)==1
    with pytest.raises(RuntimeError,match='exactly once'):
        negative.mutate_preview_packing(source+source[source.index('def draw():'):])


def test_mutation_writes_owned_copy_and_receipt_but_refuses_general_source(negative,tmp_path):
    source=(ROOT/'extension/astro_collision_lab/authoring.py').read_bytes()
    ordinary=tmp_path/'authoring.py'
    ordinary.write_bytes(source)
    with pytest.raises(RuntimeError,match='harness-owned'):
        negative.main(ordinary)
    assert ordinary.read_bytes()==source
    owned=tmp_path/'aclw-test-only'
    owned.mkdir()
    installed=owned/'authoring.py'
    installed.write_bytes(source)
    negative.main(installed)
    receipt=json.loads((owned/'preview_format_mutation.json').read_text(encoding='utf-8'))
    assert receipt['target']=='authoring.draw / batch_for_shader / pos'
    assert receipt['before_sha256']!=receipt['after_sha256']
    assert installed.read_bytes()!=source
