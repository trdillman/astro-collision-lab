"""Exercise UI-owned layout and callback contracts without a Blender process.

The fake records the layout commands the production draw code emits; Blender
RNA declarations are captured at import so callback routing is also exercised.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parents[1] / 'extension' / 'astro_collision_lab'


def load_ui():
    bpy = ModuleType('bpy')
    props = ModuleType('bpy.props')
    for name in ('BoolProperty', 'EnumProperty', 'FloatProperty', 'FloatVectorProperty',
                 'IntProperty', 'PointerProperty', 'StringProperty'):
        setattr(props, name, lambda **kwargs: kwargs)
    types = ModuleType('bpy.types')
    for name in ('Operator', 'Panel', 'PropertyGroup', 'Menu', 'AddonPreferences', 'Collection', 'Object'):
        setattr(types, name, type(name, (), {}))
    bpy.props, bpy.types = props, types
    package = ModuleType('ui_contract_package')
    package.__path__ = [str(PACKAGE)]
    presets = ModuleType('ui_contract_package.preset_specs')
    presets.galaxy_collision = presets.moon_collision = lambda: None
    modules = {'bpy': bpy, 'bpy.props': props, 'bpy.types': types,
               'ui_contract_package': package, 'ui_contract_package.preset_specs': presets}
    spec = importlib.util.spec_from_file_location('ui_contract_package.ui', PACKAGE / 'ui.py')
    ui = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(ui)
    return ui


class Layout:
    def __init__(self, events=None, parent=None):
        self.events = events if events is not None else []
        self.parent = parent
        self.enabled = True
        self.alert = False

    @property
    def active(self):
        return self.enabled and (self.parent is None or self.parent.active)

    def column(self, **kwargs):
        return Layout(self.events, self)

    row = box = column

    def prop(self, data, key, **kwargs):
        self.events.append(('prop', key, self.active, data))

    def label(self, **kwargs):
        self.events.append(('label', kwargs.get('text', ''), self.active, None))

    def operator(self, *args, **kwargs):
        return SimpleNamespace()

    def separator(self):
        pass


def body(**overrides):
    values = dict(enabled=True, is_galaxy=False, native_primitive=True,
                  is_mesh_proxy=False, kind='DISK', disk_fraction=.2)
    values.update(overrides)
    return SimpleNamespace(**values)


class UIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = load_ui()

    def draw_body(self, *, managed=False, galaxy=False):
        parent = SimpleNamespace(name='Galaxy A', acl_body=body(is_galaxy=True)) if managed else None
        obj = SimpleNamespace(type='EMPTY' if galaxy else 'MESH', acl_body=body(is_galaxy=galaxy), parent=parent)
        context = SimpleNamespace(object=obj, scene=SimpleNamespace(acl_settings=SimpleNamespace(mode='GALAXY')))
        layout = Layout()
        with patch.object(self.ui, '_source_in_active_collection', return_value=True):
            self.ui._draw_body(layout, context)
        return layout.events

    def test_managed_child_physical_controls_are_disabled_but_color_is_editable(self):
        events = self.draw_body(managed=True)
        physical = [e for e in events if e[0]=='prop' and e[1]!='color']
        self.assertTrue(physical)
        self.assertFalse(any(e[2] for e in physical), physical)
        self.assertEqual([e[2] for e in events if e[:2]==('prop', 'color')], [True])

    def test_standalone_physical_controls_remain_editable(self):
        events = self.draw_body()
        self.assertTrue(all(e[2] for e in events if e[0]=='prop'))

    def test_galaxy_root_exposes_existing_disk_rotation(self):
        events = self.draw_body(galaxy=True)
        self.assertIn(('rotation_fraction', True), [(e[1],e[2]) for e in events if e[0]=='prop'])

    def test_every_appearance_property_routes_to_redraw_without_authoring(self):
        properties = self.ui.ACLSceneSettings.__annotations__
        viewport = ModuleType('ui_contract_package.viewport')
        calls = []
        viewport.request_redraw = lambda: calls.append('redraw')
        for key in ('blend_mode', 'particle_size_px', 'physical_radius', 'color_by', 'ramp_min', 'ramp_max'):
            callback = properties[key].get('update')
            self.assertTrue(callable(callback), key)
            with patch.dict(sys.modules, {'ui_contract_package.viewport': viewport}):
                callback(None, None)
        self.assertEqual(calls, ['redraw']*6)

    def test_appearance_panel_exposes_blending(self):
        settings = SimpleNamespace(physical_radius=False, color_by='SOURCE')
        panel = self.ui.ACL_PT_appearance()
        panel.layout = Layout()
        panel.draw(SimpleNamespace(scene=SimpleNamespace(acl_settings=settings)))
        self.assertIn('blend_mode', [e[1] for e in panel.layout.events if e[0]=='prop'])

    def test_root_membership_warning_does_not_treat_empty_as_absent(self):
        root = SimpleNamespace(type='EMPTY', acl_body=body(is_galaxy=True))
        collection = SimpleNamespace(all_objects=[root])
        scene = SimpleNamespace(acl_settings=SimpleNamespace(source_collection=collection))
        with patch.object(self.ui, '_scene_collections', return_value=[collection]):
            self.assertTrue(self.ui._source_in_active_collection(scene, root))

    def test_native_numeric_limits_are_accepted_by_solver_bounds(self):
        props = self.ui.ACLSceneSettings.__annotations__
        import struct
        stored = struct.unpack('f', struct.pack('f', props['theta']['max']))[0]
        self.assertLessEqual(stored, .8)
        self.assertGreater(props['gravitational_constant']['min'], 0)

    def test_balance_sampling_can_repair_currently_invalid_spacing(self):
        runtime = ModuleType('ui_contract_package.runtime')
        package = ModuleType('ui_contract_package')
        package.__path__ = [str(PACKAGE)]
        package.runtime = runtime
        objects = tuple(SimpleNamespace(acl_body=SimpleNamespace(particle_count=50000))
                        for _ in range(2))
        bodies = tuple(SimpleNamespace(kind='SPHERE', radius=radius, particle_count=50000)
                       for radius in (1., .46))
        def extract(_scene, *, validate=True):
            if validate:
                raise ValueError('IMPACT requires common spacing across sources')
            return SimpleNamespace(mode='IMPACT'), bodies, objects
        runtime.authored_scene = extract
        operator = self.ui.ACL_OT_balance_sampling()
        operator.report = lambda *_args: None
        with patch.dict(sys.modules, {'ui_contract_package': package,
                                      'ui_contract_package.runtime': runtime}):
            result = operator.execute(SimpleNamespace(scene=object()))
        self.assertEqual(result, {'FINISHED'})
        counts = tuple(obj.acl_body.particle_count for obj in objects)
        self.assertEqual(sum(counts), 100000)
        self.assertLessEqual(self.ui._sampling_spacing_ratio((1., .46), counts), 1.01)


if __name__ == '__main__':
    unittest.main()
