"""Compound ownership regressions using Blender data substitutes, without GPU claims."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / 'extension' / 'astro_collision_lab'


def settings(**values):
    defaults = {'is_galaxy': False, 'enabled': False, 'native_primitive': False,
                'galaxy_type': 'DISK', 'kind': 'SPHERE', 'mass': 1., 'particle_count': 50000,
                'radius': 5., 'scale_length': 1., 'thickness': .4, 'disk_fraction': .2,
                'bulge_fraction': 0., 'toomre_q': 1.5, 'rotation_fraction': 1., 'seed': 0,
                'velocity': (0., 0., 0.), 'spin': (0., 0., 0.), 'material': 'ROCK'}
    defaults.update(values)
    return SimpleNamespace(**defaults)


class Mesh:
    users = 1

    def clear_geometry(self):
        pass

    def from_pydata(self, vertices, edges, faces):
        self.vertices = vertices

    def update(self):
        pass


class Object(dict):
    def __init__(self, name, data=None):
        super().__init__()
        self.name = name
        self.data = data
        self.type = 'MESH' if data is not None else 'EMPTY'
        self.acl_body = settings()
        self.children = []
        self.users_collection = []
        self._parent = None
        self.color = (.2, .4, .8, 1.)

    def as_pointer(self):
        return id(self)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent is not None:
            self._parent.children.remove(self)
        self._parent = parent
        if parent is not None:
            parent.children.append(self)


class Collection:
    def __init__(self):
        self.all_objects = []
        self.objects = SimpleNamespace(link=self.link)

    def link(self, obj):
        if not any(item is obj for item in self.all_objects):
            self.all_objects.append(obj)
            obj.users_collection.append(self)

    def unlink(self, obj):
        self.all_objects = [item for item in self.all_objects if item is not obj]
        obj.users_collection.remove(self)


@pytest.fixture
def authoring():
    fake = ModuleType('bpy')
    fake.data = SimpleNamespace(
        meshes=SimpleNamespace(new=lambda name: Mesh(), remove=lambda mesh: None),
        objects=SimpleNamespace(new=Object),
    )

    def remove(obj, do_unlink=False):
        obj.parent = None
        for collection in tuple(obj.users_collection):
            collection.unlink(obj)

    fake.data.objects.remove = remove
    package = ModuleType('compound_test_package')
    package.__path__ = [str(PACKAGE)]
    ui = ModuleType('compound_test_package.ui')
    ui._scene_collections = lambda scene: scene.collections
    spec = importlib.util.spec_from_file_location('compound_test_package.authoring', PACKAGE / 'authoring.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'bpy': fake, 'compound_test_package': package,
                                 'compound_test_package.ui': ui}):
        spec.loader.exec_module(module)
        yield module


def galaxy_fixture(authoring):
    group = Collection()
    root = Object('Galaxy')
    root.acl_body = settings(is_galaxy=True)
    group.link(root)
    scene = SimpleNamespace(acl_settings=SimpleNamespace(source_collection=group),
                            collections=[group], view_layers=[])
    authoring.synchronize(scene)
    return scene, group, root


def test_disabled_managed_child_is_restored_with_exact_root_budget(authoring):
    scene, _group, root = galaxy_fixture(authoring)
    child = root.children[0]
    child.acl_body.enabled = False
    authoring.synchronize(scene)
    assert child.acl_body.enabled
    assert sum(obj.acl_body.particle_count for obj in root.children if obj.acl_body.enabled) == 50000


def test_managed_physical_override_is_restored_without_overwriting_color(authoring):
    scene, _group, root = galaxy_fixture(authoring)
    child = root.children[0]
    child.acl_body.mass = 99.
    child.acl_body.particle_count = 42
    child.acl_body.velocity = (3., 4., 5.)
    child.color = (.9, .2, .1, .7)
    authoring.synchronize(scene)
    assert child.acl_body.mass == .2
    assert child.acl_body.particle_count == 10000
    assert child.acl_body.velocity == (0., 0., 0.)
    assert child.color == (.9, .2, .1, .7)


def test_duplicate_component_tag_rejects_before_mutating_existing_children(authoring):
    scene, group, root = galaxy_fixture(authoring)
    duplicate = Object('Duplicated Disk', Mesh())
    duplicate['acl_component'] = 'DISK'
    duplicate.acl_body = settings(enabled=True, native_primitive=True, kind='DISK')
    duplicate.parent = root
    group.link(duplicate)
    prior = [(id(obj), obj.acl_body.mass, obj.acl_body.particle_count) for obj in root.children]
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        authoring.synchronize(scene)
    assert [(id(obj), obj.acl_body.mass, obj.acl_body.particle_count) for obj in root.children] == prior


def test_deleted_managed_component_is_recreated_with_full_budget(authoring):
    scene, _group, root = galaxy_fixture(authoring)
    old_child = root.children[0]
    authoring.bpy.data.objects.remove(old_child, do_unlink=True)
    authoring.synchronize(scene)
    assert sorted(obj['acl_component'] for obj in root.children) == ['DISK', 'HALO']
    assert sum(obj.acl_body.particle_count for obj in root.children) == 50000
    assert all(obj is not old_child for obj in root.children)


def test_invalid_galaxy_outside_active_collection_is_not_synchronized(authoring):
    scene, _group, root = galaxy_fixture(authoring)
    inactive = Collection()
    other = Object('Inactive invalid galaxy')
    other.acl_body = settings(is_galaxy=True, disk_fraction=.9, bulge_fraction=.9)
    other['acl_galaxy_type'] = 'DISK'
    inactive.link(other)
    scene.collections.append(inactive)
    authoring.synchronize(scene)
    assert len(root.children) == 2
    assert not other.children
    assert other.acl_body.disk_fraction == .9


@pytest.mark.parametrize('parent_type', ['none', 'ordinary'])
def test_orphan_tagged_component_cannot_become_silent_standalone_source(authoring, parent_type):
    scene, _group, root = galaxy_fixture(authoring)
    child = root.children[0]
    child.parent = None if parent_type == 'none' else Object('Ordinary parent')
    with pytest.raises(ValueError, match='[Pp]arent|[Oo]rphan|Galaxy'):
        authoring.synchronize(scene)


def test_active_child_requires_its_galaxy_root_in_the_same_source_group(authoring):
    scene, group, root = galaxy_fixture(authoring)
    group.unlink(root)
    with pytest.raises(ValueError, match='[Cc]ollection|[Gg]roup'):
        authoring.synchronize(scene)


def test_active_root_rejects_child_linked_only_outside_active_group(authoring):
    scene, group, root = galaxy_fixture(authoring)
    child = root.children[0]
    group.unlink(child)
    other = Collection()
    other.link(child)
    scene.collections.append(other)
    with pytest.raises(ValueError, match='[Cc]ollection|[Gg]roup'):
        authoring.synchronize(scene)


def test_untagged_enabled_child_cannot_inflate_compound_budget(authoring):
    scene, group, root = galaxy_fixture(authoring)
    extra = Object('Extra disk', Mesh())
    extra.acl_body = settings(enabled=True, kind='DISK')
    extra.parent = root
    group.link(extra)
    with pytest.raises(ValueError, match='[Cc]omponent|[Gg]alaxy'):
        authoring.synchronize(scene)
