"""Panel warning follows collection membership, including Empty Galaxy roots."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = Path(__file__).resolve().parents[1] / 'extension/astro_collision_lab/ui.py'

class Layout:
    def __init__(self): self.messages = []
    def box(self): return self
    def column(self): return self
    def label(self, *, text, **kwargs): self.messages.append(text)
    def prop(self, *args, **kwargs): pass
    def operator(self, *args, **kwargs): pass


def warnings(obj, collection, linked=True):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'source_objects', '_source_in_active_collection', '_draw_body'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'_scene_collections': lambda scene: (collection,) if linked else ()}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), scope)
    layout = Layout()
    scene = NS(acl_settings=NS(source_collection=collection))
    scope['_draw_body'](layout, NS(object=obj, scene=scene))
    return 'Outside active source collection' in layout.messages


def galaxy():
    return NS(type='EMPTY', parent=None,
              acl_body=NS(is_galaxy=True, enabled=False, disk_fraction=0))


def test_galaxy_root_inside_active_collection_has_no_warning():
    root = galaxy()
    assert not warnings(root, NS(all_objects=[root]))


def test_outside_galaxy_root_still_warns():
    assert warnings(galaxy(), NS(all_objects=[]))


def test_unlinked_collection_still_warns():
    root = galaxy()
    assert warnings(root, NS(all_objects=[root]), linked=False)


def test_missing_collection_still_warns():
    assert warnings(galaxy(), None)
