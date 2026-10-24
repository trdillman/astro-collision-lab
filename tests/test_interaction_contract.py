"""Pure contracts for preparing the currently authored interaction revision."""

import json
from dataclasses import replace

import pytest

from astro_collision_lab import simulation
from astro_collision_lab.simulation import BodySpec, SimulationConfig


def bodies():
    return (
        BodySpec("Disk", kind="DISK", particle_count=30, mass=3.0,
                 galaxy_id="First galaxy", seed=17),
        BodySpec("Halo", kind="HALO", particle_count=10, mass=1.0,
                 galaxy_id="First galaxy", seed=17),
        BodySpec("Other halo", kind="HALO", particle_count=20, mass=2.0,
                 galaxy_id="Second galaxy", position=(10.0, 0.0, 0.0)),
    )


def config():
    return SimulationConfig(particle_count=60)


def test_physical_revision_ignores_names_and_renamed_membership_labels():
    from astro_collision_lab.interaction import physical_signature

    original = bodies()
    renamed = tuple(replace(b, name=f"Renamed {i}",
                            galaxy_id="A" if i < 2 else "B")
                    for i, b in enumerate(original))
    assert physical_signature(config(), original) == physical_signature(config(), renamed)
    payload = json.loads(physical_signature(config(), original))
    assert [b["seed"] for b in payload["bodies"]] == [17, 17, 0]
    assert all("name" not in b for b in payload["bodies"])


@pytest.mark.parametrize("change", [
    {"position": (0.01, 0.0, 0.0)}, {"velocity": (0.0, 0.01, 0.0)},
    {"spin": (0.0, 0.0, 0.01)}, {"radius": 1.1}, {"mass": 3.1},
    {"particle_count": 31}, {"seed": 18}, {"thickness": 0.2},
    {"scale_length": 0.3}, {"toomre_q": 1.6}, {"rotation_fraction": 0.5},
    {"material": "IRON"}, {"kind": "BULGE"},
    {"rotation": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))},
])
def test_physical_revision_detects_authored_changes(change):
    from astro_collision_lab.interaction import physical_signature

    original = bodies()
    changed = (replace(original[0], **change), *original[1:])
    assert physical_signature(config(), original) != physical_signature(config(), changed)


@pytest.mark.parametrize("change", [
    {"dt": 0.006}, {"gravitational_constant": 1.1}, {"softening": 0.04},
    {"theta": 0.5}, {"substeps": 2}, {"seed": 1}, {"precision": "float64"},
    {"mode": "IMPACT"}, {"specific_internal_energy": 0.1},
])
def test_physical_revision_detects_simulation_settings(change):
    from astro_collision_lab.interaction import physical_signature

    assert physical_signature(config(), bodies()) != physical_signature(replace(config(), **change), bodies())


def test_physical_revision_preserves_membership_and_source_order():
    from astro_collision_lab.interaction import physical_signature

    original = bodies()
    regrouped = (original[0], replace(original[1], galaxy_id="Second galaxy"), original[2])
    independent = tuple(replace(b, galaxy_id="") for b in original)
    signature = physical_signature(config(), original)
    assert signature != physical_signature(config(), regrouped)
    assert signature != physical_signature(config(), independent)
    assert signature != physical_signature(config(), tuple(reversed(original)))


def test_equilibrium_revision_removes_each_groups_global_position_and_drift():
    from astro_collision_lab.interaction import (
        equilibrium_signature,
        physical_signature,
    )

    original = bodies()
    moved = tuple(replace(b, position=(13.25, -6.5, 0.3) if i < 2 else (-8.0, 11.0, 0.0),
                          velocity=(0.7, 2.1, -4.5) if i < 2 else (-2.0, 0.5, 3.0))
                  for i, b in enumerate(original))
    assert equilibrium_signature(config(), original) == equilibrium_signature(config(), moved)
    assert physical_signature(config(), original) != physical_signature(config(), moved)


def test_equilibrium_revision_uses_mass_weighted_relative_coordinates():
    from astro_collision_lab.interaction import equilibrium_signature

    original = bodies()[:2]
    original = (replace(original[0], position=(1.0, 0.0, 0.0), velocity=(4.0, 0.0, 0.0)),
                replace(original[1], position=(5.0, 0.0, 0.0), velocity=(12.0, 0.0, 0.0)))
    payload = json.loads(equilibrium_signature(replace(config(), particle_count=40), original))
    assert [b["position"] for b in payload["bodies"]] == [[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    assert [b["velocity"] for b in payload["bodies"]] == [[-2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]


def test_equilibrium_revision_normalizes_translation_roundoff_without_changing_support():
    from astro_collision_lab.interaction import equilibrium_signature

    original = tuple(replace(b, position=(0.1 + i * 0.1, 0.0, 0.0),
                             velocity=(0.2 + i * 0.1, 0.0, 0.0)) for i, b in enumerate(bodies()[:2]))
    translated = tuple(replace(b, position=(b.position[0] + 3.0, 0.0, 0.0),
                               velocity=(b.velocity[0] + 2.0, 0.0, 0.0)) for b in original)
    signature = equilibrium_signature(config(), original)
    assert signature == equilibrium_signature(config(), translated)
    for change in ({"toomre_q": 1.7}, {"spin": (0.0, 0.0, 1.0)},
                   {"position": (0.15, 0.0, 0.0)}, {"velocity": (0.25, 0.0, 0.0)},
                   {"rotation": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))}):
        assert signature != equilibrium_signature(config(), (replace(original[0], **change), original[1]))


def test_equilibrium_revision_treats_unparented_bodies_as_separate_groups():
    from astro_collision_lab.interaction import equilibrium_signature

    original = tuple(replace(b, galaxy_id="") for b in bodies())
    moved = tuple(replace(b, position=(i * 12.0, -4.0, 2.0), velocity=(i * 3.0, 0.0, 0.0))
                  for i, b in enumerate(original))
    assert equilibrium_signature(config(), original) == equilibrium_signature(config(), moved)


@pytest.mark.parametrize("mode,kinds,tagged,expected", [
    ("IMPACT", ("DISK", "HALO"), True, "GALAXY"),
    ("IMPACT", ("BULGE",), True, "GALAXY"),
    ("IMPACT", ("DISK", "HALO"), False, "IMPACT"),
    ("COMPRESSIBLE", ("DISK", "HALO"), True, "COMPRESSIBLE"),
    ("IMPACT", ("SPHERE",), True, "IMPACT"),
    ("IMPACT", ("DISK", "SPHERE"), True, "IMPACT"),
    ("IMPACT", (), True, "IMPACT"),
    ("GALAXY", ("SPHERE",), False, "GALAXY"),
])
def test_resolved_mode_promotes_only_tagged_collisionless_sources(mode, kinds, tagged, expected):
    from astro_collision_lab.interaction import resolved_mode

    sources = tuple(BodySpec(str(i), kind=kind, galaxy_id="tagged") for i, kind in enumerate(kinds))
    assert resolved_mode(mode, sources, tagged) == expected


def test_resolved_mode_does_not_promote_unassociated_components():
    from astro_collision_lab.interaction import resolved_mode

    associated = BodySpec("Disk", kind="DISK", galaxy_id="assembly")
    independent = BodySpec("Halo", kind="HALO")
    assert resolved_mode("IMPACT", (associated, independent), True) == "IMPACT"
    assert resolved_mode("IMPACT", (independent,), True) == "IMPACT"


def test_fingerprint_covers_actual_owned_vendored_code_and_manifest(tmp_path):
    from astro_collision_lab.interaction import code_fingerprint

    (tmp_path / "runtime.py").write_text("value = 1\n")
    vendor = tmp_path / "_vendor"
    vendor.mkdir()
    (vendor / "solver.py").write_text("solver = 1\n")
    manifest = tmp_path / "blender_manifest.toml"
    manifest.write_text('version = "0.1"\n')
    (tmp_path / "_build.json").write_text('{"source_fingerprint":"claimed"}')
    initial = code_fingerprint(tmp_path)
    assert len(initial) == 64
    (tmp_path / "_build.json").write_text('{"source_fingerprint":"another claim"}')
    assert code_fingerprint(tmp_path) == initial
    for path in (tmp_path / "runtime.py", vendor / "solver.py", manifest):
        previous = code_fingerprint(tmp_path)
        path.write_text(path.read_text() + "\n# Changed\n")
        assert code_fingerprint(tmp_path) != previous
    previous = code_fingerprint(tmp_path)
    (vendor / "solver.py").rename(vendor / "renamed.py")
    assert code_fingerprint(tmp_path) != previous


def test_fingerprint_is_independent_of_directory_creation_order(tmp_path):
    from astro_collision_lab.interaction import code_fingerprint

    files = {"a.py": "a = 1", "nested/z.py": "z = 2", "blender_manifest.toml": 'version = "1"'}
    for directory, items in ((tmp_path / "first", files.items()),
                             (tmp_path / "second", reversed(tuple(files.items())))):
        for relative, contents in items:
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
    assert code_fingerprint(tmp_path / "first") == code_fingerprint(tmp_path / "second")


@pytest.mark.parametrize("receipt,expected", [
    ({"schema": 1, "key": "current", "pass": True}, False),
    ({"schema": 1, "key": "old", "pass": True}, False),
    ({"schema": 1, "key": "current", "pass": False}, False),
    ({"schema": 1, "key": "current", "pass": 1}, False),
    ({"schema": 1, "key": "current", "pass": "true"}, False),
    ({"schema": 0, "key": "current", "pass": True}, False),
    ({"schema": 2, "key": "current", "pass": True}, False),
    ({"schema": True, "key": "current", "pass": True}, False),
    ({"key": "current", "pass": True}, False),
    ({"pass": True}, False), ([], False), (None, False),
])
def test_marker_requires_current_successful_receipt(tmp_path, receipt, expected):
    from astro_collision_lab.interaction import marker_valid

    path = tmp_path / "ready.json"
    path.write_text(json.dumps(receipt))
    assert marker_valid(path, "current") is expected


def test_marker_missing_or_unreadable_json_is_not_ready(tmp_path):
    from astro_collision_lab.interaction import marker_valid

    path = tmp_path / "ready.json"
    assert marker_valid(path, "current") is False
    path.write_text("{partial")
    assert marker_valid(path, "current") is False
    path.write_bytes(b"\xff")
    assert marker_valid(path, "current") is False


def test_preflight_returns_specs_without_sampling_or_cuda():
    original = bodies()
    assert simulation.validate_scene(config(), original) == original


@pytest.mark.parametrize("change,match", [
    ({"position": (0.01, 0.0, 0.0)}, "centre"),
    ({"velocity": (0.01, 0.0, 0.0)}, "velocity"),
    ({"rotation": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))}, "orientation"),
])
def test_preflight_rejects_unsupported_compound_mismatch(change, match):
    original = bodies()
    modified = (original[0], replace(original[1], **change), original[2])
    with pytest.raises(ValueError, match=match):
        simulation.validate_scene(config(), modified)


def test_preflight_rejects_compound_velocity_mismatch_even_with_large_common_drift():
    original = tuple(replace(b, velocity=(1e8, 0.0, 0.0)) for b in bodies())
    modified = (original[0], replace(original[1], velocity=(1e8 + 1.0, 0.0, 0.0)), original[2])
    with pytest.raises(ValueError, match="velocity"):
        simulation.validate_scene(config(), modified)


def test_preflight_does_not_require_separate_galaxies_to_have_matching_transforms():
    original = bodies()
    modified = (*original[:2], replace(original[2], velocity=(3.0, 2.0, 1.0),
                                    rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))))
    assert simulation.validate_scene(config(), modified) == modified


def test_preflight_preserves_config_and_impact_spacing_guards():
    with pytest.raises(ValueError, match="dt"):
        simulation.validate_scene(replace(config(), dt=0.0), bodies())
    impact = SimulationConfig(mode="IMPACT", particle_count=60)
    uneven = (BodySpec("A", particle_count=30, radius=1.0), BodySpec("B", particle_count=30, radius=2.0))
    with pytest.raises(ValueError, match="common spacing"):
        simulation.validate_scene(impact, uneven)
