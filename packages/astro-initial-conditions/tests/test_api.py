import math

import numpy as np
import pytest

import astro_ics as ics


def test_plummer_sphere_is_deterministic_and_centred():
    sphere = ics.plummer_sphere(particle_count=4000, mass=2.0, radius=3.0, position=(1.0, 2.0, 3.0),
                                velocity=(0.5, 0.0, 0.0), seed=7)
    a, b = ics.sample([sphere]), ics.sample([sphere])
    assert len(a) == 4000
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.velocities, b.velocities)
    assert math.isclose(a.masses.sum(), 2.0)
    np.testing.assert_allclose(a.positions.mean(0), (1.0, 2.0, 3.0), atol=1e-9)
    np.testing.assert_allclose(a.velocities.mean(0), (0.5, 0.0, 0.0), atol=1e-9)
    assert np.all(np.linalg.norm(a.positions - (1, 2, 3), axis=1) <= 3.0 * 1.01)


def test_seed_changes_the_draw():
    one = ics.sample([ics.plummer_sphere(particle_count=500, seed=1)])
    two = ics.sample([ics.plummer_sphere(particle_count=500, seed=2)])
    assert not np.array_equal(one.positions, two.positions)


def test_isolated_plummer_sphere_is_close_to_virial_equilibrium():
    s = ics.sample([ics.plummer_sphere(particle_count=20_000, radius=10.0, scale_length=1.0)],
                   ics.Config(softening=0.01))
    d = s.diagnostics(ics.Config(softening=0.01))
    assert d["finite"]
    potential = d["energy"] - d["kinetic"]
    assert 2 * d["kinetic"] / -potential == pytest.approx(1.0, abs=0.1)
    assert d["bound_fraction"] > 0.99


def test_disk_galaxy_rotates_and_shares_a_group():
    disk, halo = ics.disk_galaxy("g", particle_count=6000, seed=3)
    assert disk.galaxy_id == halo.galaxy_id == "g"
    assert disk.particle_count + halo.particle_count == 6000
    s = ics.sample([disk, halo])
    assert set(np.unique(s.component)) == {0, 1}
    on_disk = s.component == 0
    lz = np.cross(s.positions[on_disk], s.velocities[on_disk])[:, 2]
    assert np.mean(lz > 0) > 0.9
    assert np.sqrt(np.mean(s.positions[on_disk, 2] ** 2)) < 0.5


def test_rotation_turns_the_disk_plane():
    tilt = ics.euler_xyz_matrix((math.pi / 2, 0.0, 0.0))
    s = ics.sample([ics.exponential_disk(particle_count=3000, radius=5.0, scale_length=1.0,
                                         thickness=0.2, rotation=tilt)])
    spread = np.sqrt(np.mean(s.positions ** 2, axis=0))
    assert spread[1] < 0.3 * spread[2]


def test_presets_round_trip():
    galaxies, config = ics.preset("galaxy_collision")
    assert config.mode == "GALAXY" and len(galaxies) == 4
    assert config.particle_count == 100_000
    moon, cfg = ics.preset("moon_collision")
    assert cfg.mode == "IMPACT" and [c.name for c in moon] == ["Proto-Earth", "Theia"]
    with pytest.raises(KeyError):
        ics.preset("nope")


def test_preview_limits_points():
    parts = ics.disk_galaxy("p", particle_count=10_000)
    rows = ics.preview(parts, limit=100)
    assert [len(r) for r in rows] == [100, 100]


def test_assess_flags_drift():
    s = ics.sample([ics.plummer_sphere(particle_count=2000)])
    base = s.diagnostics()
    assert ics.assess(base, [base], periods=6)["pass"]
    drifted = dict(base, energy=base["energy"] * 1.2)
    assert not ics.assess(base, [drifted], periods=6)["pass"]
    assert not ics.assess(base, [base], periods=2)["pass"]


@pytest.mark.parametrize("bad", [
    {"kind": "CUBE"}, {"particle_count": 0}, {"mass": -1.0}, {"radius": float("nan")},
    {"rotation": ((2, 0, 0), (0, 1, 0), (0, 0, 1))}, {"rotation_fraction": 2.0},
])
def test_bad_components_are_rejected(bad):
    with pytest.raises(ValueError):
        ics.Component(name="x", **bad)


def test_euler_identity():
    np.testing.assert_allclose(ics.euler_xyz_matrix((0, 0, 0)), np.eye(3))
