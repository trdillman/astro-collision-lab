"""Initial-condition invariants; no Blender or GPU required."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest


_path = Path(__file__).resolve().parents[1] / "preset_specs.py"
if not _path.exists():
    _path = _path.parent / "extension" / "astro_collision_lab" / "preset_specs.py"
_spec = importlib.util.spec_from_file_location("acl_preset_specs", _path)
presets = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = presets
_spec.loader.exec_module(presets)


class PresetInitialConditions(unittest.TestCase):
    def assert_balanced(self, bodies):
        for axis in range(3):
            self.assertAlmostEqual(sum(b.mass * b.location[axis] for b in bodies), 0.0)
            self.assertAlmostEqual(sum(b.mass * b.velocity[axis] for b in bodies), 0.0)

    def test_impact_has_exact_requested_budget_and_separate_bodies(self):
        spec = presets.moon_collision()
        self.assertEqual(sum(b.count for b in spec.bodies), 100_000)
        target, impactor = spec.bodies
        separation = math.dist(target.location, impactor.location)
        self.assertGreater(separation, target.radius + impactor.radius)
        self.assertLess(separation, 1.15 * (target.radius + impactor.radius))
        self.assert_balanced(spec.bodies)

    def test_iisph_preset_count_follows_volume_for_common_spacing(self):
        target,impactor=presets.moon_collision().bodies
        ratio=(target.radius**3/target.count)/(impactor.radius**3/impactor.count)
        self.assertAlmostEqual(ratio,1.0,places=4)
        self.assertEqual((target.count,impactor.count),(91130,8870))

    def test_impact_is_approaching_and_has_nonzero_impact_angular_momentum(self):
        a, b = presets.moon_collision().bodies
        relative_position = tuple(b.location[i] - a.location[i] for i in range(3))
        relative_velocity = tuple(b.velocity[i] - a.velocity[i] for i in range(3))
        self.assertLess(sum(x * v for x, v in zip(relative_position, relative_velocity)), 0.0)
        angular_z = sum(
            body.mass * (body.location[0] * body.velocity[1] - body.location[1] * body.velocity[0])
            for body in (a, b)
        )
        self.assertNotEqual(angular_z, 0.0)
        self.assertLess(abs(relative_position[1]), a.radius + b.radius)

    def test_galaxies_have_exact_budget_balanced_components_and_orbit(self):
        spec = presets.galaxy_collision()
        self.assertEqual(sum(b.count for b in spec.bodies), 100_000)
        self.assert_balanced(spec.bodies)
        first, second = spec.bodies[:2], spec.bodies[2:]
        for galaxy in (first, second):
            self.assertEqual({b.kind for b in galaxy}, {"DISK", "HALO"})
            self.assertEqual(sum(b.count for b in galaxy), 50_000)
            self.assertAlmostEqual(sum(b.mass for b in galaxy), 1.0)
            self.assertEqual(galaxy[0].location, galaxy[1].location)
            self.assertEqual(galaxy[0].velocity, galaxy[1].velocity)
            self.assertLess(galaxy[0].radius, galaxy[1].radius)
        self.assertAlmostEqual(second[0].location[1] - first[0].location[1], 0.4)
        self.assertGreater(first[0].velocity[0], second[0].velocity[0])


if __name__ == "__main__":
    unittest.main()
