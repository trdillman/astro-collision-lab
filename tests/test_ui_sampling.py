"""Exercise the pure count allocation without importing Blender or the GPU runtime."""

import ast
import math
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "extension" / "astro_collision_lab" / "ui.py"


def sampling_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_balanced_sampling_counts", "_sampling_spacing_ratio"}
    ]
    assert len(functions) == 2, "pure sampling helpers must be implemented"
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_balanced_sampling_counts"], namespace["_sampling_spacing_ratio"]


class SamplingAllocation(unittest.TestCase):
    def setUp(self):
        self.allocate, self.spacing_ratio = sampling_helpers()

    def test_preset_allocation_preserves_exact_total_and_common_spacing(self):
        counts = self.allocate((1.0, 0.46), 100000)
        self.assertEqual(counts, (91130, 8870))
        self.assertEqual(sum(counts), 100000)
        self.assertLessEqual(self.spacing_ratio((1.0, 0.46), counts), 1.01)

    def test_scaled_primitive_rebalances_without_changing_total(self):
        radii = (1.4, 0.46, 0.8)
        counts = self.allocate(radii, 100003)
        self.assertEqual(sum(counts), 100003)
        self.assertTrue(all(n >= 1 for n in counts))
        self.assertLessEqual(self.spacing_ratio(radii, counts), 1.01)
        self.assertEqual(counts, self.allocate(tuple(r * 1e100 for r in radii), 100003))

    def test_remainders_are_deterministic_and_every_source_retained(self):
        self.assertEqual(self.allocate((1, 1, 1), 10), (4, 3, 3))
        self.assertEqual(self.allocate((1, 0.0001, 0.0001), 5), (3, 1, 1))
        self.assertGreater(self.spacing_ratio((1, 0.0001, 0.0001), (3, 1, 1)), 1.01)

    def test_invalid_inputs_refused_before_authoring(self):
        for radii, total in (
            ((), 10),
            ((1, 1), 1),
            ((0, 1), 10),
            ((float("nan"),), 10),
            ((1,), 2.5),
            ((1,), True),
        ):
            with self.subTest(radii=radii, total=total):
                with self.assertRaises(ValueError):
                    self.allocate(radii, total)


if __name__ == "__main__":
    unittest.main()
