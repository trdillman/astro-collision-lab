"""CPU execution of real Warp kernels; no Windows, OpenGL or CUDA device required."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import warp as wp

wp.config.kernel_cache_dir = str(Path(tempfile.gettempdir()) / "acl-cpu-kernel-cache")
wp.init()


def load():
    path = Path(__file__).parents[1] / "extension" / "astro_collision_lab" / "device_appearance.py"
    assert path.exists(), "device packing is missing"
    spec = importlib.util.spec_from_file_location("device_appearance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Packing(unittest.TestCase):
    def setUp(self):
        self._previous_warp = sys.modules.get("warp")
        sys.modules["warp"] = wp
        self.m = load()
        self.writer = self.m.DeviceAppearanceWriter("cpu", allow_cpu=True)

    def tearDown(self):
        if self._previous_warp is None:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = self._previous_warp

    def frame(self, points=((1, 2, 3), (4, 5, 6)), **channels):
        channels["positions"] = wp.array(points, dtype=wp.vec3f, device="cpu")
        return types.SimpleNamespace(channels=channels)

    def pack(self, frame, appearance):
        result = wp.zeros((len(frame.channels["positions"]), 12), dtype=wp.float32, device="cpu")
        self.writer.write(frame, result, appearance)
        return result.numpy()

    def test_flat_layout_and_absent_velocity(self):
        out = self.pack(self.frame(), self.m.Appearance(flat_color=(0.1, 0.2, 0.3, 0.4)))
        np.testing.assert_allclose(out[:, :3], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(out[:, 4:8], [[0.1, 0.2, 0.3, 0.4]] * 2)
        np.testing.assert_allclose(out[:, [3, 8, 9, 10, 11]], 0)

    def test_physical_radii_floors_and_highlight_follow_ids(self):
        self.writer.set_metadata(
            stable_ids=[-2, 7], floor_radii=[3, 0.1], highlighted=[True, False], revision=1
        )
        frame = self.frame(
            sizes=wp.array([2.0, 1.0], dtype=wp.float32, device="cpu"),
            stable_ids=wp.array([7, -2], dtype=wp.int64, device="cpu"),
        )
        out = self.pack(
            frame,
            self.m.Appearance(
                size_source="PHYSICAL", radius_multiplier=2.0, flat_color=(0.2, 0.3, 0.4, 0.25)
            ),
        )
        np.testing.assert_allclose(out[:, 3], [4.0, 3.0])
        np.testing.assert_allclose(out[1, 4:8], [1.0, 0.63, 0.15, 0.25])

    def test_five_stop_ramp_and_nonfinite_draws_first_stop(self):
        values = wp.array([-1.0, 0.0, 0.25, 0.5, 1.0, float("nan")], dtype=wp.float32, device="cpu")
        frame = self.frame(points=[(0, 0, 0)] * 6, densities=values)
        knots = ((0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 1, 1))
        out = self.pack(
            frame,
            self.m.Appearance(
                color_source="DENSITY", ramp_low=0.0, ramp_high=1.0, ramp_knots=knots
            ),
        )
        np.testing.assert_allclose(
            out[:, 4:8], [knots[0], knots[0], knots[1], knots[2], knots[4], knots[0]]
        )

    def test_speed_auto_range_uses_finite_gpu_bounds(self):
        frame = self.frame(
            velocities=wp.array([(0, 0, 0), (3, 4, 0)], dtype=wp.vec3f, device="cpu")
        )
        out = self.pack(frame, self.m.Appearance(color_source="SPEED"))
        np.testing.assert_allclose(out[0, 4:8], self.m.DEFAULT_RAMP[0])
        np.testing.assert_allclose(out[1, 4:8], self.m.DEFAULT_RAMP[-1])
        np.testing.assert_allclose(out[:, 8:11], [[0, 0, 0], [3, 4, 0]])

    def test_distance_uses_explicit_reference_point(self):
        out = self.pack(
            self.frame(),
            self.m.Appearance(
                color_source="DISTANCE",
                reference_point=(1, 2, 3),
                ramp_low=0,
                ramp_high=np.sqrt(27),
            ),
        )
        np.testing.assert_allclose(out[0, 4:8], self.m.DEFAULT_RAMP[0])
        np.testing.assert_allclose(out[1, 4:8], self.m.DEFAULT_RAMP[-1], atol=1e-6)

    def test_source_colors_and_metadata_revision_do_not_upload_per_frame(self):
        metadata = dict(
            stable_ids=[10, 20], colors=[(0.8, 0.1, 0.2, 0.3), (0.2, 0.3, 0.9, 0.4)], revision=2
        )
        self.writer.set_metadata(**metadata)
        uploads = self.writer.report()["metadata_uploads"]
        self.writer.set_metadata(**metadata)
        frame = self.frame(stable_ids=wp.array([20, 10], dtype=wp.int64, device="cpu"))
        out = self.pack(frame, self.m.Appearance(color_source="OBJECT_COLOR"))
        np.testing.assert_allclose(out[:, 4:8], [(0.2, 0.3, 0.9, 0.4), (0.8, 0.1, 0.2, 0.3)])
        self.pack(frame, self.m.Appearance(color_source="OBJECT_COLOR"))
        assert self.writer.report()["metadata_uploads"] == uploads

    def test_missing_or_unsupported_fields_raise_before_any_write(self):
        frame = self.frame()
        with self.assertRaisesRegex(self.m.UnsupportedAppearance, "TEMPERATURE"):
            self.writer.validate(frame, self.m.Appearance(color_source="TEMPERATURE"))
        with self.assertRaisesRegex(self.m.UnsupportedAppearance, "densities"):
            self.writer.validate(frame, self.m.Appearance(color_source="DENSITY"))

    def test_double_precision_channels_and_manual_count(self):
        frame = types.SimpleNamespace(
            channels={
                "positions": wp.array([(1, 2, 3), (4, 5, 6)], dtype=wp.vec3d, device="cpu"),
                "velocities": wp.array([(0, 3, 4), (1, 2, 2)], dtype=wp.vec3d, device="cpu"),
                "sizes": wp.array([0.5, 0.25], dtype=wp.float64, device="cpu"),
                "densities": wp.array([2.0, 4.0], dtype=wp.float64, device="cpu"),
            }
        )
        result = wp.zeros((1, 12), dtype=wp.float32, device="cpu")
        self.writer.write(
            frame,
            result,
            self.m.Appearance(
                size_source="PHYSICAL", color_source="DENSITY", ramp_low=2, ramp_high=4
            ),
            count=1,
        )
        out = result.numpy()[0]
        np.testing.assert_allclose(out[:4], [1, 2, 3, 0.5])
        np.testing.assert_allclose(out[8:11], [0, 3, 4])
        np.testing.assert_allclose(out[4:8], self.m.DEFAULT_RAMP[0])

    def test_material_palette_and_explicit_frame_rgba(self):
        colors = wp.array(
            [(0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8)], dtype=wp.vec4f, device="cpu"
        )
        frame = self.frame(
            material_ids=wp.array([-1, 11], dtype=wp.int64, device="cpu"), colors=colors
        )
        material = self.pack(frame, self.m.Appearance(color_source="MATERIAL"))
        np.testing.assert_allclose(
            material[:, 4:8], [(0.35, 0.35, 0.35, 1.0), self.m.MATERIAL_PALETTE[1]]
        )
        direct = self.pack(frame, self.m.Appearance(color_source="FRAME"))
        np.testing.assert_allclose(direct[:, 4:8], colors.numpy())

    def test_fractional_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            self.writer.validate(self.frame(), self.m.Appearance(), count=1.5)


if __name__ == "__main__":
    unittest.main()
