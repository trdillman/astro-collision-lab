"""CPU checks of the public package. GPU/OpenGL paths are covered by the add-on's tests
(tests/test_cuda_gl.py fakes the drivers) and by native probes on a Windows CUDA machine."""
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

warp = pytest.importorskip("warp")
warp.config.kernel_cache_dir = str(Path(tempfile.gettempdir()) / "warp-cuda-gl-test-cache")
warp.init()

import warp_cuda_gl as wgl  # noqa: E402


def frame(points, **channels):
    channels["positions"] = warp.array(points, dtype=warp.vec3f, device="cpu")
    return types.SimpleNamespace(channels=channels)


def test_public_names():
    assert wgl.STRIDE == 12
    assert len(wgl.MATERIAL_PALETTE) == 10 and len(wgl.DEFAULT_RAMP) == 4


@pytest.mark.skipif(os.name == "nt", reason="non-Windows behaviour")
def test_no_gl_context_off_windows():
    assert wgl.current_context_key() == 0


def test_writer_refuses_cpu_without_opt_in():
    with pytest.raises(RuntimeError):
        wgl.DeviceAppearanceWriter("cpu")


def test_batch_refuses_non_cuda_device():
    with pytest.raises(RuntimeError):
        wgl.CudaGLBatch("cpu", 16)
    with pytest.raises(ValueError):
        wgl.CudaGLBatch("cpu", 0)


def test_flat_colour_pack_on_cpu():
    writer = wgl.DeviceAppearanceWriter("cpu", allow_cpu=True)
    out = warp.zeros((3, wgl.STRIDE), dtype=warp.float32, device="cpu")
    writer.write(frame(((1, 2, 3), (4, 5, 6), (7, 8, 9))), out,
                 wgl.Appearance(flat_color=(0.1, 0.2, 0.3, 1.0), uniform_radius=0.5))
    rows = out.numpy()
    np.testing.assert_allclose(rows[:, :3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(rows[:, 4:8], [[0.1, 0.2, 0.3, 1.0]] * 3, rtol=1e-6)
    assert writer.report()["host_particle_readbacks"] == 0


def test_speed_ramp_spans_low_to_high():
    writer = wgl.DeviceAppearanceWriter("cpu", allow_cpu=True)
    velocities = warp.array(((0, 0, 0), (3, 4, 0)), dtype=warp.vec3f, device="cpu")
    out = warp.zeros((2, wgl.STRIDE), dtype=warp.float32, device="cpu")
    writer.write(frame(((0, 0, 0), (1, 1, 1)), velocities=velocities), out,
                 wgl.Appearance(color_source="SPEED", ramp_low=0.0, ramp_high=5.0))
    rows = out.numpy()
    np.testing.assert_allclose(rows[0, 4:8], wgl.DEFAULT_RAMP[0], atol=1e-5)
    np.testing.assert_allclose(rows[1, 4:8], wgl.DEFAULT_RAMP[-1], atol=1e-5)


def test_missing_channel_is_reported_before_writing():
    writer = wgl.DeviceAppearanceWriter("cpu", allow_cpu=True)
    with pytest.raises(wgl.UnsupportedAppearance):
        writer.validate(frame(((0, 0, 0),)), wgl.Appearance(color_source="DENSITY"))
