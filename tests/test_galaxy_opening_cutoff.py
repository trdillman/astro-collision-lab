"""The last rejected float and its successor bracket the rounded opening test."""
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from astro_collision_lab.galaxy_walk import opening_cutoff


@wp.kernel(enable_backward=False)
def _cutoffs(reach: wp.array(dtype=float), theta: wp.array(dtype=float),
             result: wp.array(dtype=float)):
    i = wp.tid()
    result[i] = opening_cutoff(reach[i], theta[i])


@wp.kernel(enable_backward=False)
def _reference(reach: wp.array(dtype=float), theta: wp.array(dtype=float),
               distance2: wp.array(dtype=float), result: wp.array(dtype=int)):
    i = wp.tid()
    result[i] = int(reach[i] / wp.sqrt(distance2[i]) < theta[i])


def check_boundaries(device):
    rng = np.random.default_rng(113)
    edge = np.array([0, 1, 0x007fffff, 0x00800000, 0x3f000000, 0x3f800000,
                     0x7f7fffff], dtype=np.uint32).view(np.float32)
    reach = np.concatenate([np.repeat(edge, len(edge)-1), rng.integers(
        0, 0x7f800000, size=8192, dtype=np.uint32).view(np.float32)])
    theta = np.concatenate([np.tile(edge[1:], len(edge)), rng.integers(
        1, 0x7f800000, size=8192, dtype=np.uint32).view(np.float32)])
    size = len(reach)
    params = [wp.array(x, device=device) for x in (reach, theta)]
    cut = wp.empty(size, dtype=float, device=device)
    wp.launch(_cutoffs, size, inputs=[*params, cut], device=device)
    boundary = cut.numpy()
    assert np.isfinite(boundary).all() and (boundary >= 0).all()
    result = wp.empty(size, dtype=int, device=device)
    with np.errstate(over="ignore", under="ignore"):
        successor = np.nextafter(boundary, np.float32(np.inf))
    for distances, accepted in ((boundary, 0), (successor, 1)):
        wp.launch(_reference, size, inputs=[*params, wp.array(distances, device=device), result],
                  device=device)
        np.testing.assert_array_equal(result.numpy(), np.full(size, accepted))


def test_opening_cutoff_matches_reference_on_adjacent_floats():
    check_boundaries("cpu")
