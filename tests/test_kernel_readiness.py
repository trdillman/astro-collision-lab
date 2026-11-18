import pytest
from astro_collision_lab.interaction import kernel_identity
from astro_collision_lab.simulation import SimulationConfig
from astro_collision_lab.warm_worker import kernel_probe


def test_kernel_identity_is_reused_across_authored_models():
    runtime = ("Blender", (5, 2, 0), "Warp", "1.15.0", "test-device")
    code = "source-fingerprint"
    original = SimulationConfig(mode="GALAXY", particle_count=100_000, theta=.6)
    edited = SimulationConfig(mode="GALAXY", particle_count=125_000, theta=.4, dt=.002)

    assert kernel_identity(original, code_fingerprint=code, runtime=runtime) == kernel_identity(
        edited, code_fingerprint=code, runtime=runtime
    )
    assert kernel_identity(original, code_fingerprint=code, runtime=runtime) != kernel_identity(
        SimulationConfig(mode="IMPACT", particle_count=100_000),
        code_fingerprint=code, runtime=runtime,
    )


@pytest.mark.parametrize("mode", ["GALAXY", "IMPACT", "COMPRESSIBLE"])
def test_kernel_probe_is_bounded_and_valid_for_mode(mode):
    config, bodies = kernel_probe(mode)

    assert config.mode == mode
    assert config.particle_count == sum(body.particle_count for body in bodies)
    assert 0 < config.particle_count <= 512
    assert all(body.kind == "SPHERE" for body in bodies)
    assert config.validate() is None
