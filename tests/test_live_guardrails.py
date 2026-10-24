"""Live adapter regressions with real Warp CPU kernels and a solver boundary.

The boundary replaces CUDA simulation, not the production adapter or health
reduction. Particle-array readbacks are forbidden while a live step is checked.
"""

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import warp as wp

# unittest discovery does not load pytest's conftest.py. Resolve the production
# package from this checkout without requiring an externally configured path.
_extension = str(Path(__file__).resolve().parents[1] / "extension")
sys.path.insert(0, _extension)
try:
    _simulation = importlib.import_module("astro_collision_lab.simulation")
finally:
    sys.path.remove(_extension)
AstroCollisionSimulation = _simulation.AstroCollisionSimulation
SimulationConfig = _simulation.SimulationConfig

wp.config.kernel_cache_dir = str(Path(tempfile.gettempdir()) / "acl-cpu-kernel-cache")
wp.init()


class SolverBoundary:
    def __init__(self):
        self.n = 3
        self.overflow = wp.zeros(1, dtype=wp.int32, device="cpu")
        self.clamped = wp.zeros(1, dtype=wp.int32, device="cpu")
        self._tally = wp.zeros(5, dtype=wp.int32, device="cpu")
        self.rho = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        self.heat = wp.array([0.0, 0.2, 0.4], dtype=wp.float32, device="cpu")


class RuntimeBoundary:
    def __init__(self, mutation=None, error=None):
        self._solver = SolverBoundary()
        self.last_step_ms = 0.0
        self.calls = 0
        self.display_calls = 0
        self.mutation = mutation
        self.error = error
        self.closed = False
        self.hydro = SimpleNamespace(simulation_time=0.0)
        self.channels = {
            "positions": wp.array([(0.0, 0.0, 0.0)] * 3, dtype=wp.vec3, device="cpu"),
            "densities": self._solver.rho,
            "internal_energies": self._solver.heat,
        }

    def _require_solver(self):
        return self._solver

    def step(self):
        self.calls += 1
        if self.mutation is not None:
            self.mutation(self._solver)
        if self.error is not None:
            raise self.error
        self.hydro.simulation_time += 0.005

    def display_channels(self):
        self.display_calls += 1
        return self.channels

    def close(self):
        self.closed = True


class LiveGuardrails(unittest.TestCase):
    def setUp(self):
        # Other boundary tests temporarily replace the Warp module.
        self._warp_patch = patch.dict(sys.modules, {"warp": wp})
        self._warp_patch.start()
        self.addCleanup(self._warp_patch.stop)

    def simulation(self, *, mutation=None, error=None, mode="IMPACT"):
        sim = AstroCollisionSimulation(SimulationConfig(mode=mode, particle_count=3))
        sim.runtime = RuntimeBoundary(mutation=mutation, error=error)
        sim.device_context = SimpleNamespace(device=wp.get_device("cpu"), close=lambda: None)
        sim.count = 3
        sim._channels = dict(sim.runtime.channels)
        return sim

    def health_type(self):
        self.assertIsNotNone(
            importlib.util.find_spec("astro_collision_lab._gpu_health"),
            "the live device health reduction is missing",
        )
        from astro_collision_lab._gpu_health import DeviceImpactHealth

        return DeviceImpactHealth

    def test_healthy_step_reads_only_one_20_byte_status_and_commits_one_frame(self):
        sim = self.simulation()
        reads = []
        original = wp.array.numpy

        def guarded_numpy(array, *args, **kwargs):
            self.assertIs(array, sim._health_guard.status, "live particle payload readback")
            reads.append(array.size * wp.types.type_size_in_bytes(array.dtype))
            return original(array, *args, **kwargs)

        with patch.object(wp.array, "numpy", guarded_numpy):
            channels = sim.step()
        self.assertEqual(reads, [20])
        self.assertEqual(sim._frames, 1)
        self.assertEqual(sim.physical_time, sim.config.dt)
        self.assertIs(channels["positions"], sim.runtime.channels["positions"])
        self.assertIn("live_guardrail_ms", sim.stats)
        self.assertGreaterEqual(sim.stats["last_step_ms"], sim.stats["live_guardrail_ms"])
        self.assertEqual(sim.stats["live_guardrail_readback_bytes"], 20)
        self.assertEqual(sim.stats["live_guardrail_readback_count"], 1)
        self.assertTrue(all(value == 0 for value in sim.stats["live_guardrails"].values()))

    def test_each_cumulative_guard_rejects_the_step_before_adapter_publication(self):
        mutations = {
            "neighbor_overflow_total": lambda s: s.overflow.assign([7]),
            "speed_clamps": lambda s: s.clamped.assign([3]),
            "budget_bound_steps": lambda s: s._tally.assign([2, 2, 0, 2, 1]),
        }
        for reason, mutation in mutations.items():
            with self.subTest(reason=reason):
                sim = self.simulation(mutation=mutation)
                with self.assertRaisesRegex(RuntimeError, reason):
                    sim.step()
                self.assertEqual(sim.runtime.calls, 1)
                self.assertEqual(sim._frames, 0)
                self.assertEqual(sim.physical_time, 0.0)
                self.assertEqual(sim.runtime.display_calls, 0)
                with self.assertRaisesRegex(RuntimeError, "reinitializ|reset"):
                    sim.display_channels()

    def test_nonfinite_density_or_heat_rejects_a_finite_position_step(self):
        for name, reason in (
            ("rho", "nonfinite_densities"),
            ("heat", "nonfinite_internal_energies"),
        ):
            for bad in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(channel=name, value=bad):
                    sim = self.simulation(
                        mutation=lambda s, name=name, bad=bad: getattr(s, name).assign(
                            [1.0, bad, 2.0]
                        )
                    )
                    with self.assertRaisesRegex(RuntimeError, reason):
                        sim.step()
                    self.assertEqual(sim._frames, 0)
                    with self.assertRaisesRegex(RuntimeError, "reinitializ|reset"):
                        sim.display_channels()

    def test_a_failed_step_is_latched_even_if_mutated_arrays_are_later_repaired(self):
        sim = self.simulation(error=RuntimeError("non-finite particle state; step failed closed"))
        with self.assertRaisesRegex(RuntimeError, "non-finite particle state"):
            sim.step()
        sim.runtime.error = None
        with self.assertRaisesRegex(RuntimeError, "reinitializ|reset"):
            sim.step()
        with self.assertRaisesRegex(RuntimeError, "reinitializ|reset"):
            sim.display_channels()
        self.assertEqual(sim.runtime.calls, 1)
        self.assertEqual(sim._frames, 0)

    def test_counter_rejection_stays_latched_until_close_and_fresh_runtime(self):
        sim = self.simulation(mutation=lambda s: s.clamped.assign([1]))
        with self.assertRaisesRegex(RuntimeError, "speed_clamps"):
            sim.step()
        sim.runtime._solver.clamped.zero_()
        sim.runtime.mutation = None
        with self.assertRaisesRegex(RuntimeError, "reinitializ|reset"):
            sim.step()
        self.assertEqual(sim.runtime.calls, 1)
        old_runtime = sim.runtime
        sim.close()
        self.assertTrue(old_runtime.closed)
        self.assertIsNone(sim._health_guard)
        self.assertIsNone(sim.runtime)
        sim.runtime = RuntimeBoundary()
        sim.device_context = SimpleNamespace(device=wp.get_device("cpu"), close=lambda: None)
        sim.count = 3
        sim.step()
        self.assertEqual(sim._frames, 1)
        self.assertTrue(sim.display_channels())

    def test_existing_other_mode_runtime_failures_are_preserved_and_latched(self):
        for mode in ("GALAXY", "COMPRESSIBLE"):
            with self.subTest(mode=mode):
                sim = self.simulation(mode=mode, error=RuntimeError("existing solver refusal"))
                with self.assertRaisesRegex(RuntimeError, "existing solver refusal"):
                    sim.step()
                with self.assertRaisesRegex(RuntimeError, "reinitializ|reset"):
                    sim.display_channels()
                self.assertIsNone(sim._health_guard)
                self.assertEqual(sim._frames, 0)

    def test_real_kernel_preserves_cumulative_counters_and_resets_transient_health(self):
        health = self.health_type()(SimpleNamespace(device=wp.get_device("cpu")))
        solver = SolverBoundary()
        solver.overflow.assign([7])
        solver.clamped.assign([3])
        solver._tally.assign([2, 2, 0, 2, 1])
        solver.rho.assign([float("nan"), float("inf"), 1.0])
        solver.heat.assign([0.0, -float("inf"), float("nan")])
        first = health.sample(solver)
        self.assertEqual(
            first,
            {
                "neighbor_overflow_total": 7,
                "speed_clamps": 3,
                "budget_bound_steps": 1,
                "nonfinite_densities": 2,
                "nonfinite_internal_energies": 2,
            },
        )
        solver.rho.assign([1.0, 2.0, 3.0])
        solver.heat.assign([0.0, 0.2, 0.4])
        second = health.sample(solver)
        self.assertEqual(second["neighbor_overflow_total"], 7)
        self.assertEqual(second["speed_clamps"], 3)
        self.assertEqual(second["budget_bound_steps"], 1)
        self.assertEqual(second["nonfinite_densities"], 0)
        self.assertEqual(second["nonfinite_internal_energies"], 0)
        self.assertEqual(health.status.dtype, wp.int32)
        self.assertEqual(health.status.shape, (5,))


if __name__ == "__main__":
    unittest.main()
