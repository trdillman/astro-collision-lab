import unittest
from types import SimpleNamespace
from validation import solver_probe

class SolverGuardrailContracts(unittest.TestCase):
    def test_finite_moving_but_truncated_or_clamped_is_not_a_guardrail_pass(self):
        class Solver:
            neighbor_overflow=0;budget_bound_steps=0;pressure_sweeps=12
            active_outer_steps=2;max_outer_steps_used=2;dropped_bonds=0
            def neighbor_overflow_total(self):return 7
            def speed_clamps(self):return 3
        sim=SimpleNamespace(config=SimpleNamespace(mode='IMPACT'),runtime=SimpleNamespace(_require_solver=lambda:Solver()))
        result=solver_probe.final_guardrails(sim)
        self.assertFalse(result['solver_guardrails_pass'])
        self.assertEqual(result['counters']['neighbor_overflow_total'],7)
        self.assertEqual(result['counters']['speed_clamps'],3)
        self.assertFalse(result['scientific_validation'])
