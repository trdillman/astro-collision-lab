"""Validation-only support code.

Nothing in this package is imported by the product runtime or the Blender extension.
It exists so accuracy gates can be measured against references that do not share the
solver's algorithm, precision, or device.
"""

from .oracles import (
    GasState,
    NeighborReference,
    OracleUnavailableError,
    ReferenceTrajectory,
    RiemannSolution,
    TwoBodyElements,
    brute_force_neighbors,
    converged_rk4_nbody,
    direct_accelerations,
    elements_from_state,
    exact_riemann,
    kepler_two_body_state,
    strong_shock_density_ratio,
)

__all__ = [
    "GasState",
    "NeighborReference",
    "OracleUnavailableError",
    "ReferenceTrajectory",
    "RiemannSolution",
    "TwoBodyElements",
    "brute_force_neighbors",
    "converged_rk4_nbody",
    "direct_accelerations",
    "elements_from_state",
    "exact_riemann",
    "kepler_two_body_state",
    "strong_shock_density_ratio",
]
