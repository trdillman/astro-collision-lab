"""Live IISPH health reduction; particle arrays stay on the solver device."""

import warp as wp


@wp.kernel
def _impact_health(
    overflow: wp.array(dtype=wp.int32),
    clamped: wp.array(dtype=wp.int32),
    tally: wp.array(dtype=wp.int32),
    density: wp.array(dtype=wp.float32),
    heat: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i == 0:
        status[0] = overflow[0]
        status[1] = clamped[0]
        status[2] = tally[4]
    if not wp.isfinite(density[i]):
        wp.atomic_add(status, 3, 1)
    if not wp.isfinite(heat[i]):
        wp.atomic_add(status, 4, 1)


class DeviceImpactHealth:
    """Reuse a five-int status buffer, with one 20-byte read after a completed step.

    The solver owns the three cumulative counters; sampling never clears them.
    Only the two current-state nonfinite counts are reset for each sample.
    Existing runtime checks still own position/velocity and gravity-tree faults.
    """

    FIELDS = (
        "neighbor_overflow_total",
        "speed_clamps",
        "budget_bound_steps",
        "nonfinite_densities",
        "nonfinite_internal_energies",
    )
    READBACK_BYTES = 20

    def __init__(self, context):
        self.device = context.device
        self.status = wp.zeros(5, dtype=wp.int32, device=self.device)

    def sample(self, solver):
        self.status.zero_()
        wp.launch(
            _impact_health,
            dim=int(solver.n),
            inputs=[solver.overflow, solver.clamped, solver._tally, solver.rho, solver.heat],
            outputs=[self.status],
            device=self.device,
        )
        values = self.status.numpy()
        return {name: int(value) for name, value in zip(self.FIELDS, values, strict=True)}
