"""Self-gravity for the SPH particle set, recorded into the hydrodynamic derivative pass.

The tree itself is the Phase 3 backend: the same Morton keys, the same linear octree build,
the same monopole traversal, compiled from `tree.py`'s kernel set. What is new here is the
binding. `BarnesHutRuntime` owns its bodies - it allocates their positions, integrates them
and resolves their collisions - and an SPH particle is owned by the neighbour stage instead.
So this service allocates only the tree, borrows the position and mass arrays it is given,
and exposes its launches for recording rather than submitting them, which is what lets the
whole derivative evaluation stay one captured sequence.

Borrowing rather than copying is the point of the binding: gravity has to be evaluated at
the positions the drift just wrote, not at a snapshot of them, and a second copy would be a
second thing to keep in step. The spec puts it as an acceptance clause - self-gravity
coupling uses a consistent evaluation time - and the way that clause is met here is
structural rather than procedural. There is no path that evaluates gravity anywhere except
inside the same pass that evaluates the pressure gradient, from the same array.

Which sum evaluates that gravity is a measurement, not a constant. A Barnes-Hut traversal
is an approximation bought to escape an O(N^2) pair sum, and it is only worth its build
above the count where the build stops costing more than the pairs it saves. Below that
count the exact sum is both faster and exact, so `backend="auto"` times the two paths once
against each other on the bound state and keeps the winner, which is the rule
`runtime_factory._select_auto_backend` already applies to the orbital backends. The accuracy
half of that rule is free here: the direct path is the exact evaluation the tree
approximates, so preferring it can only lower the error, and the tree keeps the theta bound
it already answers to. Both paths are recorded into the same derivative pass from the same
borrowed array, so the coupling clause above is indifferent to which one ran.

The softening is a fixed Plummer length, deliberately not tied to the smoothing length.
Softening that varies per particle changes the potential's dependence on the coordinates,
so the force acquires an extra term - the gravitational analogue of the grad-h correction -
and dropping that term costs exactly the energy conservation this stage is gated on.
Adaptive gravitational softening is therefore its own deliverable with its own correction,
not a parameter change here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .model import DEFAULT_TREE_MAX_DEPTH, DEFAULT_TREE_STACK_CAPACITY
from .nbody import Precision, _dtypes, _registrars
from .runtime_foundation import DeviceContext, DeviceRequirementError
from .tree import (
    MAX_TREE_DEPTH,
    MAX_TREE_NODES,
    _next_power_of_two,
    sparse_node_capacity,
)
from .tree import (
    _kernels as _tree_kernels,
)

__all__ = [
    "GRAVITY_BACKENDS",
    "SelfGravityConfig",
    "SelfGravityDiagnostics",
    "SelfGravityRuntime",
]

#: Paths `record_stage` can write. "tree" is the Barnes-Hut traversal, "direct" the exact
#: softened pair sum, "auto" whichever of the two this device measures faster at this count.
GRAVITY_BACKENDS = ("auto", "tree", "direct")

#: Timed launches per path when `backend="auto"` calibrates. Three is the sample the orbital
#: calibration already takes, and the median of three is not worth the extra launch here:
#: the decision is a comparison between two costs that differ by a factor, not a measurement
#: reported as a figure.
CALIBRATION_SAMPLES = 3


@dataclass(frozen=True)
class SelfGravityConfig:
    """Configuration of the self-gravity stage.

    The sizes and the precision are repeated here rather than read from the neighbour
    configuration because this stage is also driven on its own, against an exact direct
    sum, where there is no neighbour stage to read them from. `HydroConfig.validate`
    checks that the two agree, so the duplication cannot become a disagreement.
    """

    capacity: int
    count: int
    gravitational_constant: float
    softening: float
    theta: float = 0.5
    max_depth: int = DEFAULT_TREE_MAX_DEPTH
    leaf_capacity: int = 8
    stack_capacity: int = DEFAULT_TREE_STACK_CAPACITY
    node_capacity: int = 0
    # Whether the multipole acceptance carries the traceless quadrupole of each node, per
    # `adr/0026-tree-quadrupole-expansion.md`. The default matches `TreeConfig` because
    # both drive the same kernels; `quadrupoles=False` reproduces the monopole-only build
    # bit for bit.
    quadrupoles: bool = True
    # Accuracy parameter of the free-fall step, `dt = sqrt(2 eta eps / |a|)`. It is the
    # time to fall a fixed fraction of the softening length from rest, which is the only
    # length the gravitational field alone defines: the Courant step is set by the sound
    # crossing time and says nothing about a cold region collapsing under its own weight.
    timestep_accuracy: float = 0.025
    precision: Precision = Precision.FLOAT32
    # Which sum `record_stage` records; see `GRAVITY_BACKENDS`. The default is the tree and
    # not "auto" on purpose. Every suite that drives this stage against a reference builds
    # this config directly and reads `accelerations` as the tree's answer -- the theta-bound
    # field test in `validation/sph_gravity_suite.py` among them -- so a default that could
    # silently substitute the exact sum would leave those bounds measuring nothing. Opting
    # in belongs to the lane that plays rather than the lane that certifies.
    backend: str = "tree"

    @property
    def resolved_node_capacity(self) -> int:
        return self.node_capacity or sparse_node_capacity(self.count, self.max_depth)

    def validate(self) -> None:
        if self.backend not in GRAVITY_BACKENDS:
            raise ValueError(f"backend must be one of {list(GRAVITY_BACKENDS)}")
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 < self.count <= self.capacity:
            raise ValueError("count must be in (0, capacity]")
        if self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be positive")
        if self.softening <= 0.0:
            # Unsoftened, the pair force diverges as two particles approach and one close
            # encounter sets the step for the whole run. A collisionless field wants the
            # softening to exist; how large it is belongs to the scene.
            raise ValueError("softening must be positive")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if not 2 <= self.max_depth <= MAX_TREE_DEPTH:
            raise ValueError(f"max_depth must be in [2, {MAX_TREE_DEPTH}]")
        if self.leaf_capacity < 1:
            raise ValueError("leaf_capacity must be positive")
        if self.timestep_accuracy <= 0.0:
            raise ValueError("timestep_accuracy must be positive")
        required_stack = 1 + 7 * self.max_depth
        if self.stack_capacity < required_stack:
            raise ValueError(
                f"stack_capacity {self.stack_capacity} is below required {required_stack}"
            )
        nodes = self.resolved_node_capacity
        minimum_nodes = self.max_depth + 1
        if nodes < minimum_nodes:
            raise ValueError(
                f"node_capacity {nodes} is below minimum occupied chain {minimum_nodes}"
            )
        if nodes > MAX_TREE_NODES:
            raise ValueError(
                f"tree node capacity {nodes} exceeds supported maximum {MAX_TREE_NODES}"
            )


@dataclass(frozen=True)
class SelfGravityDiagnostics:
    active_count: int
    node_count: int
    leaf_count: int
    max_depth: int
    stack_high_water: int
    overflow: bool
    interactions: int
    theta: float
    softening: float
    stage_ms: float
    capacity_overflow: bool = False
    submission: str = "eager"
    # The path that actually ran, and the two costs that chose it. Under "auto" the tree
    # counters above are zero when `backend` reads "direct", because no tree was built --
    # that is the measurement, not a missing one.
    backend: str = "tree"
    calibrated_tree_ms: float = 0.0
    calibrated_direct_ms: float = 0.0


class SelfGravityRuntime:
    """Build a linear octree over borrowed particles and accumulate their mutual gravity."""

    def __init__(self, device_context: DeviceContext, config: SelfGravityConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.last_stage_ms = 0.0
        self.explicit_validation_readbacks = 0
        self._wp: Any | None = None
        self._kernels: Any | None = None
        self._gravity_kernels: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._graphs: dict[str, Any] = {}
        self._positions: Any | None = None
        self._masses: Any | None = None
        self._total_nodes = config.resolved_node_capacity
        self._sort_size = _next_power_of_two(config.capacity)
        # Resolved at bind, because "auto" is answered by measuring the bound state. Until
        # then the tree is the standing answer, which is what this stage has always done.
        self._selected = "tree" if config.backend == "auto" else config.backend
        self._tree_ms = 0.0
        self._direct_ms = 0.0

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    @property
    def sort_size(self) -> int:
        return self._sort_size

    def bind(self, positions: Any, masses: Any) -> None:
        """Take the position and mass arrays another stage owns and allocate the tree.

        Nothing is copied. The arrays stay the property of whoever allocated them, which
        for the coupled runtime is the neighbour stage and the hydrodynamic stage, so the
        drift and this build always see the same coordinates.
        """

        self.device_context.initialize()
        wp = self.device_context.wp
        self._wp = wp
        np = __import__("numpy")
        config = self.config
        scalar, vector, _state_dtype = _dtypes(wp, config.precision)
        if positions.dtype != vector:
            raise ValueError("positions dtype does not match the configured precision")
        if masses.dtype != scalar:
            raise ValueError("masses dtype does not match the configured precision")
        if len(positions) < config.capacity or len(masses) < config.capacity:
            raise ValueError("bound arrays are shorter than the configured capacity")
        self._kernels = _tree_kernels(wp, config.precision)
        self._gravity_kernels = _gravity_kernels(wp, config.precision)
        device = self.device_context.device
        self._positions = positions
        self._masses = masses

        active = np.zeros(config.capacity, dtype=np.int32)
        active[: config.count] = 1
        self._arrays = {
            "accelerations": wp.zeros(config.capacity, dtype=vector, device=device),
            "active": wp.array(active, dtype=wp.int32, device=device),
            "stable_ids": wp.array(
                np.arange(config.capacity, dtype=np.int64), dtype=wp.int64, device=device
            ),
            "body_keys": wp.zeros(config.capacity, dtype=wp.int64, device=device),
            "sorted_keys": wp.zeros(self._sort_size, dtype=wp.int64, device=device),
            "sorted_indices": wp.zeros(self._sort_size, dtype=wp.int32, device=device),
            "bounds": wp.zeros(6, dtype=scalar, device=device),
            "node_levels": wp.zeros(self._total_nodes, dtype=wp.int32, device=device),
            "node_prefix": wp.zeros(self._total_nodes, dtype=wp.int64, device=device),
            "node_mass": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_moment_x": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_moment_y": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_moment_z": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_qxx": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_qxy": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_qxz": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_qyy": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_qyz": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_qzz": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_count": wp.zeros(self._total_nodes, dtype=wp.int32, device=device),
            "node_start": wp.zeros(self._total_nodes, dtype=wp.int32, device=device),
            "node_end": wp.zeros(self._total_nodes, dtype=wp.int32, device=device),
            "node_cursor": wp.zeros(1, dtype=wp.int32, device=device),
            "slot_nodes": wp.zeros(
                config.count * (config.max_depth + 1), dtype=wp.int32, device=device
            ),
            "stack_nodes": wp.zeros(
                config.capacity * config.stack_capacity, dtype=wp.int32, device=device
            ),
            "stack_levels": wp.zeros(
                config.capacity * config.stack_capacity, dtype=wp.int32, device=device
            ),
            "direct_accelerations": wp.zeros(config.capacity, dtype=vector, device=device),
            "diag": wp.zeros(8, dtype=wp.int64, device=device),
        }
        self._graphs.clear()
        if config.backend == "auto":
            # Here and not later: the caller captures `record_stage` into a graph on its
            # first step, and a capture may not be timed from inside itself. Bind is the
            # last moment the arrays exist and nothing is recording yet.
            self._calibrate()

    @property
    def selected_backend(self) -> str:
        """Which sum `record_stage` writes: ``"tree"`` or ``"direct"``, never ``"auto"``."""

        return self._selected

    def _calibrate(self) -> None:
        """Time both sums on the bound state and keep the faster.

        The comparison is the whole decision. There is no count threshold here and there
        deliberately is not one: where the tree starts paying for itself is a property of
        this device, this precision and this particle count together, and the repo's own
        sweep puts it four orders of magnitude above what an impact scene runs. A number
        copied out of that sweep would be right for the machine it was measured on and
        quietly wrong everywhere else, so the machine is asked instead.

        Cost is one warm plus `CALIBRATION_SAMPLES` timed launches of each path, once per
        bind, against the thousands of substeps a single frame of this backend then takes.

        A tree that overflowed its node or stack capacity while being timed loses outright.
        That is the same clause `_select_auto_backend` applies, and it earns its place: a
        scene whose tree cannot be built at all still has an exact sum available, so it can
        now run correctly where it previously only failed closed.
        """

        tree_ms = self._time(self._record_tree_stage)
        direct_ms = self._time(self._record_direct_stage)
        diag = self._arrays["diag"].numpy()
        overflowed = bool(int(diag[3]) or int(diag[7]))
        self._tree_ms = tree_ms
        self._direct_ms = direct_ms
        self._selected = "direct" if overflowed or direct_ms < tree_ms else "tree"
        if self._selected == "direct":
            # The timed tree run left its diagnostics behind, overflow flags included, and
            # nothing will overwrite them again because no tree will be built. Clearing
            # them keeps `_raise_tree_overflow` from failing every later step closed on a
            # tree this runtime has decided not to use.
            wp = self._require_wp()
            wp.launch(
                self._require_kernels().clear_diag,
                dim=8,
                inputs=[self._arrays["diag"]],
                device=self.device_context.device,
            )
            self._wait()

    def _time(self, record: Any) -> float:
        """Mean wall time of one recorded path, after a warm launch to exclude module load."""

        record()
        self._wait()
        started = time.perf_counter()
        for _ in range(CALIBRATION_SAMPLES):
            record()
        self._wait()
        return (time.perf_counter() - started) * 1000.0 / float(CALIBRATION_SAMPLES)

    @property
    def accelerations(self) -> Any:
        """The device array of gravitational accelerations, without a readback."""

        if "accelerations" not in self._arrays:
            raise DeviceRequirementError("SelfGravityRuntime has no bound arrays")
        return self._arrays["accelerations"]

    def evaluate(self) -> None:
        """Submit the stage on its own and wait, leaving the result on the device."""

        self._submit("gravity", self.record_stage)
        self._wait()
        self._raise_tree_overflow()

    def evaluate_direct(self) -> None:
        """Submit the exact monopole sum over the same softening, for a reference.

        This is the O(N^2) pair sum the tree approximates, run on the device at the state
        precision. It is a measurement of the tree's own approximation error and nothing
        else: the validation lane's independent reference is computed in float64 from the
        positions, not from this.
        """

        self._submit("direct", self._record_direct)
        self._wait()

    @property
    def direct_accelerations(self) -> Any:
        if "direct_accelerations" not in self._arrays:
            raise DeviceRequirementError("SelfGravityRuntime has no bound arrays")
        return self._arrays["direct_accelerations"]

    def record_stage(self) -> None:
        """Write the build and the traversal into the caller's sequence.

        The hydrodynamic stage captures its whole derivative evaluation as one graph, and a
        captured graph cannot launch another captured graph, so what is exposed is the
        recording rather than a submission - the same contract `NeighborRuntime` uses.

        Which sum gets recorded was settled at bind. It cannot be settled here: this runs
        inside the caller's capture, where a timing readback is illegal and a branch taken
        per step would be baked into the graph anyway.
        """

        if self._selected == "direct":
            self._record_direct_stage()
            return
        self._record_tree_stage()

    def _record_tree_stage(self) -> None:
        self._record_rebuild()
        self._record_traversal()

    def _record_direct_stage(self) -> None:
        """Record the exact pair sum into this stage's own acceleration array.

        Distinct from `_record_direct`, which fills the separate `direct_accelerations`
        array the accuracy suites hold the tree against. Keeping them apart is the point:
        a reference that playback also writes would no longer be independent of it.

        No clear precedes this. `direct_acceleration` assigns every entry it is launched
        over, the inactive ones to zero, which is exactly what `clear_accelerations` does
        for the tree path before its traversal accumulates.
        """

        wp = self._require_wp()
        kernels = self._require_kernels()
        config = self.config
        wp.launch(
            kernels.direct_acceleration,
            dim=config.count,
            inputs=[
                self._require_positions(),
                self._require_masses(),
                self._arrays["active"],
                config.count,
                config.gravitational_constant,
                config.softening,
                self._arrays["accelerations"],
            ],
            device=self.device_context.device,
        )

    def record_apply(self, accelerations: Any, particle_timesteps: Any) -> None:
        """Add the gravitational acceleration into a caller's total and fold in its step.

        Kept apart from `record_stage` so the stage can be driven against a reference
        without a hydrodynamic acceleration to add to, and so the ordering is written where
        it matters: the pressure gradient assigns the total acceleration, so this has to
        follow it rather than precede it.
        """

        wp = self._require_wp()
        kernels = self._require_gravity_kernels()
        config = self.config
        wp.launch(
            kernels.apply_gravity,
            dim=config.count,
            inputs=[
                self._arrays["accelerations"],
                accelerations,
                particle_timesteps,
                config.softening,
                config.timestep_accuracy,
            ],
            device=self.device_context.device,
        )

    def diagnostics(self) -> SelfGravityDiagnostics:
        self._wait()
        diag = self._arrays["diag"].numpy()
        return SelfGravityDiagnostics(
            active_count=int(diag[6]),
            node_count=int(diag[0]),
            leaf_count=int(diag[5]),
            max_depth=int(diag[1]),
            stack_high_water=int(diag[2]),
            overflow=bool(diag[3] or diag[7]),
            interactions=int(diag[4]),
            theta=self.config.theta,
            softening=self.config.softening,
            stage_ms=self.last_stage_ms,
            capacity_overflow=bool(diag[7]),
            submission="graph" if self._graphs.get("gravity") is not None else "eager",
            backend=self._selected,
            calibrated_tree_ms=self._tree_ms,
            calibrated_direct_ms=self._direct_ms,
        )

    def _raise_tree_overflow(self) -> None:
        if self._selected == "direct":
            # No tree was built, so the flags below describe nothing this runtime ran.
            return
        diag = self._arrays["diag"].numpy()
        if int(diag[7]):
            raise DeviceRequirementError(
                "SPH self-gravity node capacity overflow; operation failed closed"
            )
        if int(diag[3]):
            raise DeviceRequirementError(
                "SPH self-gravity traversal stack overflow; operation failed closed"
            )

    def close(self) -> None:
        self._arrays.clear()
        self._graphs.clear()
        self._positions = None
        self._masses = None

    def _wait(self) -> None:
        wp = self._require_wp()
        wp.synchronize_device(self.device_context.device)

    def _submit(self, name: str, record: Any) -> None:
        if not self._arrays:
            raise DeviceRequirementError("SelfGravityRuntime has no bound arrays")
        wp = self._require_wp()
        start = time.perf_counter()
        graph = self._graph(name, record)
        if graph is None:
            record()
        else:
            wp.capture_launch(graph)
        self.last_stage_ms = (time.perf_counter() - start) * 1000.0

    def _graph(self, name: str, record: Any) -> Any:
        if name in self._graphs:
            return self._graphs[name]
        wp = self._require_wp()
        device = wp.get_device(self.device_context.device)
        if not device.is_cuda:
            self._graphs[name] = None
            return None
        wp.force_load(device=device)
        with wp.ScopedCapture(device) as capture:
            record()
        self._graphs[name] = capture.graph
        return capture.graph

    def _record_rebuild(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        arrays = self._arrays
        config = self.config
        device = self.device_context.device
        positions = self._require_positions()
        masses = self._require_masses()

        wp.launch(kernels.clear_bounds, dim=6, inputs=[arrays["bounds"]], device=device)
        wp.launch(
            kernels.reduce_bounds,
            dim=config.count,
            inputs=[positions, arrays["active"], config.count, arrays["bounds"]],
            device=device,
        )
        wp.launch(kernels.finalize_bounds, dim=1, inputs=[arrays["bounds"]], device=device)
        wp.launch(
            kernels.compute_morton,
            dim=self._sort_size,
            inputs=[
                positions,
                arrays["active"],
                config.count,
                self._sort_size,
                config.max_depth,
                arrays["bounds"],
                arrays["body_keys"],
                arrays["sorted_keys"],
                arrays["sorted_indices"],
            ],
            device=device,
        )
        width = 2
        while width <= self._sort_size:
            stride = width // 2
            while stride > 0:
                wp.launch(
                    kernels.bitonic_step,
                    dim=self._sort_size,
                    inputs=[
                        arrays["sorted_keys"],
                        arrays["sorted_indices"],
                        arrays["stable_ids"],
                        self._sort_size,
                        stride,
                        width,
                    ],
                    device=device,
                )
                stride //= 2
            width *= 2

        wp.launch(kernels.clear_diag, dim=8, inputs=[arrays["diag"]], device=device)
        wp.launch(
            kernels.clear_tree,
            dim=self._total_nodes,
            inputs=[
                arrays["node_prefix"],
                arrays["node_levels"],
                arrays["node_mass"],
                arrays["node_moment_x"],
                arrays["node_moment_y"],
                arrays["node_moment_z"],
                arrays["node_qxx"],
                arrays["node_qxy"],
                arrays["node_qxz"],
                arrays["node_qyy"],
                arrays["node_qyz"],
                arrays["node_qzz"],
                arrays["node_count"],
                arrays["node_start"],
                arrays["node_end"],
                self._sort_size,
            ],
            device=device,
        )
        wp.launch(
            kernels.clear_slot_nodes,
            dim=config.count * (config.max_depth + 1),
            inputs=[arrays["slot_nodes"]],
            device=device,
        )
        wp.launch(
            kernels.clear_node_cursor,
            dim=1,
            inputs=[arrays["node_cursor"]],
            device=device,
        )
        wp.launch(
            kernels.build_sparse_topology,
            dim=config.count,
            inputs=[
                arrays["sorted_keys"],
                arrays["sorted_indices"],
                config.count,
                config.max_depth,
                self._total_nodes,
                arrays["slot_nodes"],
                arrays["node_prefix"],
                arrays["node_levels"],
                arrays["node_start"],
                arrays["node_end"],
                arrays["node_cursor"],
                arrays["diag"],
            ],
            device=device,
        )
        # Node totals, bottom up. `build_sparse_topology` gives every level a node for
        # every distinct prefix, and a level L+1 node's slot range partitions its parent's,
        # so one launch over all `_total_nodes` would re-sum all N bodies once per level and
        # give the root's N-body sum to a single thread. Summing the deepest level from
        # bodies and every shallower level from its own children is the same arithmetic in
        # O(N + nodes) work and O(depth) launches. `node_prefix` fixes the node id of level
        # L at `1 + (L - 1) * count`, so each level is a contiguous `dim=count` launch.
        quadrupoles = int(config.quadrupoles)
        wp.launch(
            kernels.accumulate_sparse_leaf_nodes,
            dim=config.count,
            inputs=[
                positions,
                masses,
                arrays["active"],
                arrays["sorted_indices"],
                config.count,
                1 + (config.max_depth - 1) * config.count,
                arrays["node_start"],
                arrays["node_end"],
                arrays["node_mass"],
                arrays["node_moment_x"],
                arrays["node_moment_y"],
                arrays["node_moment_z"],
                arrays["node_qxx"],
                arrays["node_qxy"],
                arrays["node_qxz"],
                arrays["node_qyy"],
                arrays["node_qyz"],
                arrays["node_qzz"],
                quadrupoles,
                arrays["node_count"],
            ],
            device=device,
        )
        for level in range(config.max_depth - 1, -1, -1):
            wp.launch(
                kernels.reduce_sparse_node_level,
                dim=config.count if level > 0 else 1,
                inputs=[
                    config.count,
                    level,
                    1 + (level - 1) * config.count if level > 0 else 0,
                    arrays["slot_nodes"],
                    arrays["node_start"],
                    arrays["node_end"],
                    arrays["node_mass"],
                    arrays["node_moment_x"],
                    arrays["node_moment_y"],
                    arrays["node_moment_z"],
                    arrays["node_qxx"],
                    arrays["node_qxy"],
                    arrays["node_qxz"],
                    arrays["node_qyy"],
                    arrays["node_qyz"],
                    arrays["node_qzz"],
                    quadrupoles,
                    arrays["node_count"],
                ],
                device=device,
            )
        # Both accumulators leave raw second moments in the q arrays; this is the single
        # conversion to the traceless form the traversal reads, and it zeroes empty nodes.
        wp.launch(
            kernels.finalize_sparse_node_quadrupoles,
            dim=self._total_nodes,
            inputs=[
                arrays["node_mass"],
                arrays["node_qxx"],
                arrays["node_qxy"],
                arrays["node_qxz"],
                arrays["node_qyy"],
                arrays["node_qyz"],
                arrays["node_qzz"],
                quadrupoles,
            ],
            device=device,
        )
        wp.launch(
            kernels.collect_tree_diag,
            dim=self._total_nodes,
            inputs=[
                arrays["node_count"],
                arrays["node_levels"],
                config.max_depth,
                arrays["diag"],
            ],
            device=device,
        )

    def _record_traversal(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        arrays = self._arrays
        config = self.config
        device = self.device_context.device
        wp.launch(
            kernels.reset_traversal_diag, dim=1, inputs=[arrays["diag"]], device=device
        )
        wp.launch(
            kernels.clear_accelerations,
            dim=config.count,
            inputs=[arrays["accelerations"]],
            device=device,
        )
        wp.launch(
            kernels.tree_acceleration,
            dim=config.count,
            inputs=[
                self._require_positions(),
                self._require_masses(),
                arrays["active"],
                arrays["body_keys"],
                arrays["sorted_indices"],
                config.count,
                config.gravitational_constant,
                config.softening,
                config.theta,
                config.max_depth,
                config.leaf_capacity,
                config.stack_capacity,
                config.capacity,
                int(config.quadrupoles),
                arrays["bounds"],
                arrays["node_prefix"],
                arrays["slot_nodes"],
                arrays["node_mass"],
                arrays["node_moment_x"],
                arrays["node_moment_y"],
                arrays["node_moment_z"],
                arrays["node_qxx"],
                arrays["node_qxy"],
                arrays["node_qxz"],
                arrays["node_qyy"],
                arrays["node_qyz"],
                arrays["node_qzz"],
                arrays["node_count"],
                arrays["node_start"],
                arrays["node_end"],
                arrays["stack_nodes"],
                arrays["stack_levels"],
                arrays["accelerations"],
                arrays["diag"],
            ],
            device=device,
        )

    def _record_direct(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        config = self.config
        wp.launch(
            kernels.direct_acceleration,
            dim=config.count,
            inputs=[
                self._require_positions(),
                self._require_masses(),
                self._arrays["active"],
                config.count,
                config.gravitational_constant,
                config.softening,
                self._arrays["direct_accelerations"],
            ],
            device=self.device_context.device,
        )

    def _require_wp(self):
        if self._wp is None:
            raise DeviceRequirementError("SelfGravityRuntime has no device context")
        return self._wp

    def _require_kernels(self):
        if self._kernels is None:
            raise DeviceRequirementError("SelfGravityRuntime kernels are not compiled")
        return self._kernels

    def _require_gravity_kernels(self):
        if self._gravity_kernels is None:
            raise DeviceRequirementError("SelfGravityRuntime kernels are not compiled")
        return self._gravity_kernels

    def _require_positions(self):
        if self._positions is None:
            raise DeviceRequirementError("SelfGravityRuntime has no bound arrays")
        return self._positions

    def _require_masses(self):
        if self._masses is None:
            raise DeviceRequirementError("SelfGravityRuntime has no bound arrays")
        return self._masses


def _gravity_kernels(wp, precision: Precision = Precision.FLOAT32):
    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, _warp_func = _registrars(wp, scalar, vector)
    two = scalar(2.0)

    @warp_kernel
    def apply_gravity(
        gravity: wp.array(dtype=vector),
        accelerations: wp.array(dtype=vector),
        particle_timesteps: wp.array(dtype=scalar),
        softening: scalar,
        timestep_accuracy: scalar,
    ):
        """Add the gravitational acceleration to the total and bound the step by it.

        The hydrodynamic step is a sound crossing time, which is the right bound for a
        pressure wave and no bound at all on a cold region falling together: at the start
        of a collapse the sound speed is small precisely where the acceleration is large.
        The gravitational bound is the time to fall a fixed fraction of the softening
        length from rest, `sqrt(2 eta eps / |a|)`, so the two together bound both ways a
        particle's state can change quickly. A particle feeling no gravity at all keeps the
        hydrodynamic step rather than an infinite one.
        """

        i = wp.tid()
        a = gravity[i]
        accelerations[i] = accelerations[i] + a
        magnitude = wp.length(a)
        if magnitude > scalar(0.0):
            limit = wp.sqrt(two * timestep_accuracy * softening / magnitude)
            particle_timesteps[i] = wp.min(particle_timesteps[i], limit)

    return _SelfGravityKernels(apply_gravity=apply_gravity)


@dataclass(frozen=True)
class _SelfGravityKernels:
    apply_gravity: Any
