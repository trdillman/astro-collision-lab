"""GPU-resident sparse occupied-prefix octree Barnes-Hut backend."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

from .broadphase import (
    BroadphaseConfig,
    allocate_broadphase_arrays,
    broadphase_kernels,
)
from .emitters import (
    EmitterSpec,
    emitter_kernels,
    released_at_frame,
    validate_slot_range,
)
from .force_fields import ExternalFieldSpec, field_arrays, field_kernels
from .model import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_TREE_MAX_DEPTH,
    DEFAULT_TREE_STACK_CAPACITY,
    BindingDescriptor,
)
from .nbody import (
    CollisionMode,
    Precision,
    _dtypes,
    _registrars,
    launch_hold_pinned,
    pinned_arrays,
)
from .nbody import _kernels as _nbody_kernels
from .runtime_foundation import (
    DeviceContext,
    DeviceRequirementError,
    FramePrimitiveProgress,
)
from .sampling import BodySet, expand_bindings

# Ceiling on the node arrays, stated as a storage budget rather than a round number.
# The build costs 24 fixed bytes a node (`node_levels` int32, `node_prefix` int64, and the
# int32 `node_count`/`node_start`/`node_end`) plus ten scalars — mass, the three moment
# components, and the six quadrupole components of ADR 0026. The force walk's depth-first
# copy adds a 16-byte link and ten more scalars (ADR 0132). That is 120 bytes per node at
# float32 and 200 at float64, so this cap holds node storage under 1.8 GB (2 GiB) in the
# worst case. Nothing is allocated by the cap itself: a runtime allocates
# `sparse_node_capacity(count, max_depth)` for the configuration it was given, and this
# only bounds which configurations are admissible. See ADR 0029, restated by ADR 0132 and
# ADR 0134, which raised it from 5,000,000 with the default depth so every body count the
# depth-9 default admitted (555,555) is still admitted at depth 16.
MAX_TREE_NODES = 9_000_000
MAX_TREE_BODIES = (1 << 24) - 1
MAX_TREE_DEPTH = 16
# The shipped depth (`model.DEFAULT_TREE_MAX_DEPTH`, imported above) is this maximum, per
# ADR 0134. Its stack is the least the depth needs, and the product walk is stackless
# (ADR 0132), so raising it allocates nothing.
# One authored pure-gravity substep, in order. Each phase is a fixed number of launches:
# the node moments are a leaf pass plus one reduction per level (ADR 0131), so the
# sequence no longer grows with the body count.
FRAME_PRIMITIVE_PHASES = (
    "kick_drift",
    "bounds_sort",
    "topology_clear_build",
    "node_moments",
    "tree_diagnostic_collection",
    "acceleration",
    "kick_diagnostics",
)
# The phases that build the tree, which the two-rate step runs once a frame (ADR 0144).
_TREE_BUILD_PHASES = frozenset(
    ("bounds_sort", "topology_clear_build", "node_moments", "tree_diagnostic_collection")
)
# ADR 0144: the two-rate step. A body carrying at least this share of the system's mass is a
# dominant mass, whose pull is summed directly on every substep; the rest -- the light bodies
# -- feel each other through the tree once a frame. It applies only when there are at most
# `MAX_DOMINANT_BODIES` of them and the light bodies together carry no more than
# `LIGHT_MASS_CEILING` of the mass: then the light bodies' mutual pull is at most a twentieth
# of the dominant pull, and so is the rate it changes at.
DOMINANT_MASS_FRACTION = 0.01
LIGHT_MASS_CEILING = 0.05
MAX_DOMINANT_BODIES = 8


def dominant_body_indices(masses: Any, active: Any) -> tuple[int, ...]:
    """The bodies the two-rate step treats as dominant, or ``()`` when it does not apply.

    Pure host arithmetic on the loaded masses, so the decision is made once per load and
    is the same on every machine. See `DOMINANT_MASS_FRACTION` for the rule.
    """

    live = [
        (index, float(mass))
        for index, (mass, flag) in enumerate(zip(masses, active, strict=True))
        if flag
    ]
    total = sum(mass for _, mass in live)
    if not live or not total > 0.0:
        return ()
    dominant = tuple(index for index, mass in live if mass >= DOMINANT_MASS_FRACTION * total)
    if not 1 <= len(dominant) <= MAX_DOMINANT_BODIES or len(dominant) == len(live):
        return ()
    light = total - sum(float(masses[index]) for index in dominant)
    if light > LIGHT_MASS_CEILING * total:
        return ()
    return dominant


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def sparse_node_capacity(count: int, max_depth: int) -> int:
    """Worst-case occupied prefixes: one root plus one prefix per body and level."""

    if count < 1:
        raise ValueError("count must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    return 1 + count * max_depth


@dataclass(frozen=True)
class BarnesHutConfig:
    capacity: int
    count: int
    gravitational_constant: float
    dt: float
    theta: float = 0.5
    substeps: int = 1
    softening: float = 0.0
    max_depth: int = DEFAULT_TREE_MAX_DEPTH
    leaf_capacity: int = 8
    stack_capacity: int = DEFAULT_TREE_STACK_CAPACITY
    node_capacity: int = 0
    collision_mode: CollisionMode = CollisionMode.IGNORE
    event_capacity: int = 1024
    precision: Precision = Precision.FLOAT32
    # External force fields (ADR 0045), same contract as NBodyConfig.external_fields:
    # applied after the tree traversal into the same accelerations array.
    external_fields: tuple[ExternalFieldSpec, ...] = ()
    # Emitter release schedules (ADR 0037 3.6), same contract as NBodyConfig.emitters: the
    # slots are already loaded and held inactive, and the per-frame activation runs outside
    # the captured substep in exactly the same place the direct lane runs it.
    emitters: tuple[EmitterSpec, ...] = ()
    quadrupoles: bool = True
    """Include the traceless second multipole moment in every accepted cell.

    ``False`` restores the monopole-only expansion bit-for-bit: the acceptance test,
    the interaction set, the traversal order, and the monopole arithmetic are all
    unchanged, so the two settings differ only by the added term. That is what makes
    an A/B accuracy measurement of the term itself meaningful rather than a comparison
    of two different walks.
    """

    def validate(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 < self.count <= self.capacity:
            raise ValueError("count must be in (0, capacity]")
        if self.count > MAX_TREE_BODIES:
            # The force walk packs a node's body count into the low 24 bits of one
            # word and its level into the rest (ADR 0132).
            raise ValueError(f"count must be at most {MAX_TREE_BODIES}")
        if self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        if self.softening < 0.0:
            raise ValueError("softening must be non-negative")
        if not 1 <= self.max_depth <= MAX_TREE_DEPTH:
            raise ValueError(f"max_depth must be in [1, {MAX_TREE_DEPTH}]")
        if self.leaf_capacity < 1:
            raise ValueError("leaf_capacity must be positive")
        if self.event_capacity < 1:
            raise ValueError("event_capacity must be positive")
        for field in self.external_fields:
            field.validate()
        for spec in self.emitters:
            spec.validate()
            validate_slot_range(spec, self.count)
        if self.collision_mode == CollisionMode.ABSORB:
            # Fail closed at validation: the tree lane's collision path does not support
            # absorption (it is a direct-lane feature, ADR 0037 §3.3). Falling through
            # would mis-simulate rather than absorb.
            raise ValueError(
                "the tree lane does not support CollisionMode.ABSORB; "
                "absorption is a direct-lane feature"
            )
        required_stack = 1 + 7 * self.max_depth
        if self.stack_capacity < required_stack:
            raise ValueError(
                f"stack_capacity {self.stack_capacity} is below required {required_stack}"
            )
        nodes = self.node_capacity or sparse_node_capacity(self.count, self.max_depth)
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
class BarnesHutDiagnostics:
    active_count: int
    node_count: int
    leaf_count: int
    max_depth: int
    stack_high_water: int
    overflow: bool
    interactions: int
    build_ms: float
    traversal_ms: float
    step_ms: float
    sort_size: int
    collision_events: int
    collision_overflow: bool
    collision_broadphase: str
    collision_grid_overflow: bool
    capacity_overflow: bool = False
    sort_algorithm: str = "gpu_bitonic_key_stable_id"
    submission: str = "eager"
    traversal_order: str = "morton_slot"


class BarnesHutRuntime:
    """Own a GPU linear octree, traversal scratch, and KDK integration state."""

    def __init__(self, device_context: DeviceContext, config: BarnesHutConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.step_index = 0
        # Frames finished so far. Authored frame numbers come from this counter, not from
        # `step_index // substeps`, which mis-numbers frames once `substeps` changes (SR-29).
        self.frame_index = 0
        self.last_build_ms = 0.0
        self.last_traversal_ms = 0.0
        self.last_step_ms = 0.0
        self.last_solver_primitive_ms = 0.0
        self.full_state_playback_readbacks = 0
        self.explicit_snapshot_readbacks = 0
        self.explicit_validation_readbacks = 0
        self._wp: Any | None = None
        self._np: Any | None = None
        self._kernels: Any | None = None
        self._collision_kernels: Any | None = None
        self._field_kernels: Any | None = None
        self._emitter_kernels: Any | None = None
        self._broadphase_config = BroadphaseConfig.for_capacity(config.capacity)
        self._broadphase_kernels: Any | None = None
        self._gravity_service = False
        self._arrays: dict[str, Any] = {}
        self._graphs: dict[str, Any] = {}
        # ADR 0226: the last complete tree's report counters (diag slots 0, 1, 5, 6).
        self._tree_report: tuple[int, int, int, int] | None = None
        self._stable_id_order: tuple[int, ...] = ()
        self._frame_primitive_started_at: float | None = None
        self._frame_primitive_cursor = 0
        self.last_frame_primitive_phase = ""
        self._total_nodes = config.node_capacity or sparse_node_capacity(
            config.count, config.max_depth
        )
        self._sort_size = _next_power_of_two(config.capacity)
        # The walk offsets' two-pass scan: sqrt(count) chunks of sqrt(count) slots, so
        # neither pass has a thread sum more than ~sqrt(count) values (ADR 0132).
        self._walk_chunk = max(1, math.isqrt(config.count - 1) + 1)
        self._walk_chunks = -(-config.count // self._walk_chunk)
        # ADR 0144's two-rate step: the dominant bodies of the loaded masses, and where the
        # current frame is. Empty means the single-rate step, unchanged.
        self._dominant: tuple[int, ...] = ()
        self._host_masses: Any | None = None
        self._host_active: Any | None = None
        self._frame_substep = 0

    @property
    def two_rate(self) -> bool:
        """Whether this runtime steps its light bodies' mutual gravity once a frame (ADR 0144)."""

        return bool(self._dominant)

    @property
    def dominant_bodies(self) -> tuple[int, ...]:
        return self._dominant

    @property
    def frame_primitive_substeps_differ(self) -> bool:
        """Whether a frame's substeps do unlike work, so calibration must sample a whole frame.

        Under the two-rate step only the frame's last substep builds and walks the tree.
        """

        return self.two_rate

    def _configure_two_rate(self) -> None:
        """Decide the two-rate step from the loaded masses and fill the tree's mass array.

        Structural conditions first: collisions, emitters and fields each change masses,
        activity or accelerations inside a frame, and one substep has nothing to split.
        """

        dominant: tuple[int, ...] = ()
        if self._uses_internal_work_slices and self.config.substeps >= 2:
            dominant = dominant_body_indices(self._host_masses, self._host_active)
        if dominant != self._dominant:
            self._graphs.clear()
        self._dominant = dominant
        self._frame_substep = 0
        if not dominant:
            # The single-rate step reads `masses` and never the tree's own copy.
            return
        wp = self._require_wp()
        scalar, _vector, state_dtype = _dtypes(wp, self.config.precision)
        device = self.device_context.device
        tree_masses = self._np.zeros(self.config.capacity, dtype=state_dtype)
        tree_masses[: self.config.count] = self._host_masses
        indices = self._np.full(MAX_DOMINANT_BODIES, -1, dtype=self._np.int32)
        for position, index in enumerate(dominant):
            tree_masses[index] = 0.0
            indices[position] = index
        wp.copy(self._arrays["tree_masses"], wp.array(tree_masses, dtype=scalar, device=device))
        wp.copy(
            self._arrays["dominant_indices"],
            wp.array(indices, dtype=wp.int32, device=device),
        )

    @property
    def _tree_masses(self) -> Any:
        """The masses the tree is built from: the light bodies' only, under the two-rate step."""

        return self._arrays["tree_masses" if self._dominant else "masses"]

    @property
    def frame_transaction_resumable(self) -> bool:
        """Yield between tree/KDK substeps only without an emitter finalization tail."""

        return not self.config.emitters

    @property
    def frame_primitive_cursor(self) -> int:
        """Host-known coarse phase coordinate within the current Tree substep."""

        return self._frame_primitive_cursor

    @property
    def _uses_internal_work_slices(self) -> bool:
        return (
            self.config.collision_mode == CollisionMode.IGNORE
            and not self.config.emitters
            and not self.config.external_fields
        )

    @property
    def frame_primitive_calibration_phases(self) -> tuple[str, ...]:
        """Exact authored-substep sequence calibration must observe in order."""

        if not self._uses_internal_work_slices:
            return ("solver_substep",)
        return FRAME_PRIMITIVE_PHASES

    @property
    def frame_primitive_admission_ready(self) -> bool:
        """Whether primitive calibration may currently authorize playback.

        ADR 0061 Step 6 admits pure-gravity work slices only through ProductRuntime's
        complete ordered-phase calibration and conservative synchronized maximum.
        Configurations that adapt one whole substep do not inherit this readiness; their
        existing legacy admission or refusal remains independent of the work-slice proof.
        """

        return self._uses_internal_work_slices

    @property
    def sort_size(self) -> int:
        return self._sort_size

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    def load_bindings(self, bindings: tuple[BindingDescriptor, ...]) -> None:
        self.load_bodies(
            expand_bindings(
                bindings,
                gravitational_constant=self.config.gravitational_constant,
                softening=self.config.softening,
            )
        )

    def load_bodies(self, bodies: BodySet, *, gravity_service: bool = False) -> None:
        """Allocate the tree's device state for ``bodies``.

        ``gravity_service=True`` (ADR 0218) loads it only to be driven through
        :meth:`record_gravity_pass`: the N-body kick and collision kernel modules, which the
        integrator and the broadphase need and a gravity pass does not, are not compiled or
        loaded, and the frame/substep entry points refuse to run.
        """

        count = len(bodies)
        if count != self.config.count:
            raise ValueError("body count must match BarnesHutConfig.count")
        self._frame_primitive_started_at = None
        self.device_context.initialize()
        self._wp = self.device_context.wp
        self._np = __import__("numpy")
        self._gravity_service = bool(gravity_service)
        self._kernels = _kernels(self._wp, self.config.precision)
        if self._gravity_service:
            self._collision_kernels = None
            self._broadphase_kernels = None
        else:
            self._collision_kernels = _nbody_kernels(self._wp, self.config.precision)
            self._broadphase_kernels = broadphase_kernels(self._wp, self.config.precision)
        np = self._np
        wp = self._wp
        scalar, vector, state_dtype = _dtypes(wp, self.config.precision)
        quad = _quad_dtype(wp, self.config.precision)
        device = self.device_context.device
        positions = np.zeros((self.config.capacity, 3), dtype=state_dtype)
        velocities = np.zeros((self.config.capacity, 3), dtype=state_dtype)
        masses = np.zeros(self.config.capacity, dtype=state_dtype)
        radii = np.zeros(self.config.capacity, dtype=state_dtype)
        active = np.zeros(self.config.capacity, dtype=np.int32)
        stable_ids = np.zeros(self.config.capacity, dtype=np.int64)
        self._stable_id_order = tuple(int(value) for value in bodies.stable_ids)
        positions[:count] = bodies.positions
        velocities[:count] = bodies.velocities
        masses[:count] = bodies.masses
        radii[:count] = bodies.radii
        active[:count] = bodies.active
        stable_ids[:count] = bodies.stable_ids

        self._arrays = {
            "positions": wp.array(positions, dtype=vector, device=device),
            "velocities": wp.array(velocities, dtype=vector, device=device),
            "accelerations": wp.zeros(self.config.capacity, dtype=vector, device=device),
            "direct_accelerations": wp.zeros(
                self.config.capacity, dtype=vector, device=device
            ),
            "masses": wp.array(masses, dtype=scalar, device=device),
            "radii": wp.array(radii, dtype=scalar, device=device),
            "active": wp.array(active, dtype=wp.int32, device=device),
            "stable_ids": wp.array(stable_ids, dtype=wp.int64, device=device),
            "body_keys": wp.zeros(self.config.capacity, dtype=wp.int64, device=device),
            "sorted_keys": wp.zeros(self._sort_size, dtype=wp.int64, device=device),
            "sorted_indices": wp.zeros(self._sort_size, dtype=wp.int32, device=device),
            "bounds": wp.zeros(6, dtype=scalar, device=device),
            "node_levels": wp.zeros(self._total_nodes, dtype=wp.int32, device=device),
            "node_prefix": wp.zeros(self._total_nodes, dtype=wp.int64, device=device),
            "node_mass": wp.zeros(self._total_nodes, dtype=scalar, device=device),
            "node_moment_x": wp.zeros(
                self._total_nodes, dtype=scalar, device=device
            ),
            "node_moment_y": wp.zeros(
                self._total_nodes, dtype=scalar, device=device
            ),
            "node_moment_z": wp.zeros(
                self._total_nodes, dtype=scalar, device=device
            ),
            # Traceless quadrupole about each node's own centre of mass, stored as the
            # six independent components of a symmetric tensor. Allocated unconditionally
            # so the kernel signature does not depend on the option: 24 bytes per node
            # against the 40 the monopole topology already costs.
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
                self.config.count * (self.config.max_depth + 1),
                dtype=wp.int32,
                device=device,
            ),
            # ADR 0132: the force walk's depth-first node array, rebuilt each substep in
            # the acceleration phase. It replaces the per-body traversal stack
            # (capacity * stack_capacity words twice), which only the reference walk
            # still allocates.
            "walk_bodies": wp.zeros(self.config.count, dtype=quad, device=device),
            "walk_first_level": wp.zeros(self.config.count, dtype=wp.int32, device=device),
            "walk_node_counts": wp.zeros(self.config.count, dtype=wp.int32, device=device),
            "walk_offsets": wp.zeros(self.config.count + 1, dtype=wp.int32, device=device),
            "walk_chunk_totals": wp.zeros(
                self._walk_chunks, dtype=wp.int32, device=device
            ),
            "walk_links": wp.zeros(self._total_nodes, dtype=wp.vec4i, device=device),
            # ADR 0226: what the opening test reads (mass centre and reach) is one record, and
            # what only an accepted cell reads (mass and quadrupole) is two more.
            "walk_cells": wp.zeros(self._total_nodes, dtype=quad, device=device),
            "walk_mass_diagonal": wp.zeros(self._total_nodes, dtype=quad, device=device),
            "walk_off_diagonal": wp.zeros(self._total_nodes, dtype=vector, device=device),
            "diag": wp.zeros(8, dtype=wp.int64, device=device),
            "idiag": wp.zeros(4, dtype=wp.int32, device=device),
            "idiag_totals": wp.zeros(5, dtype=wp.int32, device=device),
            "claimed": wp.zeros(self.config.capacity, dtype=wp.int32, device=device),
            "event_i": wp.zeros(
                self.config.event_capacity, dtype=wp.int32, device=device
            ),
            "event_j": wp.zeros(
                self.config.event_capacity, dtype=wp.int32, device=device
            ),
            "relative_errors": wp.zeros(
                self.config.capacity, dtype=scalar, device=device
            ),
            # ADR 0144. Allocated whatever the step, so a load that switches it on or off
            # changes which arrays a graph reads, never whether they exist.
            "tree_masses": wp.array(masses, dtype=scalar, device=device),
            "dominant_indices": wp.array(
                np.full(MAX_DOMINANT_BODIES, -1, dtype=np.int32),
                dtype=wp.int32,
                device=device,
            ),
            "slow_accelerations": wp.zeros(
                self.config.capacity, dtype=vector, device=device
            ),
            "total_accelerations": wp.zeros(
                self.config.capacity, dtype=vector, device=device
            ),
        }
        if bodies.pinned is not None and any(int(value) != 0 for value in bodies.pinned):
            self._arrays.update(
                pinned_arrays(
                    wp, device, np, bodies, state_dtype, vector, self.config.capacity
                )
            )
        if self.config.collision_mode != CollisionMode.IGNORE:
            self._arrays.update(
                {
                    f"collision_grid_{name}": array
                    for name, array in allocate_broadphase_arrays(
                        wp,
                        device,
                        self.config.precision,
                        self._broadphase_config,
                    ).items()
                }
            )
        if self.config.external_fields:
            # Same baked-array contract as the direct lane: stable device pointers so
            # the launch in _record_accelerations is legal inside graph capture.
            self._field_kernels = field_kernels(wp, self.config.precision)
            self._arrays.update(
                field_arrays(wp, self.config.precision, self.config.external_fields, device)
            )
        if self.config.emitters:
            # Precision-free and array-free, unlike the fields above: the slot block is
            # already in "active" and the only launch argument is a host integer.
            self._emitter_kernels = emitter_kernels(wp)
        self._graphs.clear()
        self._tree_report = None
        self._host_masses = masses[:count].copy()
        self._host_active = active[:count].copy()
        self._dominant = ()
        self._configure_two_rate()
        # Before the build, not after: a body an emitter releases at frame 0 has to be in
        # the tree the first traversal walks, or it would contribute no gravity for a whole
        # substep and receive none.
        self._release_emitters(0, None)
        self.rebuild_tree()
        self._compute_accelerations()
        self._frame_primitive_cursor = 0
        self.last_frame_primitive_phase = ""

    def array_dtypes(self) -> dict[str, str]:
        """Element type of every live device array.

        Validation asserts the precision contract against the arrays that exist rather
        than against a self-reported label. Unlike the direct solver, the tree's node
        reductions follow ``config.precision``: they are part of the approximation being
        measured, not part of the measurement.
        """

        return {
            name: getattr(array.dtype, "__name__", str(array.dtype))
            for name, array in self._arrays.items()
        }

    def display_source(self) -> tuple[Any, int, Precision]:
        """The live position array a display stage samples, its length, and its precision."""

        if not self._arrays:
            raise DeviceRequirementError("BarnesHutRuntime has no loaded bindings")
        return self._arrays["positions"], self.config.count, self.config.precision

    def display_channels(self) -> dict[str, Any]:
        """Every device array a display stage may sample, keyed by channel name.

        Same keys as :meth:`DirectNBodyRuntime.display_channels`, and that is a
        requirement rather than a coincidence: Auto swaps between the two lanes on body
        count alone, so a render spec that works at 64 bodies has to keep working at
        131072. A channel this lane offered and direct did not would make the swap
        visible in the viewport. Like the direct lane, this lane carries no density, so
        the mass/density size derivation cannot run here -- it does not need to:
        ``load_bodies`` already keeps a real per-body ``radii`` array alive for its own
        collision/merge kernels, and that radius IS the physical size of the body, so
        `PHYSICAL` particle size is served straight from it rather than substituted with a
        number (`adr/0023-gpu-particle-renderer-viewport-adoption.md` §7 amendment).
        """

        if not self._arrays:
            raise DeviceRequirementError("BarnesHutRuntime has no loaded bindings")
        return {
            "positions": self._arrays["positions"],
            "velocities": self._arrays["velocities"],
            "masses": self._arrays["masses"],
            "radii": self._arrays["radii"],
            # ``active`` and ``stable_ids`` for the reason `DirectNBodyRuntime` gives, and
            # under the same-keys requirement this docstring opens with: a merge ghost that
            # Auto removed below 131072 bodies and kept above it would be the worst of both.
            "active": self._arrays["active"],
            "stable_ids": self._arrays["stable_ids"],
        }

    def apply_parameters(
        self,
        *,
        gravitational_constant: float,
        softening: float,
        dt: float,
        substeps: int,
        field_strengths: tuple[float, ...] | None = None,
    ) -> None:
        """Adopt prospective solver parameters without disturbing solved state.

        ``field_strengths`` (ADR 0165) rewrites the external fields' device strengths in
        place; a strength-only edit keeps the captured graphs, which read that array.

        Same contract as :meth:`DirectNBodyRuntime.apply_parameters`, and it has to be,
        because Auto swaps between the two on body count alone: without this a scene that
        keeps its solved steps through a ``G`` edit at 64 bodies would discard them at
        131072, which is the same edit reading differently purely because of scale.

        The one extra step is dropping the captured graphs. A capture freezes the scalars
        it was recorded with — ``sub_dt``, ``G``, ``softening`` are literal kernel
        arguments — so replaying one would silently keep stepping on the old values. The
        arrays the graphs point at are untouched, so clearing the cache costs one
        re-record on the next submission and nothing else; at 65536 bodies that is the
        8.8 ms of host submission :meth:`_graph` documents, against a human-rate edit.
        """

        from .force_fields import specs_with_strengths, write_field_strengths

        new_fields = self.config.external_fields
        if field_strengths is not None:
            new_fields = specs_with_strengths(self.config.external_fields, field_strengths)
        candidate = replace(
            self.config,
            gravitational_constant=float(gravitational_constant),
            softening=float(softening),
            dt=float(dt),
            substeps=int(substeps),
            external_fields=new_fields,
        )
        candidate.validate()
        scalars_changed = (
            candidate.gravitational_constant,
            candidate.softening,
            candidate.dt,
            candidate.substeps,
        ) != (
            self.config.gravitational_constant,
            self.config.softening,
            self.config.dt,
            self.config.substeps,
        )
        strengths_changed = new_fields != self.config.external_fields
        self.config = candidate
        if field_strengths is not None:
            write_field_strengths(self._wp, self._arrays, new_fields)
        # A strength-only edit keeps the graphs: they read the field array live.
        if scalars_changed:
            self._graphs.clear()
        # ADR 0144: `substeps` decides whether the two-rate step applies, and its slow
        # acceleration was computed with the old G and softening (and field strengths).
        # Rebuild both from the solved positions whenever the two-rate step was or now is
        # on; the single-rate step keeps its existing behaviour.
        if not self._arrays:
            return
        if not (scalars_changed or strengths_changed):
            return
        was_two_rate = self.two_rate
        self._configure_two_rate()
        if was_two_rate or self.two_rate:
            self.rebuild_tree()
            self._compute_accelerations()

    def begin_frame(self) -> None:
        if not self._arrays:
            raise DeviceRequirementError("BarnesHutRuntime has no loaded bindings")
        self._refuse_gravity_service()
        wp = self._require_wp()
        device = self.device_context.device
        self._frame_primitive_started_at = time.perf_counter()
        self._frame_substep = 0
        wp.launch(
            self._collision_kernels.clear_idiag,
            dim=5,
            inputs=[self._arrays["idiag_totals"]],
            device=device,
        )

    def advance_frame_substep(self) -> None:
        """Execute, synchronize, and validate exactly one captured tree/KDK substep."""

        if not self._arrays:
            raise DeviceRequirementError("BarnesHutRuntime has no loaded bindings")
        self._refuse_gravity_service()
        wp = self._require_wp()
        device = self.device_context.device
        start = self._frame_primitive_started_at or time.perf_counter()
        self._frame_primitive_started_at = None
        opens, closes = self._two_rate_position()
        if self._dominant:
            self._submit(
                f"substep_two_rate_{int(opens)}{int(closes)}",
                lambda: self._record_two_rate_substep(opens, closes),
            )
        else:
            self._submit("substep", self._record_substep)
        self._flag_nonfinite(closes)
        wp.synchronize_device(device)
        diag = self._arrays["diag"].numpy()
        collision_totals = self._arrays["idiag_totals"].numpy()
        self.last_solver_primitive_ms = (time.perf_counter() - start) * 1000.0
        self._raise_substep_diagnostics(diag, collision_totals)
        self._complete_substep()

    def _refuse_gravity_service(self) -> None:
        if self._gravity_service:
            raise DeviceRequirementError(
                "this BarnesHutRuntime was loaded as a gravity service (ADR 0218) and has no "
                "integrator kernels; use record_gravity_pass"
            )

    def _flag_nonfinite(self, slow_too: bool) -> None:
        names = ("accelerations", "slow_accelerations") if slow_too else ("accelerations",)
        for name in names:
            self._require_wp().launch(
                self._collision_kernels.flag_nonfinite,
                dim=self.config.count,
                inputs=[
                    self._arrays[name],
                    self.config.count,
                    self._arrays["idiag_totals"],
                ],
                device=self.device_context.device,
            )

    def advance_frame_primitive(self) -> FramePrimitiveProgress:
        """Execute one synchronized coarse phase of a pure-gravity Tree substep.

        Unsupported configurations keep the existing one-whole-substep adapter. The
        coarse path is eligible for ADR 0061 Step 6's complete ordered calibration; a
        successful calibration result, rather than this executable surface, gates admission.
        """

        if not self._uses_internal_work_slices:
            self.advance_frame_substep()
            self.last_frame_primitive_phase = "solver_substep"
            return FramePrimitiveProgress(
                phase="solver_substep",
                authored_substep_complete=True,
            )
        if not self._arrays:
            raise DeviceRequirementError("BarnesHutRuntime has no loaded bindings")
        cursor = self._frame_primitive_cursor
        if not 0 <= cursor < len(FRAME_PRIMITIVE_PHASES):
            raise DeviceRequirementError("Tree frame primitive cursor is invalid")
        phase = FRAME_PRIMITIVE_PHASES[cursor]
        record = {
            "kick_drift": self._record_kick_drift,
            "bounds_sort": self._record_bounds_sort,
            "topology_clear_build": self._record_topology_clear_build,
            "node_moments": self._record_node_moments,
            "tree_diagnostic_collection": self._record_tree_diagnostic_collection,
            "acceleration": self._record_accelerations,
            "kick_diagnostics": self._record_final_kick,
        }[phase]
        opens, closes = self._two_rate_position()
        # ADR 0144: under the two-rate step a substep that does not close its frame keeps
        # every phase name, so calibration sees the same order, but its four tree phases
        # submit nothing and its acceleration is the fast force alone.
        idle = bool(self._dominant) and not closes and phase in _TREE_BUILD_PHASES
        # ADR 0226 (superseding ADR 0147's once-a-frame collection): the report counters are
        # read by nobody during play, so the phase keeps its place in the order and submits
        # nothing; `diagnostics()` counts them from the last complete tree when asked. It
        # still refuses an overflowed topology before the walk reads it.
        submits = phase != "tree_diagnostic_collection"
        graph_name = f"substep_{phase}"
        if self._dominant and phase == "kick_drift":
            graph_name = f"substep_kick_drift_two_rate_{int(opens)}"
            record = lambda: self._record_kick_drift(opens_frame=opens)  # noqa: E731
        elif self._dominant and phase == "acceleration":
            graph_name = f"substep_acceleration_two_rate_{int(closes)}"
            record = (
                self._record_accelerations if closes else self._record_fast_accelerations
            )
        elif self._dominant and phase == "kick_diagnostics":
            graph_name = f"substep_kick_diagnostics_two_rate_{int(closes)}"
            record = lambda: self._record_final_kick(closes_frame=closes)  # noqa: E731
        wp = self._require_wp()
        device = self.device_context.device
        start = self._frame_primitive_started_at or time.perf_counter()
        self._frame_primitive_started_at = None
        try:
            # Every phase is a fixed launch sequence, so each keeps one cached graph and
            # the graph cache is independent of body count.
            if not idle:
                if submits:
                    self._submit(graph_name, record)
                if phase == "kick_diagnostics":
                    self._flag_nonfinite(closes)
                wp.synchronize_device(device)
            if phase in {"tree_diagnostic_collection", "acceleration"} and not idle:
                self._raise_tree_overflow()
            if phase == "kick_diagnostics":
                diag = self._arrays["diag"].numpy()
                collision_totals = self._arrays["idiag_totals"].numpy()
                self._raise_substep_diagnostics(diag, collision_totals)
        finally:
            self.last_solver_primitive_ms = (time.perf_counter() - start) * 1000.0

        self.last_frame_primitive_phase = phase
        authored_substep_complete = phase == "kick_diagnostics"
        if authored_substep_complete:
            self._complete_substep()
            self._frame_primitive_cursor = 0
        else:
            self._frame_primitive_cursor += 1
        return FramePrimitiveProgress(
            phase=phase,
            authored_substep_complete=authored_substep_complete,
        )

    def _raise_substep_diagnostics(self, diag: Any, collision_totals: Any) -> None:
        self._raise_tree_overflow(diag)
        if int(collision_totals[3]):
            raise DeviceRequirementError(
                "collision grid bucket overflow; step failed closed"
            )
        if int(collision_totals[1]):
            raise DeviceRequirementError("collision event capacity overflow; step failed closed")
        if int(collision_totals[2]):
            raise DeviceRequirementError("non-finite acceleration detected; step failed closed")

    def finish_frame(self) -> None:
        self.frame_index += 1
        if not self.config.emitters:
            return
        frame = self.frame_index
        if self._release_emitters(frame, frame - 1):
            self.rebuild_tree()
            self._compute_accelerations()
            self._require_wp().synchronize_device(self.device_context.device)

    def step(self, steps: int = 1) -> None:
        if not self._arrays:
            raise DeviceRequirementError("BarnesHutRuntime has no loaded bindings")
        if steps < 1:
            raise ValueError("steps must be positive")
        start = time.perf_counter()
        for _ in range(steps):
            self.begin_frame()
            for _substep in range(self.config.substeps):
                self.advance_frame_substep()
            self.finish_frame()
        self.last_step_ms = (time.perf_counter() - start) * 1000.0

    def _graph(self, name: str, record: Any) -> Any:
        """Capture one launch sequence once so later submissions replay a CUDA graph.

        The build is launch-submission bound, not GPU-work bound: at 65536 bodies the 149
        launches of a rebuild cost 8.8 ms of host submission against 0.015 ms of GPU time,
        and the bitonic sort makes that count grow as ``O(log**2 N)``. Capture is legal here
        because every launch dimension comes from the frozen config, every array is
        allocated once in ``load_bindings``, and nothing inside a recorded sequence
        allocates, synchronizes, or reads back. ``load_bindings`` and ``close`` drop the
        cache because they replace or release the arrays the graphs point at; a snapshot
        restore does not, because it copies into those same arrays.
        """

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

    def _submit(self, name: str, record: Any) -> None:
        """Replay ``record``'s captured graph, or run it eagerly off CUDA."""

        graph = self._graph(name, record)
        if graph is None:
            record()
        else:
            self._require_wp().capture_launch(graph)

    def _hold_pinned(self) -> None:
        """ADR 0158: put every pinned body back at its anchor, stopped; a no-op with none."""

        launch_hold_pinned(
            self._require_wp(),
            self._collision_kernels,
            self._arrays,
            self.config.count,
            self.device_context.device,
        )

    def _record_substep(self) -> None:
        self._record_kick_drift()
        self._resolve_collisions()
        self._hold_pinned()
        self._record_rebuild()
        self._record_accelerations()
        self._record_final_kick()

    def _record_slow_kick(self) -> None:
        """ADR 0144: half a frame's kick from the slow force, at a frame's open or close."""

        wp = self._require_wp()
        wp.launch(
            self._require_kernels().kick,
            dim=self.config.count,
            inputs=[
                self._arrays["velocities"],
                self._arrays["slow_accelerations"],
                self.config.dt * 0.5,
            ],
            device=self.device_context.device,
        )

    def _record_kick_drift(self, *, opens_frame: bool = False) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        sub_dt = self.config.dt / float(self.config.substeps)
        if opens_frame:
            self._record_slow_kick()
        wp.launch(
            kernels.kick,
            dim=self.config.count,
            inputs=[
                self._arrays["velocities"],
                self._arrays["accelerations"],
                sub_dt * 0.5,
            ],
            device=device,
        )
        wp.launch(
            kernels.drift,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["active"],
                self.config.count,
                sub_dt,
            ],
            device=device,
        )
        self._hold_pinned()

    def _record_final_kick(self, *, closes_frame: bool = False) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        sub_dt = self.config.dt / float(self.config.substeps)
        wp.launch(
            kernels.kick,
            dim=self.config.count,
            inputs=[
                self._arrays["velocities"],
                self._arrays["accelerations"],
                sub_dt * 0.5,
            ],
            device=device,
        )
        if closes_frame:
            self._record_slow_kick()
        self._hold_pinned()

    def _two_rate_position(self) -> tuple[bool, bool]:
        """Whether the next substep opens and/or closes its frame under ADR 0144."""

        if not self._dominant:
            return False, False
        position = self._frame_substep
        return position == 0, position == self.config.substeps - 1

    def _substep_closes_frame(self) -> bool:
        """Whether the next substep is its frame's last, under either step."""

        if self._dominant:
            return self._two_rate_position()[1]
        substeps = max(1, int(self.config.substeps))
        return self.step_index % substeps == substeps - 1

    def _complete_substep(self) -> None:
        self.step_index += 1
        if self._dominant:
            self._frame_substep = (self._frame_substep + 1) % self.config.substeps

    def _record_two_rate_substep(self, opens: bool, closes: bool) -> None:
        """One whole ADR 0144 substep: the tree is built and walked only at a frame's close."""

        self._record_kick_drift(opens_frame=opens)
        self._record_fast_accelerations()
        if closes:
            self._record_rebuild()
            self._record_slow_accelerations()
        self._record_final_kick(closes_frame=closes)

    def rebuild_tree(self) -> None:
        """Submit a standalone build and time it.

        The substep path records `_record_rebuild` into its own graph instead of calling
        this, so the only remaining callers want the timing, which needs the synchronize.
        """

        wp = self._require_wp()
        start = time.perf_counter()
        self._submit("build", self._record_rebuild)
        wp.synchronize_device(self.device_context.device)
        self._raise_tree_overflow()
        self.last_build_ms = (time.perf_counter() - start) * 1000.0

    def record_gravity_pass(self, positions: Any, accelerations: Any) -> None:
        """Record one gravity evaluation of ``positions`` into ``accelerations`` (ADR 0218).

        The tree as a service for another solver: copy the caller's positions over this
        runtime's own, rebuild the tree on them, walk it, and copy the accelerations out.
        Every step is a launch or a device copy on the current stream, so a caller may record
        the whole pass into its own CUDA graph; nothing here synchronises or reads back.
        The masses are the ones `load_bodies` was given, in the caller's slot order, and the
        constant, softening and opening angle are this runtime's config. The caller's arrays
        must be ``count`` long with this runtime's vector dtype. Follow it with
        :meth:`record_overflow_flag` to fail closed on a build or walk that ran out of room.
        """

        wp = self._require_wp()
        count = self.config.count
        if len(positions) != count or len(accelerations) != count:
            raise ValueError("gravity pass arrays must hold exactly config.count bodies")
        wp.copy(self._arrays["positions"], positions, count=count)
        self._record_rebuild()
        self._record_walk()
        wp.copy(accelerations, self._arrays["accelerations"], count=count)

    def record_overflow_flag(self, flag: Any) -> None:
        """Record a launch that sets ``flag[0]`` to 1 if the last build or walk overflowed.

        ``flag`` is the caller's device int32 array. The launch only ever raises it, so an
        overflow in any pass between two reads survives to the read; the caller clears it.
        It covers the two conditions :meth:`check_overflow` raises on (node capacity and
        traversal), with no readback, so it is legal inside a CUDA graph capture.
        """

        wp = self._require_wp()
        wp.launch(
            self._require_kernels().fold_tree_overflow,
            dim=1,
            inputs=[self._arrays["diag"], flag],
            device=self.device_context.device,
        )

    def check_overflow(self) -> None:
        """Raise if the last recorded build or walk overflowed its capacity (one readback)."""

        self._raise_tree_overflow()

    def _record_rebuild(self) -> None:
        self._record_bounds_sort()
        self._record_topology_moments()

    def _record_bounds_sort(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device

        wp.launch(kernels.clear_bounds, dim=6, inputs=[self._arrays["bounds"]], device=device)
        wp.launch(
            kernels.reduce_bounds,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["active"],
                self.config.count,
                self._arrays["bounds"],
            ],
            device=device,
        )
        wp.launch(
            kernels.finalize_bounds,
            dim=1,
            inputs=[self._arrays["bounds"]],
            device=device,
        )
        wp.launch(
            kernels.compute_morton,
            dim=self._sort_size,
            inputs=[
                self._arrays["positions"],
                self._arrays["active"],
                self.config.count,
                self._sort_size,
                self.config.max_depth,
                self._arrays["bounds"],
                self._arrays["body_keys"],
                self._arrays["sorted_keys"],
                self._arrays["sorted_indices"],
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
                        self._arrays["sorted_keys"],
                        self._arrays["sorted_indices"],
                        self._arrays["stable_ids"],
                        self._sort_size,
                        stride,
                        width,
                    ],
                    device=device,
                )
                stride //= 2
            width *= 2

    def _record_topology_moments(self) -> None:
        """Record the whole rebuild after the sort: topology and node moments.

        The report counters are not collected here (ADR 0226): `diagnostics()` counts them
        on demand, so a rebuild costs only what the walk needs.
        """

        self._record_topology_clear_build()
        self._record_node_moments()

    def _record_topology_clear_build(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device

        wp.launch(kernels.clear_tree_flags, dim=8, inputs=[self._arrays["diag"]], device=device)
        wp.launch(
            kernels.clear_tree,
            dim=self._total_nodes,
            inputs=[
                self._arrays["node_prefix"],
                self._arrays["node_levels"],
                self._arrays["node_mass"],
                self._arrays["node_moment_x"],
                self._arrays["node_moment_y"],
                self._arrays["node_moment_z"],
                self._arrays["node_qxx"],
                self._arrays["node_qxy"],
                self._arrays["node_qxz"],
                self._arrays["node_qyy"],
                self._arrays["node_qyz"],
                self._arrays["node_qzz"],
                self._arrays["node_count"],
                self._arrays["node_start"],
                self._arrays["node_end"],
                self._sort_size,
            ],
            device=device,
        )
        wp.launch(
            kernels.clear_slot_nodes,
            dim=self.config.count * (self.config.max_depth + 1),
            inputs=[self._arrays["slot_nodes"]],
            device=device,
        )
        wp.launch(
            kernels.clear_node_cursor,
            dim=1,
            inputs=[self._arrays["node_cursor"]],
            device=device,
        )
        wp.launch(
            kernels.build_sparse_topology,
            dim=self.config.count,
            inputs=[
                self._arrays["sorted_keys"],
                self._arrays["sorted_indices"],
                self.config.count,
                self.config.max_depth,
                self._total_nodes,
                self._arrays["slot_nodes"],
                self._arrays["node_prefix"],
                self._arrays["node_levels"],
                self._arrays["node_start"],
                self._arrays["node_end"],
                self._arrays["node_cursor"],
                self._arrays["diag"],
            ],
            device=device,
        )

    def _record_node_moments(self) -> None:
        # Node totals, bottom up (ADR 0131, the same pass ADR 0124 gave the SPH tree).
        # `build_sparse_topology` gives every level a node for every distinct prefix, and a
        # level L+1 node's slot range partitions its parent's, so one launch over all
        # `_total_nodes` would re-sum all N bodies once per level and give the root's N-body
        # sum to a single thread. Summing the deepest level from bodies and every shallower
        # level from its own children is the same arithmetic in O(N + nodes) work and
        # O(depth) launches. `node_prefix` fixes the node id of level L at
        # `1 + (L - 1) * count`, so each level is a contiguous launch of at most `count`
        # threads, clipped to the nodes a smaller `node_capacity` actually allocated.
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        arrays = self._arrays
        config = self.config
        quadrupoles = int(config.quadrupoles)
        moment_arrays = [
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
        ]
        leaf_base = 1 + (config.max_depth - 1) * config.count
        leaf_dim = min(config.count, self._total_nodes - leaf_base)
        if leaf_dim > 0:
            wp.launch(
                kernels.accumulate_sparse_leaf_nodes,
                dim=leaf_dim,
                inputs=[
                    arrays["positions"],
                    self._tree_masses,
                    arrays["active"],
                    arrays["sorted_indices"],
                    config.count,
                    leaf_base,
                    arrays["node_start"],
                    arrays["node_end"],
                    *moment_arrays,
                ],
                device=device,
            )
        for level in range(config.max_depth - 1, -1, -1):
            level_base = 1 + (level - 1) * config.count if level > 0 else 0
            level_dim = min(config.count, self._total_nodes - level_base) if level > 0 else 1
            if level_dim <= 0:
                continue
            wp.launch(
                kernels.reduce_sparse_node_level,
                dim=level_dim,
                inputs=[
                    config.count,
                    level,
                    level_base,
                    arrays["slot_nodes"],
                    arrays["node_start"],
                    arrays["node_end"],
                    *moment_arrays,
                ],
                device=device,
            )
        # Both passes leave raw second moments in the q arrays; this is the single
        # conversion to the traceless form the traversal reads, and it zeroes empty nodes.
        self._record_node_quadrupole_finalize()

    def _record_node_moment_quadrupole_monolithic(self) -> None:
        """The pre-ADR 0131 one-launch accumulation, kept as the tests' reference."""

        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device

        wp.launch(
            kernels.accumulate_sparse_nodes,
            dim=self._total_nodes,
            inputs=[
                self._arrays["positions"],
                self._tree_masses,
                self._arrays["active"],
                self._arrays["sorted_indices"],
                self.config.count,
                self._arrays["node_start"],
                self._arrays["node_end"],
                self._arrays["node_mass"],
                self._arrays["node_moment_x"],
                self._arrays["node_moment_y"],
                self._arrays["node_moment_z"],
                self._arrays["node_qxx"],
                self._arrays["node_qxy"],
                self._arrays["node_qxz"],
                self._arrays["node_qyy"],
                self._arrays["node_qyz"],
                self._arrays["node_qzz"],
                int(self.config.quadrupoles),
                self._arrays["node_count"],
            ],
            device=device,
        )

    def _record_node_quadrupole_finalize(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        wp.launch(
            kernels.finalize_sparse_node_quadrupoles,
            dim=self._total_nodes,
            inputs=[
                self._arrays["node_mass"],
                self._arrays["node_qxx"],
                self._arrays["node_qxy"],
                self._arrays["node_qxz"],
                self._arrays["node_qyy"],
                self._arrays["node_qyz"],
                self._arrays["node_qzz"],
                int(self.config.quadrupoles),
            ],
            device=self.device_context.device,
        )

    def _record_tree_diagnostic_collection(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device

        wp.launch(kernels.clear_tree_report, dim=8, inputs=[self._arrays["diag"]], device=device)
        wp.launch(
            kernels.collect_tree_diag,
            dim=self._total_nodes,
            inputs=[
                self._arrays["node_count"],
                self._arrays["node_levels"],
                self.config.max_depth,
                self._arrays["diag"],
            ],
            device=device,
        )

    def _collect_tree_report(self) -> tuple[int, int, int, int]:
        """Count the last complete tree's nodes, depth, leaves and bodies (ADR 0226).

        The counters touch only diag slots 0, 1, 5 and 6, which no fail-closed check reads,
        so an eager collection between phases changes nothing the solver uses. A tree whose
        topology has been cleared but whose moments are not yet summed (a call between the
        `topology_clear_build` and `node_moments` slices) has an empty root, and then the
        previous complete tree's report is kept, so the report is never empty mid-frame.
        """

        wp = self._require_wp()
        device = self.device_context.device
        self._record_tree_diagnostic_collection()
        wp.synchronize_device(device)
        diag = self._arrays["diag"].numpy()
        report = (int(diag[0]), int(diag[1]), int(diag[5]), int(diag[6]))
        if report[3] == 0 and self._tree_report is not None:
            return self._tree_report
        self._tree_report = report
        return report

    def diagnostics(self) -> BarnesHutDiagnostics:
        wp = self._require_wp()
        wp.synchronize_device(self.device_context.device)
        node_count, max_depth, leaf_count, active_count = self._collect_tree_report()
        diag = self._arrays["diag"].numpy()
        idiag_totals = self._arrays["idiag_totals"].numpy()
        return BarnesHutDiagnostics(
            active_count=active_count,
            node_count=node_count,
            leaf_count=leaf_count,
            max_depth=max_depth,
            stack_high_water=int(diag[2]),
            interactions=int(diag[4]),
            build_ms=self.last_build_ms,
            traversal_ms=self.last_traversal_ms,
            step_ms=self.last_step_ms,
            sort_size=self._sort_size,
            collision_events=int(idiag_totals[0]),
            collision_overflow=bool(idiag_totals[1]),
            collision_broadphase=(
                "disabled"
                if self.config.collision_mode == CollisionMode.IGNORE
                else "all_pairs_fallback"
                if int(idiag_totals[4])
                else "uniform_grid"
            ),
            collision_grid_overflow=bool(idiag_totals[3]),
            capacity_overflow=bool(diag[7]),
            overflow=bool(diag[3] or diag[7]),
            submission="cuda_graph" if any(self._graphs.values()) else "eager",
        )

    def _raise_tree_overflow(self, diag: Any | None = None) -> None:
        values = self._arrays["diag"].numpy() if diag is None else diag
        if int(values[7]):
            raise DeviceRequirementError(
                "Barnes-Hut node capacity overflow; operation failed closed"
            )
        if int(values[3]):
            raise DeviceRequirementError(
                "Barnes-Hut traversal overflow; operation failed closed"
            )

    def benchmark_acceleration(self, repeats: int = 3) -> float:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        elapsed = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            self._compute_accelerations()
            self._require_wp().synchronize_device(self.device_context.device)
            elapsed += (time.perf_counter() - start) * 1000.0
        self.last_traversal_ms = elapsed / float(repeats)
        return self.last_traversal_ms

    def benchmark_direct_acceleration(self, repeats: int = 3) -> float:
        """Measure the direct validation kernel at the current matching state."""
        if repeats < 1:
            raise ValueError("repeats must be positive")
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        elapsed = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            wp.launch(
                kernels.direct_acceleration,
                dim=self.config.count,
                inputs=[
                    self._arrays["positions"],
                    self._arrays["masses"],
                    self._arrays["active"],
                    self.config.count,
                    self.config.gravitational_constant,
                    self.config.softening,
                    self._arrays["direct_accelerations"],
                ],
                device=device,
            )
            wp.synchronize_device(device)
            elapsed += (time.perf_counter() - start) * 1000.0
        return elapsed / float(repeats)

    def direct_error_sample(self) -> dict[str, float]:
        """Explicit validation boundary; this is not used during ordinary playback."""

        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        wp.launch(
            kernels.direct_acceleration,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["masses"],
                self._arrays["active"],
                self.config.count,
                self.config.gravitational_constant,
                self.config.softening,
                self._arrays["direct_accelerations"],
            ],
            device=device,
        )
        # The approximate side includes the bound force fields (ADR 0045), so the exact
        # reference must too or the sample reports the field as tree error (SR-26).
        self._record_external_fields(self._arrays["direct_accelerations"])
        approximate_array = self._total_accelerations()
        wp.launch(
            kernels.relative_error,
            dim=self.config.count,
            inputs=[
                approximate_array,
                self._arrays["direct_accelerations"],
                self._arrays["active"],
                self.config.count,
                self._arrays["relative_errors"],
            ],
            device=device,
        )
        wp.synchronize_device(device)
        values = self._arrays["relative_errors"].numpy()[: self.config.count]
        active = self._arrays["active"].numpy()[: self.config.count]
        approximate = approximate_array.numpy()[: self.config.count]
        direct = self._arrays["direct_accelerations"].numpy()[: self.config.count]
        self.explicit_validation_readbacks += 4
        errors = sorted(float(values[i]) for i in range(self.config.count) if active[i] != 0)
        angular_errors = []
        denominator_floor = 1.0e-12
        for index in range(self.config.count):
            if active[index] == 0:
                continue
            approximate_length = float(self._np.linalg.norm(approximate[index]))
            direct_length = float(self._np.linalg.norm(direct[index]))
            if approximate_length <= denominator_floor or direct_length <= denominator_floor:
                continue
            cosine = float(
                self._np.dot(approximate[index], direct[index])
                / (approximate_length * direct_length)
            )
            angular_errors.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
        angular_errors.sort()
        # The metric the GRAV-002 gate bounds, and therefore the one Auto's accuracy policy
        # has to read: per-body error against the RMS acceleration of the whole system. The
        # per-body denominator above cannot carry a theta**2 bound, because a body sitting
        # near a force null makes its own error arbitrarily large relative to its own
        # acceleration for reasons that have nothing to do with the opening angle.
        selected = active != 0
        reference = direct[selected]
        reference_rms = (
            float(self._np.sqrt(self._np.mean(self._np.einsum("ij,ij->i", reference, reference))))
            if reference.size
            else 0.0
        )
        rms_errors = (
            sorted(
                float(value)
                for value in self._np.linalg.norm(
                    approximate[selected] - reference, axis=1
                )
                / reference_rms
            )
            if reference_rms > 0.0
            else []
        )
        return {
            "median": _quantile(errors, 0.5),
            "p90": _quantile(errors, 0.9),
            "p99": _quantile(errors, 0.99),
            "max": max(errors) if errors else 0.0,
            "rms_normalized_median": _quantile(rms_errors, 0.5),
            "rms_normalized_p99": _quantile(rms_errors, 0.99),
            "rms_normalized_max": max(rms_errors) if rms_errors else 0.0,
            "reference_rms_acceleration": reference_rms,
            "angular_median_degrees": _quantile(angular_errors, 0.5),
            "angular_p90_degrees": _quantile(angular_errors, 0.9),
            "angular_p99_degrees": _quantile(angular_errors, 0.99),
            "angular_max_degrees": max(angular_errors) if angular_errors else 0.0,
            "denominator_floor": denominator_floor,
        }

    def topology_validation_snapshot(self) -> dict[str, Any]:
        """Explicit full-state validation boundary; never used by playback."""
        self._require_wp().synchronize_device(self.device_context.device)
        names = (
            "positions",
            "masses",
            "stable_ids",
            "active",
            "body_keys",
            "sorted_keys",
            "sorted_indices",
            "node_mass",
            "node_moment_x",
            "node_moment_y",
            "node_moment_z",
            "node_qxx",
            "node_qxy",
            "node_qxz",
            "node_qyy",
            "node_qyz",
            "node_qzz",
            "node_count",
            "node_start",
            "node_end",
            "node_prefix",
            "node_levels",
            "slot_nodes",
        )
        host = {name: self._arrays[name].numpy() for name in names}
        occupied_nodes = [
            index
            for index, level in enumerate(host["node_levels"])
            if int(level) >= 0 and int(host["node_count"][index]) > 0
        ]
        node_index = {device_index: index for index, device_index in enumerate(occupied_nodes)}
        node_names = (
            "node_mass",
            "node_moment_x",
            "node_moment_y",
            "node_moment_z",
            "node_qxx",
            "node_qxy",
            "node_qxz",
            "node_qyy",
            "node_qyz",
            "node_qzz",
            "node_count",
            "node_start",
            "node_end",
            "node_prefix",
            "node_levels",
        )
        for name in node_names:
            host[name] = host[name][occupied_nodes]
        host["slot_nodes"] = self._np.asarray(
            [node_index.get(int(value), -1) for value in host["slot_nodes"]],
            dtype=self._np.int32,
        )
        occupied_node_count = len(occupied_nodes)
        self.explicit_validation_readbacks += len(names) + 1
        return {
            "count": self.config.count,
            "sort_size": self._sort_size,
            "max_depth": self.config.max_depth,
            "leaf_capacity": self.config.leaf_capacity,
            "node_capacity": self._total_nodes,
            "quadrupoles": bool(self.config.quadrupoles),
            "occupied_node_count": occupied_node_count,
            **{
                name: values.tolist()
                for name, values in host.items()
            },
        }

    def _total_accelerations(self) -> Any:
        """Each body's whole acceleration: under ADR 0144 the fast and slow forces summed."""

        if not self._dominant:
            return self._arrays["accelerations"]
        self._require_wp().launch(
            self._require_kernels().add_vectors,
            dim=self.config.capacity,
            inputs=[
                self._arrays["accelerations"],
                self._arrays["slow_accelerations"],
                self._arrays["total_accelerations"],
            ],
            device=self.device_context.device,
        )
        return self._arrays["total_accelerations"]

    def snapshot(self) -> dict[str, Any]:
        positions = self._arrays["positions"].numpy()
        velocities = self._arrays["velocities"].numpy()
        accelerations = self._total_accelerations().numpy()
        masses = self._arrays["masses"].numpy()
        radii = self._arrays["radii"].numpy()
        stable_ids = self._arrays["stable_ids"].numpy()
        active = self._arrays["active"].numpy()
        self.explicit_snapshot_readbacks += 7
        bodies = []
        for index in range(self.config.count):
            bodies.append(
                {
                    "index": index,
                    "stable_id": int(stable_ids[index]),
                    "active": bool(active[index]),
                    "mass": float(masses[index]),
                    "radius": float(radii[index]),
                    "position": [float(value) for value in positions[index]],
                    "velocity": [float(value) for value in velocities[index]],
                    "acceleration": [float(value) for value in accelerations[index]],
                }
            )
        return {
            "schema": CHECKPOINT_SCHEMA_VERSION,
            "backend": "tree",
            "capacity": self.config.capacity,
            "count": self.config.count,
            "step": self.step_index,
            "frame": self.frame_index,
            "tree_config": {
                "theta": self.config.theta,
                "max_depth": self.config.max_depth,
                "leaf_capacity": self.config.leaf_capacity,
                "stack_capacity": self.config.stack_capacity,
                "node_capacity": self._total_nodes,
                "quadrupoles": bool(self.config.quadrupoles),
                "collision_mode": self.config.collision_mode.value,
                "event_capacity": self.config.event_capacity,
            },
            "recomputed_on_restore": [
                "accelerations",
                "morton_keys",
                "sort_permutation",
                "tree_topology",
                "monopoles",
                "quadrupoles",
                "traversal_scratch",
                "collision_event_scratch",
            ],
            "bodies": bodies,
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint snapshot schema")
        if snapshot.get("backend") != "tree":
            raise ValueError("checkpoint backend does not match tree runtime")
        if int(snapshot.get("capacity", -1)) != self.config.capacity:
            raise ValueError("snapshot capacity does not match runtime")
        expected_tree = {
            "theta": self.config.theta,
            "max_depth": self.config.max_depth,
            "leaf_capacity": self.config.leaf_capacity,
            "stack_capacity": self.config.stack_capacity,
            "quadrupoles": bool(self.config.quadrupoles),
            "collision_mode": self.config.collision_mode.value,
            "event_capacity": self.config.event_capacity,
        }
        recorded_tree = dict(snapshot.get("tree_config") or {})
        # Capacity describes derived topology storage, not authoritative body state.
        # Sparse-tree releases can therefore restore dense-era checkpoints and rebuild
        # the occupied topology without pretending the old allocation is still active.
        recorded_tree.pop("node_capacity", None)
        # The multipole order is not derived storage: a checkpoint written before the
        # option existed recorded monopole-only trajectories, so it is only restorable
        # into a monopole-only runtime. Defaulting the missing key to False makes that a
        # configuration mismatch rather than a silent change of approximation.
        recorded_tree.setdefault("quadrupoles", False)
        if recorded_tree != expected_tree:
            raise ValueError("snapshot tree configuration does not match runtime")
        bodies = list(snapshot.get("bodies") or ())
        if len(bodies) != self.config.count:
            raise ValueError("snapshot body count does not match runtime")
        if [int(body.get("index", -1)) for body in bodies] != list(range(self.config.count)):
            raise ValueError("snapshot body ordering is invalid")
        if tuple(int(body["stable_id"]) for body in bodies) != self._stable_id_order:
            raise ValueError("snapshot stable IDs do not match runtime ordering")
        np = self._np
        wp = self._require_wp()
        scalar, vector, state_dtype = _dtypes(wp, self.config.precision)
        device = self.device_context.device
        positions = np.asarray([body["position"] for body in bodies], dtype=state_dtype)
        velocities = np.asarray([body["velocity"] for body in bodies], dtype=state_dtype)
        masses = np.asarray([body["mass"] for body in bodies], dtype=state_dtype)
        radii = np.asarray([body["radius"] for body in bodies], dtype=state_dtype)
        stable_ids = np.asarray([body["stable_id"] for body in bodies], dtype=np.int64)
        active = np.asarray(
            [1 if body.get("active", True) else 0 for body in bodies],
            dtype=np.int32,
        )
        wp.copy(self._arrays["positions"], wp.array(positions, dtype=vector, device=device))
        wp.copy(self._arrays["velocities"], wp.array(velocities, dtype=vector, device=device))
        wp.copy(self._arrays["masses"], wp.array(masses, dtype=scalar, device=device))
        wp.copy(self._arrays["radii"], wp.array(radii, dtype=scalar, device=device))
        wp.copy(self._arrays["stable_ids"], wp.array(stable_ids, dtype=wp.int64, device=device))
        wp.copy(self._arrays["active"], wp.array(active, dtype=wp.int32, device=device))
        self.step_index = int(snapshot.get("step", 0))
        self.frame_index = int(snapshot.get("frame", self.step_index // self.config.substeps))
        self._frame_primitive_started_at = None
        # A snapshot carries its own masses and activity, so the two-rate decision is
        # remade from them, not from whatever was loaded (ADR 0144).
        self._host_masses = masses.copy()
        self._host_active = active.copy()
        self._configure_two_rate()
        self.rebuild_tree()
        self._compute_accelerations()
        self._frame_primitive_cursor = 0
        self.last_frame_primitive_phase = ""
        self.last_step_ms = 0.0

    def close(self) -> None:
        self._frame_primitive_started_at = None
        self._frame_primitive_cursor = 0
        self._frame_substep = 0
        self._dominant = ()
        self.last_frame_primitive_phase = ""
        self._arrays.clear()
        self._graphs.clear()

    def _release_emitters(self, frame: int, previous_frame: int | None) -> bool:
        """Raise every emitter's live watermark from ``previous_frame`` to ``frame``.

        The direct lane's :meth:`DirectNBodyRuntime._release_emitters` verbatim, and for
        the same reasons: only the delta block is launched, so a slot the collision path
        deactivated stays deactivated, and ``previous_frame`` is ``None`` only at load.
        """

        if not self.config.emitters:
            return False
        wp = self._require_wp()
        kernels = self._require_emitter_kernels()
        released_any = False
        for spec in self.config.emitters:
            already = 0 if previous_frame is None else released_at_frame(spec, previous_frame)
            delta = released_at_frame(spec, frame) - already
            if delta <= 0:
                continue
            wp.launch(
                kernels.activate_slots,
                dim=delta,
                inputs=[self._arrays["active"], spec.begin_index + already],
                device=self.device_context.device,
            )
            released_any = True
        return released_any

    def _require_emitter_kernels(self):
        if self._emitter_kernels is None:
            raise DeviceRequirementError("Warp emitter kernels are not loaded")
        return self._emitter_kernels

    def _compute_accelerations(self) -> None:
        self._submit("accelerate", self._record_accelerations)
        self._require_wp().synchronize_device(self.device_context.device)
        self._raise_tree_overflow()

    def _record_accelerations(self) -> None:
        if self._dominant:
            # Load and restore need both forces at the solved positions (ADR 0144).
            self._record_fast_accelerations()
            self._record_slow_accelerations()
            return
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        wp.launch(
            kernels.reset_traversal_diag,
            dim=1,
            inputs=[self._arrays["diag"]],
            device=device,
        )
        # Traversal is indexed by morton slot, so it only writes the bodies the sort placed
        # in a slot: an inactive body has no slot and would otherwise keep a stale
        # acceleration across the KDK kick.
        wp.launch(
            kernels.clear_accelerations,
            dim=self.config.count,
            inputs=[self._arrays["accelerations"]],
            device=device,
        )
        self._record_walk()
        self._record_external_fields()

    def _record_fast_accelerations(self) -> None:
        """ADR 0144's fast force: every body's pull from the dominant bodies, direct."""

        wp = self._require_wp()
        kernels = self._require_kernels()
        wp.launch(
            kernels.dominant_acceleration,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["masses"],
                self._arrays["active"],
                self._arrays["dominant_indices"],
                len(self._dominant),
                self.config.count,
                self.config.gravitational_constant,
                self.config.softening,
                self._arrays["accelerations"],
            ],
            device=self.device_context.device,
        )

    def _record_slow_accelerations(self) -> None:
        """ADR 0144's slow force: the light bodies' tree, walked by every body.

        The tree is built from `tree_masses`, so a dominant body still feels the light
        bodies here and the light bodies do not feel it twice.
        """

        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        wp.launch(
            kernels.reset_traversal_diag,
            dim=1,
            inputs=[self._arrays["diag"]],
            device=device,
        )
        wp.launch(
            kernels.clear_accelerations,
            dim=self.config.count,
            inputs=[self._arrays["slow_accelerations"]],
            device=device,
        )
        self._record_walk("slow_accelerations")

    def _record_walk(self, output: str = "accelerations") -> None:
        """ADR 0132: lay the tree out depth-first, then walk it with no stack."""

        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        count = self.config.count
        arrays = self._arrays
        wp.launch(
            kernels.prepare_walk_slots,
            dim=count,
            inputs=[
                arrays["positions"],
                self._tree_masses,
                arrays["active"],
                arrays["sorted_keys"],
                arrays["sorted_indices"],
                count,
                self.config.max_depth,
                arrays["bounds"],
                arrays["node_count"],
                self.config.leaf_capacity,
                self._total_nodes,
                arrays["walk_bodies"],
                arrays["walk_first_level"],
                arrays["walk_node_counts"],
            ],
            device=device,
        )
        wp.launch(
            kernels.total_walk_chunks,
            dim=self._walk_chunks,
            inputs=[
                arrays["walk_node_counts"],
                count,
                self._walk_chunk,
                arrays["walk_chunk_totals"],
            ],
            device=device,
        )
        wp.launch(
            kernels.offset_walk_slots,
            dim=self._walk_chunks,
            inputs=[
                arrays["walk_node_counts"],
                count,
                self._walk_chunk,
                arrays["walk_chunk_totals"],
                arrays["walk_offsets"],
            ],
            device=device,
        )
        wp.launch(
            kernels.fill_walk_nodes,
            dim=count,
            inputs=[
                count,
                self.config.max_depth,
                self._total_nodes,
                self._total_nodes,
                arrays["walk_first_level"],
                arrays["walk_offsets"],
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
                arrays["node_end"],
                arrays["sorted_keys"],
                arrays["bounds"],
                self.config.theta,
                arrays["walk_links"],
                arrays["walk_cells"],
                arrays["walk_mass_diagonal"],
                arrays["walk_off_diagonal"],
                arrays["diag"],
            ],
            device=device,
        )
        wp.launch(
            kernels.tree_acceleration_walk,
            dim=count,
            inputs=[
                arrays["sorted_indices"],
                count,
                self.config.gravitational_constant,
                self.config.softening,
                self.config.theta,
                self.config.max_depth,
                self.config.leaf_capacity,
                self._total_nodes,
                int(self.config.quadrupoles),
                arrays["walk_bodies"],
                arrays["walk_offsets"],
                arrays["walk_links"],
                arrays["walk_cells"],
                arrays["walk_mass_diagonal"],
                arrays["walk_off_diagonal"],
                arrays[output],
                arrays["diag"],
            ],
            device=device,
        )

    def _record_accelerations_stack_reference(self) -> None:
        """The pre-ADR 0132 stack walk, kept as the agreement reference for tests.

        Not on any product path: it allocates its own per-body stack on first use.
        """

        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        if "stack_nodes" not in self._arrays:
            for name in ("stack_nodes", "stack_levels"):
                self._arrays[name] = wp.zeros(
                    self.config.capacity * self.config.stack_capacity,
                    dtype=wp.int32,
                    device=device,
                )
        wp.launch(
            kernels.reset_traversal_diag,
            dim=1,
            inputs=[self._arrays["diag"]],
            device=device,
        )
        wp.launch(
            kernels.clear_accelerations,
            dim=self.config.count,
            inputs=[self._arrays["accelerations"]],
            device=device,
        )
        wp.launch(
            kernels.tree_acceleration,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["masses"],
                self._arrays["active"],
                self._arrays["body_keys"],
                self._arrays["sorted_indices"],
                self.config.count,
                self.config.gravitational_constant,
                self.config.softening,
                self.config.theta,
                self.config.max_depth,
                self.config.leaf_capacity,
                self.config.stack_capacity,
                self.config.capacity,
                int(self.config.quadrupoles),
                self._arrays["bounds"],
                self._arrays["node_prefix"],
                self._arrays["slot_nodes"],
                self._arrays["node_mass"],
                self._arrays["node_moment_x"],
                self._arrays["node_moment_y"],
                self._arrays["node_moment_z"],
                self._arrays["node_qxx"],
                self._arrays["node_qxy"],
                self._arrays["node_qxz"],
                self._arrays["node_qyy"],
                self._arrays["node_qyz"],
                self._arrays["node_qzz"],
                self._arrays["node_count"],
                self._arrays["node_start"],
                self._arrays["node_end"],
                self._arrays["stack_nodes"],
                self._arrays["stack_levels"],
                self._arrays["accelerations"],
                self._arrays["diag"],
            ],
            device=device,
        )
        self._record_external_fields()

    def _record_external_fields(self, target: Any | None = None) -> None:
        if self.config.external_fields:
            wp = self._require_wp()
            device = self.device_context.device
            # After the traversal, into the same accumulator (ADR 0045); inactive
            # bodies are skipped by the kernel, matching clear_accelerations above.
            wp.launch(
                self._field_kernels.apply_external_fields,
                dim=self.config.count,
                inputs=[
                    self._arrays["positions"],
                    self._arrays["active"],
                    self.config.count,
                    self._arrays["field_kinds"],
                    self._arrays["field_strengths"],
                    self._arrays["field_positions"],
                    self._arrays["field_axes"],
                    len(self.config.external_fields),
                    self.config.softening,
                    self._arrays["accelerations"] if target is None else target,
                ],
                device=device,
            )

    def _resolve_collisions(self) -> None:
        if self.config.collision_mode == CollisionMode.IGNORE:
            return
        wp = self._require_wp()
        kernels = self._collision_kernels
        if kernels is None:
            raise DeviceRequirementError("Warp collision kernels are not loaded")
        device = self.device_context.device
        wp.launch(
            kernels.clear_idiag,
            dim=4,
            inputs=[self._arrays["idiag"]],
            device=device,
        )
        wp.launch(
            kernels.clear_idiag,
            dim=self.config.capacity,
            inputs=[self._arrays["claimed"]],
            device=device,
        )
        wp.launch(
            self._require_broadphase_kernels().clear_grid,
            dim=self._broadphase_config.cell_capacity,
            inputs=[
                self._arrays["collision_grid_max_radius"],
                self._arrays["collision_grid_min_radius"],
                self._arrays["collision_grid_cell_counts"],
                self._arrays["collision_grid_flags"],
            ],
            device=device,
        )
        wp.launch(
            self._require_broadphase_kernels().reduce_max_radius,
            dim=self.config.count,
            inputs=[
                self._arrays["radii"],
                self._arrays["active"],
                self.config.count,
                self._arrays["collision_grid_max_radius"],
                self._arrays["collision_grid_min_radius"],
            ],
            device=device,
        )
        wp.launch(
            self._require_broadphase_kernels().bin_particles,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["active"],
                self.config.count,
                self._arrays["collision_grid_max_radius"],
                self._broadphase_config.cell_capacity,
                self._broadphase_config.bucket_capacity,
                self._arrays["collision_grid_cell_counts"],
                self._arrays["collision_grid_cell_items"],
                self._arrays["collision_grid_flags"],
            ],
            device=device,
        )
        wp.launch(
            self._require_broadphase_kernels().detect_collisions_grid,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["radii"],
                self._arrays["active"],
                self.config.count,
                self._arrays["collision_grid_max_radius"],
                self._arrays["collision_grid_min_radius"],
                self._broadphase_config.cell_capacity,
                self._broadphase_config.bucket_capacity,
                self._broadphase_config.all_pairs_threshold,
                self._broadphase_config.radius_ratio_threshold,
                self._arrays["collision_grid_cell_counts"],
                self._arrays["collision_grid_cell_items"],
                self._arrays["collision_grid_flags"],
                self._arrays["event_i"],
                self._arrays["event_j"],
                self.config.event_capacity,
                self._arrays["idiag"],
            ],
            device=device,
        )
        wp.launch(
            kernels.accumulate_broadphase_flags,
            dim=1,
            inputs=[
                self._arrays["collision_grid_flags"],
                self._arrays["idiag_totals"],
            ],
            device=device,
        )
        wp.launch(
            kernels.accumulate_idiag,
            dim=1,
            inputs=[self._arrays["idiag"], self._arrays["idiag_totals"]],
            device=device,
        )
        resolver = (
            kernels.resolve_merge
            if self.config.collision_mode == CollisionMode.MERGE
            else kernels.resolve_elastic
        )
        wp.launch(
            resolver,
            dim=self.config.event_capacity,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["masses"],
                self._arrays["radii"],
                self._arrays["stable_ids"],
                self._arrays["active"],
                self._arrays["claimed"],
                self._arrays["event_i"],
                self._arrays["event_j"],
                self.config.event_capacity,
                self._arrays["idiag"],
            ],
            device=device,
        )

    def _require_wp(self):
        if self._wp is None:
            raise DeviceRequirementError("Warp runtime is not loaded")
        return self._wp

    def _require_kernels(self):
        if self._kernels is None:
            raise DeviceRequirementError("Warp tree kernels are not loaded")
        return self._kernels

    def _require_broadphase_kernels(self):
        if self._broadphase_kernels is None:
            raise DeviceRequirementError("Warp broad-phase kernels are not loaded")
        return self._broadphase_kernels


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _quad_dtype(wp, precision: Precision):
    """Four scalars of the state precision: a body's position and mass, or a node's."""

    return wp.vec4d if precision == Precision.FLOAT64 else wp.vec4


def _kernels(wp, precision: Precision = Precision.FLOAT32):
    """Compile the tree kernels for one state precision.

    The tree's own reductions (node mass and moments) stay at the state precision: they
    are part of the approximation being measured, not part of the measurement. Only the
    direct solver's conservation diagnostics reduce in float64.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    quad = _quad_dtype(wp, precision)
    warp_kernel, warp_func = _registrars(wp, scalar, vector, quad=quad)
    sentinel_key = wp.int64(9223372036854775807)
    # Largest value strictly below 1.0 in this precision, so a normalized coordinate can
    # never round up to `scale` and index a cell one past the last.
    normalized_ceiling = scalar(
        0.999999999999999889 if precision == Precision.FLOAT64 else 0.99999994
    )
    zero = scalar(0.0)
    half_one = scalar(0.5)

    @warp_kernel
    def clear_bounds(bounds: wp.array(dtype=scalar)):
        i = wp.tid()
        if i < 3:
            bounds[i] = scalar(1.0e30)
        else:
            bounds[i] = scalar(-1.0e30)

    @warp_kernel
    def reduce_bounds(
        positions: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        count: int,
        bounds: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        if i < count and active[i] != 0:
            p = positions[i]
            wp.atomic_min(bounds, 0, p[0])
            wp.atomic_min(bounds, 1, p[1])
            wp.atomic_min(bounds, 2, p[2])
            wp.atomic_max(bounds, 3, p[0])
            wp.atomic_max(bounds, 4, p[1])
            wp.atomic_max(bounds, 5, p[2])

    @warp_kernel
    def finalize_bounds(bounds: wp.array(dtype=scalar)):
        mn = vector(bounds[0], bounds[1], bounds[2])
        mx = vector(bounds[3], bounds[4], bounds[5])
        center = (mn + mx) * half_one
        extent = wp.max(wp.max(mx[0] - mn[0], mx[1] - mn[1]), mx[2] - mn[2])
        half = wp.max(extent * half_one, scalar(1.0e-5)) * scalar(1.0001)
        bounds[0] = center[0] - half
        bounds[1] = center[1] - half
        bounds[2] = center[2] - half
        bounds[3] = center[0] + half
        bounds[4] = center[1] + half
        bounds[5] = center[2] + half

    @warp_func
    def encode_morton(p: vector, bounds: wp.array(dtype=scalar), depth: int):
        scale = 1 << depth
        extent = bounds[3] - bounds[0]
        nx = wp.clamp((p[0] - bounds[0]) / extent, zero, normalized_ceiling)
        ny = wp.clamp((p[1] - bounds[1]) / extent, zero, normalized_ceiling)
        nz = wp.clamp((p[2] - bounds[2]) / extent, zero, normalized_ceiling)
        xi = wp.int32(nx * scalar(scale))
        yi = wp.int32(ny * scalar(scale))
        zi = wp.int32(nz * scalar(scale))
        key = wp.int64(0)
        for bit in range(depth):
            key = key | (wp.int64((xi >> bit) & 1) << wp.int64(3 * bit))
            key = key | (wp.int64((yi >> bit) & 1) << wp.int64(3 * bit + 1))
            key = key | (wp.int64((zi >> bit) & 1) << wp.int64(3 * bit + 2))
        return key

    @warp_kernel
    def compute_morton(
        positions: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        count: int,
        sort_size: int,
        depth: int,
        bounds: wp.array(dtype=scalar),
        body_keys: wp.array(dtype=wp.int64),
        sorted_keys: wp.array(dtype=wp.int64),
        sorted_indices: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= sort_size:
            return
        if i < count and active[i] != 0:
            key = encode_morton(positions[i], bounds, depth)
            body_keys[i] = key
            sorted_keys[i] = key
            sorted_indices[i] = i
        else:
            if i < count:
                body_keys[i] = sentinel_key
            sorted_keys[i] = sentinel_key
            sorted_indices[i] = -1

    @warp_kernel
    def bitonic_step(
        keys: wp.array(dtype=wp.int64),
        indices: wp.array(dtype=wp.int32),
        stable_ids: wp.array(dtype=wp.int64),
        count: int,
        stride: int,
        width: int,
    ):
        i = wp.tid()
        partner = i ^ stride
        if i >= count or partner >= count or partner <= i:
            return
        key_i = keys[i]
        key_j = keys[partner]
        index_i = indices[i]
        index_j = indices[partner]
        tie_i = sentinel_key
        tie_j = sentinel_key
        if index_i >= 0:
            tie_i = stable_ids[index_i]
        if index_j >= 0:
            tie_j = stable_ids[index_j]
        greater = key_i > key_j or (key_i == key_j and tie_i > tie_j)
        ascending = (i & width) == 0
        swap = (ascending and greater) or ((not ascending) and (not greater))
        if swap:
            keys[i] = key_j
            keys[partner] = key_i
            indices[i] = index_j
            indices[partner] = index_i

    @warp_kernel
    def clear_tree(
        node_prefix: wp.array(dtype=wp.int64),
        node_levels: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        node_count: wp.array(dtype=wp.int32),
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        empty_start: int,
    ):
        i = wp.tid()
        node_prefix[i] = wp.int64(0)
        node_levels[i] = -1
        node_mass[i] = zero
        moment_x[i] = zero
        moment_y[i] = zero
        moment_z[i] = zero
        qxx[i] = zero
        qxy[i] = zero
        qxz[i] = zero
        qyy[i] = zero
        qyz[i] = zero
        qzz[i] = zero
        node_count[i] = 0
        node_start[i] = empty_start
        node_end[i] = -1

    @warp_kernel
    def clear_slot_nodes(slot_nodes: wp.array(dtype=wp.int32)):
        slot_nodes[wp.tid()] = -1

    @warp_kernel
    def clear_node_cursor(node_cursor: wp.array(dtype=wp.int32)):
        node_cursor[0] = 0

    @warp_kernel
    def build_sparse_topology(
        sorted_keys: wp.array(dtype=wp.int64),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        max_depth: int,
        node_capacity: int,
        slot_nodes: wp.array(dtype=wp.int32),
        node_prefix: wp.array(dtype=wp.int64),
        node_levels: wp.array(dtype=wp.int32),
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        node_cursor: wp.array(dtype=wp.int32),
        diag: wp.array(dtype=wp.int64),
    ):
        slot = wp.tid()
        if slot >= count or sorted_indices[slot] < 0:
            return
        key = sorted_keys[slot]
        for level in range(max_depth + 1):
            shift = wp.int64(3 * (max_depth - level))
            prefix = key >> shift
            low = wp.int32(0)
            high = wp.int32(slot)
            while low < high:
                middle = (low + high) // 2
                if (sorted_keys[middle] >> shift) < prefix:
                    low = middle + 1
                else:
                    high = middle
            first_slot = low
            node = wp.int32(0)
            if level > 0:
                node = 1 + (level - 1) * count + first_slot
            if node >= node_capacity:
                slot_nodes[level * count + slot] = -1
                diag[7] = wp.int64(1)
            else:
                slot_nodes[level * count + slot] = node
                if slot == first_slot:
                    node_prefix[node] = prefix
                    node_levels[node] = level
                    node_start[node] = slot
                    wp.atomic_add(node_cursor, 0, 1)
                wp.atomic_max(node_end, node, slot)

    @warp_kernel
    def accumulate_sparse_nodes(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        use_quadrupoles: int,
        node_count: wp.array(dtype=wp.int32),
    ):
        node = wp.tid()
        start = node_start[node]
        end = node_end[node]
        if start < 0 or end < start:
            return
        total_mass = scalar(0.0)
        total_x = scalar(0.0)
        total_y = scalar(0.0)
        total_z = scalar(0.0)
        bodies = wp.int32(0)
        for slot in range(start, end + 1):
            if slot < count:
                source = sorted_indices[slot]
                if source >= 0 and active[source] != 0:
                    mass = masses[source]
                    p = positions[source]
                    total_mass += mass
                    total_x += mass * p[0]
                    total_y += mass * p[1]
                    total_z += mass * p[2]
                    bodies += 1
        node_mass[node] = total_mass
        moment_x[node] = total_x
        moment_y[node] = total_y
        moment_z[node] = total_z
        node_count[node] = bodies

        # Second pass, about the centre of mass rather than the origin. Accumulating
        # `sum m x_a x_b` and subtracting `M d_a d_b` afterwards is algebraically the same
        # and numerically useless: at float32 the two terms agree to seven digits for any
        # node far from the origin, so the difference that carries the whole quadrupole is
        # the part that cancelled. Re-reading the range once costs a second pass over data
        # already resident and keeps every product O(cell size) instead of O(distance).
        if use_quadrupoles != 0 and total_mass > zero:
            inverse_mass = scalar(1.0) / total_mass
            center = vector(
                total_x * inverse_mass, total_y * inverse_mass, total_z * inverse_mass
            )
            sxx = scalar(0.0)
            sxy = scalar(0.0)
            sxz = scalar(0.0)
            syy = scalar(0.0)
            syz = scalar(0.0)
            szz = scalar(0.0)
            for slot in range(start, end + 1):
                if slot < count:
                    source = sorted_indices[slot]
                    if source >= 0 and active[source] != 0:
                        mass = masses[source]
                        s = positions[source] - center
                        sxx += mass * s[0] * s[0]
                        sxy += mass * s[0] * s[1]
                        sxz += mass * s[0] * s[2]
                        syy += mass * s[1] * s[1]
                        syz += mass * s[1] * s[2]
                        szz += mass * s[2] * s[2]
            trace = sxx + syy + szz
            three = scalar(3.0)
            # Traceless form: Q_ab = sum m (3 s_a s_b - delta_ab s.s).
            qxx[node] = three * sxx - trace
            qxy[node] = three * sxy
            qxz[node] = three * sxz
            qyy[node] = three * syy - trace
            qyz[node] = three * syz
            qzz[node] = three * szz - trace

    # --- bottom-up node accumulation -------------------------------------------------
    #
    # `accumulate_sparse_nodes` above solves every node as an independent range sum. That
    # is correct and it is also the whole cost of a rebuild: `build_sparse_topology` emits
    # a node at EVERY level for every distinct Morton prefix, and a level L+1 node's slot
    # range partitions its level L parent's, so a launch over all nodes re-reads all N
    # bodies once per level -- and hands the level-0 range, all N of them, to a single
    # thread. At 100,000 bodies and max_depth=9 that one kernel measured 65.5 ms, 86.4 %
    # of the rebuild, with 80 SMs idle behind one serial walk.
    #
    # The pair below uses the fact that it is a tree. The deepest level is summed from
    # bodies (its ranges are one or a few slots); every shallower level is then reduced
    # from its own children, whose totals are already final. Work becomes O(N + nodes) and
    # the critical path O(depth) instead of O(N).
    #
    # Both store RAW second moments S_ab about the node's own centre of mass in the q
    # arrays and leave the traceless conversion to `finalize_sparse_node_quadrupoles`.
    # The split is necessary here: a traceless Q cannot be reduced, because its trace is
    # identically zero and so the information the parent needs is exactly what was thrown
    # away.

    @warp_kernel
    def accumulate_sparse_leaf_nodes(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        node_base: int,
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        use_quadrupoles: int,
        node_count: wp.array(dtype=wp.int32),
    ):
        node = node_base + wp.tid()
        start = node_start[node]
        end = node_end[node]
        if start < 0 or end < start:
            return
        total_mass = scalar(0.0)
        total_x = scalar(0.0)
        total_y = scalar(0.0)
        total_z = scalar(0.0)
        bodies = wp.int32(0)
        for slot in range(start, end + 1):
            if slot < count:
                source = sorted_indices[slot]
                if source >= 0 and active[source] != 0:
                    mass = masses[source]
                    p = positions[source]
                    total_mass += mass
                    total_x += mass * p[0]
                    total_y += mass * p[1]
                    total_z += mass * p[2]
                    bodies += 1
        node_mass[node] = total_mass
        moment_x[node] = total_x
        moment_y[node] = total_y
        moment_z[node] = total_z
        node_count[node] = bodies

        if use_quadrupoles != 0 and total_mass > zero:
            inverse_mass = scalar(1.0) / total_mass
            center = vector(
                total_x * inverse_mass, total_y * inverse_mass, total_z * inverse_mass
            )
            sxx = scalar(0.0)
            sxy = scalar(0.0)
            sxz = scalar(0.0)
            syy = scalar(0.0)
            syz = scalar(0.0)
            szz = scalar(0.0)
            for slot in range(start, end + 1):
                if slot < count:
                    source = sorted_indices[slot]
                    if source >= 0 and active[source] != 0:
                        mass = masses[source]
                        s = positions[source] - center
                        sxx += mass * s[0] * s[0]
                        sxy += mass * s[0] * s[1]
                        sxz += mass * s[0] * s[2]
                        syy += mass * s[1] * s[1]
                        syz += mass * s[1] * s[2]
                        szz += mass * s[2] * s[2]
            qxx[node] = sxx
            qxy[node] = sxy
            qxz[node] = sxz
            qyy[node] = syy
            qyz[node] = syz
            qzz[node] = szz

    @warp_kernel
    def reduce_sparse_node_level(
        count: int,
        level: int,
        node_base: int,
        slot_nodes: wp.array(dtype=wp.int32),
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        use_quadrupoles: int,
        node_count: wp.array(dtype=wp.int32),
    ):
        node = node_base + wp.tid()
        start = node_start[node]
        end = node_end[node]
        if start < 0 or end < start:
            return
        child_row = (level + 1) * count
        total_mass = scalar(0.0)
        total_x = scalar(0.0)
        total_y = scalar(0.0)
        total_z = scalar(0.0)
        bodies = wp.int32(0)
        # Walk the children by jumping over each one's slot range, the same enumeration
        # the traversal uses. A node has at most eight children, so this loop is short and
        # uniform across the launch.
        cursor = start
        while cursor <= end:
            # A slot with no node at the child level (one `build_sparse_topology` skipped,
            # or one it could not place) is stepped over rather than ending the walk, so a
            # gap cannot silently truncate a parent's children. Landing anywhere inside a
            # child's range returns that child and jumps past its end, so the walk is
            # self-correcting and always advances.
            child = slot_nodes[child_row + cursor]
            next_cursor = cursor + 1
            if child >= 0:
                child_end = node_end[child]
                if child_end >= cursor:
                    next_cursor = child_end + 1
                    total_mass += node_mass[child]
                    total_x += moment_x[child]
                    total_y += moment_y[child]
                    total_z += moment_z[child]
                    bodies += node_count[child]
            cursor = next_cursor
        node_mass[node] = total_mass
        moment_x[node] = total_x
        moment_y[node] = total_y
        moment_z[node] = total_z
        node_count[node] = bodies

        # Parallel-axis shift, child centre of mass to parent centre of mass. The warning
        # on the body-level pass above is about shifting from the ORIGIN, where the offset
        # is the distance to the scene origin and the subtraction eats the whole moment.
        # Here the offset is bounded by the parent cell, so `m d_a d_b` is the same order
        # as the S it is added to -- the identical conditioning the leaf pass already has
        # when it forms `positions[source] - center`.
        if use_quadrupoles != 0 and total_mass > zero:
            inverse_mass = scalar(1.0) / total_mass
            center = vector(
                total_x * inverse_mass, total_y * inverse_mass, total_z * inverse_mass
            )
            sxx = scalar(0.0)
            sxy = scalar(0.0)
            sxz = scalar(0.0)
            syy = scalar(0.0)
            syz = scalar(0.0)
            szz = scalar(0.0)
            cursor = start
            while cursor <= end:
                child = slot_nodes[child_row + cursor]
                next_cursor = cursor + 1
                if child >= 0:
                    child_end = node_end[child]
                    if child_end >= cursor:
                        next_cursor = child_end + 1
                        child_mass = node_mass[child]
                        if child_mass > zero:
                            child_inverse = scalar(1.0) / child_mass
                            d = vector(
                                moment_x[child] * child_inverse - center[0],
                                moment_y[child] * child_inverse - center[1],
                                moment_z[child] * child_inverse - center[2],
                            )
                            sxx += qxx[child] + child_mass * d[0] * d[0]
                            sxy += qxy[child] + child_mass * d[0] * d[1]
                            sxz += qxz[child] + child_mass * d[0] * d[2]
                            syy += qyy[child] + child_mass * d[1] * d[1]
                            syz += qyz[child] + child_mass * d[1] * d[2]
                            szz += qzz[child] + child_mass * d[2] * d[2]
                cursor = next_cursor
            qxx[node] = sxx
            qxy[node] = sxy
            qxz[node] = sxz
            qyy[node] = syy
            qyz[node] = syz
            qzz[node] = szz

    @warp_kernel
    def finalize_sparse_node_quadrupoles(
        node_mass: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        use_quadrupoles: int,
    ):
        node = wp.tid()
        if use_quadrupoles == 0 or node_mass[node] <= zero:
            qxx[node] = zero
            qxy[node] = zero
            qxz[node] = zero
            qyy[node] = zero
            qyz[node] = zero
            qzz[node] = zero
            return
        sxx = qxx[node]
        sxy = qxy[node]
        sxz = qxz[node]
        syy = qyy[node]
        syz = qyz[node]
        szz = qzz[node]
        trace = sxx + syy + szz
        three = scalar(3.0)
        qxx[node] = three * sxx - trace
        qxy[node] = three * sxy
        qxz[node] = three * sxz
        qyy[node] = three * syy - trace
        qyz[node] = three * syz
        qzz[node] = three * szz - trace

    # int64 because slot 4 accumulates one entry per body-node interaction across the whole
    # launch: 131072 bodies at ~1000 interactions each is already 1.3e8, and a scene an
    # order of magnitude larger passes 2**31 and would report a negative interaction count.
    @warp_kernel
    def clear_diag(diag: wp.array(dtype=wp.int64)):
        diag[wp.tid()] = wp.int64(0)

    # The report counters (slots 0, 1, 5, 6) are cleared by the collection that fills them, run
    # only when a report is asked for (ADR 0226); the build clears only the flags it and the
    # walk set on every substep.
    @warp_kernel
    def clear_tree_flags(diag: wp.array(dtype=wp.int64)):
        slot = wp.tid()
        if not (slot == 0 or slot == 1 or slot == 5 or slot == 6):
            diag[slot] = wp.int64(0)

    @warp_kernel
    def clear_tree_report(diag: wp.array(dtype=wp.int64)):
        slot = wp.tid()
        if slot == 0 or slot == 1 or slot == 5 or slot == 6:
            diag[slot] = wp.int64(0)

    # ADR 0218: a sticky, capture-safe fail-closed signal for a caller that borrows the tree as
    # a gravity service. Slot 7 is the node-capacity overflow and slot 3 the traversal
    # overflow (the two `_raise_tree_overflow` reads back); the caller owns the flag and
    # clears it when it reads it, so an overflow in any pass of a frame survives to the read.
    @warp_kernel
    def fold_tree_overflow(diag: wp.array(dtype=wp.int64), flag: wp.array(dtype=wp.int32)):
        if diag[7] != wp.int64(0) or diag[3] != wp.int64(0):
            wp.atomic_max(flag, 0, 1)

    @warp_kernel
    def collect_tree_diag(
        node_count: wp.array(dtype=wp.int32),
        node_levels: wp.array(dtype=wp.int32),
        max_depth: int,
        diag: wp.array(dtype=wp.int64),
    ):
        node = wp.tid()
        if node == 0:
            diag[6] = wp.int64(node_count[0])
        if node_count[node] > 0:
            wp.atomic_add(diag, 0, wp.int64(1))
            wp.atomic_max(diag, 1, wp.int64(node_levels[node]))
            if node_levels[node] == max_depth:
                wp.atomic_add(diag, 5, wp.int64(1))

    @warp_kernel
    def reset_traversal_diag(diag: wp.array(dtype=wp.int64)):
        if wp.tid() == 0:
            diag[2] = wp.int64(0)
            diag[3] = wp.int64(0)
            diag[4] = wp.int64(0)

    @warp_kernel
    def direct_acceleration(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        accelerations: wp.array(dtype=vector),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0:
            accelerations[i] = vector()
            return
        p = positions[i]
        a = vector()
        for j in range(count):
            if j != i and active[j] != 0:
                r = positions[j] - p
                d2 = wp.dot(r, r) + softening * softening
                inv = scalar(1.0) / wp.sqrt(d2)
                a += r * (gravitational_constant * masses[j] * inv * inv * inv)
        accelerations[i] = a

    @warp_kernel
    def dominant_acceleration(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        dominant_indices: wp.array(dtype=wp.int32),
        dominant_count: int,
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        accelerations: wp.array(dtype=vector),
    ):
        # ADR 0144's fast force: `direct_acceleration` restricted to the dominant sources.
        i = wp.tid()
        if i >= count or active[i] == 0:
            accelerations[i] = vector()
            return
        p = positions[i]
        a = vector()
        for k in range(dominant_count):
            j = dominant_indices[k]
            if j != i and active[j] != 0:
                r = positions[j] - p
                d2 = wp.dot(r, r) + softening * softening
                inv = scalar(1.0) / wp.sqrt(d2)
                a += r * (gravitational_constant * masses[j] * inv * inv * inv)
        accelerations[i] = a

    @warp_kernel
    def add_vectors(
        first: wp.array(dtype=vector),
        second: wp.array(dtype=vector),
        total: wp.array(dtype=vector),
    ):
        i = wp.tid()
        total[i] = first[i] + second[i]

    @warp_kernel
    def clear_accelerations(accelerations: wp.array(dtype=vector)):
        accelerations[wp.tid()] = vector()

    @warp_kernel
    def tree_acceleration(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        body_keys: wp.array(dtype=wp.int64),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        theta: scalar,
        max_depth: int,
        leaf_capacity: int,
        stack_capacity: int,
        stack_stride: int,
        use_quadrupoles: int,
        bounds: wp.array(dtype=scalar),
        node_prefix: wp.array(dtype=wp.int64),
        slot_nodes: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        node_count: wp.array(dtype=wp.int32),
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        stack_nodes: wp.array(dtype=wp.int32),
        stack_levels: wp.array(dtype=wp.int32),
        accelerations: wp.array(dtype=vector),
        diag: wp.array(dtype=wp.int64),
    ):
        # Walk the tree per morton slot rather than per body index. Neighbouring slots hold
        # neighbouring bodies, so a warp shares node paths instead of diverging across the
        # whole octree. Each body still walks the same nodes in the same order, so the
        # interaction set and the accumulation order are unchanged. Padding slots and
        # inactive bodies have no slot; `clear_accelerations` zeroes them beforehand.
        slot = wp.tid()
        if slot >= count:
            return
        i = sorted_indices[slot]
        if i < 0:
            return
        # Stack entries are strided by slot, not packed per body: neighbouring slots at the
        # same depth land in neighbouring words, so a warp's push and pop coalesce into one
        # transaction instead of 32 scattered ones. Depth is the slow axis because the whole
        # warp moves through depth together. Same entries, same order, same arithmetic.
        stack_nodes[slot] = 0
        stack_levels[slot] = 0
        top = wp.int32(1)
        high = wp.int32(1)
        interactions = wp.int32(0)
        overflow = wp.int32(0)
        target_key = body_keys[i]
        p = positions[i]
        a = vector()
        root_center = vector(
            (bounds[0] + bounds[3]) * half_one,
            (bounds[1] + bounds[4]) * half_one,
            (bounds[2] + bounds[5]) * half_one,
        )
        root_half = (bounds[3] - bounds[0]) * half_one

        while top > 0:
            top -= 1
            entry = slot + top * stack_stride
            node = stack_nodes[entry]
            level = stack_levels[entry]
            count_node = node_count[node]
            if count_node == 0:
                continue
            prefix = node_prefix[node]
            target_prefix = wp.int64(0)
            if level > 0:
                target_prefix = target_key >> wp.int64(3 * (max_depth - level))
            contains_target = wp.int64(prefix) == target_prefix

            center = root_center
            half = root_half
            for path_level in range(level):
                shift = 3 * (level - 1 - path_level)
                child_code = wp.int32(
                    (prefix >> wp.int64(shift)) & wp.int64(7)
                )
                half *= half_one
                if (child_code & 1) != 0:
                    center[0] += half
                else:
                    center[0] -= half
                if (child_code & 2) != 0:
                    center[1] += half
                else:
                    center[1] -= half
                if (child_code & 4) != 0:
                    center[2] += half
                else:
                    center[2] -= half

            mass = node_mass[node]
            com = vector(
                moment_x[node] / mass,
                moment_y[node] / mass,
                moment_z[node] / mass,
            )
            r = com - p
            d2 = wp.dot(r, r) + softening * softening
            distance = wp.sqrt(d2)
            size = half + half
            # ADR 0147: the same reach the packed walk stores per cell (Barnes 1994).
            reach = size + theta * wp.length(com - center)
            is_leaf = level == max_depth or count_node <= leaf_capacity

            if is_leaf:
                start = node_start[node]
                end = node_end[node]
                cursor = start
                while cursor <= end:
                    source = sorted_indices[cursor]
                    if source >= 0 and source != i and active[source] != 0:
                        delta = positions[source] - p
                        dist2 = wp.dot(delta, delta) + softening * softening
                        inv = scalar(1.0) / wp.sqrt(dist2)
                        a += delta * (
                            gravitational_constant * masses[source] * inv * inv * inv
                        )
                        interactions += 1
                    cursor += 1
            elif (not contains_target) and reach / distance < theta:
                inv = scalar(1.0) / distance
                a += r * (gravitational_constant * mass * inv * inv * inv)
                # Added, never substituted: the acceptance test, the interaction set, the
                # traversal order and the monopole expression above are all identical to
                # the monopole-only build, so `quadrupoles=False` reproduces it bit for
                # bit and the difference between the two builds is this term alone.
                #
                # a_c += G [ -(Q r)_c / d^5 + (5/2) (r.Q.r) r_c / d^7 ], with `r` pointing
                # from the body to the cell's centre of mass and `d` the same softened
                # distance the monopole uses, so softening damps both terms together
                # rather than leaving the higher moment singular inside the softened core.
                if use_quadrupoles != 0:
                    qx = qxx[node] * r[0] + qxy[node] * r[1] + qxz[node] * r[2]
                    qy = qxy[node] * r[0] + qyy[node] * r[1] + qyz[node] * r[2]
                    qz = qxz[node] * r[0] + qyz[node] * r[1] + qzz[node] * r[2]
                    r_q_r = r[0] * qx + r[1] * qy + r[2] * qz
                    inv2 = inv * inv
                    inv5 = inv2 * inv2 * inv
                    inv7 = inv5 * inv2
                    a += vector(qx, qy, qz) * (-gravitational_constant * inv5)
                    a += r * (gravitational_constant * scalar(2.5) * r_q_r * inv7)
                interactions += 1
            else:
                cursor = node_end[node]
                while cursor >= node_start[node]:
                    child_node = slot_nodes[(level + 1) * count + cursor]
                    if child_node < 0:
                        overflow = 1
                        break
                    if top >= stack_capacity:
                        overflow = 1
                    else:
                        push = slot + top * stack_stride
                        stack_nodes[push] = child_node
                        stack_levels[push] = level + 1
                        top += 1
                        high = wp.max(high, top)
                    cursor = node_start[child_node] - 1
                if overflow != 0:
                    break

        if overflow != 0:
            accelerations[i] = vector()
            wp.atomic_max(diag, 3, wp.int64(1))
        else:
            accelerations[i] = a
        wp.atomic_max(diag, 2, wp.int64(high))
        wp.atomic_add(diag, 4, wp.int64(interactions))

    # ADR 0132: the walk as a depth-first array with skip pointers.
    #
    # `tree_acceleration` above is latency-bound, not arithmetic-bound: every pop is a
    # global stack read, and every opened node costs two dependent loads per child
    # (`slot_nodes`, then `node_start`) before the next pop can begin. Its pops already
    # visit nodes in depth-first preorder (children are pushed last-first), so the same
    # walk can be written with no stack at all: lay the nodes out in that preorder, and
    # a node's first child is the next entry while "done with this subtree" is one stored
    # index. Same nodes, same order, same arithmetic; one coalesced load per step.
    #
    # Preorder is (first slot, level). A slot starts a node at every level from the first
    # level at which its key's prefix differs from the previous slot's down to max_depth,
    # and at none if its key repeats. So node k of slot s sits at `offset[s] + k` once
    # `offset` is the exclusive prefix sum of those per-slot counts, and the node after a
    # subtree ending at slot e is the first one slot e + 1 starts: `offset[e + 1]`.
    # The sum is a two-pass chunked scan written here rather than `wp.utils.array_scan`,
    # whose scratch buffer must not be baked into a captured graph (ADR 0096).

    @warp_kernel
    def prepare_walk_slots(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        sorted_keys: wp.array(dtype=wp.int64),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        max_depth: int,
        bounds: wp.array(dtype=scalar),
        node_count: wp.array(dtype=wp.int32),
        leaf_capacity: int,
        node_capacity: int,
        walk_bodies: wp.array(dtype=quad),
        walk_first_level: wp.array(dtype=wp.int32),
        walk_node_counts: wp.array(dtype=wp.int32),
    ):
        slot = wp.tid()
        if slot >= count:
            return
        i = sorted_indices[slot]
        first = max_depth + 1
        body = quad()
        key = sorted_keys[slot]
        if i >= 0:
            if active[i] != 0:
                p = positions[i]
                body = quad(p[0], p[1], p[2], masses[i])
            if slot == 0:
                first = 0
            else:
                previous = sorted_keys[slot - 1]
                for level in range(1, max_depth + 1):
                    shift = wp.int64(3 * (max_depth - level))
                    if first > max_depth and (key >> shift) != (previous >> shift):
                        first = level
        # ADR 0228: lay out only the nodes a walk can reach. The walk never goes below a node
        # it sums as a leaf (at most `leaf_capacity` bodies, or at `max_depth`), so a node
        # whose parent is such a leaf is never visited, and neither is anything under it.
        # Laying those out cost about ten entries per body at 500,000 bodies, nine of them
        # dead, and the walk read its live entries scattered among them. A node's count only
        # shrinks with depth, so the reachable nodes a slot starts are one run of levels: from
        # `first`, if its parent is not a leaf, down to the first one that is a leaf itself.
        emitted = wp.int32(0)
        if first <= max_depth:
            reachable = wp.int32(1)
            if first > 0:
                parent_level = first - 1
                parent = wp.int32(0)
                if parent_level > 0:
                    # The parent was started by the first slot sharing its prefix.
                    shift = wp.int64(3 * (max_depth - parent_level))
                    prefix = key >> shift
                    low = wp.int32(0)
                    high = slot
                    while low < high:
                        middle = (low + high) // 2
                        if (sorted_keys[middle] >> shift) < prefix:
                            low = middle + 1
                        else:
                            high = middle
                    parent = 1 + (parent_level - 1) * count + low
                # A parent the topology could not allocate is laid out below as it always
                # was; the walk fails closed on reaching it.
                if parent < node_capacity and node_count[parent] <= leaf_capacity:
                    reachable = 0
            if reachable != 0:
                for level in range(first, max_depth + 1):
                    emitted += 1
                    node = wp.int32(0)
                    if level > 0:
                        node = 1 + (level - 1) * count + slot
                    if node >= node_capacity:
                        break
                    if node_count[node] <= leaf_capacity:
                        break
        walk_bodies[slot] = body
        walk_first_level[slot] = first
        walk_node_counts[slot] = emitted

    @warp_kernel
    def total_walk_chunks(
        walk_node_counts: wp.array(dtype=wp.int32),
        count: int,
        chunk: int,
        walk_chunk_totals: wp.array(dtype=wp.int32),
    ):
        c = wp.tid()
        total = wp.int32(0)
        for k in range(chunk):
            slot = c * chunk + k
            if slot < count:
                total += walk_node_counts[slot]
        walk_chunk_totals[c] = total

    @warp_kernel
    def offset_walk_slots(
        walk_node_counts: wp.array(dtype=wp.int32),
        count: int,
        chunk: int,
        walk_chunk_totals: wp.array(dtype=wp.int32),
        walk_offsets: wp.array(dtype=wp.int32),
    ):
        c = wp.tid()
        running = wp.int32(0)
        for previous in range(c):
            running += walk_chunk_totals[previous]
        for k in range(chunk):
            slot = c * chunk + k
            if slot < count:
                walk_offsets[slot] = running
                running += walk_node_counts[slot]
        if (c + 1) * chunk >= count:
            walk_offsets[count] = running

    @warp_kernel
    def fill_walk_nodes(
        count: int,
        max_depth: int,
        node_capacity: int,
        walk_capacity: int,
        walk_first_level: wp.array(dtype=wp.int32),
        walk_offsets: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        node_count: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        sorted_keys: wp.array(dtype=wp.int64),
        bounds: wp.array(dtype=scalar),
        theta: scalar,
        walk_links: wp.array(dtype=wp.vec4i),
        walk_cells: wp.array(dtype=quad),
        walk_mass_diagonal: wp.array(dtype=quad),
        walk_off_diagonal: wp.array(dtype=vector),
        diag: wp.array(dtype=wp.int64),
    ):
        # One writer per entry: slot s writes exactly the nodes it starts.
        slot = wp.tid()
        if slot >= count:
            return
        first = walk_first_level[slot]
        base = walk_offsets[slot]
        # ADR 0228: the slot's run of reachable levels; it may stop above max_depth.
        last = first + (walk_offsets[slot + 1] - base) - 1
        # ADR 0226: the nodes a slot starts are the ancestors of its own body, so one descent
        # along the slot's key gives every one of their geometric centres in O(depth), where
        # rebuilding each from its prefix cost O(level) per node. The same halvings and
        # signed steps run in the same order, so each centre is the same number to the bit.
        key = sorted_keys[slot]
        center = vector(
            (bounds[0] + bounds[3]) * half_one,
            (bounds[1] + bounds[4]) * half_one,
            (bounds[2] + bounds[5]) * half_one,
        )
        half = (bounds[3] - bounds[0]) * half_one
        for level in range(0, max_depth + 1):
            if level > last:
                break
            if level > 0:
                shift = wp.int64(3 * (max_depth - level))
                child_code = wp.int32((key >> shift) & wp.int64(7))
                half *= half_one
                if (child_code & 1) != 0:
                    center[0] += half
                else:
                    center[0] -= half
                if (child_code & 2) != 0:
                    center[1] += half
                else:
                    center[1] -= half
                if (child_code & 4) != 0:
                    center[2] += half
                else:
                    center[2] -= half
            if level < first:
                continue
            entry = base + level - first
            if entry >= walk_capacity:
                wp.atomic_max(diag, 3, wp.int64(1))
            else:
                node = wp.int32(0)
                if level > 0:
                    node = 1 + (level - 1) * count + slot
                if node >= node_capacity:
                    # A node the topology could not allocate; `build_sparse_topology` has
                    # already raised diag[7]. A walk that reaches it fails closed.
                    walk_links[entry] = wp.vec4i(slot, slot, entry + 1, -1)
                else:
                    end = node_end[node]
                    mass = node_mass[node]
                    walk_links[entry] = wp.vec4i(
                        slot,
                        end,
                        walk_offsets[end + 1],
                        node_count[node] | (level << 24),
                    )
                    com = vector(
                        moment_x[node] / mass,
                        moment_y[node] / mass,
                        moment_z[node] / mass,
                    )
                    # ADR 0147: a cell reaches as far as its edge plus theta times the offset
                    # of its mass from its centre (Barnes 1994), so a body just outside a cell
                    # whose mass sits on the far side still opens it.
                    reach = (half + half) + theta * wp.length(com - center)
                    walk_cells[entry] = quad(com[0], com[1], com[2], reach)
                    walk_mass_diagonal[entry] = quad(mass, qxx[node], qyy[node], qzz[node])
                    walk_off_diagonal[entry] = vector(qxy[node], qxz[node], qyz[node])

    @warp_kernel
    def tree_acceleration_walk(
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        theta: scalar,
        max_depth: int,
        leaf_capacity: int,
        walk_capacity: int,
        use_quadrupoles: int,
        walk_bodies: wp.array(dtype=quad),
        walk_offsets: wp.array(dtype=wp.int32),
        walk_links: wp.array(dtype=wp.vec4i),
        walk_cells: wp.array(dtype=quad),
        walk_mass_diagonal: wp.array(dtype=quad),
        walk_off_diagonal: wp.array(dtype=vector),
        accelerations: wp.array(dtype=vector),
        diag: wp.array(dtype=wp.int64),
    ):
        slot = wp.tid()
        if slot >= count:
            return
        i = sorted_indices[slot]
        if i < 0:
            return
        total = wp.min(walk_offsets[count], walk_capacity)
        body = walk_bodies[slot]
        p = vector(body[0], body[1], body[2])
        a = vector()
        interactions = wp.int32(0)
        overflow = wp.int32(0)
        entry = wp.int32(0)
        while entry < total:
            link = walk_links[entry]
            start = link[0]
            end = link[1]
            skip = link[2]
            packed = link[3]
            if packed < 0:
                overflow = 1
                break
            count_node = packed & 16777215
            level = packed >> 24
            if count_node == 0:
                entry = skip
                continue
            if level == max_depth or count_node <= leaf_capacity:
                # Every slot in a node's range holds a sorted body, so the self test is
                # the slot; an inactive source was stored with zero mass.
                cursor = start
                while cursor <= end:
                    if cursor != slot:
                        source = walk_bodies[cursor]
                        delta = vector(source[0], source[1], source[2]) - p
                        dist2 = wp.dot(delta, delta) + softening * softening
                        inv = scalar(1.0) / wp.sqrt(dist2)
                        a += delta * (gravitational_constant * source[3] * inv * inv * inv)
                        interactions += 1
                    cursor += 1
                entry = skip
                continue
            cell = walk_cells[entry]
            r = vector(cell[0], cell[1], cell[2]) - p
            d2 = wp.dot(r, r) + softening * softening
            distance = wp.sqrt(d2)
            contains_target = slot >= start and slot <= end
            reach = cell[3]
            if (not contains_target) and reach / distance < theta:
                moments = walk_mass_diagonal[entry]
                mass = moments[0]
                inv = scalar(1.0) / distance
                a += r * (gravitational_constant * mass * inv * inv * inv)
                if use_quadrupoles != 0:
                    oq = walk_off_diagonal[entry]
                    qx = moments[1] * r[0] + oq[0] * r[1] + oq[1] * r[2]
                    qy = oq[0] * r[0] + moments[2] * r[1] + oq[2] * r[2]
                    qz = oq[1] * r[0] + oq[2] * r[1] + moments[3] * r[2]
                    r_q_r = r[0] * qx + r[1] * qy + r[2] * qz
                    inv2 = inv * inv
                    inv5 = inv2 * inv2 * inv
                    inv7 = inv5 * inv2
                    a += vector(qx, qy, qz) * (-gravitational_constant * inv5)
                    a += r * (gravitational_constant * scalar(2.5) * r_q_r * inv7)
                interactions += 1
                entry = skip
            else:
                entry += 1

        if overflow != 0:
            accelerations[i] = vector()
            wp.atomic_max(diag, 3, wp.int64(1))
        else:
            accelerations[i] = a
        wp.atomic_add(diag, 4, wp.int64(interactions))

    @warp_kernel
    def relative_error(
        approx: wp.array(dtype=vector),
        direct: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        count: int,
        errors: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0:
            errors[i] = zero
            return
        denominator = wp.max(wp.length(direct[i]), scalar(1.0e-12))
        errors[i] = wp.length(approx[i] - direct[i]) / denominator

    @warp_kernel
    def kick(
        velocities: wp.array(dtype=vector), accelerations: wp.array(dtype=vector), dt: scalar
    ):
        i = wp.tid()
        velocities[i] = velocities[i] + accelerations[i] * dt

    @warp_kernel
    def drift(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        count: int,
        dt: scalar,
    ):
        i = wp.tid()
        if i < count and active[i] != 0:
            positions[i] = positions[i] + velocities[i] * dt

    @dataclass(frozen=True)
    class Kernels:
        clear_bounds: Any
        reduce_bounds: Any
        finalize_bounds: Any
        compute_morton: Any
        bitonic_step: Any
        clear_tree: Any
        clear_slot_nodes: Any
        clear_node_cursor: Any
        build_sparse_topology: Any
        accumulate_sparse_nodes: Any
        accumulate_sparse_leaf_nodes: Any
        reduce_sparse_node_level: Any
        finalize_sparse_node_quadrupoles: Any
        clear_diag: Any
        clear_tree_flags: Any
        clear_tree_report: Any
        collect_tree_diag: Any
        fold_tree_overflow: Any
        reset_traversal_diag: Any
        direct_acceleration: Any
        dominant_acceleration: Any
        add_vectors: Any
        clear_accelerations: Any
        tree_acceleration: Any
        prepare_walk_slots: Any
        total_walk_chunks: Any
        offset_walk_slots: Any
        fill_walk_nodes: Any
        tree_acceleration_walk: Any
        relative_error: Any
        kick: Any
        drift: Any

    return Kernels(
        clear_bounds,
        reduce_bounds,
        finalize_bounds,
        compute_morton,
        bitonic_step,
        clear_tree,
        clear_slot_nodes,
        clear_node_cursor,
        build_sparse_topology,
        accumulate_sparse_nodes,
        accumulate_sparse_leaf_nodes,
        reduce_sparse_node_level,
        finalize_sparse_node_quadrupoles,
        clear_diag,
        clear_tree_flags,
        clear_tree_report,
        collect_tree_diag,
        fold_tree_overflow,
        reset_traversal_diag,
        direct_acceleration,
        dominant_acceleration,
        add_vectors,
        clear_accelerations,
        tree_acceleration,
        prepare_walk_slots,
        total_walk_chunks,
        offset_walk_slots,
        fill_walk_nodes,
        tree_acceleration_walk,
        relative_error,
        kick,
        drift,
    )
