"""SPH neighbour service: Warp ``HashGrid`` search and the bounded adaptive-``h`` solve.

This is the H1 stage of `docs/source-audit/SPEC/06_SPH_AND_NEIGHBOR_SEARCH.md` and the
local-neighbour half of ARCH-004: it is a spatial service of its own, independent of the
gravity octree in `tree.py`, so neither structure has to become universal before the other
needs to change.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .nbody import Precision, _dtypes, _registrars
from .runtime_foundation import DeviceContext, DeviceRequirementError

MAX_NEIGHBOR_ITERATIONS = 64
MIN_HASH_DIMENSION = 8
MAX_HASH_DIMENSION = 128
# Buckets the hash table holds per particle (ADR 0130). Warp's table wraps every ``dim``
# cells, so cells one period apart share a bucket; at one bucket a particle the 100k Asteroid
# Impact's impactor sat one period from the target and its threads walked thousands of the
# target's points. Sixteen is the smallest load measured to remove that at 10k and 100k.
HASH_BUCKETS_PER_PARTICLE = 16

# Levels the adaptive backend may hold. Each level doubles the cell width, so the ladder has
# to be as tall as the *whole* `h` bracket is wide: `support_radius` is linear in `h`, so a
# bracket spanning eight decades is a radius span of 1e8 and needs 28 levels. A ladder that
# stops short does not lose correctness - the query clamps to the top level - but it loses the
# entire point, because a clamped query walks `(2r/c + 1)**3` cells again. The cap is set well
# above that so the eight-decade bracket the suite drives is covered outright; it bounds the
# per-stage cost, which is one grid build per level over every particle and measures at
# 0.7 ms per level at 262144 particles on the target device.
MAX_NEIGHBOR_LEVELS = 32

# Consecutive iterations the power-law step may run without halving the bracket before a
# bisection is forced. It trades the guarantee against the measured cost, and both move: the
# worst case is `1 + (BISECTION_CADENCE + 1) * log2(span)` iterations, while a larger cadence
# lets the power-law step keep its own estimate instead of losing it to a midpoint. The value
# is measured, not assumed - `validation/sph_neighbor_suite.py` sweeps it over uniform and
# strongly clustered samples and reports the cost at each setting.
BISECTION_CADENCE = 4

# Solve launches issued per submission. This is a budget, not the worst case: the sequence
# is captured once, so the launch count is fixed at record time and cannot depend on the
# data. `NeighborConfig.iteration_bound()` states the worst case the safeguard guarantees,
# which is far larger. The budget is the worst case measured over the configured sweep in
# `validation/sph_neighbor_suite.py` - 18 iterations, on 262144 Plummer particles from a
# bracket spanning eight decades - times `ITERATION_BUDGET_MARGIN`, rounded up. A
# deliberately mistuned cell width in the same suite reached 22 and still converged every
# particle, so the margin over everything measured is 1.8 rather than 2.0. It is a budget
# and not a guarantee in either case: a particle that does not converge inside it is
# reported as `STATE_EXHAUSTED` rather than accepted, so the cost of the budget being wrong
# is a visible diagnostic, never a silently bad smoothing length. Raising it costs one full
# launch over every particle per unit, which is why it tracks the measurement.
DEFAULT_ITERATION_BUDGET = 40
ITERATION_BUDGET_MARGIN = 2.0

# Per-particle outcome of the bounded solve. The two converged states are separated because
# they make different claims: `COUNT_CONVERGED` reached the requested neighbour band, while
# `BRACKET_CONVERGED` resolved `h` to the requested precision without the band being
# reachable - the neighbour count is an integer step function of `h`, so a band narrower
# than one particle crossing has no solution at all. The three failure states are counted
# and reported rather than clamped away.
STATE_ITERATING = 0
STATE_COUNT_CONVERGED = 1
STATE_BRACKET_CONVERGED = 2
STATE_EXHAUSTED = 3
STATE_SATURATED_MIN = 4
STATE_SATURATED_MAX = 5
# A particle whose ``h`` its caller sets and the search only counts at (ADR 0170): a solid's
# carried density decides its support, so solving it to a neighbour count would be overwritten.
STATE_HELD = 6


class SphKernel(StrEnum):
    CUBIC_SPLINE = "cubic_spline"
    WENDLAND_C2 = "wendland_c2"
    WENDLAND_C4 = "wendland_c4"


class NeighborBackend(StrEnum):
    """Which spatial structure answers the radius queries.

    Both backends answer the same question and are gated on returning the same sets; they
    differ only in how many cells a query has to walk to answer it. A uniform grid fixes one
    cell width for the whole cloud, so a particle whose search radius has grown past that
    width scans `(2r/c + 1)**3` cells to find the handful of neighbours inside `r`. The
    adaptive backend keeps a geometric ladder of widths over the same points and sends each
    query to the level whose cells are at least as wide as its own radius, which bounds that
    factor at 27 cells regardless of how far the local density has drifted from the mean.
    """

    UNIFORM_HASH_GRID = "warp_hash_grid"
    ADAPTIVE_MULTIGRID = "warp_multigrid"


@dataclass(frozen=True)
class KernelSpec:
    """One smoothing kernel under a single stated support convention.

    Every kernel in this registry is written so its support radius is ``support_ratio * h``
    with ``support_ratio == 2``. Kernels are published in both conventions in the
    literature, and mixing them silently rescales every neighbour count and every density,
    so the convention is declared once here and the search radius is derived from it rather
    than configured separately.

    ``normalization`` is the ``sigma`` in ``W(r, h) = sigma / h**3 * shape(r / h)``, chosen
    so ``W`` integrates to one over three dimensions. ``recommended_neighbors`` is the
    default target for the adaptive solve, not a threshold: fewer neighbours than the
    kernel's shape supports raises the density estimator's variance, and more invites the
    pairing instability, so each kernel carries its own default.
    """

    name: SphKernel
    support_ratio: float
    normalization: float
    recommended_neighbors: int

    def support_radius(self, smoothing_length: float) -> float:
        return self.support_ratio * smoothing_length

    def value(self, radius: float, smoothing_length: float) -> float:
        """``W(r, h)``, zero outside the support."""

        if smoothing_length <= 0.0:
            raise ValueError("smoothing_length must be positive")
        q = radius / smoothing_length
        if q >= self.support_ratio:
            return 0.0
        return self.normalization / smoothing_length**3 * _shape(self.name, q)

    def gradient(self, radius: float, smoothing_length: float) -> float:
        """``dW/dr`` at ``r``, zero outside the support and zero at the peak."""

        if smoothing_length <= 0.0:
            raise ValueError("smoothing_length must be positive")
        q = radius / smoothing_length
        if q >= self.support_ratio:
            return 0.0
        return self.normalization / smoothing_length**4 * _shape_derivative(self.name, q)


def _shape(name: SphKernel, q: float) -> float:
    if name == SphKernel.CUBIC_SPLINE:
        if q < 1.0:
            return 1.0 - 1.5 * q * q + 0.75 * q * q * q
        return 0.25 * (2.0 - q) ** 3
    if name == SphKernel.WENDLAND_C2:
        return (1.0 - 0.5 * q) ** 4 * (1.0 + 2.0 * q)
    return (1.0 - 0.5 * q) ** 6 * (1.0 + 3.0 * q + 35.0 / 12.0 * q * q)


def _shape_derivative(name: SphKernel, q: float) -> float:
    if name == SphKernel.CUBIC_SPLINE:
        if q < 1.0:
            return -3.0 * q + 2.25 * q * q
        return -0.75 * (2.0 - q) ** 2
    if name == SphKernel.WENDLAND_C2:
        return -5.0 * q * (1.0 - 0.5 * q) ** 3
    return -(7.0 / 3.0) * q * (1.0 - 0.5 * q) ** 5 * (2.0 + 5.0 * q)


# Normalizations are ``1 / (4 * pi * integral(shape(q) * q**2, 0, 2))`` in three dimensions.
# The unit tests recompute each one by quadrature rather than trusting the constant here.
KERNEL_REGISTRY: dict[SphKernel, KernelSpec] = {
    SphKernel.CUBIC_SPLINE: KernelSpec(
        name=SphKernel.CUBIC_SPLINE,
        support_ratio=2.0,
        normalization=1.0 / math.pi,
        recommended_neighbors=50,
    ),
    SphKernel.WENDLAND_C2: KernelSpec(
        name=SphKernel.WENDLAND_C2,
        support_ratio=2.0,
        normalization=21.0 / (16.0 * math.pi),
        recommended_neighbors=100,
    ),
    SphKernel.WENDLAND_C4: KernelSpec(
        name=SphKernel.WENDLAND_C4,
        support_ratio=2.0,
        normalization=495.0 / (256.0 * math.pi),
        recommended_neighbors=200,
    ),
}


def hash_dimension(count: int) -> int:
    """Hash table side length: the smallest cube holding ``HASH_BUCKETS_PER_PARTICLE`` buckets
    a particle, so the table's period stays long against the scene and occupancy stays flat.

    The upper clamp bounds the table at ``128**3`` cells, a start and an end offset each,
    about 16 MB; it is reached at 131,072 particles. Past that the load per bucket grows
    again; the exit is a coarser level in a multi-level grid, not a larger table.
    """

    if count < 1:
        raise ValueError("count must be positive")
    buckets = HASH_BUCKETS_PER_PARTICLE * count
    dimension = max(1, round(buckets ** (1.0 / 3.0)))
    while dimension**3 < buckets:
        dimension += 1
    while dimension > 1 and (dimension - 1) ** 3 >= buckets:
        dimension -= 1
    return min(MAX_HASH_DIMENSION, max(MIN_HASH_DIMENSION, dimension))


@dataclass(frozen=True)
class NeighborConfig:
    capacity: int
    count: int
    h_init: float
    h_min: float
    h_max: float
    kernel: SphKernel = SphKernel.CUBIC_SPLINE
    target_neighbors: int = 0
    neighbor_tolerance: int = 2
    max_iterations: int = DEFAULT_ITERATION_BUDGET
    h_relative_tolerance: float = 1.0e-4
    neighbor_list_capacity: int = 0
    precision: Precision = Precision.FLOAT32
    backend: NeighborBackend = NeighborBackend.UNIFORM_HASH_GRID
    levels: int = 0

    @property
    def spec(self) -> KernelSpec:
        return KERNEL_REGISTRY[self.kernel]

    @property
    def resolved_target_neighbors(self) -> int:
        return self.target_neighbors or self.spec.recommended_neighbors

    @property
    def cell_width(self) -> float:
        """Grid bucketing width, matched to the search radius the initial ``h`` implies.

        Correctness does not depend on this: ``hash_grid_query`` derives its cell range from
        the radius it is given, so a particle whose ``h`` has grown to ``h_max`` still sees
        every neighbour. Only the number of cells scanned depends on the match, which is why
        the width follows the typical radius rather than the worst case.
        """

        return self.spec.support_radius(self.h_init)

    @property
    def base_cell_width(self) -> float:
        """Width of the finest level: the search radius the smallest allowed ``h`` implies.

        The ladder starts here rather than at ``h_init`` because a level coarser than a
        query's radius is what the query needs, and the finest radius any particle can ask
        for is the one at ``h_min``. Starting coarser would leave the densest particles
        walking cells that hold far more points than they will keep.
        """

        return self.spec.support_radius(self.h_min)

    @property
    def resolved_levels(self) -> int:
        """Levels needed for the coarsest reachable radius to land on a level at least as
        wide as itself, given each level doubles the one below it."""

        if self.levels:
            return self.levels
        span = self.spec.support_radius(self.h_max) / self.base_cell_width
        return min(MAX_NEIGHBOR_LEVELS, 1 + max(0, math.ceil(math.log2(max(span, 1.0)))))

    def level_widths(self) -> list[float]:
        return [self.base_cell_width * (2.0**level) for level in range(self.resolved_levels)]

    def validate(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 < self.count <= self.capacity:
            raise ValueError("count must be in (0, capacity]")
        if self.h_min <= 0.0:
            raise ValueError("h_min must be positive")
        if not self.h_min <= self.h_init <= self.h_max:
            raise ValueError("h_init must be in [h_min, h_max]")
        if self.target_neighbors < 0:
            raise ValueError("target_neighbors must be non-negative")
        if self.resolved_target_neighbors < 1:
            raise ValueError("resolved target_neighbors must be positive")
        if self.neighbor_tolerance < 0:
            raise ValueError("neighbor_tolerance must be non-negative")
        if not 1 <= self.max_iterations <= MAX_NEIGHBOR_ITERATIONS:
            raise ValueError(f"max_iterations must be in [1, {MAX_NEIGHBOR_ITERATIONS}]")
        if not 0.0 < self.h_relative_tolerance < 1.0:
            raise ValueError("h_relative_tolerance must be in (0, 1)")
        if self.neighbor_list_capacity < 0:
            raise ValueError("neighbor_list_capacity must be non-negative")
        if self.levels < 0:
            raise ValueError("levels must be non-negative")
        if not 1 <= self.resolved_levels <= MAX_NEIGHBOR_LEVELS:
            raise ValueError(f"resolved levels must be in [1, {MAX_NEIGHBOR_LEVELS}]")

    def bracket_span(self) -> float:
        """Opening bracket width in units of the width the solve must reach.

        Both the worst-case bound and the measured convergence law are stated against this
        number rather than against ``h_max / h_min``, because what the safeguard shrinks is
        the interval and what it is shrinking towards is the relative tolerance.
        """

        return (self.h_max - self.h_min) / (self.h_relative_tolerance * self.h_min)

    def iteration_bound(self) -> int:
        """Worst-case iterations the safeguarded solve can spend resolving ``h``.

        The neighbour count is monotone in ``h`` by set inclusion, so every iteration keeps a
        true bracket, and the safeguard takes the power-law step only while the bracket keeps
        halving. After ``BISECTION_CADENCE`` iterations that fail to halve it, a bisection is
        forced instead, and its midpoint evaluation halves the bracket on the next iteration.
        The width therefore halves at least once every ``BISECTION_CADENCE + 1`` iterations,
        from ``h_max - h_min`` down to ``h_relative_tolerance * h_min``, which is the law
        below. One iteration is added for the opening evaluation at ``h_init``, which
        establishes the bracket rather than shrinking it.

        This is the guarantee, not the expectation. The power-law step inverts the ``h**3``
        neighbour-count law exactly for a locally uniform sample, so the measured cost is a
        handful of iterations; the proof reports both, and the ratio is the margin.
        """

        halvings = max(1, math.ceil(math.log2(max(self.bracket_span(), 2.0))))
        return 1 + (BISECTION_CADENCE + 1) * halvings


@dataclass(frozen=True)
class NeighborDiagnostics:
    active_count: int
    neighbor_min: int
    neighbor_max: int
    neighbor_mean: float
    count_converged: int
    bracket_converged: int
    exhausted: int
    saturated_min: int
    saturated_max: int
    iterations_used: int
    iteration_bound: int
    list_overflow: int
    stage_ms: float
    target_neighbors: int
    neighbor_tolerance: int
    kernel: str
    support_ratio: float
    cell_width: float
    hash_dimension: int
    search_backend: str = "warp_hash_grid"
    submission: str = "eager"
    levels: int = 1
    level_width_max: float = 0.0
    held: int = 0

    @property
    def unconverged(self) -> int:
        return self.exhausted + self.saturated_min + self.saturated_max


class NeighborRuntime:
    """Own a Warp ``HashGrid``, the per-particle smoothing lengths, and the bounded solve.

    The whole stage is submitted as one recorded sequence: the grid build, one bounded
    solve launch, the final count, and the diagnostic reduction. The iteration *bound* is
    fixed rather than data dependent because an early exit decided on the host would need a
    readback per iteration, which would both stall the device and make the sequence illegal
    to capture. The bound is carried into the kernel instead, so each thread stops at its
    own particle's convergence: the exit is per particle and on the device, and the budget
    still costs nothing to record.
    """

    def __init__(self, device_context: DeviceContext, config: NeighborConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.last_stage_ms = 0.0
        self.explicit_validation_readbacks = 0
        self._wp: Any | None = None
        self._np: Any | None = None
        self._kernels: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._graphs: dict[str, Any] = {}
        self._grid: Any | None = None
        self._grids: list[Any] = []
        self._hash_dimension = hash_dimension(config.count)
        self._level_count = (
            config.resolved_levels
            if config.backend is NeighborBackend.ADAPTIVE_MULTIGRID
            else 1
        )

    @property
    def hash_dimension(self) -> int:
        return self._hash_dimension

    def load_points(self, positions: Any) -> None:
        """Upload particle positions and allocate the solve state.

        Positions arrive as a plain array rather than as ``BindingDescriptor`` values: the
        Emitter role that will own SPH particles is gated on this very requirement, so the
        service is built against the geometry it needs and bound to the product graph when
        that role becomes runnable.
        """

        self.device_context.initialize()
        self._wp = self.device_context.wp
        self._np = __import__("numpy")
        np = self._np
        wp = self._wp
        self._kernels = _kernels(wp, self.config.precision)
        scalar, vector, state_dtype = _dtypes(wp, self.config.precision)
        device = self.device_context.device
        supplied = np.asarray(positions, dtype=state_dtype)
        if supplied.shape != (self.config.count, 3):
            raise ValueError(
                f"positions must have shape ({self.config.count}, 3), got {supplied.shape}"
            )
        points = np.zeros((self.config.capacity, 3), dtype=state_dtype)
        points[: self.config.count] = supplied
        list_slots = self.config.count * max(self.config.neighbor_list_capacity, 1)

        self._arrays = {
            "positions": wp.array(points, dtype=vector, device=device),
            "smoothing_lengths": wp.zeros(self.config.capacity, dtype=scalar, device=device),
            "bracket_low": wp.zeros(self.config.capacity, dtype=scalar, device=device),
            "bracket_high": wp.zeros(self.config.capacity, dtype=scalar, device=device),
            "bracket_reference": wp.zeros(self.config.capacity, dtype=scalar, device=device),
            "stalled": wp.zeros(self.config.capacity, dtype=wp.int32, device=device),
            "neighbor_counts": wp.zeros(self.config.capacity, dtype=wp.int32, device=device),
            "state": wp.zeros(self.config.capacity, dtype=wp.int32, device=device),
            "iterations": wp.zeros(self.config.capacity, dtype=wp.int32, device=device),
            "neighbor_lists": wp.full(list_slots, -1, dtype=wp.int32, device=device),
            "diag": wp.zeros(11, dtype=wp.int32, device=device),
        }
        self._grids = [
            wp.HashGrid(
                self._hash_dimension,
                self._hash_dimension,
                self._hash_dimension,
                device=device,
                dtype=scalar,
            )
            for _level in range(self._level_count)
        ]
        self._grid = self._grids[0]
        if self.config.backend is NeighborBackend.ADAPTIVE_MULTIGRID:
            # The handles are read once here rather than rebuilt per submission: a grid's id
            # is fixed at construction and `build` only refills what it points at, so a
            # captured sequence can hold them.
            self._arrays["grid_ids"] = wp.array(
                np.array([grid.id for grid in self._grids], dtype=np.uint64),
                dtype=wp.uint64,
                device=device,
            )
        self._graphs.clear()

    def solve(self) -> None:
        """Submit the neighbour stage and wait for it, leaving the result on the device."""

        if not self._arrays:
            raise DeviceRequirementError("NeighborRuntime has no loaded points")
        wp = self._require_wp()
        start = time.perf_counter()
        self.build_grid()
        self._submit("neighbors", self.record_stage)
        wp.synchronize_device(self.device_context.device)
        self.last_stage_ms = (time.perf_counter() - start) * 1000.0

    def build_grid(self) -> None:
        """Rebuild the grid from the current positions, issued by the host, never captured.

        ``wp.HashGrid.build`` must not be recorded into a CUDA graph (ADR 0096). Warp 1.15.0
        fills one function-local ``static`` descriptor per dtype and copies it to the device,
        and sorts through a temporary buffer cached per stream and freed when a larger sort
        runs there. A graph that captured the build replays the copy from whichever grid of
        the same dtype was built last anywhere in the process - the fragment-boundary
        diagnostic's, or a second SPH system's - and the sort into a buffer it may no longer
        own. Issued here, both are resolved when the build runs. The build is ordered on the
        stream like any launch, so it still reads the positions a preceding replay wrote,
        without a host synchronization.
        """

        if "positions" not in self._arrays:
            raise DeviceRequirementError("NeighborRuntime has no loaded points")
        config = self.config
        points = self._arrays["positions"][: config.count]
        if config.backend is NeighborBackend.ADAPTIVE_MULTIGRID:
            for level, width in enumerate(config.level_widths()):
                self._grids[level].build(points, width)
        else:
            self._require_grid().build(points, config.cell_width)

    def record_stage(self, held_lengths: Any | None = None) -> None:
        """Write the neighbour search into the caller's sequence, against the built grid.

        A stage built on top of this one - the hydrodynamic core in `hydro.py` - has to
        re-solve ``h`` inside its own step, and a captured graph cannot launch another
        captured graph. Exposing the recording rather than the submission lets that stage
        capture the search as part of its own sequence. The grid build is not part of it:
        the caller issues `build_grid` first, outside the capture, for the reason given there.

        ``held_lengths``, when given, is a per-particle ``h`` the caller has already decided:
        a positive entry is taken as that particle's length and only counted at, never
        solved (`STATE_HELD`). A solid's continuity density sets its support (ADR 0170);
        solving it as well was thrown away, and for a lone fragment it was the search
        hunting out to ``h_max`` every step.
        """

        self._record_search(held_lengths)

    @property
    def grid_id(self) -> Any:
        """The ``wp.uint64`` grid handle kernels take as their first argument."""

        return self._require_grid().id

    @property
    def positions(self) -> Any:
        """The device position array the grid is built from.

        Returned rather than copied: a stage that integrates positions has to move the same
        array the next grid build reads, and a second copy is a second source of truth.
        """

        if "positions" not in self._arrays:
            raise DeviceRequirementError("NeighborRuntime has no loaded points")
        return self._arrays["positions"]

    @property
    def smoothing_length_array(self) -> Any:
        """The device array of solved smoothing lengths, without a readback."""

        if "smoothing_lengths" not in self._arrays:
            raise DeviceRequirementError("NeighborRuntime has no loaded points")
        return self._arrays["smoothing_lengths"]

    def _graph(self, name: str, record: Any) -> Any:
        """Capture one launch sequence once so later submissions replay a CUDA graph.

        Every launch dimension comes from the frozen config and every solve-state array is
        allocated once in ``load_points``, so the recorded sequence never synchronizes or
        reads back. The grid build is not in the sequence (see `build_grid`); it used to be,
        and ``wp.HashGrid.build`` is not allocation-free - its radix sort reserves temporary
        device memory sized from the point count, and on a memory-pool device a pool-growing
        allocation *during* an active stream capture serializes against the capture and can
        stall it. That is why ``record`` is run once, untimed, outside ``ScopedCapture``
        before it is captured: any allocation the sequence performs sizes the pool here, where
        growing it is legal. With the build moved out the searched sequence allocates nothing
        known, and the warm-up is kept as the cheap guard against one that is added later - a
        one-time cost per capture, not per replay. It writes only to the pre-allocated arrays
        the captured pass overwrites - ``reset_state`` re-initializes the solve state - so the
        post-capture device state, and every solved output, is identical to a capture with no
        warm-up.
        """

        if name in self._graphs:
            return self._graphs[name]
        wp = self._require_wp()
        device = wp.get_device(self.device_context.device)
        if not device.is_cuda:
            self._graphs[name] = None
            return None
        wp.force_load(device=device)
        # Warm up the pool: run the sequence once outside capture so any allocation it makes
        # sizes the mempool here, where a pool-growing allocation is legal, instead of inside
        # capture, where it can stall.
        record()
        with wp.ScopedCapture(device) as capture:
            record()
        self._graphs[name] = capture.graph
        return capture.graph

    def _submit(self, name: str, record: Any) -> None:
        graph = self._graph(name, record)
        if graph is None:
            record()
        else:
            self._require_wp().capture_launch(graph)

    def _record_solve(self) -> None:
        """The whole stage issued eagerly, build then search - what one `solve` computes."""

        self.build_grid()
        self._record_search()

    def _record_search(self, held_lengths: Any | None = None) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        config = self.config
        device = self.device_context.device
        grid = self._require_grid()
        spec = config.spec
        count = config.count
        if config.backend is NeighborBackend.ADAPTIVE_MULTIGRID:
            solve_sweep = kernels.solve_sweep_leveled
            finalize = kernels.finalize_leveled
            search = [self._arrays["grid_ids"], grid.id, config.base_cell_width, self._level_count]
        else:
            solve_sweep = kernels.solve_sweep
            finalize = kernels.finalize
            search = [grid.id]
        wp.launch(
            kernels.reset_state,
            dim=count,
            inputs=[
                self._arrays["smoothing_lengths"],
                self._arrays["bracket_low"],
                self._arrays["bracket_high"],
                self._arrays["bracket_reference"],
                self._arrays["stalled"],
                self._arrays["neighbor_counts"],
                self._arrays["state"],
                self._arrays["iterations"],
                config.h_init,
                config.h_min,
                config.h_max,
            ],
            device=device,
        )
        if held_lengths is not None:
            wp.launch(
                kernels.hold_state,
                dim=count,
                inputs=[held_lengths, self._arrays["smoothing_lengths"], self._arrays["state"]],
                device=device,
            )
        wp.launch(kernels.clear_diag, dim=1, inputs=[self._arrays["diag"]], device=device)
        # One launch, not `max_iterations` of them: the budget is carried into the kernel
        # and each thread runs its own particle's iterations, stopping as soon as that
        # particle settles. See the note above `solve_sweep` for why the barriers the
        # per-iteration launches imposed were not carrying anything.
        wp.launch(
            solve_sweep,
            dim=count,
            inputs=[
                *search,
                self._arrays["positions"],
                self._arrays["smoothing_lengths"],
                self._arrays["bracket_low"],
                self._arrays["bracket_high"],
                self._arrays["bracket_reference"],
                self._arrays["stalled"],
                self._arrays["neighbor_counts"],
                self._arrays["state"],
                self._arrays["iterations"],
                spec.support_ratio,
                config.resolved_target_neighbors,
                config.neighbor_tolerance,
                BISECTION_CADENCE,
                config.h_relative_tolerance,
                config.h_min,
                config.h_max,
                config.max_iterations,
            ],
            device=device,
        )
        wp.launch(
            finalize,
            dim=count,
            inputs=[
                *search,
                self._arrays["positions"],
                self._arrays["smoothing_lengths"],
                self._arrays["neighbor_counts"],
                self._arrays["state"],
                self._arrays["neighbor_lists"],
                self._arrays["diag"],
                spec.support_ratio,
                config.resolved_target_neighbors,
                config.neighbor_tolerance,
                config.neighbor_list_capacity,
                config.h_min,
                config.h_max,
            ],
            device=device,
        )
        wp.launch(
            kernels.reduce_diag,
            dim=count,
            inputs=[
                self._arrays["neighbor_counts"],
                self._arrays["state"],
                self._arrays["iterations"],
                self._arrays["diag"],
            ],
            device=device,
        )

    def diagnostics(self) -> NeighborDiagnostics:
        wp = self._require_wp()
        wp.synchronize_device(self.device_context.device)
        diag = self._arrays["diag"].numpy()
        count = self.config.count
        return NeighborDiagnostics(
            active_count=count,
            neighbor_min=int(diag[0]),
            neighbor_max=int(diag[1]),
            neighbor_mean=float(diag[2]) / float(count),
            count_converged=int(diag[3]),
            bracket_converged=int(diag[4]),
            exhausted=int(diag[5]),
            saturated_min=int(diag[6]),
            saturated_max=int(diag[7]),
            iterations_used=int(diag[8]),
            iteration_bound=self.config.iteration_bound(),
            list_overflow=int(diag[9]),
            held=int(diag[10]),
            stage_ms=self.last_stage_ms,
            target_neighbors=self.config.resolved_target_neighbors,
            neighbor_tolerance=self.config.neighbor_tolerance,
            kernel=str(self.config.kernel),
            support_ratio=self.config.spec.support_ratio,
            cell_width=self.config.cell_width,
            hash_dimension=self._hash_dimension,
            submission="cuda_graph" if any(self._graphs.values()) else "eager",
            search_backend=str(self.config.backend),
            levels=self._level_count,
            level_width_max=self.config.level_widths()[-1]
            if self.config.backend is NeighborBackend.ADAPTIVE_MULTIGRID
            else self.config.cell_width,
        )

    def smoothing_lengths(self) -> list[float]:
        """Explicit validation readback of the solved ``h``."""

        return [float(value) for value in self._read("smoothing_lengths")]

    def neighbor_counts(self) -> list[int]:
        """Explicit validation readback of the neighbour count at the solved ``h``."""

        return [int(value) for value in self._read("neighbor_counts")]

    def states(self) -> list[int]:
        """Explicit validation readback of the per-particle convergence outcome."""

        return [int(value) for value in self._read("state")]

    def iterations(self) -> list[int]:
        """Explicit validation readback of the per-particle iteration count.

        The diagnostics report only the maximum, which is what the bound has to cover. The
        distribution is what shows whether the tail grows with the particle count, so the
        validation suite reads it here rather than inferring it from the maximum.
        """

        return [int(value) for value in self._read("iterations")]

    def neighbor_lists(self) -> list[list[int]]:
        """Explicit validation readback of the neighbour sets, for oracle comparison."""

        capacity = self.config.neighbor_list_capacity
        if capacity < 1:
            raise DeviceRequirementError(
                "neighbor_list_capacity must be configured before reading neighbour lists"
            )
        self.explicit_validation_readbacks += 1
        wp = self._require_wp()
        wp.synchronize_device(self.device_context.device)
        flat = self._arrays["neighbor_lists"].numpy()
        counts = self._arrays["neighbor_counts"].numpy()
        lists: list[list[int]] = []
        for index in range(self.config.count):
            stored = min(int(counts[index]), capacity)
            window = flat[index * capacity : index * capacity + stored]
            lists.append([int(value) for value in window])
        return lists

    def _read(self, name: str) -> Any:
        self.explicit_validation_readbacks += 1
        wp = self._require_wp()
        wp.synchronize_device(self.device_context.device)
        return self._arrays[name].numpy()[: self.config.count]

    def close(self) -> None:
        self._arrays.clear()
        self._graphs.clear()
        self._grid = None
        self._grids = []

    def _require_wp(self):
        if self._wp is None:
            raise DeviceRequirementError("Warp runtime is not loaded")
        return self._wp

    def _require_kernels(self):
        if self._kernels is None:
            raise DeviceRequirementError("Warp neighbour kernels are not loaded")
        return self._kernels

    def _require_grid(self):
        if self._grid is None:
            raise DeviceRequirementError("HashGrid is not allocated")
        return self._grid


def _kernels(wp, precision: Precision = Precision.FLOAT32):
    """Compile the neighbour kernels for one state precision.

    The search is a geometric predicate, so it runs at the state precision rather than in a
    float64 diagnostic lane: a reference computed at a different precision would disagree
    with the device about pairs that sit within round-off of the support radius, and the
    validation suite bounds that gap explicitly instead of hiding it behind a mixed lane.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, warp_func = _registrars(wp, scalar, vector)
    half = scalar(0.5)

    @warp_kernel
    def clear_diag(diag: wp.array(dtype=wp.int32)):
        diag[0] = 2147483647
        for slot in range(1, 11):
            diag[slot] = 0

    @warp_kernel
    def reset_state(
        smoothing_lengths: wp.array(dtype=scalar),
        bracket_low: wp.array(dtype=scalar),
        bracket_high: wp.array(dtype=scalar),
        bracket_reference: wp.array(dtype=scalar),
        stalled: wp.array(dtype=wp.int32),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        h_init: scalar,
        h_min: scalar,
        h_max: scalar,
    ):
        i = wp.tid()
        smoothing_lengths[i] = h_init
        bracket_low[i] = h_min
        bracket_high[i] = h_max
        # Twice the opening bracket width, so the first iteration always passes the
        # halving test and the opening power-law step runs unsafeguarded.
        bracket_reference[i] = scalar(2.0) * (h_max - h_min)
        stalled[i] = 0
        neighbor_counts[i] = 0
        state[i] = STATE_ITERATING
        iterations[i] = 0

    @warp_kernel
    def hold_state(
        held_lengths: wp.array(dtype=scalar),
        smoothing_lengths: wp.array(dtype=scalar),
        state: wp.array(dtype=wp.int32),
    ):
        # After `reset_state`: a held particle leaves the sweep before its first iteration
        # (the sweep runs only `STATE_ITERATING`) and `finalize` counts it at this length.
        i = wp.tid()
        length = held_lengths[i]
        if length > scalar(0.0):
            smoothing_lengths[i] = length
            state[i] = STATE_HELD

    @warp_func
    def count_within(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        i: int,
        radius: scalar,
    ):
        p = positions[i]
        query = wp.hash_grid_query(grid, p, radius)
        # `int(0)` rather than `0` is Warp's declaration syntax for a variable that a
        # dynamic loop may reassign; a bare literal is compiled as a constant and the
        # codegen rejects the mutation with "Error mutating a constant ... inside a
        # dynamic loop". Ruff reads it as a redundant call, so the rule is suppressed here.
        j = int(0)  # noqa: UP018
        found = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i and wp.length(positions[j] - p) <= radius:
                found += 1
        return found

    @warp_func
    def level_for(radius: scalar, base_width: scalar, level_count: int):
        """Coarsest-but-one level whose cells are at least as wide as the query radius.

        Widths double, so this is a logarithm. Clamping at the top is what keeps the
        structure correct rather than merely fast: a radius past the last level still gets
        answered, it just walks more cells, exactly as the uniform grid does everywhere.
        """

        ratio = wp.max(radius / base_width, scalar(1.0))
        level = int(wp.ceil(wp.log2(ratio)))
        if level > level_count - 1:
            level = level_count - 1
        return level

    @warp_func
    def count_within_leveled(
        grids: wp.array(dtype=wp.uint64),
        base_width: scalar,
        level_count: int,
        positions: wp.array(dtype=vector),
        i: int,
        radius: scalar,
    ):
        return count_within(
            grids[level_for(radius, base_width, level_count)], positions, i, radius
        )

    @warp_func
    def advance(
        i: int,
        h: scalar,
        found: int,
        smoothing_lengths: wp.array(dtype=scalar),
        bracket_low: wp.array(dtype=scalar),
        bracket_high: wp.array(dtype=scalar),
        bracket_reference: wp.array(dtype=scalar),
        stalled: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        target: int,
        tolerance: int,
        cadence: int,
        h_relative_tolerance: scalar,
        h_min: scalar,
        h_max: scalar,
    ):
        """One safeguarded step of the ``h`` solve, given the count at the current ``h``.

        Shared by both backends without modification: which structure produced ``found`` is
        not something the solve can see, which is what makes "the two backends return the
        same smoothing lengths" a property of the search rather than of the iteration.
        """

        low = bracket_low[i]
        high = bracket_high[i]
        settled = int(0)  # noqa: UP018
        if found > target + tolerance:
            high = h
        elif found < target - tolerance:
            low = h
        else:
            state[i] = STATE_COUNT_CONVERGED
            settled = 1
        if settled == 0:
            bracket_low[i] = low
            bracket_high[i] = high
            if high - low <= h_relative_tolerance * h:
                # A collapsed bracket only says `h` is resolved to the requested precision.
                # Whether that `h` answers the request is a separate question, and the answer
                # is no when the count is still outside the band and the bracket collapsed
                # onto a domain limit that never moved: the limit is what stopped the solve,
                # not the neighbour target. Reporting that as convergence would hide a whole
                # cloud of particles whose `h` is pinned at the wrong end of the allowed
                # range.
                if found > target + tolerance and low <= h_min:
                    state[i] = STATE_SATURATED_MIN
                elif found < target - tolerance and high >= h_max:
                    state[i] = STATE_SATURATED_MAX
                else:
                    state[i] = STATE_BRACKET_CONVERGED
                settled = 1
        if settled == 0:
            # The neighbour count of a locally uniform sample grows as h**3, so the exact
            # inverse of that law is the step for this problem, and it lands on the answer in
            # one evaluation wherever the sample is locally uniform. Bisection is strictly
            # worse whenever the model holds, because it throws the estimate away for a
            # midpoint, so the safeguard is a cadence rather than a per-iteration veto: the
            # model step runs while it keeps halving the bracket, and only after ``cadence``
            # consecutive iterations that fail to halve it does a bisection take over. That
            # still forces a halving at least every ``cadence + 1`` iterations, which is what
            # bounds the solve.
            width = high - low
            proposed = h * wp.cbrt(scalar(target) / wp.max(scalar(found), scalar(1.0)))
            if width <= half * bracket_reference[i]:
                bracket_reference[i] = width
                stalled[i] = 0
            else:
                waited = stalled[i] + 1
                if waited >= cadence:
                    proposed = half * (low + high)
                    waited = 0
                stalled[i] = waited
            if proposed <= low or proposed >= high:
                proposed = half * (low + high)
            smoothing_lengths[i] = proposed

    @warp_kernel
    def solve_step(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        bracket_low: wp.array(dtype=scalar),
        bracket_high: wp.array(dtype=scalar),
        bracket_reference: wp.array(dtype=scalar),
        stalled: wp.array(dtype=wp.int32),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        support_ratio: scalar,
        target: int,
        tolerance: int,
        cadence: int,
        h_relative_tolerance: scalar,
        h_min: scalar,
        h_max: scalar,
    ):
        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        if state[i] != STATE_ITERATING:
            return
        h = smoothing_lengths[i]
        found = count_within(grid, positions, i, support_ratio * h)
        neighbor_counts[i] = found
        iterations[i] = iterations[i] + 1
        advance(
            i,
            h,
            found,
            smoothing_lengths,
            bracket_low,
            bracket_high,
            bracket_reference,
            stalled,
            state,
            target,
            tolerance,
            cadence,
            h_relative_tolerance,
            h_min,
            h_max,
        )

    @warp_kernel
    def solve_step_leveled(
        grids: wp.array(dtype=wp.uint64),
        order_grid: wp.uint64,
        base_width: scalar,
        level_count: int,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        bracket_low: wp.array(dtype=scalar),
        bracket_high: wp.array(dtype=scalar),
        bracket_reference: wp.array(dtype=scalar),
        stalled: wp.array(dtype=wp.int32),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        support_ratio: scalar,
        target: int,
        tolerance: int,
        cadence: int,
        h_relative_tolerance: scalar,
        h_min: scalar,
        h_max: scalar,
    ):
        """The same step, with the query routed to the level that matches its own radius.

        ``order_grid`` is the finest level and supplies the traversal order only. Every level
        holds every particle, so which one enumerates them is a memory-locality choice and
        not a correctness one, and taking it from the finest level keeps the ordering the
        uniform backend already has.
        """

        slot = wp.tid()
        i = wp.hash_grid_point_id(order_grid, slot)
        if i < 0:
            return
        if state[i] != STATE_ITERATING:
            return
        h = smoothing_lengths[i]
        found = count_within_leveled(
            grids, base_width, level_count, positions, i, support_ratio * h
        )
        neighbor_counts[i] = found
        iterations[i] = iterations[i] + 1
        advance(
            i,
            h,
            found,
            smoothing_lengths,
            bracket_low,
            bracket_high,
            bracket_reference,
            stalled,
            state,
            target,
            tolerance,
            cadence,
            h_relative_tolerance,
            h_min,
            h_max,
        )

    # The two kernels below run the whole budget for one particle inside one thread, which
    # is what the stage actually issues. `solve_step` and `solve_step_leveled` are kept
    # because they are the per-iteration definition these sweeps must agree with, and
    # `tests/unit/test_neighbor_sweep_fold.py` holds them to it.
    #
    # Folding is sound because a particle's iteration reads `positions`, which the solve
    # never writes, and otherwise touches only its own slot of every array: no particle's
    # step depends on any other particle's step, and the grid is built once before the
    # solve and not rebuilt between iterations. The barrier between the launches was
    # therefore carrying nothing. What it cost was a `max_iterations`-by-count grid of
    # thread dispatches, in which every lane whose particle had already settled did nothing
    # but resolve its point id, load its own state and return.
    #
    # The budget is unchanged in kind: still a bound fixed at record time, independent of
    # the data, so the sequence stays legal to capture, and a particle that does not settle
    # inside it is still left `STATE_ITERATING` for `finalize` to report as
    # `STATE_EXHAUSTED`. What changes is only that a settled lane retires instead of being
    # re-dispatched.

    @warp_kernel
    def solve_sweep(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        bracket_low: wp.array(dtype=scalar),
        bracket_high: wp.array(dtype=scalar),
        bracket_reference: wp.array(dtype=scalar),
        stalled: wp.array(dtype=wp.int32),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        support_ratio: scalar,
        target: int,
        tolerance: int,
        cadence: int,
        h_relative_tolerance: scalar,
        h_min: scalar,
        h_max: scalar,
        max_iterations: int,
    ):
        """Every iteration of one particle's solve, in one thread, in one launch."""

        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        # `int(0)` rather than `0` for the same reason `count_within` declares its cursor
        # that way: a bare literal is a constant the codegen refuses to mutate in a loop.
        iteration = int(0)  # noqa: UP018
        while iteration < max_iterations:
            if state[i] != STATE_ITERATING:
                break
            h = smoothing_lengths[i]
            found = count_within(grid, positions, i, support_ratio * h)
            neighbor_counts[i] = found
            iterations[i] = iterations[i] + 1
            advance(
                i,
                h,
                found,
                smoothing_lengths,
                bracket_low,
                bracket_high,
                bracket_reference,
                stalled,
                state,
                target,
                tolerance,
                cadence,
                h_relative_tolerance,
                h_min,
                h_max,
            )
            iteration += 1

    @warp_kernel
    def solve_sweep_leveled(
        grids: wp.array(dtype=wp.uint64),
        order_grid: wp.uint64,
        base_width: scalar,
        level_count: int,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        bracket_low: wp.array(dtype=scalar),
        bracket_high: wp.array(dtype=scalar),
        bracket_reference: wp.array(dtype=scalar),
        stalled: wp.array(dtype=wp.int32),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        support_ratio: scalar,
        target: int,
        tolerance: int,
        cadence: int,
        h_relative_tolerance: scalar,
        h_min: scalar,
        h_max: scalar,
        max_iterations: int,
    ):
        """The same sweep, with each query routed to the level that matches its radius."""

        slot = wp.tid()
        i = wp.hash_grid_point_id(order_grid, slot)
        if i < 0:
            return
        iteration = int(0)  # noqa: UP018
        while iteration < max_iterations:
            if state[i] != STATE_ITERATING:
                break
            h = smoothing_lengths[i]
            found = count_within_leveled(
                grids, base_width, level_count, positions, i, support_ratio * h
            )
            neighbor_counts[i] = found
            iterations[i] = iterations[i] + 1
            advance(
                i,
                h,
                found,
                smoothing_lengths,
                bracket_low,
                bracket_high,
                bracket_reference,
                stalled,
                state,
                target,
                tolerance,
                cadence,
                h_relative_tolerance,
                h_min,
                h_max,
            )
            iteration += 1

    @warp_kernel
    def finalize(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        neighbor_lists: wp.array(dtype=wp.int32),
        diag: wp.array(dtype=wp.int32),
        support_ratio: scalar,
        target: int,
        tolerance: int,
        list_capacity: int,
        h_min: scalar,
        h_max: scalar,
    ):
        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        h = smoothing_lengths[i]
        radius = support_ratio * h
        p = positions[i]
        query = wp.hash_grid_query(grid, p, radius)
        # `int(0)` rather than `0` is Warp's declaration syntax for a variable that a
        # dynamic loop may reassign; a bare literal is compiled as a constant and the
        # codegen rejects the mutation with "Error mutating a constant ... inside a
        # dynamic loop". Ruff reads it as a redundant call, so the rule is suppressed here.
        j = int(0)  # noqa: UP018
        found = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i and wp.length(positions[j] - p) <= radius:
                if list_capacity > 0:
                    if found < list_capacity:
                        neighbor_lists[i * list_capacity + found] = j
                    else:
                        wp.atomic_add(diag, 9, 1)
                found += 1
        # The count has to describe the smoothing length that survived the loop, not the
        # one the last iteration was evaluating, or the reported pair would not be the one
        # the density stage will use.
        neighbor_counts[i] = found
        if state[i] == STATE_ITERATING:
            if found < target - tolerance and h >= h_max:
                state[i] = STATE_SATURATED_MAX
            elif found > target + tolerance and h <= h_min:
                state[i] = STATE_SATURATED_MIN
            else:
                state[i] = STATE_EXHAUSTED

    @warp_kernel
    def finalize_leveled(
        grids: wp.array(dtype=wp.uint64),
        order_grid: wp.uint64,
        base_width: scalar,
        level_count: int,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        neighbor_lists: wp.array(dtype=wp.int32),
        diag: wp.array(dtype=wp.int32),
        support_ratio: scalar,
        target: int,
        tolerance: int,
        list_capacity: int,
        h_min: scalar,
        h_max: scalar,
    ):
        slot = wp.tid()
        i = wp.hash_grid_point_id(order_grid, slot)
        if i < 0:
            return
        h = smoothing_lengths[i]
        radius = support_ratio * h
        p = positions[i]
        grid = grids[level_for(radius, base_width, level_count)]
        query = wp.hash_grid_query(grid, p, radius)
        # `int(0)` rather than `0` is Warp's declaration syntax for a variable that a
        # dynamic loop may reassign; a bare literal is compiled as a constant and the
        # codegen rejects the mutation with "Error mutating a constant ... inside a
        # dynamic loop". Ruff reads it as a redundant call, so the rule is suppressed here.
        j = int(0)  # noqa: UP018
        found = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i and wp.length(positions[j] - p) <= radius:
                if list_capacity > 0:
                    if found < list_capacity:
                        neighbor_lists[i * list_capacity + found] = j
                    else:
                        wp.atomic_add(diag, 9, 1)
                found += 1
        neighbor_counts[i] = found
        if state[i] == STATE_ITERATING:
            if found < target - tolerance and h >= h_max:
                state[i] = STATE_SATURATED_MAX
            elif found > target + tolerance and h <= h_min:
                state[i] = STATE_SATURATED_MIN
            else:
                state[i] = STATE_EXHAUSTED

    @warp_kernel
    def reduce_diag(
        neighbor_counts: wp.array(dtype=wp.int32),
        state: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        diag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        found = neighbor_counts[i]
        wp.atomic_min(diag, 0, found)
        wp.atomic_max(diag, 1, found)
        wp.atomic_add(diag, 2, found)
        outcome = state[i]
        if outcome == STATE_COUNT_CONVERGED:
            wp.atomic_add(diag, 3, 1)
        elif outcome == STATE_BRACKET_CONVERGED:
            wp.atomic_add(diag, 4, 1)
        elif outcome == STATE_EXHAUSTED:
            wp.atomic_add(diag, 5, 1)
        elif outcome == STATE_SATURATED_MIN:
            wp.atomic_add(diag, 6, 1)
        elif outcome == STATE_HELD:
            wp.atomic_add(diag, 10, 1)
        else:
            wp.atomic_add(diag, 7, 1)
        wp.atomic_max(diag, 8, iterations[i])

    @dataclass(frozen=True)
    class Kernels:
        clear_diag: Any
        reset_state: Any
        solve_step: Any
        solve_step_leveled: Any
        solve_sweep: Any
        solve_sweep_leveled: Any
        finalize: Any
        finalize_leveled: Any
        reduce_diag: Any
        hold_state: Any

    return Kernels(
        clear_diag,
        reset_state,
        solve_step,
        solve_step_leveled,
        solve_sweep,
        solve_sweep_leveled,
        finalize,
        finalize_leveled,
        reduce_diag,
        hold_state,
    )
