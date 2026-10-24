"""Warp-only direct N-body runtime for the Phase 2 orbital slice."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, replace
from enum import StrEnum
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
    BindingDescriptor,
    Precision,
)
from .runtime_foundation import DeviceContext, DeviceRequirementError
from .sampling import BodySet, expand_bindings
from .shearing_box import ShearingBoxSpec, shearing_arrays, shearing_kernels


class CollisionMode(StrEnum):
    IGNORE = "ignore"
    MERGE = "merge"
    ELASTIC = "elastic"
    # Sphere-boundary accretion (ADR 0037 §3.3): particles inside a flagged Body's OWN
    # radius are absorbed into it, conserving mass and momentum barycentrically. Direct
    # lane only; the tree lane's collision path refuses it (see tree.py).
    ABSORB = "absorb"


def _dtypes(wp: Any, precision: Precision) -> tuple[Any, Any, str]:
    """Warp scalar type, Warp vector type, and NumPy dtype name for a precision."""

    if precision == Precision.FLOAT64:
        return wp.float64, wp.vec3d, "float64"
    return wp.float32, wp.vec3, "float32"


def _registrars(wp: Any, scalar: Any, vector: Any, **types: Any) -> tuple[Any, Any]:
    """Return ``wp.kernel``/``wp.func`` replacements that resolve parametrized annotations.

    ``from __future__ import annotations`` stringizes every annotation, and Warp evaluates
    those strings against the callable's real closure only. A precision name that appears
    just in the signature is not a closure variable, so ``dtype=vector`` would resolve for
    the kernels whose body also happens to mention ``vector`` and raise ``NameError`` for
    the ones that do not. Resolving against an explicit namespace here makes registration
    independent of what the body says. ``types`` adds further precision-dependent names.
    """

    namespace = {"wp": wp, "scalar": scalar, "vector": vector, **types}

    def resolve(func: Any) -> Any:
        func.__annotations__ = inspect.get_annotations(func, eval_str=True, locals=namespace)
        return func

    def register_kernel(func: Any) -> Any:
        return wp.kernel(resolve(func))

    def register_func(func: Any) -> Any:
        return wp.func(resolve(func))

    return register_kernel, register_func


@dataclass(frozen=True)
class NBodyConfig:
    capacity: int
    count: int
    gravitational_constant: float
    dt: float
    substeps: int = 1
    softening: float = 0.0
    collision_mode: CollisionMode = CollisionMode.IGNORE
    event_capacity: int = 1024
    precision: Precision = Precision.FLOAT32
    # External force fields (ADR 0045): one spec per active FIELD binding, applied by a
    # kernel launched after gravity into the same accelerations array. Empty means the
    # runtime this class described before the Force role existed, byte for byte.
    external_fields: tuple[ExternalFieldSpec, ...] = ()
    # Emitter release schedules (ADR 0037 3.6): one spec per active EMITTER binding. These
    # add nothing to `capacity` and nothing to `count` -- an emitter's particles are drawn
    # by the ordinary cloud expansion at load and held inactive -- so a spec only names a
    # slot range that is already loaded. Empty means the runtime this class described
    # before the Emitter role existed, byte for byte.
    emitters: tuple[EmitterSpec, ...] = ()
    # A local shearing box (ADR 0202): Hill's equations in a frame turning at ``omega`` with
    # shearing-periodic boundaries, in place of an inertial open space. ``None`` is the runtime
    # this class described before the option existed, byte for byte: no extra array, no extra
    # launch, and the same ``drift`` and ``acceleration`` kernels.
    shearing_box: ShearingBoxSpec | None = None

    def validate(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 < self.count <= self.capacity:
            raise ValueError("count must be in (0, capacity]")
        if self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        if self.softening < 0.0:
            raise ValueError("softening must be non-negative")
        if self.event_capacity < 1:
            raise ValueError("event_capacity must be positive")
        for field in self.external_fields:
            field.validate()
        for spec in self.emitters:
            spec.validate()
            validate_slot_range(spec, self.count)
        if self.shearing_box is not None:
            self.shearing_box.validate()
            # Each of these assumes an inertial, unwrapped space the box does not have; refusing
            # here puts the reason in plain words ahead of a run that would quietly be wrong.
            if self.collision_mode != CollisionMode.IGNORE:
                raise ValueError(
                    "a shearing box cannot resolve collisions: a body that leaves through one "
                    "side re-enters through the other, so the collision stage's neighbours are "
                    "wrong. Set the System's Collision to Ignore"
                )
            if self.external_fields:
                raise ValueError(
                    "a shearing box cannot carry force fields: a field's position means "
                    "nothing in a box that repeats. Remove the Force members"
                )
            if self.emitters:
                raise ValueError(
                    "a shearing box cannot carry emitters: the release schedule is written for "
                    "an open space. Remove the Emitter members"
                )


@dataclass(frozen=True)
class NBodyDiagnostics:
    active_count: int
    total_energy: float
    kinetic_energy: float
    potential_energy: float
    linear_momentum: tuple[float, float, float]
    angular_momentum: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    max_acceleration: float
    collision_events: int
    collision_overflow: bool
    collision_broadphase: str
    collision_grid_overflow: bool
    step_ms: float


class DirectNBodyRuntime:
    """Owns device arrays for direct O(N^2) gravity and KDK stepping."""

    def __init__(self, device_context: DeviceContext, config: NBodyConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.step_index = 0
        # Frames finished so far. Authored frame numbers come from this counter, not from
        # `step_index // substeps`, which mis-numbers frames once `substeps` changes (SR-29).
        self.frame_index = 0
        self.last_step_ms = 0.0
        self.last_solver_primitive_ms = 0.0
        self.full_state_playback_readbacks = 0
        self.explicit_snapshot_readbacks = 0
        self._wp: Any | None = None
        self._np: Any | None = None
        self._arrays: dict[str, Any] = {}
        # Captured CUDA graphs keyed by launch-sequence name. Populated lazily on the first
        # step() and replayed thereafter; invalidated whenever the arrays they point at are
        # replaced (load_bodies, close) or the baked scalar arguments change (apply_parameters).
        self._graphs: dict[str, Any] = {}
        self._kernels: Any | None = None
        self._field_kernels: Any | None = None
        self._emitter_kernels: Any | None = None
        self._shearing_kernels: Any | None = None
        self._broadphase_config = BroadphaseConfig.for_capacity(config.capacity)
        self._broadphase_kernels: Any | None = None
        self._stable_id_order: tuple[int, ...] = ()
        self._frame_primitive_started_at: float | None = None

    @property
    def frame_transaction_resumable(self) -> bool:
        """Yield between KDK substeps only when no unbounded emitter tail exists."""

        return not self.config.emitters

    def load_bindings(self, bindings: tuple[BindingDescriptor, ...]) -> None:
        self.load_bodies(
            expand_bindings(
                bindings,
                gravitational_constant=self.config.gravitational_constant,
                softening=self.config.softening,
                # In a shearing box velocities are measured from the circular shear flow, so
                # a body authored at rest is on a circular orbit (ADR 0202); zero otherwise.
                shear_flow_omega=(
                    0.0 if self.config.shearing_box is None else self.config.shearing_box.omega
                ),
            )
        )

    def load_bodies(self, bodies: BodySet) -> None:
        if len(bodies) != self.config.count:
            raise ValueError("body count must match NBodyConfig.count")
        if self.config.collision_mode == CollisionMode.ABSORB and (
            bodies.body_flags is None
            or not any(int(value) != 0 for value in bodies.body_flags)
        ):
            # Fail closed at load rather than degrade silently to IGNORE: an ABSORB
            # scene with no flagged Body has no sink, so every collision would pass
            # through unresolved - the same mismatch the tree lane refuses at validation.
            raise ValueError(
                "CollisionMode.ABSORB requires at least one flagged Body element; "
                "mark a binding with role BODY and the ANALYTICAL representation"
            )
        self._frame_primitive_started_at = None
        self.device_context.initialize()
        self._wp = self.device_context.wp
        self._np = __import__("numpy")
        self._kernels = _kernels(self._wp, self.config.precision)
        self._broadphase_kernels = broadphase_kernels(self._wp, self.config.precision)
        wp = self._wp
        scalar, vector, state_dtype = _dtypes(wp, self.config.precision)
        device = self.device_context.device
        self._stable_id_order = tuple(int(value) for value in bodies.stable_ids)
        self._arrays = {
            "positions": wp.array(
                bodies.positions.astype(state_dtype),
                dtype=vector,
                device=device,
            ),
            "velocities": wp.array(
                bodies.velocities.astype(state_dtype),
                dtype=vector,
                device=device,
            ),
            "accelerations": wp.zeros(self.config.capacity, dtype=vector, device=device),
            "masses": wp.array(
                bodies.masses.astype(state_dtype),
                dtype=scalar,
                device=device,
            ),
            "radii": wp.array(
                bodies.radii.astype(state_dtype),
                dtype=scalar,
                device=device,
            ),
            "stable_ids": wp.array(
                bodies.stable_ids,
                dtype=wp.int64,
                device=device,
            ),
            "active": wp.array(
                bodies.active,
                dtype=wp.int32,
                device=device,
            ),
            # 1 marks a Body in the ADR 0037 §3.3 sense (the accretion sink of ABSORB
            # mode), 0 an ordinary particle. Sampled once by `expand_bindings`; a BodySet
            # built without flags carries none and nothing ever absorbs.
            "body_flags": wp.array(
                self._np.asarray(bodies.body_flags, dtype=self._np.int32)
                if bodies.body_flags is not None
                else self._np.zeros(len(bodies), dtype=self._np.int32),
                dtype=wp.int32,
                device=device,
            ),
            # Reductions are always float64: summing float32 contributions in float32
            # accumulates round-off with body count, which would make the conservation
            # diagnostics report the reduction's error instead of the solver's.
            "diag": wp.zeros(16, dtype=wp.float64, device=device),
            "idiag": wp.zeros(4, dtype=wp.int32, device=device),
            # Persists across substeps within one step() call, unlike "idiag" above which
            # _resolve_collisions clears and rewrites fresh every substep: [0]=collision
            # event total, [1]=overflow flag, folded in by the accumulate_idiag kernel,
            # [2]=non-finite acceleration flag written by flag_nonfinite. All three are
            # read back together as one 20-byte transfer at the end of step().
            # [0:3] retain the existing event/non-finite totals. [3] records grid
            # bucket overflow and [4] records any all-pairs fallback, both folded on the
            # device so ordinary playback still performs one bounded readback.
            "idiag_totals": wp.zeros(5, dtype=wp.int32, device=device),
            # Per-body claim counters for serializing collision resolution; cleared before
            # every _resolve_collisions() call.
            "claimed": wp.zeros(self.config.capacity, dtype=wp.int32, device=device),
            "event_i": wp.zeros(self.config.event_capacity, dtype=wp.int32, device=device),
            "event_j": wp.zeros(self.config.event_capacity, dtype=wp.int32, device=device),
        }
        if bodies.pinned is not None and any(int(value) != 0 for value in bodies.pinned):
            self._arrays.update(
                pinned_arrays(wp, device, self._np, bodies, state_dtype, vector)
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
        if self.config.collision_mode == CollisionMode.ABSORB:
            # Absorption-sink scratch, cleared every substep. The accumulators are
            # float64 in both precision lanes for the same reason "diag" is: they sum up
            # to N particle contributions per Body, and summing float32 terms in float32
            # would put the reduction's round-off inside the conservation contract the
            # gate bakes against. Allocated only for ABSORB so the other modes carry no
            # per-body memory cost for a sink they never launch.
            self._arrays.update(
                {
                    "absorb_targets": wp.zeros(
                        self.config.capacity, dtype=wp.int32, device=device
                    ),
                    "absorb_mass": wp.zeros(
                        self.config.capacity, dtype=wp.float64, device=device
                    ),
                    "absorb_momentum": wp.zeros(
                        self.config.capacity, dtype=wp.vec3d, device=device
                    ),
                    "absorb_position": wp.zeros(
                        self.config.capacity, dtype=wp.vec3d, device=device
                    ),
                    "absorb_r3": wp.zeros(
                        self.config.capacity, dtype=wp.float64, device=device
                    ),
                }
            )
        if self.config.external_fields:
            # Field parameters are baked device arrays, so the launch appended in
            # _compute_accelerations is legal inside CUDA graph capture; the config is
            # frozen, so they never change for the life of this runtime.
            self._field_kernels = field_kernels(wp, self.config.precision)
            self._arrays.update(
                field_arrays(wp, self.config.precision, self.config.external_fields, device)
            )
        if self.config.emitters:
            # No baked arrays to allocate, unlike the fields above: the slot block is
            # already in "active", and the only launch argument is a host integer the
            # release schedule computes per frame.
            self._emitter_kernels = emitter_kernels(wp)
        if self.config.shearing_box is not None:
            if "pinned" in self._arrays:
                raise ValueError(
                    "a shearing box cannot hold a body still: the frame turns and the box "
                    "repeats, so there is no fixed place to pin it. Turn Pinned off"
                )
            # The box's clock is one device scalar (a captured graph replays its launch
            # arguments, so the time cannot be an argument). It starts at zero, which is also
            # what a restored snapshot overwrites.
            self._arrays.update(shearing_arrays(wp, device))
            self._shearing_kernels = shearing_kernels(wp, self.config.precision)
            # A body authored outside the box is the same body expressed in another copy of it:
            # bring it in (with the shear shift) before the first force is evaluated.
            self._launch_shear_wrap()
        # A graph captured against the previous bindings points at arrays this reload just
        # replaced; drop the cache so the next step() recaptures against the new arrays.
        self._graphs.clear()
        # Frame 0 is the timeline's first frame, so an emitter whose authored range opens
        # there is already releasing before the run takes a single step. Applied ahead of
        # the acceleration below so those bodies are in the very first field evaluation
        # rather than joining it a substep late.
        self._release_emitters(0, None)
        self._compute_accelerations()

    def array_dtypes(self) -> dict[str, str]:
        """Element type of every live device array.

        Validation asserts the precision contract against the arrays that exist rather
        than against a self-reported label: state arrays must follow
        ``config.precision`` while ``diag`` must stay float64 in both lanes.
        """

        return {
            name: getattr(array.dtype, "__name__", str(array.dtype))
            for name, array in self._arrays.items()
        }

    def display_source(self) -> tuple[Any, int, Precision]:
        """The live position array a display stage samples, its length, and its precision.

        The array itself, not a copy: the display sample is a device kernel over the state
        the solver is integrating, so handing back a copy would stage a stale frame.
        """

        if not self._arrays:
            raise DeviceRequirementError("DirectNBodyRuntime has no loaded bindings")
        return self._arrays["positions"], self.config.count, self.config.precision

    def display_channels(self) -> dict[str, Any]:
        """Every device array a display stage may sample, keyed by channel name.

        The sibling of :meth:`display_source`, and device handles for the same reason: a
        render mode that colours by speed or sizes by mass needs those arrays sampled by
        the same strided kernel, and routing them through a host accessor would put a full
        readback on the draw path — the one thing `adr/0003-display-readback-boundary.md`
        forbids. Absent keys are absent capabilities, not errors: this lane carries no
        density, so the mass/density size derivation cannot run here. It does not need to:
        ``load_bodies`` already keeps a real per-body ``radii`` array alive for its own
        collision/merge kernels, and that radius IS the physical size of the body, so
        `PHYSICAL` particle size is served straight from it rather than substituted with a
        number (`adr/0023-gpu-particle-renderer-viewport-adoption.md` §7 amendment).

        ``active`` is offered for a different purpose than the rest: it is not something
        the viewport draws, it is what decides *which bodies* the viewport draws at all.
        Every kernel in this module already honours it --- gravity skips an inactive body,
        integration returns early for it, and merge/absorb are how a body becomes inactive
        --- so a display lane that ignored it drew bodies this solver had destroyed, frozen
        at wherever they died. Offering it here is what lets `DisplayStage` compact the
        sample onto the live population (ADR 0046). Stable ids ride along with it because a
        compacted sample is no longer row-aligned with the descriptor's binding order, so a
        consumer that needs identity --- the scrub cache --- has to read it from the sample.
        """

        if not self._arrays:
            raise DeviceRequirementError("DirectNBodyRuntime has no loaded bindings")
        return {
            "positions": self._arrays["positions"],
            "velocities": self._arrays["velocities"],
            "masses": self._arrays["masses"],
            "radii": self._arrays["radii"],
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

        ``field_strengths`` (ADR 0165) gives every external field's new strength, in field
        order. They are written into the existing device array in place, so a strength-only
        edit keeps the captured graphs: the graph already reads that array.

        Every value here is read fresh at launch: ``gravitational_constant`` and
        ``softening`` are arguments to the acceleration kernel, and ``dt``/``substeps``
        size the substep the next :meth:`step` call takes. So rebinding the config changes
        the steps still to come and leaves the arrays holding the steps already taken.

        Nothing is allocated or freed, and the fields that size the arrays — capacity,
        count, precision, event capacity — are not reachable from here, so the state the
        solver is mid-integration on cannot be resized underneath it. Validation runs on
        the candidate before it is adopted, so a rejected value leaves the runtime on the
        parameters it was already stepping with rather than on a half-applied set.
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
        self.config = candidate
        if field_strengths is not None:
            write_field_strengths(self._wp, self._arrays, new_fields)
        # sub_dt and the kick/acceleration scalar arguments are baked into the captured
        # substep graph at capture time, so a parameter edit must force a recapture on the
        # next step() rather than replay a graph carrying the superseded dt/G/softening.
        # A strength-only edit keeps the graphs: the field array it rewrote is read live.
        if scalars_changed:
            self._graphs.clear()

    def begin_frame(self) -> None:
        if not self._arrays:
            raise DeviceRequirementError("DirectNBodyRuntime has no loaded bindings")
        wp = self._require_wp()
        kernels = self._require_kernels()
        self._frame_primitive_started_at = time.perf_counter()
        # The totals span the authored frame, including substeps resumed on later timer
        # ticks. Clearing here (and nowhere in ``advance_frame_substep``) preserves that
        # accumulation while still making every frame's diagnostics independent.
        wp.launch(
            kernels.clear_idiag,
            dim=5,
            inputs=[self._arrays["idiag_totals"]],
            device=self.device_context.device,
        )

    def advance_frame_substep(self) -> None:
        """Execute, synchronize, and validate exactly one captured KDK substep."""

        if not self._arrays:
            raise DeviceRequirementError("DirectNBodyRuntime has no loaded bindings")
        wp = self._require_wp()
        kernels = self._require_kernels()
        start = self._frame_primitive_started_at or time.perf_counter()
        self._frame_primitive_started_at = None
        self._submit("substep", self._record_substep)
        self.step_index += 1
        # ``atomic_max`` drops NaN comparisons, so reduce a non-finite flag on-device and
        # carry only the five-int diagnostic record back across the host boundary.
        wp.launch(
            kernels.flag_nonfinite,
            dim=self.config.count,
            inputs=[
                self._arrays["accelerations"],
                self.config.count,
                self._arrays["idiag_totals"],
            ],
            device=self.device_context.device,
        )
        wp.synchronize_device(self.device_context.device)
        totals = self._arrays["idiag_totals"].numpy()
        self.last_solver_primitive_ms = (time.perf_counter() - start) * 1000.0
        if int(totals[3]):
            raise DeviceRequirementError(
                "collision grid bucket overflow; step failed closed"
            )
        if int(totals[1]):
            raise DeviceRequirementError("collision event capacity overflow; step failed closed")
        if int(totals[2]):
            raise DeviceRequirementError("non-finite acceleration detected; step failed closed")

    def finish_frame(self) -> None:
        """Run frame-only emitter activation exactly once after all substeps succeed."""

        self.frame_index += 1
        if not self.config.emitters:
            return
        frame = self.frame_index
        if self._release_emitters(frame, frame - 1):
            self._compute_accelerations()
            self._require_wp().synchronize_device(self.device_context.device)

    def step(self, steps: int = 1) -> None:
        if not self._arrays:
            raise DeviceRequirementError("DirectNBodyRuntime has no loaded bindings")
        if steps < 1:
            raise ValueError("steps must be positive")
        start = time.perf_counter()
        for _ in range(steps):
            self.begin_frame()
            for _substep in range(self.config.substeps):
                self.advance_frame_substep()
            self.finish_frame()
        self.last_step_ms = (time.perf_counter() - start) * 1000.0

    def diagnostics(self) -> NBodyDiagnostics:
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        # idiag_totals holds the whole-step accumulation (see step()); "idiag" itself is
        # scratch that _resolve_collisions clears and rewrites every substep.
        collision_totals = self._arrays["idiag_totals"].numpy()
        wp.launch(kernels.clear_diag, dim=16, inputs=[self._arrays["diag"]], device=device)
        wp.launch(kernels.clear_idiag, dim=4, inputs=[self._arrays["idiag"]], device=device)
        wp.launch(
            kernels.diagnostics,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["accelerations"],
                self._arrays["masses"],
                self._arrays["active"],
                self.config.count,
                self.config.gravitational_constant,
                self.config.softening,
                self._arrays["diag"],
                self._arrays["idiag"],
            ],
            device=device,
        )
        if self.config.shearing_box is not None:
            # In a shearing box the potential energy is the mutual pull of the bodies in the box
            # (its images are not summed here) plus Hill's tidal and vertical potentials, so the
            # total is Hill's energy of the box: conserved for bodies that do not interact and do
            # not cross a side, and not otherwise (a crossing changes it by the shear shift).
            wp.launch(
                self._require_shearing_kernels().add_tidal_energy,
                dim=self.config.count,
                inputs=[
                    self._arrays["positions"],
                    self._arrays["masses"],
                    self._arrays["active"],
                    self.config.count,
                    self.config.shearing_box.omega,
                    self._arrays["diag"],
                ],
                device=device,
            )
        wp.synchronize_device(device)
        diag = self._arrays["diag"].numpy()
        idiag = self._arrays["idiag"].numpy()
        mass = float(diag[9])
        center = (0.0, 0.0, 0.0)
        if mass != 0.0:
            center = (float(diag[6] / mass), float(diag[7] / mass), float(diag[8] / mass))
        return NBodyDiagnostics(
            active_count=int(idiag[0]),
            total_energy=float(diag[0] + diag[1]),
            kinetic_energy=float(diag[0]),
            potential_energy=float(diag[1]),
            linear_momentum=(float(diag[2]), float(diag[3]), float(diag[4])),
            angular_momentum=(float(diag[10]), float(diag[11]), float(diag[12])),
            center_of_mass=center,
            max_acceleration=float(diag[5]),
            collision_events=int(collision_totals[0])
            if self.config.collision_mode != CollisionMode.IGNORE
            else 0,
            collision_overflow=bool(collision_totals[1])
            if self.config.collision_mode != CollisionMode.IGNORE
            else False,
            collision_broadphase=(
                "disabled"
                if self.config.collision_mode == CollisionMode.IGNORE
                else "all_pairs_fallback"
                if int(collision_totals[4])
                else "uniform_grid"
            ),
            collision_grid_overflow=bool(collision_totals[3]),
            step_ms=self.last_step_ms,
        )

    def bake_samples(self, sample_count: int, steps_per_sample: int = 1) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for _ in range(sample_count):
            samples.append(self.snapshot())
            self.step(steps_per_sample)
        return samples

    def snapshot(self) -> dict[str, Any]:
        positions = self._arrays["positions"].numpy()
        velocities = self._arrays["velocities"].numpy()
        accelerations = self._arrays["accelerations"].numpy()
        masses = self._arrays["masses"].numpy()
        radii = self._arrays["radii"].numpy()
        stable_ids = self._arrays["stable_ids"].numpy()
        active = self._arrays["active"].numpy()
        body_flags = self._arrays["body_flags"].numpy()
        self.explicit_snapshot_readbacks += 8
        bodies = []
        for index in range(self.config.count):
            bodies.append(
                {
                    "index": index,
                    "stable_id": int(stable_ids[index]),
                    "active": bool(active[index]),
                    "body_flag": int(body_flags[index]),
                    "mass": float(masses[index]),
                    "radius": float(radii[index]),
                    "position": [float(value) for value in positions[index]],
                    "velocity": [float(value) for value in velocities[index]],
                    "acceleration": [float(value) for value in accelerations[index]],
                }
            )
        snapshot: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA_VERSION,
            "backend": "direct",
            "capacity": self.config.capacity,
            "count": self.config.count,
            "collision_mode": self.config.collision_mode.value,
            "step": self.step_index,
            "frame": self.frame_index,
            "recomputed_on_restore": ["accelerations", "collision_event_scratch"],
            "bodies": bodies,
        }
        if self.config.shearing_box is not None:
            # The box's clock (the wrapped q Omega Lx t) is state the bodies alone do not carry:
            # a restore that left it at zero would slide every image back to its t = 0 place.
            # Present only for a box, so a snapshot of any other run is the bytes it always was.
            snapshot["shear_offset"] = float(self._arrays["shear_offset"].numpy()[0])
            snapshot["shearing_box"] = self.config.shearing_box.to_parameters()
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint snapshot schema")
        if snapshot.get("backend") != "direct":
            raise ValueError("checkpoint backend does not match direct runtime")
        if int(snapshot.get("capacity", -1)) != self.config.capacity:
            raise ValueError("snapshot capacity does not match runtime")
        if str(snapshot.get("collision_mode")) != self.config.collision_mode.value:
            raise ValueError("snapshot collision mode does not match runtime")
        stored_box = snapshot.get("shearing_box")
        runtime_box = (
            None if self.config.shearing_box is None else self.config.shearing_box.to_parameters()
        )
        if stored_box != runtime_box:
            raise ValueError("snapshot shearing box does not match runtime")
        if runtime_box is not None and "shear_offset" not in snapshot:
            raise ValueError("a shearing-box snapshot must carry the box's clock")
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
        # Default 0 rather than error: a pre-ABSORB checkpoint carries no flags, and the
        # collision-mode match above already guarantees such a snapshot only restores
        # into a runtime that never launches the absorption sink.
        body_flags = np.asarray(
            [int(body.get("body_flag", 0)) for body in bodies],
            dtype=np.int32,
        )
        wp.copy(self._arrays["positions"], wp.array(positions, dtype=vector, device=device))
        wp.copy(self._arrays["velocities"], wp.array(velocities, dtype=vector, device=device))
        wp.copy(self._arrays["masses"], wp.array(masses, dtype=scalar, device=device))
        wp.copy(self._arrays["radii"], wp.array(radii, dtype=scalar, device=device))
        wp.copy(self._arrays["stable_ids"], wp.array(stable_ids, dtype=wp.int64, device=device))
        wp.copy(self._arrays["active"], wp.array(active, dtype=wp.int32, device=device))
        wp.copy(self._arrays["body_flags"], wp.array(body_flags, dtype=wp.int32, device=device))
        if runtime_box is not None:
            # In place, like every array above: a captured substep graph already points at it.
            wp.copy(
                self._arrays["shear_offset"],
                wp.array([float(snapshot["shear_offset"])], dtype=wp.float64, device=device),
            )
        self.step_index = int(snapshot.get("step", 0))
        self.frame_index = int(snapshot.get("frame", self.step_index // self.config.substeps))
        self._frame_primitive_started_at = None
        wp.launch(
            self._require_kernels().clear_idiag,
            dim=4,
            inputs=[self._arrays["idiag"]],
            device=device,
        )
        wp.launch(
            self._require_kernels().clear_idiag,
            dim=5,
            inputs=[self._arrays["idiag_totals"]],
            device=device,
        )
        self._compute_accelerations()
        self.last_step_ms = 0.0

    def close(self) -> None:
        self._frame_primitive_started_at = None
        self._arrays.clear()
        self._graphs.clear()

    def _record_substep(self) -> None:
        """The launch sequence of one KDK substep, recorded for graph capture or run eagerly.

        Called inside ``wp.ScopedCapture`` on the first substep of a fresh config and via
        ``capture_launch`` thereafter. Every launch dimension is fixed by the frozen config
        and every array was allocated once in :meth:`load_bodies`; ``_resolve_collisions``
        and ``_compute_accelerations`` only launch kernels (no allocation, synchronize, or
        readback), so the whole sequence is capture-legal. ``sub_dt`` and the acceleration
        kernel's scalar arguments are read from ``self.config`` here and baked into the
        graph at capture — :meth:`apply_parameters` drops the cache when they change.
        """

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
        if self.config.shearing_box is not None:
            # The exact flow of Hill's equations replaces the straight-line drift, then the
            # box's clock advances and anything that left through a side is brought back in.
            self._launch_shear_drift(sub_dt)
        else:
            wp.launch(
                kernels.drift,
                dim=self.config.count,
                inputs=[
                    self._arrays["positions"],
                    self._arrays["velocities"],
                    self._arrays["active"],
                    sub_dt,
                ],
                device=device,
            )
        launch_hold_pinned(wp, kernels, self._arrays, self.config.count, device)
        self._resolve_collisions()
        launch_hold_pinned(wp, kernels, self._arrays, self.config.count, device)
        self._compute_accelerations()
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
        launch_hold_pinned(wp, kernels, self._arrays, self.config.count, device)

    def _graph(self, name: str, record: Any) -> Any:
        """Capture one launch sequence once so later submissions replay a CUDA graph.

        The substep is launch-submission bound at scale: its kick/drift/resolve/accelerate/
        kick sequence issues many host-side launches whose submission cost dominates the
        tiny per-kernel GPU time as body count grows, and ``step()`` reissues it every
        substep of every frame. Capture is legal because every launch dimension comes from
        the frozen config, every array is allocated once in :meth:`load_bodies`, and nothing
        inside a recorded sequence allocates, synchronizes, or reads back. :meth:`load_bodies`
        and :meth:`close` drop the cache because they replace or release the arrays the
        graphs point at, and :meth:`apply_parameters` drops it because the scalar arguments
        are baked in; a snapshot restore does not, because it copies into those same arrays.
        Off CUDA the sequence is cached as ``None`` and runs eagerly.
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
        """Replay ``record``'s captured graph, or run it eagerly off CUDA.

        On the first submission of a name the capture inside :meth:`_graph` records the
        sequence without executing it, so the ``capture_launch`` below is what actually
        runs this substep; later submissions replay the cached graph directly.
        """

        graph = self._graph(name, record)
        if graph is None:
            record()
        else:
            self._require_wp().capture_launch(graph)

    def _release_emitters(self, frame: int, previous_frame: int | None) -> bool:
        """Raise every emitter's live watermark from ``previous_frame`` to ``frame``.

        ``previous_frame`` is ``None`` at load, where nothing has been released yet.
        Returns whether any slot was activated, so the caller can skip the acceleration
        refresh on the frames where the population did not move.

        Only the delta is launched. `released_at_frame` is monotone non-decreasing, so the
        block ``[released(previous), released(frame))`` has never been live before and the
        watermark below it is left exactly as the solver left it --- which is what keeps a
        particle a collision merged permanently dead instead of resurrecting it the next
        time the schedule is evaluated.
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

    def _launch_shear_wrap(self) -> None:
        """Bring every body that left the box back in, with the shear shift (ADR 0202)."""

        spec = self.config.shearing_box
        self._require_wp().launch(
            self._require_shearing_kernels().shear_wrap,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["active"],
                self._arrays["shear_offset"],
                spec.box_x,
                spec.box_y,
                spec.shear_velocity,
            ],
            device=self.device_context.device,
        )

    def _launch_shear_drift(self, dt: float) -> None:
        """One drift of the shearing box: the exact epicycle map, the clock, the boundary."""

        wp = self._require_wp()
        kernels = self._require_shearing_kernels()
        spec = self.config.shearing_box
        device = self.device_context.device
        wp.launch(
            kernels.epicycle_drift,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["active"],
                spec.omega,
                dt,
            ],
            device=device,
        )
        # The clock moves to the end of the step before the boundary reads it: a body that
        # crossed during this step crossed at (about) its end, and the offset is a function of t.
        wp.launch(
            kernels.advance_shear_offset,
            dim=1,
            inputs=[
                self._arrays["shear_offset"],
                wp.float64(spec.shear_velocity * dt),
                wp.float64(spec.box_y),
            ],
            device=device,
        )
        self._launch_shear_wrap()

    def _compute_accelerations(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        if self.config.shearing_box is not None:
            spec = self.config.shearing_box
            wp.launch(
                self._require_shearing_kernels().acceleration_images,
                dim=self.config.count,
                inputs=[
                    self._arrays["positions"],
                    self._arrays["masses"],
                    self._arrays["active"],
                    self.config.count,
                    self.config.gravitational_constant,
                    self.config.softening,
                    spec.box_x,
                    spec.box_y,
                    spec.images,
                    self._arrays["shear_offset"],
                    self._arrays["accelerations"],
                ],
                device=self.device_context.device,
            )
            return
        wp.launch(
            kernels.acceleration,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["masses"],
                self._arrays["active"],
                self.config.count,
                self.config.gravitational_constant,
                self.config.softening,
                self._arrays["accelerations"],
            ],
            device=self.device_context.device,
        )
        if self.config.external_fields:
            # After gravity, into the same accumulator (ADR 0045). Zero fields launch
            # nothing, so a field-free capture is byte-identical to the pre-Force one.
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
                    self._arrays["accelerations"],
                ],
                device=self.device_context.device,
            )

    def _resolve_collisions(self) -> None:
        if self.config.collision_mode == CollisionMode.IGNORE:
            return
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        wp.launch(kernels.clear_idiag, dim=4, inputs=[self._arrays["idiag"]], device=device)
        wp.launch(
            kernels.clear_idiag,
            dim=self.config.capacity,
            inputs=[self._arrays["claimed"]],
            device=device,
        )
        if self.config.collision_mode == CollisionMode.ABSORB:
            self._resolve_absorption(wp, kernels, device)
            return
        broadphase = self._broadphase_config
        grid_kernels = self._require_broadphase_kernels()
        grid_max_radius = self._arrays["collision_grid_max_radius"]
        grid_min_radius = self._arrays["collision_grid_min_radius"]
        grid_counts = self._arrays["collision_grid_cell_counts"]
        grid_items = self._arrays["collision_grid_cell_items"]
        grid_flags = self._arrays["collision_grid_flags"]
        wp.launch(
            grid_kernels.clear_grid,
            dim=broadphase.cell_capacity,
            inputs=[grid_max_radius, grid_min_radius, grid_counts, grid_flags],
            device=device,
        )
        wp.launch(
            grid_kernels.reduce_max_radius,
            dim=self.config.count,
            inputs=[
                self._arrays["radii"],
                self._arrays["active"],
                self.config.count,
                grid_max_radius,
                grid_min_radius,
            ],
            device=device,
        )
        wp.launch(
            grid_kernels.bin_particles,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["active"],
                self.config.count,
                grid_max_radius,
                broadphase.cell_capacity,
                broadphase.bucket_capacity,
                grid_counts,
                grid_items,
                grid_flags,
            ],
            device=device,
        )
        wp.launch(
            grid_kernels.detect_collisions_grid,
            dim=self.config.count,
            inputs=[
                self._arrays["positions"],
                self._arrays["radii"],
                self._arrays["active"],
                self.config.count,
                grid_max_radius,
                grid_min_radius,
                broadphase.cell_capacity,
                broadphase.bucket_capacity,
                broadphase.all_pairs_threshold,
                broadphase.radius_ratio_threshold,
                grid_counts,
                grid_items,
                grid_flags,
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
            inputs=[grid_flags, self._arrays["idiag_totals"]],
            device=device,
        )
        # Fold this substep's event count/overflow flag into the whole-step totals before
        # the next substep's clear_idiag call above zeroes "idiag" again.
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

    def _resolve_absorption(self, wp: Any, kernels: Any, device: Any) -> None:
        """One substep of the ABSORB lane. Caller has already cleared idiag and claimed.

        Order matters: Body-Body contact merges first (so a swallowed Body cannot also
        accrete this substep), then every particle inside a surviving Body's own radius
        is accumulated and deactivated, then each Body folds its accumulators in. No
        event list and no per-substep cap: the sink absorbs everything inside the
        boundary in this one substep, which is the whole point of the accumulation
        design (plan corrections, 2026-08-10).
        """

        count = self.config.count
        wp.launch(
            kernels.clear_absorb,
            dim=self.config.capacity,
            inputs=[
                self._arrays["absorb_mass"],
                self._arrays["absorb_momentum"],
                self._arrays["absorb_position"],
                self._arrays["absorb_r3"],
                self._arrays["absorb_targets"],
            ],
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
            dim=count,
            inputs=[
                self._arrays["radii"],
                self._arrays["active"],
                count,
                self._arrays["collision_grid_max_radius"],
                self._arrays["collision_grid_min_radius"],
            ],
            device=device,
        )
        wp.launch(
            self._require_broadphase_kernels().bin_particles,
            dim=count,
            inputs=[
                self._arrays["positions"],
                self._arrays["active"],
                count,
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
            self._require_broadphase_kernels().detect_body_collisions_grid,
            dim=count,
            inputs=[
                self._arrays["positions"],
                self._arrays["radii"],
                self._arrays["active"],
                self._arrays["body_flags"],
                count,
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
            kernels.resolve_merge,
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
        # A body merge changes the surviving centre/radius and deactivates its peer, so
        # rebuild the preallocated grid before querying particle-to-body containment.
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
            dim=count,
            inputs=[
                self._arrays["radii"],
                self._arrays["active"],
                count,
                self._arrays["collision_grid_max_radius"],
                self._arrays["collision_grid_min_radius"],
            ],
            device=device,
        )
        wp.launch(
            self._require_broadphase_kernels().bin_particles,
            dim=count,
            inputs=[
                self._arrays["positions"],
                self._arrays["active"],
                count,
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
            self._require_broadphase_kernels().detect_absorption_grid,
            dim=count,
            inputs=[
                self._arrays["positions"],
                self._arrays["radii"],
                self._arrays["active"],
                self._arrays["body_flags"],
                self._arrays["stable_ids"],
                count,
                self._arrays["collision_grid_max_radius"],
                self._arrays["collision_grid_min_radius"],
                self._broadphase_config.cell_capacity,
                self._broadphase_config.bucket_capacity,
                self._broadphase_config.all_pairs_threshold,
                self._broadphase_config.radius_ratio_threshold,
                self._arrays["collision_grid_cell_counts"],
                self._arrays["collision_grid_cell_items"],
                self._arrays["collision_grid_flags"],
                self._arrays["absorb_targets"],
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
        # Fold this substep's event count into the whole-step totals; absorption has no
        # event list, so the overflow slot idiag[2] stays 0 by construction.
        wp.launch(
            kernels.accumulate_idiag,
            dim=1,
            inputs=[self._arrays["idiag"], self._arrays["idiag_totals"]],
            device=device,
        )
        wp.launch(
            kernels.resolve_absorb,
            dim=count,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["masses"],
                self._arrays["radii"],
                self._arrays["active"],
                self._arrays["absorb_targets"],
                count,
                self._arrays["absorb_mass"],
                self._arrays["absorb_momentum"],
                self._arrays["absorb_position"],
                self._arrays["absorb_r3"],
            ],
            device=device,
        )
        wp.launch(
            kernels.finalize_absorption,
            dim=count,
            inputs=[
                self._arrays["positions"],
                self._arrays["velocities"],
                self._arrays["masses"],
                self._arrays["radii"],
                self._arrays["active"],
                self._arrays["body_flags"],
                count,
                self._arrays["absorb_mass"],
                self._arrays["absorb_momentum"],
                self._arrays["absorb_position"],
                self._arrays["absorb_r3"],
            ],
            device=device,
        )

    def _require_wp(self):
        if self._wp is None:
            raise DeviceRequirementError("Warp runtime is not loaded")
        return self._wp

    def _require_kernels(self):
        if self._kernels is None:
            raise DeviceRequirementError("Warp kernels are not loaded")
        return self._kernels

    def _require_broadphase_kernels(self):
        if self._broadphase_kernels is None:
            raise DeviceRequirementError("Warp broad-phase kernels are not loaded")
        return self._broadphase_kernels

    def _require_emitter_kernels(self):
        if self._emitter_kernels is None:
            raise DeviceRequirementError("Warp emitter kernels are not loaded")
        return self._emitter_kernels

    def _require_shearing_kernels(self):
        if self._shearing_kernels is None:
            raise DeviceRequirementError("Warp shearing-box kernels are not loaded")
        return self._shearing_kernels


def pinned_arrays(
    wp: Any, device: Any, np: Any, bodies: BodySet, state_dtype: Any, vector: Any, length: int = 0
) -> dict[str, Any]:
    """Device arrays a runtime needs to hold its pinned bodies still (ADR 0158).

    ``pinned`` is the per-body flag and ``pin_positions`` the place each flagged body is
    held at -- the position it was authored at, copied once at load. ``length`` pads both to
    a runtime that allocates more slots than it has bodies (a pad slot is never pinned).
    Shared by the direct and the tree runtime so the two cannot disagree about what pinned
    means.
    """

    count = len(bodies)
    size = max(int(length), count)
    flags = np.zeros(size, dtype=np.int32)
    flags[:count] = np.asarray(bodies.pinned, dtype=np.int32)
    anchors = np.zeros((size, 3), dtype=np.float64)
    anchors[:count] = np.asarray(bodies.positions, dtype=np.float64)
    return {
        "pinned": wp.array(flags, dtype=wp.int32, device=device),
        "pin_positions": wp.array(anchors.astype(state_dtype), dtype=vector, device=device),
    }


def launch_hold_pinned(
    wp: Any, kernels: Any, arrays: dict[str, Any], count: int, device: Any
) -> None:
    """Put every pinned body back at its anchor with zero velocity; a no-op with none."""

    if "pinned" not in arrays:
        return
    wp.launch(
        kernels.hold_pinned,
        dim=count,
        inputs=[
            arrays["positions"],
            arrays["velocities"],
            arrays["pinned"],
            arrays["pin_positions"],
        ],
        device=device,
    )


def _kernels(wp, precision: Precision = Precision.FLOAT32):
    """Compile the direct-solver kernels for one state precision.

    State-carrying arrays follow ``precision``; the ``diag`` reduction buffer is float64
    in both lanes so a conservation metric reports the solver's error and not the error
    of summing N contributions.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, _ = _registrars(wp, scalar, vector)

    @warp_kernel
    def acceleration(
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
        pi = positions[i]
        ai = vector()
        for j in range(count):
            if i != j and active[j] != 0:
                r = positions[j] - pi
                # Two coincident bodies at softening=0.0 would divide by zero here; the
                # clamp floor matches the one _resolve_collisions already uses for the same
                # r=0 case (ADR 0025).
                dist2 = wp.max(wp.dot(r, r) + softening * softening, scalar(1.0e-12))
                inv_dist = scalar(1.0) / wp.sqrt(dist2)
                ai += r * (gravitational_constant * masses[j] * inv_dist * inv_dist * inv_dist)
        accelerations[i] = ai

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
        dt: scalar,
    ):
        i = wp.tid()
        if active[i] == 0:
            return
        positions[i] = positions[i] + velocities[i] * dt

    @warp_kernel
    def hold_pinned(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        pinned: wp.array(dtype=wp.int32),
        pin_positions: wp.array(dtype=vector),
    ):
        # ADR 0158. A pinned body is a fixed attractor: put it back where it was authored and
        # stop it, after every launch that could have moved it (drift, a collision
        # resolution, the closing kick). Restoring rather than skipping the integration means
        # nothing that acts on positions or velocities -- today or later -- can move it.
        i = wp.tid()
        if pinned[i] != 0:
            positions[i] = pin_positions[i]
            velocities[i] = vector()

    @warp_kernel
    def clear_diag(values: wp.array(dtype=wp.float64)):
        values[wp.tid()] = wp.float64(0.0)

    @warp_kernel
    def clear_idiag(values: wp.array(dtype=wp.int32)):
        values[wp.tid()] = 0

    @warp_kernel
    def flag_nonfinite(
        values: wp.array(dtype=vector),
        count: int,
        totals: wp.array(dtype=wp.int32),
    ):
        # One bit out of the whole array: did any component stop being a real number. The
        # reduction is atomic_max over 0/1 rather than a count because the caller only ever
        # asks whether the step is trustworthy, and a max needs no ordering to be exact.
        i = wp.tid()
        if i >= count:
            return
        if not wp.isfinite(values[i]):
            wp.atomic_max(totals, 2, 1)

    @warp_kernel
    def accumulate_idiag(idiag: wp.array(dtype=wp.int32), totals: wp.array(dtype=wp.int32)):
        # Single-thread launch: fold one substep's collision reading into the whole-step
        # totals. "idiag" is cleared and rewritten fresh every substep by
        # _resolve_collisions, so this is the only place the per-substep numbers survive.
        wp.atomic_add(totals, 0, idiag[1])
        wp.atomic_max(totals, 1, idiag[2])

    @warp_kernel
    def accumulate_broadphase_flags(
        flags: wp.array(dtype=wp.int32), totals: wp.array(dtype=wp.int32)
    ):
        wp.atomic_max(totals, 3, flags[0])
        wp.atomic_max(totals, 4, flags[1])

    @warp_kernel
    def diagnostics(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        accelerations: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        diag: wp.array(dtype=wp.float64),
        idiag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0:
            return
        m = masses[i]
        p = positions[i]
        v = velocities[i]
        a = accelerations[i]
        kinetic = wp.float64(scalar(0.5) * m * wp.dot(v, v))
        # The pairwise potential is a sum over N, so it accumulates in float64 even when
        # the terms themselves are evaluated at the solver's precision.
        potential = wp.float64(0.0)
        for j in range(count):
            if i != j and active[j] != 0:
                r = positions[j] - p
                dist = wp.sqrt(wp.max(wp.dot(r, r) + softening * softening, scalar(1.0e-12)))
                potential += wp.float64(
                    scalar(-0.5) * gravitational_constant * m * masses[j] / dist
                )
        angular = wp.cross(p, v * m)
        wp.atomic_add(diag, 0, kinetic)
        wp.atomic_add(diag, 1, potential)
        wp.atomic_add(diag, 2, wp.float64(m * v[0]))
        wp.atomic_add(diag, 3, wp.float64(m * v[1]))
        wp.atomic_add(diag, 4, wp.float64(m * v[2]))
        wp.atomic_max(diag, 5, wp.float64(wp.length(a)))
        wp.atomic_add(diag, 6, wp.float64(m * p[0]))
        wp.atomic_add(diag, 7, wp.float64(m * p[1]))
        wp.atomic_add(diag, 8, wp.float64(m * p[2]))
        wp.atomic_add(diag, 9, wp.float64(m))
        wp.atomic_add(diag, 10, wp.float64(angular[0]))
        wp.atomic_add(diag, 11, wp.float64(angular[1]))
        wp.atomic_add(diag, 12, wp.float64(angular[2]))
        wp.atomic_add(idiag, 0, 1)

    # Retained as the all-pairs ORACLE for tests/unit/test_collision_equivalence.py; the product
    # path launches the broadphase grid detector instead (SR-28). Not launched at runtime.
    @warp_kernel
    def detect_collisions_all_pairs(
        positions: wp.array(dtype=vector),
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        event_i: wp.array(dtype=wp.int32),
        event_j: wp.array(dtype=wp.int32),
        event_capacity: int,
        idiag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0:
            return
        for j in range(i + 1, count):
            if active[j] != 0:
                r = positions[j] - positions[i]
                limit = radii[i] + radii[j]
                if wp.dot(r, r) <= limit * limit:
                    event_index = wp.atomic_add(idiag, 1, 1)
                    if event_index < event_capacity:
                        event_i[event_index] = i
                        event_j[event_index] = j
                    else:
                        wp.atomic_max(idiag, 2, 1)

    @warp_kernel
    def resolve_merge(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        radii: wp.array(dtype=scalar),
        stable_ids: wp.array(dtype=wp.int64),
        active: wp.array(dtype=wp.int32),
        claimed: wp.array(dtype=wp.int32),
        event_i: wp.array(dtype=wp.int32),
        event_j: wp.array(dtype=wp.int32),
        event_capacity: int,
        idiag: wp.array(dtype=wp.int32),
    ):
        e = wp.tid()
        if e >= event_capacity or e >= idiag[1]:
            return
        i = event_i[e]
        j = event_j[e]
        if active[i] == 0 or active[j] == 0:
            return
        # Two events in the same substep can share a body (e.g. a chain collision); a
        # plain read-modify-write from both event threads would race. First-writer-wins
        # via atomic_add: a thread only resolves its event if it is the sole claimant of
        # BOTH bodies this substep, so each body is touched by at most one resolved event
        # per substep. Losers simply skip — their collision is picked up again next
        # substep if the bodies are still overlapping.
        claim_i = wp.atomic_add(claimed, i, 1)
        claim_j = wp.atomic_add(claimed, j, 1)
        if claim_i != 0 or claim_j != 0:
            return
        keep = i
        drop = j
        if stable_ids[j] < stable_ids[i]:
            keep = j
            drop = i
        total_mass = masses[i] + masses[j]
        positions[keep] = (positions[i] * masses[i] + positions[j] * masses[j]) / total_mass
        velocities[keep] = (velocities[i] * masses[i] + velocities[j] * masses[j]) / total_mass
        masses[keep] = total_mass
        cube = scalar(3.0)
        radii[keep] = wp.pow(wp.pow(radii[i], cube) + wp.pow(radii[j], cube), scalar(1.0) / cube)
        active[drop] = 0

    @warp_kernel
    def resolve_elastic(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        radii: wp.array(dtype=scalar),
        stable_ids: wp.array(dtype=wp.int64),
        active: wp.array(dtype=wp.int32),
        claimed: wp.array(dtype=wp.int32),
        event_i: wp.array(dtype=wp.int32),
        event_j: wp.array(dtype=wp.int32),
        event_capacity: int,
        idiag: wp.array(dtype=wp.int32),
    ):
        e = wp.tid()
        if e >= event_capacity or e >= idiag[1]:
            return
        i = event_i[e]
        j = event_j[e]
        if active[i] == 0 or active[j] == 0:
            return
        # Approach check (SR-04, ADR 0186): a pair that is still overlapping but already
        # separating is left alone; re-applying the impulse would flip it back into
        # approach and make overlapping bodies jitter. Placed BEFORE the claim so a
        # separating pair never blocks another event on the same body this substep.
        if wp.dot(velocities[i] - velocities[j], positions[i] - positions[j]) >= scalar(0.0):
            return
        # See resolve_merge for the claim protocol: only the sole claimant of both bodies
        # this substep resolves the pair, avoiding a racing read-modify-write.
        claim_i = wp.atomic_add(claimed, i, 1)
        claim_j = wp.atomic_add(claimed, j, 1)
        if claim_i != 0 or claim_j != 0:
            return
        delta = positions[i] - positions[j]
        dist2 = wp.max(wp.dot(delta, delta), scalar(1.0e-12))
        vi = velocities[i]
        vj = velocities[j]
        mi = masses[i]
        mj = masses[j]
        two = scalar(2.0)
        factor_i = (two * mj / (mi + mj)) * (wp.dot(vi - vj, delta) / dist2)
        factor_j = (two * mi / (mi + mj)) * (wp.dot(vj - vi, -delta) / dist2)
        velocities[i] = vi - delta * factor_i
        velocities[j] = vj + delta * factor_j

    @warp_kernel
    def clear_absorb(
        acc_mass: wp.array(dtype=wp.float64),
        acc_momentum: wp.array(dtype=wp.vec3d),
        acc_position: wp.array(dtype=wp.vec3d),
        acc_r3: wp.array(dtype=wp.float64),
        targets: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        acc_mass[i] = wp.float64(0.0)
        acc_momentum[i] = wp.vec3d()
        acc_position[i] = wp.vec3d()
        acc_r3[i] = wp.float64(0.0)
        targets[i] = -1

    @warp_kernel
    def resolve_absorb(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        targets: wp.array(dtype=wp.int32),
        count: int,
        acc_mass: wp.array(dtype=wp.float64),
        acc_momentum: wp.array(dtype=wp.vec3d),
        acc_position: wp.array(dtype=wp.vec3d),
        acc_r3: wp.array(dtype=wp.float64),
    ):
        # The accumulation half of the sink: every targeted particle folds its mass,
        # momentum, mass-weighted position, and cubed radius into its Body's
        # accumulators and deactivates itself. atomic_add rather than resolve_merge's
        # first-writer-wins claim, so ALL inside-boundary particles are absorbed in one
        # substep - the claim protocol would cap the sink at one particle per substep.
        # masses[i] is left as-is, matching the merge lane: totals are defined over
        # active == True only.
        i = wp.tid()
        if i >= count:
            return
        b = targets[i]
        if b < 0 or active[i] == 0 or active[b] == 0:
            return
        m = wp.float64(masses[i])
        v = velocities[i]
        p = positions[i]
        wp.atomic_add(acc_mass, b, m)
        wp.atomic_add(
            acc_momentum,
            b,
            wp.vec3d(m * wp.float64(v[0]), m * wp.float64(v[1]), m * wp.float64(v[2])),
        )
        wp.atomic_add(
            acc_position,
            b,
            wp.vec3d(m * wp.float64(p[0]), m * wp.float64(p[1]), m * wp.float64(p[2])),
        )
        r = wp.float64(radii[i])
        wp.atomic_add(acc_r3, b, r * r * r)
        active[i] = 0

    @warp_kernel
    def finalize_absorption(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        body_flags: wp.array(dtype=wp.int32),
        count: int,
        acc_mass: wp.array(dtype=wp.float64),
        acc_momentum: wp.array(dtype=wp.vec3d),
        acc_position: wp.array(dtype=wp.vec3d),
        acc_r3: wp.array(dtype=wp.float64),
    ):
        # One thread per Body folds its accumulators in - the same barycentric math as
        # resolve_merge (mass sum, momentum-weighted velocity and position, cube-additive
        # radius), evaluated in float64 and narrowed once at the store.
        b = wp.tid()
        if b >= count or body_flags[b] == 0 or active[b] == 0:
            return
        accreted = acc_mass[b]
        if accreted == wp.float64(0.0):
            # Nothing accumulated - or only zero-mass particles were, whose r³ this
            # return discards. Precondition: particle masses are positive (descriptor
            # validation), so accreted == 0 really means "no absorption this substep";
            # it also keeps the barycentric divisions below away from a zero divisor.
            return
        m0 = wp.float64(masses[b])
        new_mass = m0 + accreted
        v = velocities[b]
        p = positions[b]
        momentum = acc_momentum[b]
        weighted = acc_position[b]
        velocities[b] = vector(
            scalar((m0 * wp.float64(v[0]) + momentum[0]) / new_mass),
            scalar((m0 * wp.float64(v[1]) + momentum[1]) / new_mass),
            scalar((m0 * wp.float64(v[2]) + momentum[2]) / new_mass),
        )
        positions[b] = vector(
            scalar((m0 * wp.float64(p[0]) + weighted[0]) / new_mass),
            scalar((m0 * wp.float64(p[1]) + weighted[1]) / new_mass),
            scalar((m0 * wp.float64(p[2]) + weighted[2]) / new_mass),
        )
        r0 = wp.float64(radii[b])
        radii[b] = scalar(
            wp.pow(r0 * r0 * r0 + acc_r3[b], wp.float64(1.0) / wp.float64(3.0))
        )
        masses[b] = scalar(new_mass)

    @dataclass(frozen=True)
    class Kernels:
        acceleration: Any
        kick: Any
        drift: Any
        hold_pinned: Any
        clear_diag: Any
        clear_idiag: Any
        flag_nonfinite: Any
        accumulate_idiag: Any
        accumulate_broadphase_flags: Any
        diagnostics: Any
        detect_collisions_all_pairs: Any
        resolve_merge: Any
        resolve_elastic: Any
        clear_absorb: Any
        resolve_absorb: Any
        finalize_absorption: Any

    # Keyword construction: sixteen same-typed slots means a transposition would compile
    # and silently mis-bind every kernel after it. Names make that impossible.
    return Kernels(
        acceleration=acceleration,
        kick=kick,
        drift=drift,
        hold_pinned=hold_pinned,
        clear_diag=clear_diag,
        clear_idiag=clear_idiag,
        flag_nonfinite=flag_nonfinite,
        accumulate_idiag=accumulate_idiag,
        accumulate_broadphase_flags=accumulate_broadphase_flags,
        diagnostics=diagnostics,
        detect_collisions_all_pairs=detect_collisions_all_pairs,
        resolve_merge=resolve_merge,
        resolve_elastic=resolve_elastic,
        clear_absorb=clear_absorb,
        resolve_absorb=resolve_absorb,
        finalize_absorption=finalize_absorption,
    )
