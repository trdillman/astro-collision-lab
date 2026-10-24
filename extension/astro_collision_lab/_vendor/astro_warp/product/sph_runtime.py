"""Adapt the SPH stages to the product runtime contract.

`sph.py`, `hydro.py` and `selfgravity.py` are validated against physics gates that hand
them explicit arrays and advance them at the Courant step. The product timeline hands a
runtime authored bindings and advances it by whole frames. This module is the only place
that translates between the two, so neither side has to know about the other: nothing here
adds physics, and nothing in the stages below learns what a binding or a frame is.

Two translations carry the whole file.

*Geometry to smoothing length.* `expand_bindings` already gives each sampled particle a
radius equal to the mean interparticle spacing the authored radius and count imply. A
support ball of radius ``R`` around a particle at spacing ``s`` holds ``(4/3)pi R**3 / s**3``
particles, so the ``h`` that puts the configured neighbour target inside the support is
``h = (s / kappa) * (3 N / (4 pi)) ** (1/3)`` with ``kappa`` the kernel's support radius at
``h = 1``. Every length the stages need follows from the scene rather than from a constant:
the bracket is that ``h`` scaled up and down by one authored factor, and the gravitational
softening falls back to it when the context leaves softening at zero, because a tree with
zero softening is rejected outright.

*Courant step to frame.* The hydrodynamic step length is whatever the signal velocity
allows, which is not the frame length and does not divide it. A frame is therefore
sub-cycled until the SPH clock reaches the frame boundary, with each individual step capped
at ``seconds_per_frame / substeps`` so the authored substep count still bounds how coarsely
a frame may be crossed. Overshoot is at most one capped step and is carried into the next
frame rather than discarded, so the clock never drifts further than that from frame time.
The substep ceiling is a responsiveness budget, not physics: it exists so a cloud whose
Courant step has collapsed fails with both numbers named instead of hanging Blender.

That sub-cycle is expressed once, as a frame transaction the host may drive one step at a
time (ADR 0109): `begin_frame`, then `advance_frame_primitive` until it reports the frame
crossed, then `finish_frame`. `step` is those same three calls run without yielding. One
`HydroRuntime.step` is one synchronized primitive --- it ends on a device synchronize and
raises on the tree overflow flag --- so the host gets its thread back between them and
repaints while a hard frame is still being crossed, instead of standing still for every
step the frame turns out to need.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

from .hydro import EOS_REGISTRY, EquationOfState, HydroConfig, HydroRuntime
from .model import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_TREE_MAX_DEPTH,
    DEFAULT_TREE_STACK_CAPACITY,
    BindingDescriptor,
)
from .nbody import Precision
from .runtime_foundation import (
    DeviceContext,
    DeviceRequirementError,
    FramePrimitiveProgress,
)
from .sampling import BodySet, binding_body_count, expand_bindings
from .selfgravity import GRAVITY_BACKENDS, SelfGravityConfig
from .solids import SolidConfig
from .sph import KERNEL_REGISTRY, NeighborConfig, NeighborRuntime, SphKernel

# Steps one frame may spend before the sub-cycle gives up. This is a non-termination guard
# and nothing else: it catches a Courant step driven to zero or near-zero by a degenerate
# state, where the sub-cycle would otherwise never reach the frame's end, and it fails with
# the measured step in the message rather than hanging.
#
# It is deliberately not a responsiveness budget, and the comment that said it was has been
# measured false twice over (ADR 0110). It never binds: the worst frame of the Asteroid
# Impact template's own scene -- the contact frame, 10,000 particles -- takes 68 steps, 60x
# under this number, and across 112 measured frames the count never once approached it. And
# it could not serve as one if it did bind, because responsiveness is a property of how long
# the host waits between primitives, which the scheduler's own tick budget governs
# (``_service_scrub_frontier``, ADR 0067), not of how many steps a frame contains. A frame
# that legitimately needs many small steps is slow, which the artist is told; it is not
# unusable, and refusing to integrate it would be refusing the physics rather than reporting
# its cost (ADR 0067 Decision 2).
DEFAULT_MAX_SUBSTEPS_PER_FRAME = 4096


class SphTimestepError(RuntimeError):
    """Raised when a frame cannot be crossed inside the substep ceiling."""


@dataclass(frozen=True)
class SphProductConfig:
    """Everything the SPH stages need that is not derived from the scene geometry."""

    count: int
    gravitational_constant: float
    softening: float
    seconds_per_frame: float
    substeps: int
    precision: Precision = Precision.FLOAT32
    kernel: SphKernel = SphKernel.CUBIC_SPLINE
    target_neighbors: int = 0
    neighbor_tolerance: int = 2
    # Factor between the initial smoothing length and each end of the solve bracket. One
    # authored number rather than two absolute lengths, so the bracket follows the scene.
    smoothing_bracket: float = 16.0
    # Uniform specific internal energy. Zero means derive it from the cloud's own virial
    # balance rather than start the gas arbitrarily hot or cold.
    internal_energy: float = 0.0
    equation_of_state: EquationOfState = EquationOfState.IDEAL_GAS
    alpha_viscosity: float = 1.0
    beta_viscosity: float = 2.0
    balsara_switch: bool = True
    courant_number: float = 0.3
    theta: float = 0.5
    max_depth: int = DEFAULT_TREE_MAX_DEPTH
    leaf_capacity: int = 8
    stack_capacity: int = DEFAULT_TREE_STACK_CAPACITY
    node_capacity: int = 0
    max_substeps_per_frame: int = DEFAULT_MAX_SUBSTEPS_PER_FRAME
    # Which self-gravity sum the stage records; see `selfgravity.GRAVITY_BACKENDS`. "auto"
    # here and "tree" on `SelfGravityConfig` is the deliberate split: this is the lane that
    # plays a scene, where the only question is which sum is faster, and the exact sum is
    # both faster and exact for every particle count an impact scene actually runs. The
    # stage's own default stays the tree so the suites that certify the tree keep measuring
    # it.
    gravity_backend: str = "auto"
    # Solid response. Absent is the Phase 4 cloud: every particle is a fluid and no
    # constitutive stage is recorded. Present is the Phase 6 impact: the bodies carry
    # materials, and `shells` says which binding is differentiated into which layers.
    # A tuple of pairs rather than a dict so the config stays a frozen value.
    materials: Any = None
    shells: tuple[tuple[str, tuple[tuple[float, int, float], ...]], ...] = ()
    flaws_per_particle: int = 10
    # Fragment linking length, in smoothing lengths. Expressed as a multiple rather than a
    # distance because "are these two particles still attached" is a question about the
    # resolution that resolved them: the same scene run at twice the particle count would
    # answer it differently for a fixed number of metres.
    linking_length_factor: float = 2.0
    # Damped steps each body takes towards its own discrete equilibrium before frame 1, and
    # the share of its velocity it keeps per step (ADR 0168). Zero is no settle: a gas cloud
    # is authored to collapse, and only the impact plan asks for one.
    settle_steps: int = 0
    settle_damping: float = 0.98

    @property
    def shell_map(self) -> dict[str, tuple[tuple[float, int, float], ...]]:
        return {binding_id: layers for binding_id, layers in self.shells}

    def validate(self) -> None:
        if self.count < 1:
            raise ValueError("count must be positive")
        if self.seconds_per_frame <= 0.0:
            raise ValueError("seconds_per_frame must be positive")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        if self.smoothing_bracket <= 1.0:
            raise ValueError("smoothing_bracket must exceed 1")
        if self.internal_energy < 0.0:
            raise ValueError("internal_energy must be non-negative")
        if self.max_substeps_per_frame < 1:
            raise ValueError("max_substeps_per_frame must be positive")
        if self.gravity_backend not in GRAVITY_BACKENDS:
            raise ValueError(f"gravity_backend must be one of {list(GRAVITY_BACKENDS)}")
        # Either both or neither. A registry with no layered binding would record the
        # constitutive stages over a cloud in which every particle is the fluid sentinel,
        # and shells with no registry would name material ids nothing can resolve.
        if (self.materials is None) != (not self.shells):
            raise ValueError("materials and shells must be configured together")
        if self.linking_length_factor <= 0.0:
            raise ValueError("linking_length_factor must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if not 0.0 <= self.settle_damping < 1.0:
            raise ValueError("settle_damping must be in [0, 1)")


@dataclass(frozen=True)
class SphDiagnostics:
    """Hydrodynamic and neighbour scalars in one flat record.

    Flat because the product layer serialises diagnostics with `dataclasses.asdict` into a
    bake manifest; nesting the two stage records would put the field a reader wants two
    levels down for no gain.
    """

    active_count: int
    frozen_count: int
    density_min: float
    density_max: float
    pressure_max: float
    sound_speed_max: float
    timestep: float
    sph_steps_taken: int
    simulation_time: float
    frames_advanced: int
    substeps_last_frame: int
    step_ms: float
    eos: str
    gamma: float
    alpha_viscosity: float
    beta_viscosity: float
    balsara_switch: bool
    courant_number: float
    evolves_internal_energy: bool
    self_gravity: bool
    smoothing_length_min: float
    smoothing_length_max: float
    neighbor_min: int
    neighbor_max: int
    neighbor_mean: float
    unconverged: int
    iterations_used: int
    iteration_bound: int
    energy_floor_hits: int
    reach_clamps: int
    submission: str


class SphProductRuntime:
    """Drive the SPH stages from authored bindings on the product timeline.

    The duck type is the one `ProductRuntime` calls: `step`, `snapshot`, `restore_snapshot`,
    `diagnostics`, `close`, plus `last_step_ms`; and, since ADR 0109, the resumable frame
    transaction `runtime_foundation.PrimitiveFrameRuntime` describes --- `begin_frame`,
    `advance_frame_primitive`, `finish_frame`, `last_solver_primitive_ms` --- which is what
    lets the host draw between the steps that cross one frame. Body dictionaries carry the
    same keys the orbital runtimes publish, so the display and bake paths need no SPH
    branch, with the SPH state appended rather than substituted.
    """

    BACKEND = 'sph'

    def __init__(self, device_context: DeviceContext, config: SphProductConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.frames_advanced = 0
        self.last_step_ms = 0.0
        self.last_substeps = 0
        self.last_solver_primitive_ms = 0.0
        self.explicit_snapshot_readbacks = 0
        self.neighbors: NeighborRuntime | None = None
        self.hydro: HydroRuntime | None = None
        self._bodies: BodySet | None = None
        self._frozen: Any | None = None
        self._smoothing_scale = 0.0
        # The open authored frame, or None when no frame transaction is open. The simulation
        # time the frame must reach is fixed when it opens rather than recomputed per step,
        # so a frame cannot move its own finish line while it is being crossed.
        self._frame_target: float | None = None
        self._frame_substeps = 0
        self._frame_elapsed_ms = 0.0

    # -- construction -----------------------------------------------------------------

    def load_bindings(self, bindings: tuple[BindingDescriptor, ...]) -> None:
        self.load_bodies(
            expand_bindings(
                bindings,
                self.config.shell_map or None,
                gravitational_constant=self.config.gravitational_constant,
                softening=self.config.softening,
            ),
            body_counts=tuple(binding_body_count(binding) for binding in bindings),
        )

    def load_bodies(self, bodies: BodySet, body_counts: tuple[int, ...] | None = None) -> None:
        """Upload ``bodies``, settled first when the config asks for it (ADR 0168).

        ``body_counts`` splits the set into its bindings, in order, so each settled body's
        centre of mass goes back where it was authored. None treats the set as one body.
        """

        np = __import__("numpy")
        config = self.config
        if len(bodies) != config.count:
            raise ValueError(f"expected {config.count} bodies, got {len(bodies)}")
        spec = KERNEL_REGISTRY[config.kernel]
        # Median rather than mean: a scene that mixes one analytical body into a cloud has
        # one radius orders of magnitude away from the spacing, and a mean would follow it.
        spacing = float(np.median(bodies.radii))
        if not spacing > 0.0 or not math.isfinite(spacing):
            raise ValueError("SPH bodies must carry a positive interparticle spacing")
        target = config.target_neighbors or spec.recommended_neighbors
        h_init = (spacing / spec.support_radius(1.0)) * (
            3.0 * float(target) / (4.0 * math.pi)
        ) ** (1.0 / 3.0)
        self._smoothing_scale = h_init
        neighbor_config = NeighborConfig(
            capacity=config.count,
            count=config.count,
            h_init=h_init,
            h_min=h_init / config.smoothing_bracket,
            h_max=h_init * config.smoothing_bracket,
            kernel=config.kernel,
            target_neighbors=config.target_neighbors,
            neighbor_tolerance=config.neighbor_tolerance,
            precision=config.precision,
        )
        gravity_config = SelfGravityConfig(
            capacity=config.count,
            count=config.count,
            gravitational_constant=config.gravitational_constant,
            # A tree with zero softening is rejected, and a context that never set one has
            # no length of its own to offer, so the resolution the gas is already smoothed
            # at is the only non-arbitrary choice.
            softening=float(config.softening) or h_init,
            theta=config.theta,
            max_depth=config.max_depth,
            leaf_capacity=config.leaf_capacity,
            stack_capacity=config.stack_capacity,
            node_capacity=config.node_capacity,
            precision=config.precision,
            backend=config.gravity_backend,
        )
        eos = EOS_REGISTRY[config.equation_of_state]
        # One SolidConfig over the same sizes and the same precision as the neighbour
        # service, because `HydroRuntime` builds its `SolidRuntime` over its own
        # `NeighborRuntime`: the solid and the fluid see one neighbourhood, which is the
        # whole point of coupling them here rather than running two solvers.
        solid_config = (
            None
            if config.materials is None
            else SolidConfig(
                capacity=config.count,
                count=config.count,
                registry=config.materials,
                precision=config.precision,
                flaws_per_particle=config.flaws_per_particle,
                # The strength's force and gradient in the fluid's density form (ADR 0167).
                number_density=True,
                # The plain velocity gradient is the exact adjoint of the stress force, so
                # the elastic work the stress rate stores is the work the force does; the
                # moment-corrected one is not, and a body at rest gained energy (ADR 0170).
                correct_velocity_gradient=False,
            )
        )
        hydro_config = HydroConfig(
            neighbors=neighbor_config,
            eos=eos,
            alpha_viscosity=config.alpha_viscosity,
            beta_viscosity=config.beta_viscosity,
            balsara_switch=config.balsara_switch,
            courant_number=config.courant_number,
            max_timestep=config.seconds_per_frame / config.substeps,
            # Material bodies are sampled at one spacing with per-material masses (ADR
            # 0166), which only a number density reads without smearing one material's
            # mass into its neighbour.
            number_density=config.materials is not None,
            # A solid's density is integrated and its smoothing length follows it, as in
            # miluphcuda and Benz-Asphaug codes: the summed density of a lattice jumps each
            # time a neighbour crosses a fixed-count support, and pressure did work for
            # nothing (ADR 0170).
            solid_continuity=config.materials is not None,
            gravity=gravity_config,
            solid=solid_config,
        )
        internal_energy = (config.internal_energy if config.materials is not None else
                           config.internal_energy or _virial_internal_energy(
                               bodies, config.gravitational_constant, eos.gamma))
        energies = np.full(config.count, internal_energy, dtype=np.float64)
        # An inactive binding contributes particles that still hold the field they carry but
        # are never integrated, which is exactly what the stage calls frozen. Dropping them
        # instead would change every remaining particle's density.
        frozen = (bodies.active == 0).astype(np.int32)
        neighbors = NeighborRuntime(self.device_context, neighbor_config)
        hydro = HydroRuntime(neighbors, hydro_config)
        hydro.load_state(
            bodies.positions,
            bodies.velocities,
            bodies.masses,
            energies,
            frozen=frozen,
            material_ids=bodies.material_ids,
        )
        hydro.prime()
        if config.settle_steps:
            settled = hydro.relax(config.settle_steps, config.settle_damping)
            masses = np.asarray(bodies.masses, dtype=np.float64)
            authored = np.asarray(bodies.positions, dtype=np.float64)
            counts = body_counts if body_counts is not None else (len(bodies),)
            if sum(counts) != len(bodies):
                raise ValueError(f"body_counts sum to {sum(counts)}, not {len(bodies)}")
            start = 0
            for count in counts:
                # Each body back on its authored centre of mass: the settle changes a body's
                # shape, never where it is, and the bodies' pull on each other while settling
                # is not part of the scene.
                part = slice(start, start + count)
                start += count
                if count == 0:
                    # A Force binding contributes no bodies (ADR 0045).
                    continue
                weights = masses[part, None] / masses[part].sum()
                settled[part] += (weights * authored[part]).sum(0) - (
                    weights * settled[part]
                ).sum(0)
            # In place, not reloaded: the stress the settle built is what holds the settled
            # shape, and a reload would zero it and let the body snap again.
            hydro.restart_motion(settled, bodies.velocities, energies)
            bodies = replace(bodies, positions=settled)
        self._frozen = frozen
        self._bodies = bodies
        self.neighbors = neighbors
        self.hydro = hydro
        self.frames_advanced = 0
        self.last_substeps = 0
        self.last_step_ms = 0.0
        self._abandon_frame()

    # -- timeline ---------------------------------------------------------------------

    @property
    def frame_transaction_resumable(self) -> bool:
        """Can the host have its thread back between the steps that cross one frame?

        Yes, and the unit it gets back between is one `HydroRuntime.step`, which already
        ends by synchronizing the device and raising on the tree overflow flag --- the
        exact contract `runtime_foundation.ResumableFrameRuntime` asks of a primitive
        (ADR 0109). False before any state is loaded, because a runtime with no clock to
        cross would otherwise be told mid-frame what it should have been told before the
        frame opened.
        """

        return self.hydro is not None

    frame_primitive_sequence_adaptive = True
    """One frame is a sequence of identical steps whose *length* nothing authors.

    The reason this runtime declares no `frame_primitive_calibration_phases`. ADR 0061's
    calibration proves an ordered sequence of unlike backend slices, each of which must be
    visited before the whole is bounded; here every slice is the same kick-drift-kick step
    through the same stages, and the only thing that varies is how many the signal velocity
    asks for. Declaring a phase sequence would assert a structure this loop does not have,
    and one bound covers every step of it without that assertion.
    """

    def begin_frame(self) -> None:
        """Open a frame transaction at the simulation time one more frame ends on."""

        self._require_hydro()
        if self._frame_target is not None:
            raise RuntimeError("an SPH frame transaction is already open")
        self._frame_target = (
            float(self.frames_advanced + 1) * self.config.seconds_per_frame
        )
        self._frame_substeps = 0
        self._frame_elapsed_ms = 0.0
        self.last_solver_primitive_ms = 0.0

    def advance_frame_primitive(self) -> FramePrimitiveProgress:
        """Cross at most one Courant step, and say whether that finished the frame.

        At most one, because a frame whose predecessor overshot it may already be crossed
        when it opens: that frame costs no step, and reporting it complete without running
        one is the truthful answer rather than a spare step taken to have something to time.
        """

        hydro = self._require_hydro()
        target = self._frame_target
        if target is None:
            raise RuntimeError("no SPH frame transaction is open")
        slack = self._frame_slack()
        started = time.perf_counter()
        if hydro.simulation_time + slack < target:
            if self._frame_substeps >= self.config.max_substeps_per_frame:
                raise SphTimestepError(
                    f"one frame of {self.config.seconds_per_frame} s needed more than "
                    f"{self.config.max_substeps_per_frame} SPH steps at a Courant step of "
                    f"{hydro.timestep()} s; coarsen the cloud, shorten the frame, or raise "
                    "max_substeps_per_frame"
                )
            hydro.step()
            self._frame_substeps += 1
        self.last_solver_primitive_ms = (time.perf_counter() - started) * 1000.0
        self._frame_elapsed_ms += self.last_solver_primitive_ms
        return FramePrimitiveProgress(
            phase="sph_courant_step",
            authored_substep_complete=hydro.simulation_time + slack >= target,
        )

    def finish_frame(self) -> None:
        """Close the frame the primitives above crossed, and count it."""

        if self._frame_target is None:
            raise RuntimeError("no SPH frame transaction is open")
        self.last_step_ms = self._frame_elapsed_ms
        self.last_substeps = self._frame_substeps
        self.frames_advanced += 1
        self._frame_target = None

    def step(self, steps: int = 1) -> None:
        """Advance ``steps`` product frames by sub-cycling the Courant step.

        The same three calls the host drives one primitive per tick, run here without
        yielding between them. Expressed this way rather than as a second loop because two
        definitions of "how many steps cross a frame" can drift apart and one cannot: a
        caller that steps whole frames and a caller that slices them reach the identical
        simulation time through the identical sequence of `HydroRuntime.step` calls.

        The substep ceiling applies per authored frame here, where before a multi-frame
        call pooled it. Per frame is what the constant has always said it means --- steps
        one frame may spend before the sub-cycle gives up --- and pooling let one
        pathological frame spend a later frame's allowance without anything noticing.
        """

        if steps < 1:
            raise ValueError("steps must be positive")
        self._require_hydro()
        started = time.perf_counter()
        taken = 0
        for _frame in range(steps):
            self.begin_frame()
            while not self.advance_frame_primitive().authored_substep_complete:
                pass
            self.finish_frame()
            taken += self.last_substeps
        self.last_step_ms = (time.perf_counter() - started) * 1000.0
        self.last_substeps = taken

    def _frame_slack(self) -> float:
        # The clock is compared with a slack proportional to the largest step allowed, so a
        # frame that has already been crossed to within rounding does not buy another step.
        return (self.config.seconds_per_frame / self.config.substeps) * 1.0e-6

    def _abandon_frame(self) -> None:
        """Drop any open frame transaction, because the clock underneath it moved."""

        self._frame_target = None
        self._frame_substeps = 0
        self._frame_elapsed_ms = 0.0
        self.last_solver_primitive_ms = 0.0

    # -- state ------------------------------------------------------------------------

    def display_source(self) -> tuple[Any, int, Precision]:
        """The live position array a display stage samples, its length, and its precision.

        The neighbour runtime owns the positions the hydro stages integrate, so the
        display samples that array rather than a hydro-side copy of it.
        """

        return (
            self._require_neighbors().positions,
            self.config.count,
            self.config.precision,
        )

    def display_channels(self) -> dict[str, Any]:
        """Every device array a display stage may sample, keyed by channel name.

        The only lane that carries density and specific internal energy, so the only one
        where `PHYSICAL` particle size — ``solids.particle_radius(mass, density)`` — is
        offerable and where D13 Energy can include committed specific internal energy
        (`adr/0023-gpu-particle-renderer-viewport-adoption.md` §7). ``HydroRuntime`` owns
        the split: positions live on the neighbour runtime it integrates through, fluid
        channels live on the hydro arrays, and optional Solid damage/persistent material ID
        handles come from the coupled Solid runtime. All are device handles; this path never
        calls the validation/snapshot read accessors.
        """

        return self._require_hydro().display_channels()

    def fragment_report(self) -> dict[str, Any]:
        """What the bodies broke into, as of now.

        Host work over one readback, and deliberately not a graph stage: connectivity is a
        question about a finished state, not a per-step derivative, so running it inside the
        step loop would pay for it every substep to answer it once. `ProductRuntime` calls
        this at bake time and the answer rides in the manifest's observations.

        The linking length is the configured multiple of the *median* smoothing length. A
        shattered body has a long tail of isolated particles whose smoothing lengths have
        expanded to the bracket ceiling, and a mean would let that tail set the length at
        which the remnant is judged connected.
        """

        np = __import__("numpy")
        from .impact import cluster_fragments

        hydro = self._require_hydro()
        neighbors = self._require_neighbors()
        bodies = self._require_bodies()
        count = self.config.count
        smoothing = neighbors.smoothing_length_array.numpy()[:count]
        self.explicit_snapshot_readbacks += 3
        report = cluster_fragments(
            np.asarray(hydro.positions()[:count], dtype=np.float64),
            np.asarray(bodies.masses[:count], dtype=np.float64),
            self.config.linking_length_factor * float(np.median(smoothing)),
            velocities=np.asarray(hydro.velocities()[:count], dtype=np.float64),
            gravitational_constant=self.config.gravitational_constant,
        )
        return report.to_dict()

    def snapshot(self) -> dict[str, Any]:
        hydro = self._require_hydro()
        neighbors = self._require_neighbors()
        bodies = self._require_bodies()
        frozen = self._require_frozen()
        positions = hydro.positions()
        velocities = hydro.velocities()
        accelerations = hydro.accelerations()
        energies = hydro.internal_energies()
        densities = hydro.densities()
        pressures = hydro.pressures()
        smoothing = neighbors.smoothing_length_array.numpy()[: self.config.count]
        self.explicit_snapshot_readbacks += 7
        np = __import__("numpy")
        material_ids = (
            None
            if bodies.material_ids is None
            else np.asarray(bodies.material_ids, dtype=np.int64)
        )
        records = []
        for index in range(self.config.count):
            record = {
                "index": index,
                "stable_id": int(bodies.stable_ids[index]),
                "active": not bool(frozen[index]),
                "mass": float(bodies.masses[index]),
                "radius": float(bodies.radii[index]),
                "position": [float(value) for value in positions[index]],
                "velocity": [float(value) for value in velocities[index]],
                "acceleration": [float(value) for value in accelerations[index]],
                "internal_energy": float(energies[index]),
                "density": float(densities[index]),
                "pressure": float(pressures[index]),
                "smoothing_length": float(smoothing[index]),
            }
            if material_ids is not None:
                record["material_id"] = int(material_ids[index])
            records.append(record)
        solid_state = hydro.capture_solid_state()
        snapshot: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA_VERSION,
            "backend": self.BACKEND,
            "capacity": self.config.count,
            "count": self.config.count,
            "collision_mode": "ignore",
            "step": self.frames_advanced,
            "simulation_time": hydro.simulation_time,
            "sph_steps_taken": hydro.steps_taken,
            # Smoothing lengths and the derived fields are re-solved from the restored
            # positions rather than trusted from the snapshot: they are a function of the
            # state, and a stale one would make the first restored step read a field the
            # geometry no longer supports.
            "recomputed_on_restore": [
                "smoothing_lengths",
                "densities",
                "pressures",
                "accelerations",
            ],
            "bodies": records,
        }
        if solid_state is not None:
            # Integrated solid state (SR-12): stress, damage, activated flaws, distention and
            # the continuity-density bookkeeping. Additive, so a snapshot written before this
            # key existed still restores - as intact material, which is all it ever held.
            snapshot["solid_state"] = {
                name: values.tolist() for name, values in solid_state.items()
            }
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint snapshot schema")
        if snapshot.get("backend") != self.BACKEND:
            raise ValueError("checkpoint backend does not match SPH runtime")
        if int(snapshot.get("count", -1)) != self.config.count:
            raise ValueError("snapshot body count does not match runtime")
        bodies = list(snapshot.get("bodies") or ())
        if len(bodies) != self.config.count:
            raise ValueError("snapshot body count does not match runtime")
        if [int(body.get("index", -1)) for body in bodies] != list(range(self.config.count)):
            raise ValueError("snapshot body ordering is invalid")
        source = self._require_bodies()
        if [int(body["stable_id"]) for body in bodies] != [
            int(value) for value in source.stable_ids
        ]:
            raise ValueError("snapshot stable IDs do not match runtime ordering")
        np = __import__("numpy")
        hydro = self._require_hydro()
        material_ids = None
        if hydro.solid is not None:
            if source.material_ids is None:
                raise ValueError("solid-backed snapshot has no runtime material identities")
            # The checkpoint carries the persistent material identity needed to rebuild
            # SolidRuntime. Accept a pre-field snapshot only by recovering the same immutable
            # identities from this runtime's loaded BodySet; a present mismatch is corruption,
            # not a new material assignment to apply during restore.
            source_material_ids = np.asarray(source.material_ids, dtype=np.int64)
            material_ids = np.asarray(
                [
                    body.get("material_id", int(source_material_ids[index]))
                    for index, body in enumerate(bodies)
                ],
                dtype=np.int64,
            )
            if not bool(np.array_equal(material_ids, source_material_ids)):
                raise ValueError("snapshot material IDs do not match runtime ordering")
        solid_state = snapshot.get("solid_state")
        stage = getattr(hydro, "stage_solid_state", None)
        # SR-68: reject a bad solid payload before `load_state` replaces live state. The shapes
        # depend on arrays `load_state` allocates, so a runtime that has not loaded yet cannot
        # pre-validate and is left to the post-load check.
        if (
            hydro.solid is not None
            and solid_state is not None
            and stage is not None
            and getattr(hydro.solid, "_arrays", True)
        ):
            stage(solid_state)
        # Reload rather than patch the device arrays in place: `load_state` is the one path
        # that reallocates the derivative arrays and rebinds the tree, so restoring through
        # it cannot leave a stage holding an array the others no longer write.
        hydro.load_state(
            np.asarray([body["position"] for body in bodies], dtype=np.float64),
            np.asarray([body["velocity"] for body in bodies], dtype=np.float64),
            np.asarray([body["mass"] for body in bodies], dtype=np.float64),
            np.asarray([body["internal_energy"] for body in bodies], dtype=np.float64),
            frozen=self._frozen,
            material_ids=material_ids,
        )
        if hydro.solid is not None and solid_state is not None:
            hydro.restore_solid_state(solid_state)
        hydro.prime()
        hydro.simulation_time = float(snapshot.get("simulation_time", 0.0))
        hydro.steps_taken = int(snapshot.get("sph_steps_taken", 0))
        self.frames_advanced = int(snapshot.get("step", 0))
        self.last_substeps = 0
        self.last_step_ms = 0.0
        # A restore moves the clock the open frame was being crossed against, so whatever
        # partial frame was in flight no longer describes anything. The host's own
        # partial-frame coordinate is cleared by the same restore on its side.
        self._abandon_frame()

    def diagnostics(self) -> SphDiagnostics:
        hydro = self._require_hydro()
        neighbors = self._require_neighbors()
        hydro_diagnostics = hydro.diagnostics()
        neighbor_diagnostics = neighbors.diagnostics()
        smoothing = neighbors.smoothing_length_array.numpy()[: self.config.count]
        pressures = hydro.pressures()
        return SphDiagnostics(
            active_count=self.config.count - int(self._require_frozen().sum()),
            frozen_count=int(self._require_frozen().sum()),
            density_min=hydro_diagnostics.density_min,
            density_max=hydro_diagnostics.density_max,
            pressure_max=float(pressures.max()) if self.config.count else 0.0,
            sound_speed_max=hydro_diagnostics.sound_speed_max,
            timestep=hydro_diagnostics.timestep,
            sph_steps_taken=hydro_diagnostics.steps_taken,
            simulation_time=hydro_diagnostics.simulation_time,
            frames_advanced=self.frames_advanced,
            substeps_last_frame=self.last_substeps,
            step_ms=self.last_step_ms,
            eos=hydro_diagnostics.eos,
            gamma=hydro_diagnostics.gamma,
            alpha_viscosity=hydro_diagnostics.alpha_viscosity,
            beta_viscosity=hydro_diagnostics.beta_viscosity,
            balsara_switch=hydro_diagnostics.balsara_switch,
            courant_number=hydro_diagnostics.courant_number,
            evolves_internal_energy=hydro_diagnostics.evolves_internal_energy,
            self_gravity=hydro.gravity is not None,
            smoothing_length_min=float(smoothing.min()) if self.config.count else 0.0,
            smoothing_length_max=float(smoothing.max()) if self.config.count else 0.0,
            neighbor_min=neighbor_diagnostics.neighbor_min,
            neighbor_max=neighbor_diagnostics.neighbor_max,
            neighbor_mean=neighbor_diagnostics.neighbor_mean,
            unconverged=neighbor_diagnostics.unconverged,
            iterations_used=neighbor_diagnostics.iterations_used,
            iteration_bound=neighbor_diagnostics.iteration_bound,
            energy_floor_hits=hydro.energy_floor_hits(),
            reach_clamps=hydro.reach_clamps(),
            submission=hydro_diagnostics.submission,
        )

    def close(self) -> None:
        if self.hydro is not None:
            self.hydro.close()
        if self.neighbors is not None:
            self.neighbors.close()
        self.hydro = None
        self.neighbors = None
        self._bodies = None
        self._frozen = None

    # -- guards -----------------------------------------------------------------------

    def _require_hydro(self) -> HydroRuntime:
        if self.hydro is None:
            raise DeviceRequirementError("SphProductRuntime has no loaded state")
        return self.hydro

    def _require_neighbors(self) -> NeighborRuntime:
        if self.neighbors is None:
            raise DeviceRequirementError("SphProductRuntime has no loaded state")
        return self.neighbors

    def _require_bodies(self) -> BodySet:
        if self._bodies is None:
            raise DeviceRequirementError("SphProductRuntime has no loaded state")
        return self._bodies

    def _require_frozen(self) -> Any:
        if self._frozen is None:
            raise DeviceRequirementError("SphProductRuntime has no loaded state")
        return self._frozen


def _virial_internal_energy(bodies: BodySet, gravitational_constant: float, gamma: float) -> float:
    """Specific internal energy that puts the cloud in virial balance.

    For a self-gravitating gas sphere the scalar virial theorem reads
    ``3 (gamma - 1) U + W = 0``, and a uniform sphere of mass ``M`` and radius ``R`` has
    ``W = -(3/5) G M**2 / R``. Dividing the resulting ``U`` by ``M`` gives
    ``u = G M / (5 (gamma - 1) R)``: the energy at which pressure support balances the
    cloud's own gravity, rather than a number chosen to make a particular scene look right.

    Zero whenever the balance is undefined - no mass, no extent, or a closure that does not
    evolve a thermal energy at all - because a fabricated value would be indistinguishable
    from a measured one downstream.
    """

    np = __import__("numpy")
    total_mass = float(np.sum(bodies.masses))
    if total_mass <= 0.0 or gravitational_constant <= 0.0 or gamma <= 1.0:
        return 0.0
    centre = np.average(bodies.positions, axis=0, weights=bodies.masses)
    radius = float(np.max(np.linalg.norm(bodies.positions - centre, axis=1)))
    if not radius > 0.0 or not math.isfinite(radius):
        return 0.0
    return gravitational_constant * total_mass / (5.0 * (gamma - 1.0) * radius)
