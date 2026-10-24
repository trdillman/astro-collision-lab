"""Compressible SPH core: density summation, the EOS registry, and the hydrodynamic force.

This is the H2 stage of `docs/source-audit/SPEC/06_SPH_AND_NEIGHBOR_SEARCH.md`. It owns
SPH-002 (the compressible SPH core that has to exist and be validated before any implicit
or incompressible variant) and SPH-003 (a runtime equation-of-state and sound-speed
registry). It consumes the H1 neighbour service in `sph.py` without reimplementing it: the
smoothing lengths, the `HashGrid`, and the support convention all come from there, so a
change to the search backend does not touch the physics in this module.

The scheme is explicit, energy-integrating SPH with the symmetric pressure form, the
Monaghan-Gingold artificial viscosity, and the Balsara shear switch. `adr/0014-sph-hydro-
core-and-shock-validation.md` records why each of those is the form used, what the scheme
deliberately does not carry - the `grad-h` correction terms - and what the omission costs
in measured conservation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .nbody import Precision, _dtypes, _registrars
from .runtime_foundation import DeviceRequirementError
from .selfgravity import SelfGravityConfig, SelfGravityRuntime
from .solids import SolidConfig, SolidRuntime, _matrix_type, _solid_registrars
from .sph import KERNEL_REGISTRY, NeighborBackend, NeighborConfig, NeighborRuntime, SphKernel

# Device-side kernel selector. The smoothing kernel is a runtime configuration value, so it
# reaches the device as an integer rather than by specialising the Warp module a third time
# on top of precision: recompiling per kernel shape would put a first-run compile in front
# of every kernel switch in the UI for a branch that is uniform across the whole launch.
KERNEL_ID_CUBIC_SPLINE = 0
KERNEL_ID_WENDLAND_C2 = 1
KERNEL_ID_WENDLAND_C4 = 2

_KERNEL_IDS: dict[SphKernel, int] = {
    SphKernel.CUBIC_SPLINE: KERNEL_ID_CUBIC_SPLINE,
    SphKernel.WENDLAND_C2: KERNEL_ID_WENDLAND_C2,
    SphKernel.WENDLAND_C4: KERNEL_ID_WENDLAND_C4,
}

EOS_ID_IDEAL_GAS = 0
EOS_ID_ISOTHERMAL = 1
EOS_ID_POLYTROPIC = 2

# Reduction slots. Only order-independent reductions live here: `min` and `max` are exactly
# associative in floating point, so the result does not depend on which lane wins the race.
# Conservation sums are deliberately absent - a float `atomic_add` is not reproducible, and
# the conservation claims are gated, so the validation suite sums them on the host from an
# explicit readback instead.
REDUCE_DENSITY_MIN = 0
REDUCE_DENSITY_MAX = 1
REDUCE_SOUND_SPEED_MAX = 2
REDUCE_TIMESTEP = 3
REDUCE_SLOTS = 4


class EquationOfState(StrEnum):
    IDEAL_GAS = "ideal_gas"
    ISOTHERMAL = "isothermal"
    POLYTROPIC = "polytropic"


@dataclass(frozen=True)
class EosSpec:
    """One closure relating pressure and sound speed to the evolved state.

    The registry exists because the closure is the part of a compressible scheme that
    changes per problem while the force kernel does not. Each spec answers two questions
    with the same parameters - the pressure and the sound speed - because taking them from
    different closures is the classic way to get a stable-looking run whose timestep is
    silently wrong.

    ``evolves_internal_energy`` is a property of the closure, not a preference. Only the
    ideal-gas law leaves the thermal energy free; the barotropic closures fix the pressure
    from the density alone, so integrating a thermal energy alongside them would produce a
    number that never feeds back and would read as if the run were adiabatic when it is not.
    """

    name: EquationOfState
    gamma: float = 5.0 / 3.0
    reference_sound_speed: float = 1.0
    polytropic_constant: float = 1.0

    @property
    def evolves_internal_energy(self) -> bool:
        return self.name == EquationOfState.IDEAL_GAS

    def validate(self) -> None:
        adiabatic = self.name in (EquationOfState.IDEAL_GAS, EquationOfState.POLYTROPIC)
        if adiabatic and not self.gamma > 1.0:
            raise ValueError("gamma must exceed 1")
        if self.name == EquationOfState.ISOTHERMAL and self.reference_sound_speed <= 0.0:
            raise ValueError("reference_sound_speed must be positive")
        if self.name == EquationOfState.POLYTROPIC and self.polytropic_constant <= 0.0:
            raise ValueError("polytropic_constant must be positive")

    def pressure(self, density: float, internal_energy: float) -> float:
        if density < 0.0:
            raise ValueError("density must be non-negative")
        if self.name == EquationOfState.IDEAL_GAS:
            return (self.gamma - 1.0) * density * internal_energy
        if self.name == EquationOfState.ISOTHERMAL:
            return self.reference_sound_speed**2 * density
        return self.polytropic_constant * density**self.gamma

    def sound_speed(self, density: float, internal_energy: float) -> float:
        if self.name == EquationOfState.ISOTHERMAL:
            return self.reference_sound_speed
        if density <= 0.0:
            return 0.0
        return math.sqrt(self.gamma * self.pressure(density, internal_energy) / density)


# Defaults only. A caller that needs different parameters constructs an `EosSpec` directly;
# the registry names the closures the runtime knows how to compile, which is what SPH-003
# asks for, and fixes one set of parameters per name so a report that names a closure
# without repeating its parameters is still reproducible.
EOS_REGISTRY: dict[EquationOfState, EosSpec] = {
    EquationOfState.IDEAL_GAS: EosSpec(name=EquationOfState.IDEAL_GAS, gamma=5.0 / 3.0),
    EquationOfState.ISOTHERMAL: EosSpec(
        name=EquationOfState.ISOTHERMAL, reference_sound_speed=1.0
    ),
    EquationOfState.POLYTROPIC: EosSpec(
        name=EquationOfState.POLYTROPIC, gamma=5.0 / 3.0, polytropic_constant=1.0
    ),
}


@dataclass(frozen=True)
class HydroConfig:
    """Configuration of the hydrodynamic stage, including the neighbour stage it drives."""

    neighbors: NeighborConfig
    eos: EosSpec = EOS_REGISTRY[EquationOfState.IDEAL_GAS]
    # Monaghan-Gingold artificial viscosity. `alpha` sets the bulk term that spreads a shock
    # over a few smoothing lengths and `beta` the quadratic term that stops interpenetration
    # in strong shocks; the pair 1 and 2 is the setting the scheme was published with and is
    # the setting the Sod and Sedov sections are measured at.
    alpha_viscosity: float = 1.0
    beta_viscosity: float = 2.0
    # Regulariser in `mu_ij`, in units of the mean smoothing length squared. It bounds the
    # viscous term for a pair approaching along a line as the separation goes to zero; it is
    # not a tuning knob for the shock profile.
    viscosity_softening: float = 0.01
    # Balsara shear switch. Off means the bulk viscosity acts in pure shear as well, which
    # damps differential rotation that carries no shock at all.
    balsara_switch: bool = True
    balsara_epsilon: float = 1.0e-4
    courant_number: float = 0.3
    # Fixed step, in the same time unit as the state. Zero means the step is the Courant
    # step measured from the signal velocity. A fixed step exists so a convergence study can
    # separate the spatial error from the temporal one.
    fixed_timestep: float = 0.0
    max_timestep: float = 0.0
    # Floor on the specific internal energy. A strong rarefaction can drive the energy
    # equation negative in one step; the floor keeps the sound speed real and is counted, so
    # a run that leans on it is visible rather than silently cold.
    internal_energy_floor: float = 0.0
    # Number density: `rho_i = m_i sum_j W_ij`, with the pressure gradient and energy
    # equation that belong to it (ADR 0166). Off is the mass sum, which the shock gates are
    # measured with and which is the same number wherever neighbours share a mass. On is
    # for bodies sampled at one spacing with masses that differ by material - a layered
    # planet, or two bodies of different particle mass in contact - where the mass sum
    # reads one side's mass into the other.
    number_density: bool = False
    # Continuity density for the solid particles (ADR 0170): a particle with a material
    # carries its density as state, starts at its material's reference density and moves
    # by the continuity equation, and its smoothing length follows that density instead of
    # being re-solved to an integer neighbour count. The summation density a rock body is
    # sampled into reads 10 % off its reference at rest - several GPa of pressure noise -
    # and every re-solved `h` moves it again with no work done for it. Off keeps the
    # summation density every fluid gate is measured with; on needs a solid stage.
    solid_continuity: bool = False
    # Self-gravity. Absent means the particles feel pressure and viscosity only, which is
    # what the shock tests want: Sod and Sedov are gravity-free by construction, and adding
    # a field they do not have would change what those gates measure.
    gravity: SelfGravityConfig | None = None
    # Solid response. Absent means every particle is a fluid, which is what the shock and
    # collapse gates want. Present means the bound materials carry shear strength, and the
    # constitutive stages of `solids.py` are recorded into this stage's own derivative
    # evaluation rather than driven as a second solver over a second neighbourhood.
    solid: SolidConfig | None = None

    @property
    def count(self) -> int:
        return self.neighbors.count

    @property
    def capacity(self) -> int:
        return self.neighbors.capacity

    @property
    def precision(self) -> Precision:
        return self.neighbors.precision

    def validate(self) -> None:
        self.neighbors.validate()
        self.eos.validate()
        if self.alpha_viscosity < 0.0 or self.beta_viscosity < 0.0:
            raise ValueError("viscosity coefficients must be non-negative")
        if not 0.0 < self.viscosity_softening < 1.0:
            raise ValueError("viscosity_softening must be in (0, 1)")
        if not 0.0 < self.balsara_epsilon < 1.0:
            raise ValueError("balsara_epsilon must be in (0, 1)")
        if not 0.0 < self.courant_number <= 1.0:
            raise ValueError("courant_number must be in (0, 1]")
        if self.fixed_timestep < 0.0:
            raise ValueError("fixed_timestep must be non-negative")
        if self.max_timestep < 0.0:
            raise ValueError("max_timestep must be non-negative")
        if self.internal_energy_floor < 0.0:
            raise ValueError("internal_energy_floor must be non-negative")
        if self.gravity is not None:
            self.gravity.validate()
            # The gravity stage repeats the sizes and the precision so it can also be
            # driven on its own. Repeated is fine; disagreeing is not - a tree built over
            # a different count would silently leave the tail of the particles massless.
            if (
                self.gravity.count != self.neighbors.count
                or self.gravity.capacity != self.neighbors.capacity
                or self.gravity.precision != self.neighbors.precision
            ):
                raise ValueError(
                    "gravity sizes and precision must match the neighbour configuration"
                )
        if self.solid_continuity and self.solid is None:
            raise ValueError("solid_continuity needs a solid stage: it is the solid's density")
        if self.solid is not None:
            self.solid.validate()
            # Same argument as the gravity stage: the solid config repeats the sizes so it
            # can also be driven standalone by the `adr/0022` suite, and a mismatch would
            # leave the tail of the particles without a material rather than fail.
            if (
                self.solid.count != self.neighbors.count
                or self.solid.capacity != self.neighbors.capacity
                or self.solid.precision != self.neighbors.precision
            ):
                raise ValueError(
                    "solid sizes and precision must match the neighbour configuration"
                )
            # The fold computes S1 with this stage's switch, and S6 is the deviatoric half of
            # the stress whose isotropic half this stage's pressure gradient carries: both
            # halves of one divergence in one form, or the coupling makes energy (ADR 0167).
            if self.solid.number_density != self.number_density:
                raise ValueError(
                    "the solid stage's number_density must match the fluid's: the stress "
                    "divergence and its velocity gradient are one pair with the pressure "
                    "gradient"
                )


@dataclass(frozen=True)
class HydroDiagnostics:
    active_count: int
    density_min: float
    density_max: float
    sound_speed_max: float
    timestep: float
    steps_taken: int
    simulation_time: float
    stage_ms: float
    eos: str
    gamma: float
    alpha_viscosity: float
    beta_viscosity: float
    balsara_switch: bool
    courant_number: float
    evolves_internal_energy: bool
    submission: str = "eager"


class HydroRuntime:
    """Own the hydrodynamic state and submit one recorded kick-drift-kick step.

    The neighbour search is not re-recorded here: `NeighborRuntime.record_stage()` writes its
    launches into whatever sequence is being captured. A step is two graphs around one grid
    build the host issues itself, because a captured build replays another grid's descriptor
    (ADR 0096, `NeighborRuntime.build_grid`): the timestep commit, the first kick and the
    drift; then the build over the drifted positions; then the adaptive-``h`` search, the
    density, the closure, the force, and the second kick. The build is ordered on the stream
    behind the drift, so the split costs a submission, not a synchronization. Nothing in the
    step reads back, including the timestep - the Courant step lives in a device scalar and
    is committed at the top of the next step, which is why a step can be replayed without the
    host learning anything about the state.

    The force kernels query one grid, captured once at construction (`_record_derivatives`
    freezes ``self.neighbors.grid_id`` into the replayed graph), so hydro only supports the
    uniform hash-grid neighbour backend - the adaptive multigrid backend is not routed to and
    is rejected below rather than silently searching the wrong level.
    """

    def __init__(self, neighbors: NeighborRuntime, config: HydroConfig) -> None:
        config.validate()
        if neighbors.config is not config.neighbors:
            raise ValueError("HydroConfig.neighbors must be the runtime's own configuration")
        if neighbors.config.backend is not NeighborBackend.UNIFORM_HASH_GRID:
            raise ValueError(
                "HydroRuntime requires the uniform hash-grid neighbour backend; the adaptive "
                "multigrid backend is not supported by the hydro force kernels"
            )
        self.neighbors = neighbors
        self.config = config
        self.gravity: SelfGravityRuntime | None = (
            None
            if config.gravity is None
            else SelfGravityRuntime(neighbors.device_context, config.gravity)
        )
        # Over this stage's own neighbour service, not a second one: the constitutive
        # stages sum over the same pairs the pressure gradient does, and a solid stage
        # with its own grid would give the two halves of one impact two neighbourhoods.
        self.solid: SolidRuntime | None = (
            None if config.solid is None else SolidRuntime(neighbors, config.solid)
        )
        self.steps_taken = 0
        self.simulation_time = 0.0
        self.last_stage_ms = 0.0
        self.explicit_validation_readbacks = 0
        self._wp: Any | None = None
        self._kernels: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._graphs: dict[str, Any] = {}

    def load_state(
        self,
        positions: Any,
        velocities: Any,
        masses: Any,
        internal_energies: Any,
        frozen: Any = None,
        material_ids: Any = None,
    ) -> None:
        """Upload the hydrodynamic state and allocate the derivative arrays.

        ``material_ids`` is required exactly when the configuration carries a solid stage,
        and a negative entry marks a particle that stays a fluid.

        ``frozen`` marks particles that take part in every sum but are never integrated.
        They are the rigid-wall boundary: an interior particle within a support radius of
        the wall still sees a full neighbourhood, so its density and its pressure gradient
        are the interior ones rather than a free surface's. A free surface is a physical
        boundary condition of its own - it launches a rarefaction into the domain from the
        first step - which is why a bounded problem needs this rather than a wider box.
        """

        self.neighbors.load_points(positions)
        wp = self.neighbors.device_context.wp
        self._wp = wp
        np = __import__("numpy")
        config = self.config
        scalar, vector, state_dtype = _dtypes(wp, config.precision)
        self._kernels = _hydro_kernels(wp, config.precision)
        device = self.neighbors.device_context.device
        count = config.count
        capacity = config.capacity

        def padded_vectors(values: Any, label: str) -> Any:
            supplied = np.asarray(values, dtype=state_dtype)
            if supplied.shape != (count, 3):
                raise ValueError(f"{label} must have shape ({count}, 3), got {supplied.shape}")
            padded = np.zeros((capacity, 3), dtype=state_dtype)
            padded[:count] = supplied
            return wp.array(padded, dtype=vector, device=device)

        def padded_scalars(values: Any, label: str, positive: bool) -> Any:
            supplied = np.asarray(values, dtype=state_dtype)
            if supplied.shape != (count,):
                raise ValueError(f"{label} must have shape ({count},), got {supplied.shape}")
            if positive and not bool(np.all(supplied > 0.0)):
                raise ValueError(f"{label} must be strictly positive")
            if not positive and not bool(np.all(supplied >= 0.0)):
                raise ValueError(f"{label} must be non-negative")
            padded = np.zeros(capacity, dtype=state_dtype)
            padded[:count] = supplied
            return wp.array(padded, dtype=scalar, device=device)

        def zeros_scalar() -> Any:
            return wp.zeros(capacity, dtype=scalar, device=device)

        def padded_flags(values: Any) -> Any:
            padded = np.zeros(capacity, dtype=np.int32)
            if values is not None:
                supplied = np.asarray(values, dtype=np.int32)
                if supplied.shape != (count,):
                    raise ValueError(f"frozen must have shape ({count},), got {supplied.shape}")
                padded[:count] = supplied
            return wp.array(padded, dtype=wp.int32, device=device)

        self._arrays = {
            "frozen": padded_flags(frozen),
            "velocities": padded_vectors(velocities, "velocities"),
            "masses": padded_scalars(masses, "masses", True),
            "internal_energies": padded_scalars(internal_energies, "internal_energies", False),
            "predicted_velocities": wp.zeros(capacity, dtype=vector, device=device),
            "predicted_energies": zeros_scalar(),
            "densities": zeros_scalar(),
            "omegas": zeros_scalar(),
            "pressures": zeros_scalar(),
            "sound_speeds": zeros_scalar(),
            "divergence": zeros_scalar(),
            "curl": wp.zeros(capacity, dtype=vector, device=device),
            "balsara": zeros_scalar(),
            "accelerations": wp.zeros(capacity, dtype=vector, device=device),
            "energy_rates": zeros_scalar(),
            "density_rates": zeros_scalar(),
            "particle_timesteps": zeros_scalar(),
            "reach": zeros_scalar(),
            "reduce": wp.zeros(REDUCE_SLOTS, dtype=scalar, device=device),
            "timestep": wp.zeros(2, dtype=scalar, device=device),
            "floor_hits": wp.zeros(1, dtype=wp.int32, device=device),
            "reach_clamps": wp.zeros(1, dtype=wp.int32, device=device),
        }
        if self.solid is not None:
            if material_ids is None:
                raise ValueError("a solid-backed HydroRuntime needs material_ids")
            ids = np.asarray(material_ids, dtype=np.int64)
            if ids.shape != (count,):
                raise ValueError(f"material_ids must have shape ({count},), got {ids.shape}")
            # The solid stage needs a host density only to size the Weibull flaw population,
            # which is a property of the undeformed body: at load time the particles sit at
            # their material's reference density by construction, and the device array it
            # builds from this is swapped for the SPH density immediately below. A fluid
            # particle draws no flaws, so its placeholder is never read. An unknown id falls
            # through to `SolidRuntime.load_state`, which names it.
            reference_of = {
                material.material_id: material.reference_density
                for material in config.solid.registry.materials
            }
            reference = np.array(
                [1.0 if value < 0 else reference_of.get(int(value), 1.0)
                 for value in ids.tolist()],
                dtype=state_dtype,
            )
            self.solid.load_state(
                positions=positions,
                masses=masses,
                densities=reference,
                internal_energies=internal_energies,
                material_ids=ids,
            )
            # Bound to the predicted state, not the committed one: every rate the recorded
            # stages evaluate is a rate of the predicted state, which is what the
            # hydrodynamic force and energy kernels already read.
            self.solid.bind_continuum(
                {
                    "velocities": self._arrays["predicted_velocities"],
                    "masses": self._arrays["masses"],
                    "densities": self._arrays["densities"],
                    "omegas": self._arrays["omegas"],
                    "internal_energies": self._arrays["predicted_energies"],
                    "pressures": self._arrays["pressures"],
                    "sound_speeds": self._arrays["sound_speeds"],
                    "reach": self._arrays["reach"],
                    "timesteps": self._arrays["timestep"],
                    # ADR 0171. The heat the yield and the cracks release is a change of
                    # state, not a rate, so it lands in the committed energy; the predicted
                    # one is scratch the next predictor overwrites.
                    "heated_energies": self._arrays["internal_energies"],
                }
            )
            if config.solid_continuity:
                # ADR 0170. A particle with a material starts at its reference density -
                # the density it was sampled at - rather than at the summation density,
                # which reads its own lattice and free surface as GPa of pressure. A fluid
                # particle (negative id) keeps the summation density, so its mask is 0.
                # `reference_lengths` 0 means "not yet captured": the first evaluation
                # takes the solved `h` as the length that belongs to that density.
                mask = np.zeros(capacity, dtype=np.int32)
                mask[:count] = (ids >= 0).astype(np.int32)
                start = np.zeros(capacity, dtype=state_dtype)
                start[:count] = reference
                self._arrays.update(
                    {
                        "continuity": wp.array(mask, dtype=wp.int32, device=device),
                        "continuity_densities": wp.array(start, dtype=scalar, device=device),
                        "predicted_continuity_densities": zeros_scalar(),
                        "reference_lengths": zeros_scalar(),
                        "reference_densities": zeros_scalar(),
                        "held_lengths": zeros_scalar(),
                    }
                )
        if self.gravity is not None:
            # Bound, not copied: the tree has to be built over the coordinates the drift
            # writes, which are the neighbour stage's own array. Bound last, because the
            # solid stage above reloads the shared neighbour points, which allocates a new
            # position array: bound before it, gravity summed the frame-0 copy for the
            # whole run while the drift moved the other one (ADR 0169).
            self.gravity.bind(self.neighbors.positions, self._arrays["masses"])
        self._graphs.clear()
        self.steps_taken = 0
        self.simulation_time = 0.0

    def prime(self) -> None:
        """Evaluate the derivatives once so the first step has an acceleration and a step."""

        self._require_state()
        start = time.perf_counter()
        self.neighbors.build_grid()
        self._submit("prime", self._record_derivatives)
        self.last_stage_ms = (time.perf_counter() - start) * 1000.0
        self._wait()
        if self.gravity is not None:
            self.gravity._raise_tree_overflow()

    def step(self) -> None:
        """Advance one kick-drift-kick step at the committed Courant timestep."""

        self._require_state()
        start = time.perf_counter()
        self._submit("drift", self._record_drift)
        self.neighbors.build_grid()
        self._submit("forces", self._record_forces)
        self.last_stage_ms = (time.perf_counter() - start) * 1000.0
        self._wait()
        if self.gravity is not None:
            self.gravity._raise_tree_overflow()
        self.steps_taken += 1
        self.simulation_time += self.committed_timestep()

    def advance(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _index in range(steps):
            self.step()

    def relax(self, steps: int, damping: float) -> Any:
        """Damp the loaded state towards rest and return where the particles settled.

        The settle an authored body takes before its first frame (ADR 0168). A free surface
        reads about half a neighbourhood, so a body sampled at its reference density starts
        with its outer layer under tension; integrated as authored, that layer snaps inward
        in the first steps and the body rings. Here the velocities start at zero and every
        step keeps ``damping`` of them, so the body slides into the discrete equilibrium of
        its own sample instead. The settled stress, damage and distention stay where they
        are: the elastic stress is what holds the settled shape, and a body reloaded without
        it snaps a second time (measured: the shear energy rebuilt from zero to its settled
        value within 25 steps). The caller hands the motion back with `restart_motion`.

        One host read at the end and one velocity copy per step, both while the scene is
        being built. Not a validation readback, so it does not tick the proof counter.
        """

        if steps < 0:
            raise ValueError("steps must be non-negative")
        if not 0.0 <= damping < 1.0:
            raise ValueError("damping must be in [0, 1)")
        self._require_state()
        np = __import__("numpy")
        velocities = self._arrays["velocities"]
        velocities.zero_()
        # The viscous term reads the velocities, so the first step needs the derivatives of
        # the state it actually starts from.
        self.prime()
        for _index in range(steps):
            self.step()
            velocities.assign(velocities.numpy() * damping)
        self._wait()
        return np.asarray(self.neighbors.positions.numpy()[: self.config.count], dtype=np.float64)

    def restart_motion(self, positions: Any, velocities: Any, internal_energies: Any) -> None:
        """Start the clock again from a settled body, keeping its material state.

        Written into the existing arrays rather than reloaded, so the stress a settle left
        behind survives and every captured graph still points at the arrays it recorded.
        """

        self._require_state()
        wp = self._require_wp()
        np = __import__("numpy")
        _scalar, _vector, state_dtype = _dtypes(wp, self.config.precision)
        count, capacity = self.config.count, self.config.capacity

        def padded(values: Any, shape: tuple[int, ...], label: str) -> Any:
            supplied = np.asarray(values, dtype=state_dtype)
            if supplied.shape != shape:
                raise ValueError(f"{label} must have shape {shape}, got {supplied.shape}")
            out = np.zeros((capacity, *shape[1:]), dtype=state_dtype)
            out[:count] = supplied
            return out

        self.neighbors.positions.assign(padded(positions, (count, 3), "positions"))
        self._arrays["velocities"].assign(padded(velocities, (count, 3), "velocities"))
        self._arrays["internal_energies"].assign(
            padded(internal_energies, (count,), "internal_energies")
        )
        self.steps_taken = 0
        self.simulation_time = 0.0
        self.prime()

    def _wait(self) -> None:
        wp = self._require_wp()
        wp.synchronize_device(self.neighbors.device_context.device)

    def _require_state(self) -> None:
        if not self._arrays:
            raise DeviceRequirementError("HydroRuntime has no loaded state")

    def _submit(self, name: str, record: Any) -> None:
        self._require_state()
        wp = self._require_wp()
        graph = self._graph(name, record)
        if graph is None:
            record()
        else:
            wp.capture_launch(graph)

    def _graph(self, name: str, record: Any) -> Any:
        if name in self._graphs:
            return self._graphs[name]
        wp = self._require_wp()
        device = wp.get_device(self.neighbors.device_context.device)
        if not device.is_cuda:
            self._graphs[name] = None
            return None
        wp.force_load(device=device)
        with wp.ScopedCapture(device) as capture:
            record()
        self._graphs[name] = capture.graph
        return capture.graph

    def _record_derivatives(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        config = self.config
        device = self.neighbors.device_context.device
        count = config.count
        spec = config.neighbors.spec
        kernel_id = _KERNEL_IDS[config.neighbors.kernel]
        eos = config.eos
        eos_id = _EOS_IDS[eos.name]
        arrays = self._arrays
        continuity = config.solid_continuity
        if continuity:
            # Before the search, so it can skip them: a continuity particle's support
            # follows its own density, `h ~ rho**(-1/3)`, so a body at rest keeps the
            # support it was given and the kernel sums it feeds stop jumping whenever one
            # neighbour crosses the integer-count edge (ADR 0170). Solving its `h` to a
            # count as well was work thrown away, and for a lone fragment that cannot reach
            # the count it was the search hunting out to `h_max` every step.
            wp.launch(
                kernels.hold_smoothing_lengths,
                dim=count,
                inputs=[
                    arrays["continuity"],
                    arrays["continuity_densities"],
                    arrays["density_rates"],
                    arrays["timestep"],
                    arrays["frozen"],
                    arrays["predicted_continuity_densities"],
                    arrays["reference_lengths"],
                    arrays["reference_densities"],
                    arrays["held_lengths"],
                    config.neighbors.h_min,
                    config.neighbors.h_max,
                ],
                device=device,
            )
            self.neighbors.record_stage(arrays["held_lengths"])
        else:
            self.neighbors.record_stage()
        grid_id = self.neighbors.grid_id
        positions = self.neighbors.positions
        smoothing_lengths = self.neighbors.smoothing_length_array
        if continuity:
            # A particle not yet held takes the length the search just solved for it as
            # the one that belongs to its starting density.
            wp.launch(
                kernels.capture_reference_lengths,
                dim=count,
                inputs=[
                    arrays["continuity"],
                    arrays["predicted_continuity_densities"],
                    smoothing_lengths,
                    arrays["reference_lengths"],
                    arrays["reference_densities"],
                ],
                device=device,
            )
        wp.launch(
            kernels.clear_reduce,
            dim=1,
            inputs=[arrays["reduce"], arrays["floor_hits"], arrays["reach_clamps"]],
            device=device,
        )
        wp.launch(
            kernels.predict_state,
            dim=count,
            inputs=[
                arrays["velocities"],
                arrays["internal_energies"],
                arrays["accelerations"],
                arrays["energy_rates"],
                arrays["timestep"],
                arrays["frozen"],
                arrays["predicted_velocities"],
                arrays["predicted_energies"],
                1 if eos.evolves_internal_energy else 0,
            ],
            device=device,
        )
        wp.launch(
            kernels.seed_reach,
            dim=count,
            inputs=[smoothing_lengths, arrays["reach"], spec.support_ratio],
            device=device,
        )
        # One walk of the grid for `scatter_reach`, `compute_density` and
        # `compute_velocity_gradients`: all three query the same sphere about the same
        # centre and keep the same neighbours, so the search was being paid three times to
        # answer one question. The three kernels are still compiled and still the definition
        # this fold is proven bit-for-bit against.
        wp.launch(
            kernels.compute_density_and_gradients,
            dim=count,
            inputs=[
                grid_id,
                positions,
                arrays["predicted_velocities"],
                smoothing_lengths,
                arrays["masses"],
                arrays["reach"],
                arrays["densities"],
                arrays["omegas"],
                arrays["divergence"],
                arrays["curl"],
                kernel_id,
                spec.normalization,
                spec.support_ratio,
                1 if config.number_density else 0,
            ],
            device=device,
        )
        if continuity:
            # The divergence and curl above stay the traversal's: they only steer the
            # shear switch. The density and its grad-h factor are replaced, because the
            # carried density is not a function of `h` at all.
            wp.launch(
                kernels.apply_continuity_density,
                dim=count,
                inputs=[
                    arrays["continuity"],
                    arrays["predicted_continuity_densities"],
                    arrays["densities"],
                    arrays["omegas"],
                ],
                device=device,
            )
        wp.launch(
            kernels.apply_equation_of_state,
            dim=count,
            inputs=[
                arrays["densities"],
                arrays["predicted_energies"],
                arrays["pressures"],
                arrays["sound_speeds"],
                arrays["reduce"],
                eos_id,
                eos.gamma,
                eos.reference_sound_speed,
                eos.polytropic_constant,
            ],
            device=device,
        )
        if self.solid is not None:
            # Over the gas closure rather than instead of it: the condensed kernel returns
            # early on a negative material index, so a fluid particle keeps what the launch
            # above wrote. Here rather than later because the elastic wave speed lands in
            # `sound_speeds`, and the force kernel's Courant bound reads it.
            self.solid.record_equation_of_state()
        # The shear switch, and only the switch, has to wait for the closure: it reads
        # `sound_speeds`, which the launch above and the solid override write. The gradients
        # it switches on were computed in the traversal, which needed neither.
        wp.launch(
            kernels.apply_balsara,
            dim=count,
            inputs=[
                smoothing_lengths,
                arrays["sound_speeds"],
                arrays["divergence"],
                arrays["curl"],
                arrays["balsara"],
                config.balsara_epsilon,
                1 if config.balsara_switch else 0,
            ],
            device=device,
        )
        forces = kernels.compute_hydro_forces
        force_inputs = [
            grid_id,
            positions,
            arrays["predicted_velocities"],
            smoothing_lengths,
            arrays["reach"],
            arrays["masses"],
            arrays["densities"],
            arrays["omegas"],
            arrays["pressures"],
            arrays["sound_speeds"],
            arrays["balsara"],
            arrays["accelerations"],
            arrays["energy_rates"],
            arrays["particle_timesteps"],
            arrays["reach_clamps"],
            kernel_id,
            spec.normalization,
            spec.support_ratio,
            config.alpha_viscosity,
            config.beta_viscosity,
            config.viscosity_softening,
            config.courant_number,
            1 if eos.evolves_internal_energy else 0,
            1 if config.number_density else 0,
            arrays["density_rates"],
        ]
        if self.solid is not None:
            # The solid's velocity gradient (S1) opens the query this kernel opens, over
            # the arrays `bind_continuum` gave it - these ones - so it is computed in the
            # same walk rather than a second one (ADR 0127). Nothing either half writes is
            # read by the other, and the solid reads `gradients` only from S2 on.
            forces = kernels.compute_hydro_forces_and_velocity_gradient
            force_inputs.extend(self.solid.velocity_gradient_outputs())
            # Cracked rock cannot pull: the pressure this walk carries is cut by the
            # solid's damage where it is negative (ADR 0172).
            force_inputs.append(self.solid.tension_damage())
        wp.launch(forces, dim=count, inputs=force_inputs, device=device)
        if self.solid is not None:
            # Same placement argument as the gravity block below, and ahead of it only
            # because the deviatoric divergence is the other half of the same contact
            # stress the pressure gradient just wrote.
            self.solid.record_stage()
            self.solid.record_apply(arrays["accelerations"])
        if self.gravity is not None:
            # After the pressure gradient, because that kernel assigns the acceleration
            # rather than accumulating into it, and before the timestep reduction, so the
            # free-fall bound is part of the step the reduction publishes.
            self.gravity.record_stage()
            self.gravity.record_apply(
                arrays["accelerations"], arrays["particle_timesteps"]
            )
        wp.launch(
            kernels.reduce_timestep,
            dim=count,
            inputs=[arrays["particle_timesteps"], arrays["reduce"]],
            device=device,
        )
        wp.launch(
            kernels.publish_timestep,
            dim=1,
            inputs=[
                arrays["reduce"],
                arrays["timestep"],
                config.fixed_timestep,
                config.max_timestep,
            ],
            device=device,
        )

    def _record_drift(self) -> None:
        """The first half of a step: commit the timestep, kick, drift. The grid build follows."""

        wp = self._require_wp()
        kernels = self._require_kernels()
        config = self.config
        device = self.neighbors.device_context.device
        arrays = self._arrays
        positions = self.neighbors.positions
        wp.launch(kernels.commit_timestep, dim=1, inputs=[arrays["timestep"]], device=device)
        self._record_kick()
        wp.launch(
            kernels.drift,
            dim=config.count,
            inputs=[positions, arrays["velocities"], arrays["timestep"], arrays["frozen"]],
            device=device,
        )

    def _record_forces(self) -> None:
        """The second half of a step, against the grid built over the drifted positions."""

        self._record_derivatives()
        self._record_kick()

    def _record_kick(self) -> None:
        wp = self._require_wp()
        kernels = self._require_kernels()
        config = self.config
        device = self.neighbors.device_context.device
        count = config.count
        arrays = self._arrays
        wp.launch(
            kernels.kick,
            dim=count,
            inputs=[
                arrays["velocities"],
                arrays["internal_energies"],
                arrays["accelerations"],
                arrays["energy_rates"],
                arrays["timestep"],
                arrays["frozen"],
                arrays["floor_hits"],
                config.internal_energy_floor,
                1 if config.eos.evolves_internal_energy else 0,
            ],
            device=device,
        )
        if config.solid_continuity:
            # The density is kicked like the thermal energy it is paired with, so the two
            # half steps move `rho` and `u` by the same rate the work was done at.
            wp.launch(
                kernels.kick_density,
                dim=count,
                inputs=[
                    arrays["continuity"],
                    arrays["continuity_densities"],
                    arrays["density_rates"],
                    arrays["timestep"],
                    arrays["frozen"],
                    arrays["reference_densities"],
                ],
                device=device,
            )

    def timestep(self) -> float:
        """The step the next derivative evaluation would take, as a host float.

        This is the Courant candidate in slot 0, published by the derivative evaluation that
        just ran - it is what the *next* step will use, not the step that was just integrated.
        For advancing ``simulation_time`` use `committed_timestep`.
        """

        self._wait()
        return float(self._arrays["timestep"].numpy()[0])

    def committed_timestep(self) -> float:
        """The step the last `step()` actually integrated, as a host float."""

        self._wait()
        return float(self._arrays["timestep"].numpy()[1])

    def diagnostics(self) -> HydroDiagnostics:
        self._wait()
        reduce = self._arrays["reduce"].numpy()
        eos = self.config.eos
        return HydroDiagnostics(
            active_count=self.config.count,
            density_min=float(reduce[REDUCE_DENSITY_MIN]),
            density_max=float(reduce[REDUCE_DENSITY_MAX]),
            sound_speed_max=float(reduce[REDUCE_SOUND_SPEED_MAX]),
            timestep=float(self._arrays["timestep"].numpy()[0]),
            steps_taken=self.steps_taken,
            simulation_time=self.simulation_time,
            stage_ms=self.last_stage_ms,
            eos=str(eos.name),
            gamma=eos.gamma,
            alpha_viscosity=self.config.alpha_viscosity,
            beta_viscosity=self.config.beta_viscosity,
            balsara_switch=self.config.balsara_switch,
            courant_number=self.config.courant_number,
            evolves_internal_energy=eos.evolves_internal_energy,
            submission="cuda_graph" if any(self._graphs.values()) else "eager",
        )

    def energy_floor_hits(self) -> int:
        self._wait()
        return int(self._arrays["floor_hits"].numpy()[0])

    def reach_clamps(self) -> int:
        """Pairs the last force pass dropped because only one side could see the other.

        Zero means the pair set was exactly symmetric, which is what makes the momentum
        sum exact. ``scatter_reach`` is constructed so this is always zero rather than
        merely small, so a non-zero count is not a tolerance to widen - it means a pair
        escaped the reach the scatter pass was supposed to guarantee, and the criterion
        itself is what has to be re-examined.
        """

        self._wait()
        return int(self._arrays["reach_clamps"].numpy()[0])

    def search_radii(self) -> Any:
        return self._read("reach")

    @property
    def frozen_array(self) -> Any:
        """The device ``frozen`` flag array, without a readback.

        Mirrors ``NeighborRuntime.smoothing_length_array``: returned rather than
        copied, so a diagnostic that only needs to know which slots are frozen
        (e.g. the T5a eligible-inputs stage in ``fragment_boundary.py``) reads
        the very array the solver already maintains instead of taking a second,
        potentially stale copy through ``_read``.
        """

        if "frozen" not in self._arrays:
            raise DeviceRequirementError("HydroRuntime has no loaded state")
        return self._arrays["frozen"]

    def display_channels(self) -> dict[str, Any]:
        """Device handles for the viewport, deliberately bypassing the read accessors.

        Every other accessor on this class routes through :meth:`_read`, which copies to
        the host and increments ``explicit_validation_readbacks``. That counter is the
        subject of a proof gate, and a viewport that ticked it once per frame would make
        the gate meaningless — so the display path takes the arrays themselves and samples
        them on the device (`adr/0023-gpu-particle-renderer-viewport-adoption.md` §7).

        ``velocities`` is the committed array, not ``predicted_velocities``: the predicted
        state is mid-step scratch that only has meaning inside a substep, and drawing from
        it would show the solver's intermediate guess rather than the frame it solved.
        """

        return {
            "positions": self.neighbors.positions,
            "velocities": self._arrays["velocities"],
            "masses": self._arrays["masses"],
            "densities": self._arrays["densities"],
            # Specific internal energy is authoritative state even for barotropic closures
            # that do not evolve it. It is never relabelled as Temperature: v1 has no
            # accepted EOS/heat-capacity mapping from one to the other.
            "internal_energies": self._arrays["internal_energies"],
            **({} if self.solid is None else self.solid.color_source_channels()),
        }

    def positions(self) -> Any:
        return self._read_vectors(self.neighbors.positions)

    def velocities(self) -> Any:
        return self._read_vectors(self._arrays["velocities"])

    def accelerations(self) -> Any:
        return self._read_vectors(self._arrays["accelerations"])

    def densities(self) -> Any:
        return self._read("densities")

    def grad_h_factors(self) -> Any:
        return self._read("omegas")

    def pressures(self) -> Any:
        return self._read("pressures")

    def sound_speeds(self) -> Any:
        return self._read("sound_speeds")

    def internal_energies(self) -> Any:
        return self._read("internal_energies")

    def energy_rates(self) -> Any:
        return self._read("energy_rates")

    def masses(self) -> Any:
        return self._read("masses")

    def divergences(self) -> Any:
        return self._read("divergence")

    def balsara_factors(self) -> Any:
        return self._read("balsara")

    def gravitational_accelerations(self) -> Any:
        """The gravitational part of the acceleration alone, for the energy accounting."""

        if self.gravity is None:
            raise DeviceRequirementError("HydroRuntime has no gravity stage")
        return self._read_vectors(self.gravity.accelerations)

    def _read(self, name: str) -> Any:
        """Explicit validation readback of one per-particle field."""

        self.explicit_validation_readbacks += 1
        self._wait()
        return self._arrays[name].numpy()[: self.config.count]

    def _read_vectors(self, array: Any) -> Any:
        self.explicit_validation_readbacks += 1
        self._wait()
        return array.numpy()[: self.config.count]

    # Continuity-density bookkeeping a solid-backed body integrates (ADR 0170); the first
    # evaluation after a load would otherwise re-capture the reference lengths from a
    # deformed body.
    CONTINUITY_STATE = (
        "continuity_densities",
        "reference_lengths",
        "reference_densities",
        "held_lengths",
    )

    def capture_solid_state(self) -> dict[str, Any] | None:
        """Host copies of every integrated solid quantity, or ``None`` with no solid stage."""

        if self.solid is None:
            return None
        self._wait()
        count = self.config.count
        state = dict(self.solid.capture_state())
        for name in self.CONTINUITY_STATE:
            if name in self._arrays:
                state[name] = self._arrays[name].numpy()[:count].copy()
        return state

    def restore_solid_state(self, state: dict[str, Any]) -> None:
        """Write `capture_solid_state` back in place; call after `load_state`, before `prime`."""

        if self.solid is None:
            raise ValueError("this HydroRuntime has no solid stage to restore")
        self._wait()
        # Stage and validate everything first, so a rejected restore writes nothing.
        continuity = self.stage_continuity_state(state)
        self.solid.restore_state(state)
        count = self.config.count
        for name, values in continuity.items():
            target = self._arrays[name]
            buffer = target.numpy()
            buffer[:count] = values
            target.assign(buffer)

    def stage_solid_state(self, state: dict[str, Any]) -> None:
        """Validate a whole `capture_solid_state` result without writing anything."""

        if self.solid is None:
            raise ValueError("this HydroRuntime has no solid stage to restore")
        self.solid.stage_state(state)
        self.stage_continuity_state(state)

    def stage_continuity_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate the continuity-density fields; return them as host arrays.

        Required whenever this runtime allocated them (a solid-continuity run): without them
        the first evaluation re-captures the reference lengths from the deformed body, which
        is the fresh-reference state a history restore must not silently fall back to. A
        runtime without continuity arrays ignores those keys.
        """

        np = __import__("numpy")
        count = self.config.count
        staged: dict[str, Any] = {}
        for name in self.CONTINUITY_STATE:
            if name not in self._arrays:
                continue
            if name not in state:
                raise ValueError(f"solid state is missing required continuity field {name!r}")
            buffer = self._arrays[name].numpy()
            supplied = np.asarray(state[name], dtype=buffer.dtype)
            if supplied.shape != (count, *buffer.shape[1:]):
                raise ValueError(
                    f"continuity state {name!r} has shape {supplied.shape}, "
                    f"expected {(count, *buffer.shape[1:])}"
                )
            # Densities and lengths are non-negative; 0 is the "not yet captured" marker.
            if not bool(np.isfinite(supplied).all()) or bool((supplied < 0).any()):
                raise ValueError(f"continuity state {name!r} must be finite and non-negative")
            staged[name] = supplied
        return staged

    def close(self) -> None:
        if self.gravity is not None:
            self.gravity.close()
        if self.solid is not None:
            self.solid.close()
        self._arrays.clear()
        self._graphs.clear()

    def _require_wp(self):
        if self._wp is None:
            raise DeviceRequirementError("Warp runtime is not loaded")
        return self._wp

    def _require_kernels(self):
        if self._kernels is None:
            raise DeviceRequirementError("Warp hydro kernels are not loaded")
        return self._kernels


_EOS_IDS: dict[EquationOfState, int] = {
    EquationOfState.IDEAL_GAS: EOS_ID_IDEAL_GAS,
    EquationOfState.ISOTHERMAL: EOS_ID_ISOTHERMAL,
    EquationOfState.POLYTROPIC: EOS_ID_POLYTROPIC,
}

# CUDA launch bounds for the coupled force traversal, per state precision (ADR 0127).
# Unbounded, float32 compiles to 144 registers: one 256-thread block per SM, a third of
# the resident warps either walk it replaced ran at, and a walk that is waiting on
# neighbour gathers is paid for in warps. (256, 2) caps it at 128 registers for two
# blocks, at a 24-byte spill, and is what makes one walk cheaper than two - 3.93 ms against
# 6.95 ms at 100,000 particles, where unbounded saved 0.36 ms. float64 is left alone: it
# is arithmetic-bound at 40 ms, unbounded already saves 2.5 ms, and the cap makes one walk
# 0.8 to 1.4 ms slower than the two separate ones. `launch_bounds` also sets a maximum
# block of 256 threads, which is `wp.launch`'s default and what `_record_derivatives`
# launches with.
_COUPLED_FORCE_LAUNCH_BOUNDS: dict[Precision, tuple[int, int] | None] = {
    Precision.FLOAT32: (256, 2),
    Precision.FLOAT64: None,
}


def _hydro_kernels(wp, precision: Precision = Precision.FLOAT32):
    """Compile the hydrodynamic kernels for one state precision.

    The smoothing kernel and the closure are runtime integers rather than specialisations:
    both branches are uniform across a launch, so they cost a predicted branch, and making
    them compile-time would multiply the module count by nine for no measurable gain.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, warp_func = _registrars(wp, scalar, vector)
    zero = scalar(0.0)
    half = scalar(0.5)
    one = scalar(1.0)
    two = scalar(2.0)
    three = scalar(3.0)
    # The least share of its reference density a carried density keeps (ADR 0176).
    density_floor = scalar(0.01)

    @warp_func
    def kernel_shape(kernel_id: int, q: scalar):
        t = one - half * q
        u = two - q
        shape = t * t * t * t * t * t * (one + three * q + scalar(35.0 / 12.0) * q * q)
        if kernel_id == KERNEL_ID_CUBIC_SPLINE:
            if q < one:
                shape = one - scalar(1.5) * q * q + scalar(0.75) * q * q * q
            else:
                shape = scalar(0.25) * u * u * u
        if kernel_id == KERNEL_ID_WENDLAND_C2:
            shape = t * t * t * t * (one + two * q)
        return shape

    @warp_func
    def kernel_shape_derivative(kernel_id: int, q: scalar):
        t = one - half * q
        u = two - q
        shape = scalar(-7.0 / 3.0) * q * t * t * t * t * t * (two + scalar(5.0) * q)
        if kernel_id == KERNEL_ID_CUBIC_SPLINE:
            # Kept as a branch rather than a conditional expression: Warp's code generator
            # is being asked to emit a device branch here, and the readable form is the one
            # that matches `kernel_shape` above.
            if q < one:  # noqa: SIM108
                shape = scalar(-3.0) * q + scalar(2.25) * q * q
            else:
                shape = scalar(-0.75) * u * u
        if kernel_id == KERNEL_ID_WENDLAND_C2:
            shape = scalar(-5.0) * q * t * t * t
        return shape

    @warp_func
    def kernel_value(
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        radius: scalar,
        smoothing_length: scalar,
    ):
        q = radius / smoothing_length
        if q >= support_ratio:
            return zero
        h3 = smoothing_length * smoothing_length * smoothing_length
        return normalization / h3 * kernel_shape(kernel_id, q)

    @warp_func
    def kernel_derivative(
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        radius: scalar,
        smoothing_length: scalar,
    ):
        q = radius / smoothing_length
        if q >= support_ratio:
            return zero
        h2 = smoothing_length * smoothing_length
        return normalization / (h2 * h2) * kernel_shape_derivative(kernel_id, q)

    # The solid stage's velocity gradient (S1), restated for the one traversal that computes
    # it alongside the pressure gradient; see `compute_hydro_forces_and_velocity_gradient`.
    # Restated rather than shared because it is not quite this module's arithmetic:
    # `solids.py` names its non-dyadic constant as a lane-typed object, where the fluid
    # derivative above writes it inline and Warp rounds an inline literal through float32 even
    # in the float64 lane. The copy here is `solids._solid_kernels`' `shape_derivative`,
    # `kernel_gradient` and `invariant_j2` operation for operation, and the fold's test holds
    # the two bit for bit against `solids`' own `compute_velocity_gradient`.
    matrix = _matrix_type(wp, precision)
    solid_kernel, solid_func = _solid_registrars(wp, scalar, vector, matrix)
    negative_seven_thirds = scalar(-7.0 / 3.0)
    moment_condition_floor = scalar(3.4526698e-4)

    @warp_func
    def solid_shape_derivative(kernel_id: int, q: scalar):
        t = one - half * q
        u = two - q
        shape = negative_seven_thirds * q * t * t * t * t * t * (two + scalar(5.0) * q)
        if kernel_id == KERNEL_ID_CUBIC_SPLINE:
            if q < one:  # noqa: SIM108
                shape = scalar(-3.0) * q + scalar(2.25) * q * q
            else:
                shape = scalar(-0.75) * u * u
        if kernel_id == KERNEL_ID_WENDLAND_C2:
            shape = scalar(-5.0) * q * t * t * t
        return shape

    @warp_func
    def solid_kernel_gradient(
        kernel_id: int, normalization: scalar, support_ratio: scalar, r: scalar, h: scalar
    ):
        if h <= zero or r <= zero:
            return zero
        q = r / h
        if q >= support_ratio:
            return zero
        h2 = h * h
        return normalization / (h2 * h2) * solid_shape_derivative(kernel_id, q)

    @solid_func
    def solid_invariant_j2(stress: matrix):
        total = zero
        for row in range(3):
            for column in range(3):
                total = total + stress[row, column] * stress[row, column]
        return half * total

    @warp_kernel
    def clear_reduce(
        reduce: wp.array(dtype=scalar),
        floor_hits: wp.array(dtype=wp.int32),
        reach_clamps: wp.array(dtype=wp.int32),
    ):
        reduce[REDUCE_DENSITY_MIN] = scalar(3.0e38)
        reduce[REDUCE_DENSITY_MAX] = scalar(0.0)
        reduce[REDUCE_SOUND_SPEED_MAX] = scalar(0.0)
        reduce[REDUCE_TIMESTEP] = scalar(3.0e38)
        floor_hits[0] = 0
        reach_clamps[0] = 0

    @warp_kernel
    def predict_state(
        velocities: wp.array(dtype=vector),
        internal_energies: wp.array(dtype=scalar),
        accelerations: wp.array(dtype=vector),
        energy_rates: wp.array(dtype=scalar),
        timestep: wp.array(dtype=scalar),
        frozen: wp.array(dtype=wp.int32),
        predicted_velocities: wp.array(dtype=vector),
        predicted_energies: wp.array(dtype=scalar),
        evolve_energy: int,
    ):
        """Carry the velocity and the thermal energy the rest of the way to the step's end.

        The kick-drift-kick pair evaluates the derivatives once, after the drift, and that
        evaluation stands for the end of the step. The positions are there already - the
        drift put them there - but the velocity has only had its first half kick, so it is
        still half a step behind, and so is the energy. Handing the force pass that stale
        velocity is not a rounding matter: the artificial viscosity and the whole thermal
        equation are functions of it, so the scheme silently drops to first order and the
        total energy walks in one direction instead of oscillating. The Sedov blast made
        that visible - the error halved exactly with the step rather than quartering.

        The missing half is ``a * dt / 2`` with the acceleration at the step's end, which is
        precisely what this evaluation is about to produce and therefore cannot use. Using
        the acceleration from the previous evaluation instead is wrong by
        ``(a(t+dt) - a(t)) * dt / 2``, which is second order, so the prediction is accurate
        enough to restore the scheme's own order without costing a second force pass.

        These are scratch fields, not state. The integrator's own half kicks still act on
        ``velocities`` and ``internal_energies``, so nothing here can drift the trajectory;
        the prediction only decides where the derivatives are sampled.
        """

        i = wp.tid()
        step = half * timestep[1]
        if frozen[i] != 0:
            predicted_velocities[i] = velocities[i]
            predicted_energies[i] = internal_energies[i]
            return
        predicted_velocities[i] = velocities[i] + accelerations[i] * step
        if evolve_energy != 0:
            predicted_energies[i] = internal_energies[i] + energy_rates[i] * step
        else:
            predicted_energies[i] = internal_energies[i]

    @warp_kernel
    def seed_reach(
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        support_ratio: scalar,
    ):
        i = wp.tid()
        reach[i] = support_ratio * smoothing_lengths[i]

    @warp_kernel
    def scatter_reach(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        support_ratio: scalar,
    ):
        """Push each particle's own support out to everyone standing inside it.

        A pair interacts out to the *larger* of the two supports, so a particle whose own
        support is small still has to search far enough to find a large-support partner.
        It cannot find one by looking: anything it would have to discover sits, by
        definition, beyond the radius it would be searching with. So the large-support
        particle announces itself instead. After ``seed_reach`` has set every radius to
        its own support, this pass raises ``reach[j]`` to ``support * h_i`` for every j
        that particle i can already see, which is exactly the set of pairs j could not
        have found alone. The two together leave

            reach[i] >= support * max(h_i, h_j)

        for every pair within that distance, which is the whole criterion, reached in one
        pass with no iteration count to tune. ``wp.atomic_max`` is what keeps it
        reproducible - unlike a float sum, a max is exactly order-independent, so the
        radii do not depend on how the launch happens to interleave.

        The grid returns a superset of the sphere, so the distance is re-checked here:
        without it a particle just outside the support would still be handed a radius,
        which is harmless for correctness but inflates the force pass for no reason.
        """

        i = wp.tid()
        p = positions[i]
        span = support_ratio * smoothing_lengths[i]
        query = wp.hash_grid_query(grid, p, span)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i and wp.length(p - positions[j]) <= span:
                wp.atomic_max(reach, j, span)

    @warp_kernel
    def compute_density(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        omegas: wp.array(dtype=scalar),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        number_density: int,
    ):
        """Summation density and the grad-h factor that belongs with it.

        ``number_density`` selects ``rho_i = m_i sum_j W_ij`` over ``sum_j m_j W_ij``
        (ADR 0166). The two agree wherever neighbours share a mass; where they do not - a
        heavy core particle beside a light mantle one at the same spacing - the mass sum
        reads the core's mass into the mantle and the number form does not. Each neighbour
        is then weighted by ``m_i`` rather than ``m_j``, here and in the grad-h sum.

        A variable smoothing length makes the density at ``i`` depend on ``h_i``, and
        ``h_i`` in turn on that density. Differentiating the pressure force without
        accounting for that coupling leaves a term out, and the omission does not shrink
        with resolution: it is set by how fast ``h`` varies, not by how fine the sampling
        is. That term is ``Omega``.

        The smoothing length is set here by a neighbour-count target, which is the same
        constraint as ``rho h**3 = const`` up to the discreteness of the count, so
        ``dh/drho = -h / (3 rho)``. Substituting that into the usual definition and using
        ``sum_j m_j W = rho`` collapses the whole factor to a single extra sum:

            Omega_i = -1 / (3 rho_i) * sum_j m_j r_ij W'(r_ij, h_i)

        which is exactly ``1`` in the continuum limit - integrate ``4 pi r**3 W'`` by parts
        against a normalised kernel - and departs from it only where the sampling is
        discrete or ``h`` is changing. That is why it costs one accumulator here rather
        than a pass of its own.
        """

        i = wp.tid()
        p = positions[i]
        h = smoothing_lengths[i]
        radius = support_ratio * h
        # The self contribution is the whole reason a summation density is positive
        # definite; leaving it out makes an isolated particle massless. It drops out of the
        # grad-h sum instead, where its `r` is zero.
        total = masses[i] * kernel_value(kernel_id, normalization, support_ratio, zero, h)
        grad_h = zero
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i:
                r = wp.length(positions[j] - p)
                if r <= radius:
                    m_j = masses[j]
                    if number_density != 0:
                        m_j = masses[i]
                    total += m_j * kernel_value(
                        kernel_id, normalization, support_ratio, r, h
                    )
                    grad_h += (
                        m_j
                        * r
                        * kernel_derivative(kernel_id, normalization, support_ratio, r, h)
                    )
        densities[i] = total
        omega = one
        if total > zero:
            omega = -grad_h / (three * total)
        # A non-positive factor would flip the sign of every pressure force on this
        # particle. It cannot happen for a monotone kernel with enough neighbours, so
        # treating it as "no correction" keeps a degenerate configuration stable rather
        # than explosive.
        if omega <= zero:
            omega = one
        omegas[i] = omega

    @warp_kernel
    def apply_equation_of_state(
        densities: wp.array(dtype=scalar),
        internal_energies: wp.array(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        sound_speeds: wp.array(dtype=scalar),
        reduce: wp.array(dtype=scalar),
        eos_id: int,
        gamma: scalar,
        reference_sound_speed: scalar,
        polytropic_constant: scalar,
    ):
        i = wp.tid()
        rho = densities[i]
        pressure = zero
        sound_speed = zero
        if eos_id == EOS_ID_ISOTHERMAL:
            pressure = reference_sound_speed * reference_sound_speed * rho
            sound_speed = reference_sound_speed
        else:
            if eos_id == EOS_ID_IDEAL_GAS:
                pressure = (gamma - one) * rho * internal_energies[i]
            else:
                pressure = polytropic_constant * wp.pow(rho, gamma)
            if rho > zero:
                sound_speed = wp.sqrt(wp.max(gamma * pressure / rho, zero))
        pressures[i] = pressure
        sound_speeds[i] = sound_speed
        wp.atomic_min(reduce, REDUCE_DENSITY_MIN, rho)
        wp.atomic_max(reduce, REDUCE_DENSITY_MAX, rho)
        wp.atomic_max(reduce, REDUCE_SOUND_SPEED_MAX, sound_speed)

    @warp_kernel
    def compute_velocity_gradients(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        sound_speeds: wp.array(dtype=scalar),
        divergence: wp.array(dtype=scalar),
        curl: wp.array(dtype=vector),
        balsara: wp.array(dtype=scalar),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        balsara_epsilon: scalar,
        balsara_enabled: int,
        number_density: int,
    ):
        i = wp.tid()
        p = positions[i]
        v = velocities[i]
        h = smoothing_lengths[i]
        radius = support_ratio * h
        rho = densities[i]
        div = zero
        rot = vector(zero, zero, zero)
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i:
                offset = p - positions[j]
                r = wp.length(offset)
                if r > zero and r <= radius:
                    gradient = kernel_derivative(
                        kernel_id, normalization, support_ratio, r, h
                    ) / r
                    grad = offset * gradient
                    relative = velocities[j] - v
                    # Weighted as the density was, so dividing by it leaves a volume.
                    m_j = masses[j]
                    if number_density != 0:
                        m_j = masses[i]
                    div += m_j * wp.dot(relative, grad)
                    rot += m_j * wp.cross(relative, grad)
        if rho > zero:
            div = div / rho
            rot = rot / rho
        else:
            div = zero
            rot = vector(zero, zero, zero)
        divergence[i] = div
        curl[i] = rot
        if balsara_enabled == 0:
            balsara[i] = one
        else:
            # Balsara's shear switch: the bulk viscosity is kept where the flow is
            # compressing and removed where the same velocity difference is pure shear.
            # The epsilon term carries the dimensions of a velocity gradient so the switch
            # stays finite in a uniform flow where both measures vanish.
            magnitude_div = wp.abs(div)
            magnitude_rot = wp.length(rot)
            floor = balsara_epsilon * sound_speeds[i] / h
            denominator = magnitude_div + magnitude_rot + floor
            if denominator > zero:
                balsara[i] = magnitude_div / denominator
            else:
                balsara[i] = zero

    @warp_kernel
    def compute_density_and_gradients(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        masses: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        omegas: wp.array(dtype=scalar),
        divergence: wp.array(dtype=scalar),
        curl: wp.array(dtype=vector),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        number_density: int,
    ):
        """``scatter_reach``, ``compute_density`` and ``compute_velocity_gradients`` in one
        walk of the grid.

        Those three kernels ask the grid the *same question*. Each opens a query at
        ``support_ratio * smoothing_lengths[i]`` about ``positions[i]`` and keeps the
        neighbours whose distance is within it - the same centre, the same radius, the same
        acceptance test, three times over. Only what they do with an accepted neighbour
        differs: one raises that neighbour's reach, one adds its kernel weight, one adds its
        velocity difference. Walking the cells and re-measuring the distances is the
        expensive part and it was being paid three times to answer one question.

        Nothing forced them apart. The three write disjoint outputs; none reads anything
        another writes. ``compute_velocity_gradients`` appeared to need the density pass
        ahead of it, but it only reads ``densities[i]`` - its own entry, and only to divide
        by it after the loop - and that value is ``total``, sitting in a register here. Its
        one genuine cross-kernel dependency is ``sound_speeds[i]``, which the equation of
        state writes, and that is used solely by the Balsara switch; ``apply_balsara`` takes
        the switch out to its own per-particle launch so the dependency is still honoured
        without a traversal having to wait behind it.

        The arithmetic is unchanged rather than merely equivalent. Every accumulator keeps
        the same operand sequence it had in its own kernel, because the neighbours arrive in
        the order one query returns them and that query is the one all three were already
        making. ``wp.length(p - positions[j])`` replaces the density pass's
        ``wp.length(positions[j] - p)``, which is the same number: negation is exact and the
        components are squared. ``kernel_derivative`` is evaluated once and shared by the
        grad-h sum and the velocity gradient, which is the same call with the same arguments
        both passes already made separately.
        """

        # Grid order: consecutive threads take particles that share cells, so a warp walks
        # the same neighbours together. Nothing below depends on which thread holds `i`:
        # every write is to entry `i`, or an `atomic_max`, whose result has no order.
        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        p = positions[i]
        v = velocities[i]
        h = smoothing_lengths[i]
        radius = support_ratio * h
        # See `compute_density`: the self contribution is what makes a summation density
        # positive definite, and it drops out of the grad-h sum on its own where `r` is zero.
        total = masses[i] * kernel_value(kernel_id, normalization, support_ratio, zero, h)
        grad_h = zero
        div = zero
        rot = vector(zero, zero, zero)
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i:
                offset = p - positions[j]
                r = wp.length(offset)
                if r <= radius:
                    # `scatter_reach`: announce this particle's support to everyone standing
                    # inside it, so a small-support neighbour can still find the pair.
                    wp.atomic_max(reach, j, radius)
                    m_j = masses[j]
                    if number_density != 0:
                        m_j = masses[i]
                    total += m_j * kernel_value(
                        kernel_id, normalization, support_ratio, r, h
                    )
                    derivative = kernel_derivative(
                        kernel_id, normalization, support_ratio, r, h
                    )
                    grad_h += m_j * r * derivative
                    if r > zero:
                        grad = offset * (derivative / r)
                        relative = velocities[j] - v
                        div += m_j * wp.dot(relative, grad)
                        rot += m_j * wp.cross(relative, grad)
        densities[i] = total
        omega = one
        if total > zero:
            omega = -grad_h / (three * total)
        if omega <= zero:
            omega = one
        omegas[i] = omega
        if total > zero:
            div = div / total
            rot = rot / total
        else:
            div = zero
            rot = vector(zero, zero, zero)
        divergence[i] = div
        curl[i] = rot

    @warp_kernel
    def apply_balsara(
        smoothing_lengths: wp.array(dtype=scalar),
        sound_speeds: wp.array(dtype=scalar),
        divergence: wp.array(dtype=scalar),
        curl: wp.array(dtype=vector),
        balsara: wp.array(dtype=scalar),
        balsara_epsilon: scalar,
        balsara_enabled: int,
    ):
        """The shear switch, lifted out of the gradient traversal it used to end.

        It reads only entry ``i`` of arrays the traversal has already written, so it costs a
        launch and no search. Standing on its own is what lets the traversal run before the
        equation of state while the switch still runs after it, which is the only ordering
        either of them actually requires: ``sound_speeds[i]`` is the elastic or gas wave
        speed and it is not known until the closure - gas, then solid over the top of it -
        has been applied.
        """

        i = wp.tid()
        if balsara_enabled == 0:
            balsara[i] = one
        else:
            # Balsara's shear switch: the bulk viscosity is kept where the flow is
            # compressing and removed where the same velocity difference is pure shear.
            # The epsilon term carries the dimensions of a velocity gradient so the switch
            # stays finite in a uniform flow where both measures vanish.
            magnitude_div = wp.abs(divergence[i])
            magnitude_rot = wp.length(curl[i])
            floor = balsara_epsilon * sound_speeds[i] / smoothing_lengths[i]
            denominator = magnitude_div + magnitude_rot + floor
            if denominator > zero:
                balsara[i] = magnitude_div / denominator
            else:
                balsara[i] = zero

    @warp_kernel
    def compute_hydro_forces(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        omegas: wp.array(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        sound_speeds: wp.array(dtype=scalar),
        balsara: wp.array(dtype=scalar),
        accelerations: wp.array(dtype=vector),
        energy_rates: wp.array(dtype=scalar),
        particle_timesteps: wp.array(dtype=scalar),
        reach_clamps: wp.array(dtype=wp.int32),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        alpha: scalar,
        beta: scalar,
        viscosity_softening: scalar,
        courant_number: scalar,
        evolve_energy: int,
        number_density: int,
        density_rates: wp.array(dtype=scalar),
    ):
        # Grid order, as in `compute_density_and_gradients`.
        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        p = positions[i]
        v = velocities[i]
        h_i = smoothing_lengths[i]
        rho_i = densities[i]
        c_i = sound_speeds[i]
        f_i = balsara[i]
        # `P / (Omega rho**2)` is carried per particle, and each particle's term is paired
        # with the gradient of *its own* kernel below. That split is what makes the grad-h
        # correction expressible at all - a single averaged gradient has nowhere to put a
        # per-particle factor - and it is still exactly antisymmetric across the pair, so
        # momentum is conserved to round-off for every pair the search finds from both
        # sides.
        term_i = zero
        if rho_i > zero:
            term_i = pressures[i] / (omegas[i] * rho_i * rho_i)
        acceleration = vector(zero, zero, zero)
        energy_rate = zero
        density_rate = zero
        signal_max = two * c_i
        radius = reach[i]
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i:
                offset = p - positions[j]
                r = wp.length(offset)
                h_j = smoothing_lengths[j]
                # The pair reaches as far as the *larger* of the two supports, because the
                # mean gradient below is still non-zero out there: past `support * h_i` only
                # the `h_j` half survives, and dropping it truncates every pair that spans a
                # smoothing-length contrast - a contact discontinuity, exactly where it
                # matters. `scatter_reach` is what makes that criterion discoverable from
                # both sides, and it is built to leave no exceptions. Should one appear the
                # pair is dropped rather than counted once, so the set stays symmetric and
                # momentum stays exact, and `reach_clamps` records it instead of letting it
                # pass silently.
                span = support_ratio * wp.max(h_i, h_j)
                limit = wp.min(span, wp.min(radius, reach[j]))
                if r > zero and r <= span and r > limit:
                    wp.atomic_add(reach_clamps, 0, 1)
                if r > zero and r <= limit:
                    rho_j = densities[j]
                    direction = offset / r
                    gradient_i = kernel_derivative(
                        kernel_id, normalization, support_ratio, r, h_i
                    )
                    gradient_j = kernel_derivative(
                        kernel_id, normalization, support_ratio, r, h_j
                    )
                    grad_i = direction * gradient_i
                    grad_j = direction * gradient_j
                    # The viscosity is not a gradient of a thermodynamic quantity, so it has
                    # no `Omega` of its own and keeps the symmetric mean gradient.
                    grad_bar = (grad_i + grad_j) * half
                    relative = v - velocities[j]
                    approach = wp.dot(relative, offset)
                    viscosity = zero
                    mu = zero
                    if approach < zero:
                        h_bar = half * (h_i + h_j)
                        mu = h_bar * approach / (r * r + viscosity_softening * h_bar * h_bar)
                        c_bar = half * (c_i + sound_speeds[j])
                        rho_bar = half * (rho_i + rho_j)
                        if rho_bar > zero:
                            viscosity = (-alpha * c_bar * mu + beta * mu * mu) / rho_bar
                            viscosity = viscosity * (half * (f_i + balsara[j]))
                    term_j = zero
                    if rho_j > zero:
                        term_j = pressures[j] / (omegas[j] * rho_j * rho_j)
                    if number_density == 0:
                        acceleration -= (
                            grad_i * (masses[j] * term_i)
                            + grad_j * (masses[j] * term_j)
                            + grad_bar * (masses[j] * viscosity)
                        )
                        if evolve_energy != 0:
                            # The thermal half pairs `term_i` with `grad_i` for the same
                            # reason the force does, which is what makes the kinetic energy
                            # this pair removes equal to the internal energy it deposits.
                            energy_rate += masses[j] * (
                                term_i * wp.dot(relative, grad_i)
                                + half * viscosity * wp.dot(relative, grad_bar)
                            )
                        # The continuity rate is the sum the pressure work above multiplies
                        # by `term_i`, so a carried density and the energy move as
                        # `du = P / rho**2 drho` with nothing left over (ADR 0170).
                        density_rate += masses[j] * wp.dot(relative, grad_i)
                    else:
                        # The pressure gradient that belongs with a number density (ADR
                        # 0166): each side's term is weighted by its own volume squared,
                        # `V = m / rho`, so `m_i a_i` carries `m_i**2 term_i` and
                        # `m_j**2 term_j` and is still antisymmetric across the pair. In a
                        # uniform pressure it sums the gradient of the number density,
                        # which an even lattice holds flat across an interface; the mass
                        # form sums the gradient of the mass density, which jumps there.
                        m_i = masses[i]
                        m_j = masses[j]
                        acceleration -= (
                            grad_i * (m_i * term_i)
                            + grad_j * (m_j * (m_j / m_i) * term_j)
                            + grad_bar * (m_j * viscosity)
                        )
                        if evolve_energy != 0:
                            energy_rate += m_i * term_i * wp.dot(
                                relative, grad_i
                            ) + m_j * half * viscosity * wp.dot(relative, grad_bar)
                        density_rate += m_i * wp.dot(relative, grad_i)
                    # Monaghan's signal velocity. `mu` is non-positive for an approaching
                    # pair, so the viscous term raises the estimate exactly where the
                    # viscosity is active and the step has to shrink.
                    signal = c_i + sound_speeds[j] - three * mu
                    signal_max = wp.max(signal_max, signal)
        accelerations[i] = acceleration
        energy_rates[i] = energy_rate
        density_rates[i] = density_rate
        if signal_max > zero:
            particle_timesteps[i] = courant_number * h_i / signal_max
        else:
            particle_timesteps[i] = scalar(3.0e38)

    coupled_bounds = _COUPLED_FORCE_LAUNCH_BOUNDS[precision]

    @solid_kernel(**({"launch_bounds": coupled_bounds} if coupled_bounds else {}))
    def compute_hydro_forces_and_velocity_gradient(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        omegas: wp.array(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        sound_speeds: wp.array(dtype=scalar),
        balsara: wp.array(dtype=scalar),
        accelerations: wp.array(dtype=vector),
        energy_rates: wp.array(dtype=scalar),
        particle_timesteps: wp.array(dtype=scalar),
        reach_clamps: wp.array(dtype=wp.int32),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        alpha: scalar,
        beta: scalar,
        viscosity_softening: scalar,
        courant_number: scalar,
        evolve_energy: int,
        number_density: int,
        density_rates: wp.array(dtype=scalar),
        correct: int,
        gradients: wp.array(dtype=matrix),
        correction_failures: wp.array(dtype=wp.int32),
        damage: wp.array(dtype=scalar),
    ):
        """`compute_hydro_forces` and the solid stage's S1 in one walk of the grid.

        The solid's velocity-gradient kernel opens the same query this one does - radius
        `reach[i]` about `positions[i]`, on the same grid, over the same bound arrays - and
        neither reads anything the other writes. So a coupled step was paying for the search
        twice. The force half below is `compute_hydro_forces` line for line but for the
        tension cut; the S1 half is `solids.compute_velocity_gradient` line for line,
        including the parts that differ from the force half on purpose: the strict
        `r < limit` admission, the `h_j` and `rho_j` skips, and `offset * (derivative / r)`
        rather than `direction * derivative`. Both reference kernels stay compiled, and the
        fold is held bit for bit against them on an undamaged body.

        The tension cut (ADR 0172): a damaged particle carries `(1 - D)` of a negative
        pressure and all of a positive one, because a crack does not transmit tension but a
        crushed body still pushes back. The standalone stage applies it in
        `solids.compute_stress_forces`; coupled, the pressure is carried here, so it is
        applied here, to the force and to its work alike. ``pressures`` itself stays the
        closure's, which is what S3's yield surface, S4's strain and S5's crush curve read.
        At ``D = 0`` the factor is exactly one, so an undamaged step is unchanged bit for bit.
        """

        # Grid order, as in `compute_density_and_gradients`.
        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        p = positions[i]
        v = velocities[i]
        h_i = smoothing_lengths[i]
        rho_i = densities[i]
        c_i = sound_speeds[i]
        f_i = balsara[i]
        term_i = zero
        if rho_i > zero:
            pressure_i = pressures[i]
            if pressure_i < zero:
                pressure_i = pressure_i * (one - damage[i])
            term_i = pressure_i / (omegas[i] * rho_i * rho_i)
        acceleration = vector(zero, zero, zero)
        energy_rate = zero
        density_rate = zero
        signal_max = two * c_i
        raw = matrix(zero)
        moment = matrix(zero)
        # S1's own-volume weight under a number density, as `solids.compute_velocity_gradient`
        # forms it (ADR 0167); `validate` holds the solid's switch to this one.
        own = zero
        if rho_i > zero:
            own = masses[i] / rho_i
        radius = reach[i]
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i:
                offset = p - positions[j]
                r = wp.length(offset)
                h_j = smoothing_lengths[j]
                span = support_ratio * wp.max(h_i, h_j)
                limit = wp.min(span, wp.min(radius, reach[j]))
                if r > zero and r <= span and r > limit:
                    wp.atomic_add(reach_clamps, 0, 1)
                if r > zero and r <= limit:
                    rho_j = densities[j]
                    direction = offset / r
                    gradient_i = kernel_derivative(
                        kernel_id, normalization, support_ratio, r, h_i
                    )
                    gradient_j = kernel_derivative(
                        kernel_id, normalization, support_ratio, r, h_j
                    )
                    grad_i = direction * gradient_i
                    grad_j = direction * gradient_j
                    grad_bar = (grad_i + grad_j) * half
                    relative = v - velocities[j]
                    approach = wp.dot(relative, offset)
                    viscosity = zero
                    mu = zero
                    if approach < zero:
                        h_bar = half * (h_i + h_j)
                        mu = h_bar * approach / (r * r + viscosity_softening * h_bar * h_bar)
                        c_bar = half * (c_i + sound_speeds[j])
                        rho_bar = half * (rho_i + rho_j)
                        if rho_bar > zero:
                            viscosity = (-alpha * c_bar * mu + beta * mu * mu) / rho_bar
                            viscosity = viscosity * (half * (f_i + balsara[j]))
                    term_j = zero
                    if rho_j > zero:
                        pressure_j = pressures[j]
                        if pressure_j < zero:
                            pressure_j = pressure_j * (one - damage[j])
                        term_j = pressure_j / (omegas[j] * rho_j * rho_j)
                    if number_density == 0:
                        acceleration -= (
                            grad_i * (masses[j] * term_i)
                            + grad_j * (masses[j] * term_j)
                            + grad_bar * (masses[j] * viscosity)
                        )
                        if evolve_energy != 0:
                            energy_rate += masses[j] * (
                                term_i * wp.dot(relative, grad_i)
                                + half * viscosity * wp.dot(relative, grad_bar)
                            )
                        density_rate += masses[j] * wp.dot(relative, grad_i)
                    else:
                        m_i = masses[i]
                        m_j = masses[j]
                        acceleration -= (
                            grad_i * (m_i * term_i)
                            + grad_j * (m_j * (m_j / m_i) * term_j)
                            + grad_bar * (m_j * viscosity)
                        )
                        if evolve_energy != 0:
                            energy_rate += m_i * term_i * wp.dot(
                                relative, grad_i
                            ) + m_j * half * viscosity * wp.dot(relative, grad_bar)
                        density_rate += m_i * wp.dot(relative, grad_i)
                    signal = c_i + sound_speeds[j] - three * mu
                    signal_max = wp.max(signal_max, signal)
                # S1. `solids.compute_velocity_gradient` skips a particle with no support
                # outright, so its half is gated on `h_i` rather than the force half being
                # cut short.
                if h_i > zero and h_j > zero and r > zero and r < limit:
                    rho_s = densities[j]
                    if rho_s > zero:
                        derivative = solid_kernel_gradient(
                            kernel_id, normalization, support_ratio, r, h_i
                        )
                        if derivative != zero:
                            grad = offset * (derivative / r)
                            weight = masses[j] / rho_s
                            if number_density != 0:
                                weight = own
                            raw = raw + wp.outer(velocities[j] - v, grad) * weight
                            moment = moment + wp.outer(positions[j] - p, grad) * weight
        accelerations[i] = acceleration
        energy_rates[i] = energy_rate
        density_rates[i] = density_rate
        if signal_max > zero:
            particle_timesteps[i] = courant_number * h_i / signal_max
        else:
            particle_timesteps[i] = scalar(3.0e38)
        if h_i > zero:
            gradient = raw
            if correct != 0:
                frobenius_squared = two * solid_invariant_j2(moment)
                scale_cubed = frobenius_squared * wp.sqrt(frobenius_squared)
                if wp.abs(wp.determinant(moment)) > moment_condition_floor * scale_cubed:
                    gradient = raw * wp.inverse(moment)
                else:
                    wp.atomic_add(correction_failures, 0, 1)
            gradients[i] = gradient

    @warp_kernel
    def reduce_timestep(
        particle_timesteps: wp.array(dtype=scalar),
        reduce: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        wp.atomic_min(reduce, REDUCE_TIMESTEP, particle_timesteps[i])

    @warp_kernel
    def publish_timestep(
        reduce: wp.array(dtype=scalar),
        timestep: wp.array(dtype=scalar),
        fixed_timestep: scalar,
        max_timestep: scalar,
    ):
        # Slot 0 is the step the next `commit_timestep` will adopt; slot 1 is the step the
        # current kick-drift-kick pair is using. Keeping them apart is what makes the two
        # half kicks share one step even though the Courant estimate changes between them.
        candidate = reduce[REDUCE_TIMESTEP]
        if fixed_timestep > zero:
            candidate = fixed_timestep
        if max_timestep > zero:
            candidate = wp.min(candidate, max_timestep)
        if candidate >= scalar(1.0e38):
            # No pair gave a signal (cold, isolated particles) and no limit was configured: the
            # reduce still holds its 3.0e38 sentinel. Hold the step in force instead of
            # adopting it, or the drift multiplies a velocity by 3e38 and positions go to inf.
            candidate = timestep[1]
        timestep[0] = candidate

    @warp_kernel
    def commit_timestep(timestep: wp.array(dtype=scalar)):
        timestep[1] = timestep[0]

    @warp_kernel
    def kick(
        velocities: wp.array(dtype=vector),
        internal_energies: wp.array(dtype=scalar),
        accelerations: wp.array(dtype=vector),
        energy_rates: wp.array(dtype=scalar),
        timestep: wp.array(dtype=scalar),
        frozen: wp.array(dtype=wp.int32),
        floor_hits: wp.array(dtype=wp.int32),
        internal_energy_floor: scalar,
        evolve_energy: int,
    ):
        i = wp.tid()
        if frozen[i] != 0:
            return
        step = half * timestep[1]
        velocities[i] = velocities[i] + accelerations[i] * step
        if evolve_energy != 0:
            updated = internal_energies[i] + energy_rates[i] * step
            if updated < internal_energy_floor:
                updated = internal_energy_floor
                wp.atomic_add(floor_hits, 0, 1)
            internal_energies[i] = updated

    @warp_kernel
    def drift(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        timestep: wp.array(dtype=scalar),
        frozen: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if frozen[i] != 0:
            return
        positions[i] = positions[i] + velocities[i] * timestep[1]

    @warp_kernel
    def hold_smoothing_lengths(
        continuity: wp.array(dtype=wp.int32),
        densities: wp.array(dtype=scalar),
        density_rates: wp.array(dtype=scalar),
        timestep: wp.array(dtype=scalar),
        frozen: wp.array(dtype=wp.int32),
        predicted_densities: wp.array(dtype=scalar),
        reference_lengths: wp.array(dtype=scalar),
        reference_densities: wp.array(dtype=scalar),
        held_lengths: wp.array(dtype=scalar),
        h_min: scalar,
        h_max: scalar,
    ):
        """Predict a carried density to the step's end and the ``h`` it holds (ADR 0170).

        The prediction is `predict_state`'s, for the density: the derivatives stand for the
        step's end and the density has only had its first half kick. The length is the one
        the search solved at the first evaluation, scaled as ``rho**(-1/3)`` from there, so
        a particle keeps a fixed mass inside its support - the property the integer count
        only approximates one neighbour at a time. It is handed to the search as held; 0
        leaves the particle to be solved (a fluid particle, or one not yet captured).
        """

        i = wp.tid()
        held_lengths[i] = zero
        if continuity[i] == 0:
            return
        rho = densities[i]
        if frozen[i] == 0:
            # `kick_density`'s floor, for the same half step: the prediction is the density
            # the pressure and the forces are evaluated at, and a fragment thinning fast
            # enough to be predicted through zero took a negative density into both (ADR 0175).
            rho = wp.max(rho + density_rates[i] * half * timestep[1], half * rho)
            # And its lower bound, for the same reason (ADR 0176).
            rho = wp.max(rho, density_floor * reference_densities[i])
        predicted_densities[i] = rho
        if reference_lengths[i] <= zero:
            return
        length = reference_lengths[i] * wp.pow(
            reference_densities[i] / rho, scalar(1.0 / 3.0)
        )
        held_lengths[i] = wp.clamp(length, h_min, h_max)

    @warp_kernel
    def capture_reference_lengths(
        continuity: wp.array(dtype=wp.int32),
        predicted_densities: wp.array(dtype=scalar),
        smoothing_lengths: wp.array(dtype=scalar),
        reference_lengths: wp.array(dtype=scalar),
        reference_densities: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        if continuity[i] != 0 and reference_lengths[i] <= zero:
            reference_lengths[i] = smoothing_lengths[i]
            reference_densities[i] = predicted_densities[i]

    @warp_kernel
    def apply_continuity_density(
        continuity: wp.array(dtype=wp.int32),
        predicted_densities: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        omegas: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        if continuity[i] != 0:
            densities[i] = predicted_densities[i]
            omegas[i] = one

    @warp_kernel
    def kick_density(
        continuity: wp.array(dtype=wp.int32),
        densities: wp.array(dtype=scalar),
        density_rates: wp.array(dtype=scalar),
        timestep: wp.array(dtype=scalar),
        frozen: wp.array(dtype=wp.int32),
        reference_densities: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        if continuity[i] == 0 or frozen[i] != 0:
            return
        current = densities[i]
        # A half step cannot take more than half the density away: a rate that would is a
        # particle torn loose in one step, and a zero or negative density has no pressure.
        kicked = wp.max(current + density_rates[i] * half * timestep[1], half * current)
        # Nor can the halving go on for ever. A shattered fragment thinning step after step
        # reached a density whose square float32 cannot hold, and its stress force became
        # zero times infinity for every particle in its support (ADR 0176). 0 before the
        # particle's reference is captured, so the floor waits for it.
        densities[i] = wp.max(kicked, density_floor * reference_densities[i])

    return _HydroKernels(
        clear_reduce=clear_reduce,
        predict_state=predict_state,
        seed_reach=seed_reach,
        scatter_reach=scatter_reach,
        compute_density=compute_density,
        apply_equation_of_state=apply_equation_of_state,
        compute_velocity_gradients=compute_velocity_gradients,
        compute_density_and_gradients=compute_density_and_gradients,
        apply_balsara=apply_balsara,
        compute_hydro_forces=compute_hydro_forces,
        compute_hydro_forces_and_velocity_gradient=compute_hydro_forces_and_velocity_gradient,
        reduce_timestep=reduce_timestep,
        publish_timestep=publish_timestep,
        commit_timestep=commit_timestep,
        kick=kick,
        drift=drift,
        hold_smoothing_lengths=hold_smoothing_lengths,
        capture_reference_lengths=capture_reference_lengths,
        apply_continuity_density=apply_continuity_density,
        kick_density=kick_density,
    )


@dataclass(frozen=True)
class _HydroKernels:
    clear_reduce: Any
    predict_state: Any
    seed_reach: Any
    scatter_reach: Any
    compute_density: Any
    apply_equation_of_state: Any
    compute_velocity_gradients: Any
    compute_density_and_gradients: Any
    apply_balsara: Any
    compute_hydro_forces: Any
    compute_hydro_forces_and_velocity_gradient: Any
    reduce_timestep: Any
    publish_timestep: Any
    commit_timestep: Any
    kick: Any
    drift: Any
    hold_smoothing_lengths: Any
    capture_reference_lengths: Any
    apply_continuity_density: Any
    kick_density: Any


def sph_kernel_ids() -> dict[SphKernel, int]:
    """The device-side selector for each registered smoothing kernel."""

    return dict(_KERNEL_IDS)


def eos_ids() -> dict[EquationOfState, int]:
    """The device-side selector for each registered closure."""

    return dict(_EOS_IDS)


def kernel_registry_covers_device_ids() -> bool:
    """Every registered smoothing kernel has a device selector, and nothing else does."""

    return set(_KERNEL_IDS) == set(KERNEL_REGISTRY)
