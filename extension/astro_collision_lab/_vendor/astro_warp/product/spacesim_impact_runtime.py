"""The product runtime for the realtime rock-impact solver (ADR 0193).

`spacesim_impact.SpaceSimImpactSolver` is the engine: SpaceSim's incompressible pressure and
fracture, with bond-based strength and a two-rate clock. It is a VFX method, not a scientific
one, built so a rock-sized impact plays in the viewport with every particle simulated and
drawn. This module is what makes it an add-on backend rather than a script: it turns the
bindings a System compiles into the engine's particles, publishes the engine's state on the
same display channels every other backend publishes, and speaks the resumable frame protocol
`ProductRuntime` plays with.

It is shaped like `giant_impact_runtime`, and shares its body layout on purpose:

* **One resolution.** The pressure solve has one smoothing length, so every body is sampled at
  the one common spacing `sample_giant_impact_bodies` derives, with the idle slots, stable ids
  and layered masses that function already guarantees. A body authored finer than that uses
  fewer particles and the rest of its slots stay idle for the life of the run; the descriptor's
  capacity and the display's row count never change.
* **One primitive per frame.** Gravity splits across the frame's cadence intervals, so the smallest
  unit of work that leaves the state consistent is a whole frame. ``config.substeps`` is
  therefore 1: one resumable primitive per authored frame, one captured CUDA graph replay
  inside it. How many outer (pressure) and inner (bond) steps that frame holds is sized from
  the resolution, the meeting speed and the stiffest material, never from a slider.

Two things this module owns that the giant-impact one does not:

* **Material table.** The engine wants a shear modulus, a yield limit and a rest density per
  material, in code units. The plan carries the registry in code units already (the role lane
  converts each preset once, ADR 0041), so :func:`spacesim_material_table` only reads fields.
* **Fault cells.** A body is pre-fractured into cells, the nearest of N seeds drawn from the
  body's own particles; a bond crossing two cells breaks at the body's fault strain. The cells
  are assigned here, on the host, seeded from the binding so a reload draws the same cracks.

The engine imports Warp at module scope, so this module does not import it: compile sizes a
frame with :func:`spacesim_impact_sized_steps`, a Warp-free mirror of the engine's rule, and
only a loaded runtime touches the engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from .giant_impact_runtime import (
    GiantImpactBodies,
    giant_impact_layout_scale,
    sample_giant_impact_bodies,
)
from .impact_frame import ImpactFrameRuntime
from .materials import MaterialRegistry, StrengthModel
from .model import (
    CHECKPOINT_SCHEMA_VERSION,
    BindingDescriptor,
    Precision,
    runtime_stable_id,
)
from .runtime_foundation import DeviceContext
from .sampling import BodySet
from .spacesim_precision import finite_float32

BACKEND = "spacesim_impact"
MODEL = "spacesim_impact"
# 1.1.0 (ADR 0199): central gravity half-kicks. 1.2.0 (ADR 0204): a snapshot carries the
# reference length of every bond that has yielded. 1.3.0 (ADR 0204 amended): and each particle's
# thermal energy, and a frame may take more outer steps than planned. 1.4.0 (ADR 0220): each outer
# step is sized on the device from the fastest particle, so a frame's steps differ in length.
# 1.5.0 (ADR 0218): from 32,768 particles up the default gravity is the Barnes-Hut tree, so a
# large scene's trajectory differs from the all-pairs sum a 1.4.0 checkpoint was played with.
MODEL_VERSION = "1.5.0"

# The engine's sizing constants, mirrored so compile can size a frame before a device exists.
# `spacesim_impact` imports Warp at module scope, so asking it would need the very device compile
# is trying to avoid. `test_spacesim_impact_backend` pins every one of them, and
# `spacesim_step_counts` against `spacesim_impact.step_counts`, so the mirror cannot drift out
# from under the compile-time check (ADR 0108's pattern).
_SMOOTHING = 1.5
_COLLISION_STEP_FACTOR = 0.2
_EJECTA_SPEED_MARGIN = 1.5
_BOND_LIMIT_CROSSINGS = 0.70
_BOND_DT_SAFETY = 0.4

# Ceilings on one frame's work, past which a run is refused rather than played slowly. The
# shipped Asteroid Impact frame is 8 outer steps of 20 inner steps each, 160 inner steps in all;
# these leave a generous margin over it while still refusing a frame no interactive budget holds.
DEFAULT_MAX_OUTER_STEPS = 32
DEFAULT_MAX_INNER_STEPS_PER_FRAME = 1024

# One fault cell per this many particles of a body, at most `MAX_FAULT_CELLS`, and the strain a
# bond crossing two cells breaks at (ADR 0193 decision 5).
FAULT_CELL_PARTICLES = 400
MAX_FAULT_CELLS = 48
DEFAULT_FAULT_STRAIN = 0.003


class SpaceSimImpactTimestepError(RuntimeError):
    """Raised when a frame would need more steps than the ceilings allow."""


# -- sizing -----------------------------------------------------------------------------------


def spacesim_step_counts(
    *, frame_dt: float, spacing: float, characteristic_speed: float, max_wave_speed: float
) -> tuple[int, int]:
    """`spacesim_impact.step_counts` with the engine's default settings, mirrored Warp-free.

    ``(outer, inner)`` where inner is per outer step and always even. The engine measures the
    bond network's real stability limit when it builds and may then cut the inner step finer or
    coarser than this estimate; the outer count is exact either way.
    """

    h = _SMOOTHING * spacing
    v = _EJECTA_SPEED_MARGIN * max(characteristic_speed, 0.0)
    outer = max(1, math.ceil(frame_dt * v / (_COLLISION_STEP_FACTOR * h) - 1.0e-9))
    if max_wave_speed <= 0.0:
        inner = 1  # all fluid: no bonds to resolve
    else:
        limit = _BOND_LIMIT_CROSSINGS * spacing / max_wave_speed
        inner = max(1, math.ceil((frame_dt / outer) / (_BOND_DT_SAFETY * limit) - 1.0e-9))
    return outer, inner + inner % 2


def _wave_speed(shear_modulus: float, reference_density: float) -> float:
    """`spacesim_impact.wave_speed`: the bond network's longitudinal speed, ``sqrt(3 mu / rho)``."""

    return math.sqrt(3.0 * shear_modulus / reference_density)


def _shell_pairs(shells: Any) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Shells as ``(binding_id, layers)`` pairs, from either the config's tuple or a mapping."""

    if not shells:
        return ()
    items = shells.items() if isinstance(shells, Mapping) else shells
    return tuple((str(binding_id), tuple(layers)) for binding_id, layers in items)


def spacesim_material_table(
    materials: MaterialRegistry | None, shells: Any
) -> tuple[tuple[tuple[float, float, float], ...], dict[int, int]]:
    """Every material the shells use, as ``(shear_modulus, yield_limit, reference_density)``.

    Returns ``(records, id_map)``: ``records`` is ordered by material id and ``id_map`` takes a
    registry material id to its row, which is the index the engine's ``material_ids`` use. The
    values are in code units as the registry carries them. A material with no shear modulus, or
    with no strength model, is a fluid (shear 0): it forms no bonds and is held by the pressure
    solve and gravity alone. The yield limit is the Von Mises yield stress; a material the bond
    model cannot express is refused here, by name, instead of by the engine at load.
    """

    if materials is None:
        raise ValueError("the spacesim_impact backend needs a material registry")
    used = sorted(
        {int(layer[1]) for _binding_id, layers in _shell_pairs(shells) for layer in layers}
    )
    if not used:
        raise ValueError("the spacesim_impact backend needs the material layers of its bodies")
    records: list[tuple[float, float, float]] = []
    for material_id in used:
        try:
            material = materials.get(material_id)
        except KeyError as exc:
            raise ValueError(f"a body layer names an unknown material: {exc}") from exc
        density = float(material.reference_density)
        if not (math.isfinite(density) and density > 0.0):
            raise ValueError(f"{material.name}: reference_density must be positive and finite")
        shear = float(material.shear_modulus)
        if shear <= 0.0 or material.strength is StrengthModel.NONE:
            records.append((0.0, 0.0, density))
            continue
        # A bond inside one fault cell breaks at the material's Von Mises yield strain
        # (ADR 0193 decision 5), so a solid needs that model and a positive yield stress. Both
        # are refused here, by name, so compile refuses what Play would (ADR 0108).
        if material.strength is not StrengthModel.VON_MISES:
            raise ValueError(
                f"{material.name}: the rock-impact backend breaks a solid at its Von Mises "
                f"yield strain, and this material uses the {material.strength.value} model"
            )
        yield_limit = float(material.parameters.get("yield_stress", 0.0))
        if not (math.isfinite(yield_limit) and yield_limit > 0.0):
            raise ValueError(
                f"{material.name}: the rock-impact backend needs a positive yield stress to "
                f"break this solid at"
            )
        records.append((shear, yield_limit, density))
    return tuple(records), {material_id: row for row, material_id in enumerate(used)}


def spacesim_material_melt_energies(
    materials: MaterialRegistry | None, shells: Any
) -> tuple[float, ...]:
    """The shock heat at which each material of :func:`spacesim_material_table` has softened to
    nothing, in the same row order (ADR 0204).

    The ``melt_energy`` strength parameter, in code units as the registry carries it. A fluid, a
    material that does not declare one, and a registry of descriptors written before the
    parameter existed all give ``0``: a yield stress that does not soften with heat.
    """

    if materials is None:
        raise ValueError("the spacesim_impact backend needs a material registry")
    used = sorted(
        {int(layer[1]) for _binding_id, layers in _shell_pairs(shells) for layer in layers}
    )
    energies: list[float] = []
    for material_id in used:
        material = materials.get(material_id)
        energy = float(material.parameters.get("melt_energy", 0.0))
        if not (math.isfinite(energy) and energy >= 0.0):
            raise ValueError(f"{material.name}: melt_energy must be finite and not negative")
        energies.append(energy if float(material.shear_modulus) > 0.0 else 0.0)
    return tuple(energies)


def spacesim_impact_sized_steps(
    bindings: tuple[BindingDescriptor, ...],
    *,
    seconds_per_frame: float,
    gravitational_constant: float,
    materials: MaterialRegistry | None,
    shells: Any,
) -> tuple[int, int]:
    """``(outer, inner per outer)`` a frame of this scene needs, without sampling or a device.

    What `SpaceSimImpactRuntime.load_layout` computes from its sampled layout, computed from the
    authored bindings instead, so compile asks the question Play asks (ADR 0108). The common
    spacing and the characteristic speed are the giant-impact layout's, and the fastest wave is
    the stiffest material the bodies are made of.
    """

    scale = giant_impact_layout_scale(bindings, gravitational_constant)
    records, _ = spacesim_material_table(materials, shells)
    fastest = max(_wave_speed(shear, density) for shear, _yield, density in records)
    return spacesim_step_counts(
        frame_dt=float(seconds_per_frame),
        spacing=scale.spacing,
        characteristic_speed=scale.characteristic_speed,
        max_wave_speed=fastest,
    )


def spacesim_impact_timestep_refusal(
    *,
    seconds_per_frame: float,
    outer: int,
    inner: int,
    max_outer_steps: int = DEFAULT_MAX_OUTER_STEPS,
    max_inner_steps_per_frame: int = DEFAULT_MAX_INNER_STEPS_PER_FRAME,
) -> str | None:
    """Why this frame cannot be run, or ``None`` if it can.

    One wording for the compile-time refusal and the load-time one, so the artist is told the
    same thing wherever the scene is stopped. ``inner`` is per outer step; the ceiling is on
    the frame's total. Both knobs an artist can reach are named, because the step counts grow
    with the frame length and shrink with the particle spacing.
    """

    outer = int(outer)
    inner = int(inner)
    if outer > int(max_outer_steps):
        return (
            f"a frame of {seconds_per_frame:g} s needs {outer} pressure steps at this "
            f"resolution and impact speed, above the ceiling of {int(max_outer_steps)}; "
            f"shorten Seconds per Frame or use fewer particles"
        )
    total = outer * inner
    if total > int(max_inner_steps_per_frame):
        return (
            f"a frame of {seconds_per_frame:g} s needs {total} bond steps ({outer} pressure "
            f"steps of {inner}) at this resolution and material stiffness, above the ceiling "
            f"of {int(max_inner_steps_per_frame)}; shorten Seconds per Frame or use fewer "
            f"particles"
        )
    return None


# -- configuration ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceSimImpactConfig:
    """Everything the engine needs that is not derived from the bodies themselves."""

    count: int
    gravitational_constant: float
    seconds_per_frame: float
    precision: Precision = Precision.FLOAT32
    materials: Any = None
    shells: tuple[tuple[str, tuple[tuple[float, int, float], ...]], ...] = ()
    # Fragment linking length, in smoothing lengths, as `astro.diagnostics.fragments` means it
    # everywhere. 0.75 is one and a half particle spacings: neighbours in a settled body are one
    # spacing apart, so this links a body and not the gap between two.
    linking_length_factor: float = 0.75
    fault_strain: float = DEFAULT_FAULT_STRAIN
    fault_cell_particles: int = FAULT_CELL_PARTICLES
    max_outer_steps: int = DEFAULT_MAX_OUTER_STEPS
    max_inner_steps_per_frame: int = DEFAULT_MAX_INNER_STEPS_PER_FRAME
    use_graph: bool = True
    # Lab controls: None preserves the pinned dependency defaults.
    tree_theta: float | None = None
    gravity_softening: float | None = None

    @property
    def substeps(self) -> int:
        """Resumable primitives per authored frame: one, a whole engine frame."""

        return 1

    @property
    def shell_map(self) -> dict[str, tuple[tuple[float, int, float], ...]]:
        return {binding_id: layers for binding_id, layers in self.shells}

    def validate(self) -> None:
        if self.count < 1:
            raise ValueError("count must be positive")
        if not math.isfinite(self.gravitational_constant) or self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be positive")
        if not math.isfinite(self.seconds_per_frame) or self.seconds_per_frame <= 0.0:
            raise ValueError("seconds_per_frame must be positive")
        if self.precision != Precision.FLOAT32:
            # The engine's state is float32 end to end. Accepting float64 would publish a
            # precision label the arrays do not have.
            raise ValueError("the spacesim_impact backend runs float32 state only")
        if (self.materials is None) != (not self.shells):
            raise ValueError("materials and shells must be configured together")
        if self.materials is None:
            raise ValueError("the spacesim_impact backend needs materials and shells")
        if not (math.isfinite(self.linking_length_factor) and self.linking_length_factor > 0.0):
            raise ValueError("linking_length_factor must be positive")
        if not (math.isfinite(self.fault_strain) and self.fault_strain > 0.0):
            raise ValueError("fault_strain must be positive and finite")
        if self.fault_cell_particles < 1:
            raise ValueError("fault_cell_particles must be positive")
        if self.max_outer_steps < 1:
            raise ValueError("max_outer_steps must be positive")
        if self.max_inner_steps_per_frame < 1:
            raise ValueError("max_inner_steps_per_frame must be positive")
        # The materials are the part of the plan an author can get wrong by hand, and the one
        # the engine would otherwise refuse at load with no body or layer to name.
        spacesim_material_table(self.materials, self.shells)


# -- the sampled scene ------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceSimImpactBodies:
    """The sampled scene: the giant-impact layout plus each live particle's fault cell.

    The properties below pass the layout's fields through, so this stands wherever a
    `GiantImpactBodies` does (the colour authority reads ``bodies``; the runtime reads the rest).
    ``fault_cell_ids`` has one entry per live engine particle, the cell within its own body.
    """

    layout: GiantImpactBodies
    fault_cell_ids: Any

    @property
    def bodies(self) -> BodySet:
        return self.layout.bodies

    @property
    def solver_slots(self) -> Any:
        return self.layout.solver_slots

    @property
    def source(self) -> Any:
        return self.layout.source

    @property
    def body_ids(self) -> Any:
        return self.layout.body_ids

    @property
    def centres(self) -> Any:
        return self.layout.centres

    @property
    def spacing(self) -> float:
        return self.layout.spacing

    @property
    def characteristic_speed(self) -> float:
        return self.layout.characteristic_speed

    @property
    def body_names(self) -> tuple[str, ...]:
        return self.layout.body_names

    @property
    def active_count(self) -> int:
        return self.layout.active_count


def fault_cell_count(particles: int, fault_cell_particles: int) -> int:
    """How many cells a body of ``particles`` live particles is cracked into: one per
    ``fault_cell_particles`` (rounded up), at least one and at most `MAX_FAULT_CELLS`."""

    return max(1, min(MAX_FAULT_CELLS, math.ceil(int(particles) / int(fault_cell_particles))))


def assign_fault_cells(points: Any, cells: int, rng: Any) -> Any:
    """Which fault cell each point of one body is in: the nearest of ``cells`` seeds drawn from
    the body's own points. The engine's own `fault_cells` rule, restated Warp-free; 0 or 1 cell
    means the body is one piece."""

    np = __import__("numpy")
    n = points.shape[0]
    if cells <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)
    seeds = points[rng.choice(n, size=min(cells, n), replace=False)]
    best = np.full(n, np.inf)
    owner = np.zeros(n, dtype=np.int32)
    for k, seed in enumerate(seeds):
        d2 = ((points - seed) ** 2).sum(-1)
        closer = d2 < best
        best[closer] = d2[closer]
        owner[closer] = k
    return owner


def sample_spacesim_bodies(
    bindings: tuple[BindingDescriptor, ...],
    shells: dict[str, tuple[tuple[float, int, float], ...]] | None = None,
    *,
    gravitational_constant: float,
    fault_cell_particles: int = FAULT_CELL_PARTICLES,
) -> SpaceSimImpactBodies:
    """Sample every body at one common spacing, and crack each into fault cells.

    The layout is `sample_giant_impact_bodies`' (same refusals, same common spacing, idle slots,
    stable ids, layered masses and material ids); the cells are drawn from each body's live
    particles with a generator seeded by the binding, so the same scene always cracks the same
    way and no other body's cells move when one body is edited.
    """

    np = __import__("numpy")
    shells = shells or {}
    layout = sample_giant_impact_bodies(
        bindings, shells, gravitational_constant=gravitational_constant
    )
    for binding in bindings:
        if not shells.get(binding.binding_id):
            name = binding.object_name or binding.binding_id
            raise ValueError(f"{name}: a spacesim_impact body needs its material layers")
    live_positions = layout.bodies.positions[layout.solver_slots]
    cell_ids = np.zeros(layout.active_count, dtype=np.int32)
    for ordinal, binding in enumerate(bindings):
        chosen = np.flatnonzero(layout.body_ids == ordinal)
        rng = np.random.default_rng(
            [int(binding.sampling_seed), runtime_stable_id(binding.binding_id), 1]
        )
        cell_ids[chosen] = assign_fault_cells(
            live_positions[chosen],
            fault_cell_count(len(chosen), fault_cell_particles),
            rng,
        )
    return SpaceSimImpactBodies(layout=layout, fault_cell_ids=cell_ids)


# -- the runtime ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceSimImpactDiagnostics:
    """The engine's scalars in one flat record (serialised with `dataclasses.asdict`)."""

    active_count: int
    idle_count: int
    particle_spacing: float
    smoothing_length: float
    outer_steps: int
    inner_steps: int
    # The inner (bond) step and the outer (pressure) step, in code time.
    timestep: float
    outer_timestep: float
    bond_step_limit: float
    bond_count: int
    frames_advanced: int
    simulation_time: float
    step_ms: float
    speed_clamps: int
    neighbor_overflow: int
    # Reference bonds cut at load because a row passed its 63-bond capacity (longest first).
    dropped_bonds: int
    submission: str
    # Adaptive outer stepping (ADR 0220): the outer count of the last frame, the most any frame
    # has run, how many frames ran more than the planned ``outer_steps``, and how many steps the
    # ``max_outer_steps`` budget held longer than the fastest particle's reach.
    active_outer_steps: int = 0
    max_outer_steps_used: int = 0
    adaptive_frames: int = 0
    budget_bound_steps: int = 0


def _kernels(wp: Any) -> Any:
    @wp.kernel
    def publish(
        source: wp.array(dtype=wp.int32),
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        heat: wp.array(dtype=float),
        damage: wp.array(dtype=float),
        rho: wp.array(dtype=float),
        out_x: wp.array(dtype=wp.vec3),
        out_v: wp.array(dtype=wp.vec3),
        out_e: wp.array(dtype=float),
        out_d: wp.array(dtype=float),
        out_rho: wp.array(dtype=float),
    ):
        i = wp.tid()
        j = source[i]
        out_x[i] = x[j]
        out_v[i] = v[j]
        out_e[i] = heat[j]
        out_d[i] = damage[j]
        out_rho[i] = rho[j]

    return publish


def _bond_count(solver: Any) -> int:
    """How many bonds the engine captured at load: every bond has two row entries."""

    return int(solver.bond_count)


class SpaceSimImpactRuntime(ImpactFrameRuntime):
    """One realtime rock-impact scene on the device, played one whole frame at a time."""

    _ENGINE_LABEL = "SpaceSimImpactRuntime"
    # ``internal_energies`` is the engine's shock heat: the viscous dissipation of the impact,
    # cooling over a few frames. It is not a thermodynamic energy, but it is what the Energy
    # colour mode is for. ``damage`` is the fraction of a particle's bonds that have broken, the
    # number the Damage colour mode shows. ``densities`` is the SPH mass density the engine
    # already computes at the top of every outer step (in the material table's own density
    # unit), gathered through the same source map as the heat. It is the density of the
    # positions that step opened with, so it trails the published positions by under one outer
    # step: no extra pass is spent on it.
    _DISPLAY_CHANNELS = (
        "positions",
        "velocities",
        "masses",
        "radii",
        "active",
        "stable_ids",
        "internal_energies",
        "damage",
        "densities",
        "material_ids",
    )
    _LIVE_STATE_READBACKS = 4

    def __init__(self, device_context: DeviceContext, config: SpaceSimImpactConfig) -> None:
        config.validate()
        self.device_context = device_context
        self.config = config
        self.frames_advanced = 0
        self.last_step_ms = 0.0
        self.last_solver_primitive_ms = 0.0
        self.explicit_snapshot_readbacks = 0
        self.full_state_playback_readbacks = 0
        self._layout: SpaceSimImpactBodies | None = None
        self._solver: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._publish: Any | None = None
        self._frame_started_at: float | None = None

    # -- loading ------------------------------------------------------------------------

    def load_bindings(self, bindings: tuple[BindingDescriptor, ...]) -> SpaceSimImpactBodies:
        layout = sample_spacesim_bodies(
            bindings,
            self.config.shell_map or None,
            gravitational_constant=self.config.gravitational_constant,
            fault_cell_particles=self.config.fault_cell_particles,
        )
        self.load_layout(layout)
        return layout

    def _refuse_timestep(self, outer: int, inner: int) -> None:
        refusal = spacesim_impact_timestep_refusal(
            seconds_per_frame=self.config.seconds_per_frame,
            outer=outer,
            inner=inner,
            max_outer_steps=self.config.max_outer_steps,
            max_inner_steps_per_frame=self.config.max_inner_steps_per_frame,
        )
        if refusal is not None:
            raise SpaceSimImpactTimestepError(refusal)

    def load_layout(self, layout: SpaceSimImpactBodies) -> None:
        if len(layout.bodies) != self.config.count:
            raise ValueError("body count must match SpaceSimImpactConfig.count")
        np = __import__("numpy")
        records, id_map = spacesim_material_table(self.config.materials, self.config.shells)
        # The same call compile makes on the same numbers, so a frame compile admitted is a
        # frame this accepts and the reverse (ADR 0108). It runs before anything is allocated.
        estimated = spacesim_step_counts(
            frame_dt=self.config.seconds_per_frame,
            spacing=layout.spacing,
            characteristic_speed=layout.characteristic_speed,
            max_wave_speed=max(_wave_speed(shear, rho) for shear, _yield, rho in records),
        )
        self._refuse_timestep(*estimated)
        bodies = layout.bodies
        live = layout.solver_slots
        finite_float32(bodies.positions, "positions")
        finite_float32(bodies.velocities, "velocities")
        staged_masses = finite_float32(bodies.masses, "masses")
        staged_radii = finite_float32(bodies.radii, "radii")
        finite_float32(bodies.masses[live], "live masses", positive=True)
        finite_float32(bodies.radii[live], "live radii", positive=True)
        with np.errstate(over="ignore", invalid="ignore"):
            finite_float32(
                bodies.masses[live] * self.config.gravitational_constant, "gravity masses"
            )
        from . import spacesim_impact as engine

        self.device_context.initialize()
        wp = self.device_context.wp
        device = self.device_context.device
        registry_ids = np.asarray(bodies.material_ids, dtype=np.int64)[live]
        known = np.asarray(sorted(id_map), dtype=np.int64)
        rows = np.searchsorted(known, registry_ids)
        found = known[np.minimum(rows, known.size - 1)] == registry_ids
        if not np.all(found):
            raise ValueError("a body layer names a material outside the material table")
        melts = spacesim_material_melt_energies(self.config.materials, self.config.shells)
        materials = tuple(
            engine.SpaceSimMaterial(
                shear_modulus=shear,
                yield_limit=limit,
                reference_density=rho,
                melt_energy=melt,
            )
            for (shear, limit, rho), melt in zip(records, melts, strict=True)
        )
        # The planned outer count is the floor of a frame's; the config's ceiling is the most a
        # frame may take when the fastest particle asks for more, step by step (ADR 0220).
        settings = replace(
            engine.SpaceSimImpactSettings(),
            frame_dt=float(self.config.seconds_per_frame),
            max_outer_steps=int(self.config.max_outer_steps),
            tree_theta=(engine.SpaceSimImpactSettings().tree_theta if self.config.tree_theta is None
                        else float(self.config.tree_theta)),
            softening=(engine.SpaceSimImpactSettings().softening if self.config.gravity_softening is None
                       else float(self.config.gravity_softening)/layout.spacing),
        )
        self.close()
        solver = engine.SpaceSimImpactSolver(
            bodies.positions[live],
            bodies.velocities[live],
            bodies.masses[live],
            layout.body_ids,
            rows,
            materials,
            spacing=layout.spacing,
            characteristic_speed=layout.characteristic_speed,
            gravitational_constant=self.config.gravitational_constant,
            fault_cell_ids=layout.fault_cell_ids,
            fault_strain=self.config.fault_strain,
            settings=settings,
            device=device,
            use_graph=self.config.use_graph,
        )
        # The solver cuts its inner step from the stability limit it measures on the bonds it
        # just captured, which can differ from the estimate above in either direction, so the
        # ceilings are checked again on what it will actually run.
        self._refuse_timestep(solver.outer_steps, solver.inner_steps)
        solver.settle(centres=layout.centres)
        self._solver = solver
        self._layout = layout
        self._publish = _kernels(wp)
        count = self.config.count
        self._arrays = {
            "positions": wp.zeros(count, dtype=wp.vec3, device=device),
            "velocities": wp.zeros(count, dtype=wp.vec3, device=device),
            "internal_energies": wp.zeros(count, dtype=float, device=device),
            "damage": wp.zeros(count, dtype=float, device=device),
            "densities": wp.zeros(count, dtype=float, device=device),
            "masses": wp.array(staged_masses, dtype=float, device=device),
            "radii": wp.array(staged_radii, dtype=float, device=device),
            "active": wp.array(bodies.active.astype(np.int32), dtype=wp.int32, device=device),
            "stable_ids": wp.array(bodies.stable_ids, dtype=wp.int64, device=device),
            # The solver keeps its particles in its own (Z-) order; a slot's source is a caller
            # index, so it is mapped through ``internal_index`` once here.
            "source": wp.array(
                np.asarray(solver.internal_index, dtype=np.int32)[
                    np.asarray(layout.source, dtype=np.int64)
                ],
                dtype=wp.int32,
                device=device,
            ),
            "material_ids": wp.array(
                np.asarray(bodies.material_ids, dtype=np.int64), dtype=wp.int64, device=device
            ),
        }
        self._publish_state()
        wp.synchronize_device(device)
        self.frames_advanced = 0
        self._frame_started_at = None

    def _publish_state(self) -> None:
        solver = self._require_solver()
        wp = self.device_context.wp
        a = self._arrays
        wp.launch(
            self._publish,
            dim=self.config.count,
            inputs=[a["source"], solver.x, solver.v, solver.heat, solver.damage, solver.rho],
            outputs=[
                a["positions"],
                a["velocities"],
                a["internal_energies"],
                a["damage"],
                a["densities"],
            ],
            device=self.device_context.device,
        )

    def snapshot(self) -> dict[str, Any]:
        np = __import__("numpy")
        solver = self._require_solver()
        layout = self._layout
        assert layout is not None
        bodies = layout.bodies
        state = solver.state()
        checkpoint = solver.checkpoint()
        # Four arrays, and the bond rows the broken-bond mask is read from.
        self.explicit_snapshot_readbacks += 5
        source = np.asarray(layout.source, dtype=np.int64)
        positions = state["positions"][source]
        velocities = state["velocities"][source]
        heat = state["heat"][source]
        damage = state["damage"][source]
        records = []
        for index in range(self.config.count):
            record = {
                "index": index,
                "stable_id": int(bodies.stable_ids[index]),
                "active": bool(bodies.active[index]),
                "mass": float(bodies.masses[index]),
                "radius": float(bodies.radii[index]),
                "position": [float(value) for value in positions[index]],
                "velocity": [float(value) for value in velocities[index]],
                "internal_energy": float(heat[index]),
                "damage": float(damage[index]),
            }
            if bodies.material_ids is not None:
                record["material_id"] = int(bodies.material_ids[index])
            records.append(record)
        masks = np.asarray(checkpoint["broken"], dtype=np.uint64)
        plastic = checkpoint["plastic"]
        energy = np.asarray(checkpoint["energy"], dtype=np.float32)
        return {
            "schema": CHECKPOINT_SCHEMA_VERSION,
            "backend": BACKEND,
            "model": MODEL,
            "model_version": MODEL_VERSION,
            "capacity": self.config.count,
            "count": self.config.count,
            "collision_mode": "ignore",
            "step": self.frames_advanced,
            "simulation_time": self.frames_advanced * self.config.seconds_per_frame,
            # The engine's state beyond the particles: its frame clock and, for every particle
            # with a broken bond, one bit per reference bond. Particles are named by their
            # index among the live ones (the engine's own), and a particle with nothing broken
            # is left out, so an undamaged scene writes an empty list.
            #
            # ADR 0204: the reference length of every bond that has yielded is state too. One
            # ``[particle, slot, length]`` per such bond, named by its lower-indexed end, the
            # length the float32 value the device holds (a JSON double carries it exactly).
            #
            # ADR 0204, again: each particle's persistent thermal energy (what the yield softens
            # with), sparse like the masks: one ``[particle, energy]`` for each that is not zero.
            "engine": {
                "frame": int(checkpoint["frame"]),
                "broken": [[int(i), int(masks[i])] for i in np.flatnonzero(masks)],
                "plastic": [
                    [int(i), int(k), float(length)]
                    for i, k, length in zip(
                        plastic["particles"], plastic["slots"], plastic["lengths"], strict=True
                    )
                ],
                "energy": [[int(i), float(energy[i])] for i in np.flatnonzero(energy)],
                # ADR 0220: the largest pressure acceleration the last pressure solve left. It
                # sizes the first outer step after a restore, so it is state too.
                "pressure_accel": float(checkpoint["pressure_accel"]),
                "bond_count": _bond_count(solver),
            },
            # The neighbour lists are rebuilt from the positions every outer step, and gravity
            # from the positions the restored frame opens with.
            "recomputed_on_restore": ["neighbour_grid", "gravity"],
            "bodies": records,
        }

    def _validated_restore(self, snapshot: dict[str, Any]) -> tuple[Any, ...]:
        """Check a snapshot against this runtime and return what the engine restores from.

        Everything a snapshot can get wrong is refused here, before the first device write: a
        restore that stopped half way would leave positions from one frame beside bonds from
        another.
        """

        np = __import__("numpy")
        layout = self._layout
        assert layout is not None
        if snapshot.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint snapshot schema")
        if snapshot.get("backend") != BACKEND:
            raise ValueError("checkpoint backend does not match the spacesim_impact runtime")
        if snapshot.get("model") != MODEL:
            raise ValueError("checkpoint model does not match the spacesim_impact runtime")
        if snapshot.get("model_version") != MODEL_VERSION:
            raise ValueError(
                f"checkpoint model version {snapshot.get('model_version')!r} is not the "
                f"installed {MODEL_VERSION!r}"
            )
        records: list[dict[str, Any]] = list(snapshot.get("bodies") or ())
        if int(snapshot.get("count", -1)) != self.config.count or len(records) != self.config.count:
            raise ValueError("snapshot body count does not match runtime")
        if [int(record.get("index", -1)) for record in records] != list(range(self.config.count)):
            raise ValueError("snapshot body ordering is invalid")
        if [int(record.get("stable_id", -1)) for record in records] != [
            int(value) for value in layout.bodies.stable_ids
        ]:
            raise ValueError("snapshot stable IDs do not match runtime ordering")
        if [bool(record.get("active")) for record in records] != [
            bool(value) for value in layout.bodies.active
        ]:
            raise ValueError("snapshot idle slots do not match runtime layout")
        if layout.bodies.material_ids is not None and [
            record.get("material_id") for record in records
        ] != [int(value) for value in layout.bodies.material_ids]:
            raise ValueError("snapshot material ids do not match runtime layout")
        engine = snapshot.get("engine")
        if not isinstance(engine, dict):
            raise ValueError("snapshot carries no engine checkpoint")
        frame = engine.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0:
            raise ValueError("snapshot engine frame is invalid")
        if int(snapshot.get("step", -1)) != int(frame):
            raise ValueError("snapshot step does not match its engine frame")
        if engine.get("bond_count") != _bond_count(self._require_solver()):
            raise ValueError("snapshot bond count does not match the bonds this scene captured")
        live = layout.solver_slots
        broken = np.zeros(layout.active_count, dtype=np.uint64)
        entries = engine.get("broken")
        if not isinstance(entries, (list, tuple)):
            raise ValueError("snapshot broken-bond list is invalid")
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("snapshot broken-bond entry is invalid")
            index, mask = entry
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (index, mask)):
                raise ValueError("snapshot broken-bond entry is invalid")
            if not 0 <= int(index) < layout.active_count or int(index) in seen:
                raise ValueError("snapshot broken-bond index is out of range or repeated")
            if not 0 < int(mask) < 1 << 64:
                raise ValueError("snapshot broken-bond mask is out of range")
            reference_count = int(self._require_solver().reference_bond_counts[int(index)])
            if int(mask) >> reference_count:
                raise ValueError("snapshot breaks a bond the reference row does not have")
            seen.add(int(index))
            broken[int(index)] = np.uint64(int(mask))
        plastic_entries = engine.get("plastic")
        if not isinstance(plastic_entries, (list, tuple)):
            raise ValueError("snapshot plastic-bond list is invalid")
        plastic_particles: list[int] = []
        plastic_slots: list[int] = []
        plastic_lengths: list[float] = []
        named: set[tuple[int, int]] = set()
        reference_counts = self._require_solver().reference_bond_counts
        for entry in plastic_entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError("snapshot plastic-bond entry is invalid")
            index, slot, length = entry
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (index, slot)):
                raise ValueError("snapshot plastic-bond entry is invalid")
            if isinstance(length, bool) or not isinstance(length, (int, float)):
                raise ValueError("snapshot plastic-bond entry is invalid")
            if not 0 <= int(index) < layout.active_count:
                raise ValueError("snapshot plastic-bond index is out of range")
            if not 0 <= int(slot) < int(reference_counts[int(index)]):
                raise ValueError("snapshot plastic bond names a slot the reference row lacks")
            if (int(index), int(slot)) in named:
                raise ValueError("snapshot plastic bond is repeated")
            if not (math.isfinite(float(length)) and float(length) > 0.0):
                raise ValueError("snapshot plastic-bond length must be positive and finite")
            named.add((int(index), int(slot)))
            plastic_particles.append(int(index))
            plastic_slots.append(int(slot))
            plastic_lengths.append(float(length))
        energy_entries = engine.get("energy")
        if not isinstance(energy_entries, (list, tuple)):
            raise ValueError("snapshot thermal-energy list is invalid")
        thermal = np.zeros(layout.active_count, dtype=np.float64)
        warmed: set[int] = set()
        for entry in energy_entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("snapshot thermal-energy entry is invalid")
            index, value = entry
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError("snapshot thermal-energy entry is invalid")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("snapshot thermal-energy entry is invalid")
            if not 0 <= int(index) < layout.active_count or int(index) in warmed:
                raise ValueError("snapshot thermal-energy index is out of range or repeated")
            if not (math.isfinite(float(value)) and float(value) > 0.0):
                raise ValueError("snapshot thermal energy must be positive and finite")
            finite_float32(float(value), "snapshot thermal energy")
            warmed.add(int(index))
            thermal[int(index)] = float(value)
        pressure_accel = engine.get("pressure_accel")
        if isinstance(pressure_accel, bool) or not isinstance(pressure_accel, (int, float)):
            raise ValueError("snapshot pressure acceleration is invalid")
        if not (math.isfinite(float(pressure_accel)) and float(pressure_accel) >= 0.0):
            raise ValueError("snapshot pressure acceleration must be non-negative and finite")
        finite_float32(float(pressure_accel), "snapshot pressure acceleration")
        try:
            positions = np.asarray([records[i]["position"] for i in live], dtype=np.float64)
            velocities = np.asarray([records[i]["velocity"] for i in live], dtype=np.float64)
            heat = np.asarray([records[i]["internal_energy"] for i in live], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("snapshot body records are malformed") from exc
        for name, array, shape in (
            ("positions", positions, (layout.active_count, 3)),
            ("velocities", velocities, (layout.active_count, 3)),
            ("heat", heat, (layout.active_count,)),
        ):
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"snapshot {name} must have shape {shape} and finite values")
            finite_float32(array, f"snapshot {name}")
        plastic = {
            "particles": np.asarray(plastic_particles, dtype=np.int64),
            "slots": np.asarray(plastic_slots, dtype=np.int64),
            "lengths": np.asarray(plastic_lengths, dtype=np.float64),
        }
        engine_checkpoint = {
            "frame": int(frame),
            "broken": broken,
            "plastic": plastic,
            "energy": thermal,
            "pressure_accel": float(pressure_accel),
        }
        return positions, velocities, heat, engine_checkpoint

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        solver = self._require_solver()
        positions, velocities, heat, engine = self._validated_restore(snapshot)
        # The engine validates the shapes, finiteness and the bits against its reference rows
        # again before it writes, so a bad mask is refused on both sides of this call.
        solver.restore(positions, velocities, heat, engine)
        self._publish_state()
        self.device_context.wp.synchronize_device(self.device_context.device)
        self.frames_advanced = int(snapshot.get("step", 0))
        self._frame_started_at = None
        self.last_step_ms = 0.0

    def diagnostics(self) -> SpaceSimImpactDiagnostics:
        solver = self._require_solver()
        layout = self._layout
        assert layout is not None
        return SpaceSimImpactDiagnostics(
            active_count=layout.active_count,
            idle_count=self.config.count - layout.active_count,
            particle_spacing=float(layout.spacing),
            smoothing_length=float(solver.h),
            outer_steps=int(solver.outer_steps),
            inner_steps=int(solver.inner_steps),
            timestep=float(solver.dt_inner),
            outer_timestep=float(solver.dt_outer),
            bond_step_limit=float(solver.bond_step_limit),
            bond_count=_bond_count(solver),
            frames_advanced=self.frames_advanced,
            simulation_time=self.frames_advanced * self.config.seconds_per_frame,
            step_ms=self.last_step_ms,
            speed_clamps=int(solver.speed_clamps()),
            # Read live: the engine's own attribute is the count at load, and a neighbour list
            # that overflows later in the run is the one worth seeing.
            neighbor_overflow=solver.neighbor_overflow_total(),
            dropped_bonds=int(solver.dropped_bonds),
            submission=solver.submission,
            active_outer_steps=int(solver.active_outer_steps),
            max_outer_steps_used=int(solver.max_outer_steps_used),
            adaptive_frames=int(solver.adaptive_frames),
            budget_bound_steps=int(solver.budget_bound_steps),
        )

    def close(self) -> None:
        if self._solver is not None and self.device_context.state.name == "READY":
            self.device_context.wp.synchronize_device(self.device_context.device)
        # ADR 0218: the engine's gravity tree owns device buffers of its own to let go of.
        release = getattr(self._solver, "close", None)
        if release is not None:
            release()
        self._solver = None
        self._layout = None
        self._arrays = {}
