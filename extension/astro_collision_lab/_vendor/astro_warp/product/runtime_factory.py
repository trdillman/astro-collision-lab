"""Compiled-plan product runtime factory."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from .backend_registry import (
    BACKENDS,
    Assembly,
    BackendSpec,
    ColorFacts,
    Family,
    backend_spec,
    find_backend,
)
from .backend_registry import COMMON_RUNTIME_MODULES as REGISTRY_COMMON_RUNTIME_MODULES
from .bake import BakeRevision, BakeSession
from .cache import SCHEMA_V2 as BINARY_BAKE_SCHEMA
from .color_bake import ColorBakeFrame, carry_bake_frame
from .color_by import (
    GEOMETRY_BEARING_ROLES,
    RAW_COLOR_BAKE_CHANNELS,
    PartitionSources,
    availability_matrix,
    infer_d13_role,
)
from .color_carriage import ColorPartitionProvenance, ColorProvenance
from .compiler import GraphCompiler
from .display import (
    DEFAULT_DISPLAY_CAP,
    POSITIONS_ONLY,
    DisplayChannelSpec,
    DisplaySample,
    DisplayStage,
)

# Two modules export a `specs_from_bindings`, each resolving its own role's bindings against
# the same binding tuple. Aliased rather than renamed at the source: the unqualified name is
# right inside each module, and only this file, which reads both, has to tell them apart.
from .emitters import specs_from_bindings as emitter_specs_from_bindings
from .force_fields import specs_from_bindings
from .giant_impact_runtime import GiantImpactConfig, sample_giant_impact_bodies
from .impact import predict_encounter
from .invalidation import retroactive_state_fingerprint, solved_state_fingerprint
from .materials import MaterialRegistry, registry_from_manifest
from .memory_admission import admit_scene, allocation_failure_guard
from .model import (
    AUTO_SMALL_SCENE_DIRECT_LIMIT,
    DEFAULT_TREE_MAX_DEPTH,
    DEFAULT_TREE_STACK_CAPACITY,
    DEFAULT_TREE_THETA,
    SCHEMA_VERSION,
    STAGE_ORDER,
    BindingDescriptor,
    BindingRole,
    BodyRepresentation,
    RuntimePlan,
    SimulationDescriptor,
    StageFamily,
    TimelineMapping,
)
from .modules import (
    RETIRED_ENGINE_BACKENDS,
    RETIRED_ENGINE_MESSAGE,
    RETIRED_ENGINE_MODULES,
    module_version,
)
from .nbody import CollisionMode, DirectNBodyRuntime, NBodyConfig
from .plan_policy import SHEARING_BOX_MODULE, plan_policy_errors, shearing_box_from_steps
from .runtime_foundation import BackendRuntime, DeviceContext, FramePrimitiveProgress
from .sampling import BodySet, active_body_count, binding_body_count, body_count, expand_bindings
from .shearing_box import ShearingBoxSpec
from .spacesim_impact_runtime import (
    DEFAULT_FAULT_STRAIN,
    FAULT_CELL_PARTICLES,
    SpaceSimImpactConfig,
    sample_spacesim_bodies,
)
from .tree import BarnesHutConfig, BarnesHutRuntime, sparse_node_capacity

ALLOWED_RUNTIME_MODULES = {
    "astro.context.units",
    "astro.inputs.analytical_bodies",
    "astro.gravity.direct_all_pairs",
    "astro.gravity.barnes_hut_tree",
    "astro.gravity.auto_select",
    "astro.gravity.newtonian",
    "astro.gravity.tree_newtonian",
    "astro.gravity.auto_newtonian",
    "astro.force.external_field",
    "astro.frame.shearing_box",
    "astro.integrators.kdk_leapfrog",
    "astro.emitter.activate",
    "astro.collision.point",
    "astro.diagnostics.orbital_scalars",
    "astro.diagnostics.fragments",
    "astro.inputs.giant_impact_bodies",
    "astro.giant_impact.neighbor_grid",
    "astro.giant_impact.pbf_pressure",
    "astro.giant_impact.tiled_gravity",
    "astro.giant_impact.moon_guide",
    "astro.integrators.giant_impact_frame",
    "astro.diagnostics.giant_impact_scalars",
    "astro.inputs.spacesim_bodies",
    "astro.spacesim.neighbor_grid",
    "astro.spacesim.iisph_pressure",
    "astro.spacesim.bond_strength",
    "astro.spacesim.tiled_gravity",
    "astro.integrators.spacesim_two_rate",
    "astro.diagnostics.spacesim_scalars",
    "astro.outputs.immutable_bake",
}

CALIBRATION_SCHEMA = "astro-warp-auto-calibration-v4"
# What a re-bake of the same inputs on the same device can and cannot be expected to
# reproduce, per executed backend. Written into every bake manifest, because "deterministic"
# without the qualifier is the claim that gets believed and then quietly violated by a
# floating-point atomic. Each statement is a claim about the kernels in nbody.py / tree.py:
# whether the accumulation order that produces the value is fixed by the code or chosen by
# the device scheduler.
BACKEND_DETERMINISM = {
    "direct": {
        "state_advance": (
            "bit-reproducible on the same device, driver, and Warp version: every thread "
            "sums the whole body list in one fixed array order and no atomic accumulates "
            "into the acceleration, position, or velocity arrays"
        ),
        "diagnostics": (
            "reproducible to floating-point round-off only: the scalar reductions use "
            "atomic adds, whose completion order the device schedules"
        ),
        "collision_events": (
            "the physical outcome is order-independent; the slot each event lands in is "
            "handed out by an atomic counter, so the recorded event order may differ"
        ),
    },
    "tree": {
        "state_advance": (
            "reproducible to floating-point round-off only: node mass and moment "
            "aggregation uses atomic adds, so a node centre of mass, and the force read "
            "from it, depend on device scheduling order"
        ),
        "diagnostics": (
            "reproducible to floating-point round-off only: the scalar reductions use "
            "atomic adds, whose completion order the device schedules"
        ),
        "collision_events": (
            "the physical outcome is order-independent; the slot each event lands in is "
            "handed out by an atomic counter, so the recorded event order may differ"
        ),
    },
    "giant_impact": {
        "state_advance": (
            "reproducible to floating-point round-off only: the density sums, the pressure "
            "constraint's neighbour sums and the moon guide's debris reductions use atomic "
            "adds, so two runs agree to round-off and not bit for bit; the number of pressure "
            "substeps a frame spends is fixed when the scene loads, from the resolution and "
            "the authored impact speed, so it does not follow the state"
        ),
        "diagnostics": (
            "reproducible to floating-point round-off only: the scalars are reductions over "
            "the state above"
        ),
        "collision_events": (
            "no collision stage runs; bodies meet through the position-based pressure "
            "constraint that keeps each one a body"
        ),
        "fragment_diagnostics": (
            "the fragment partition is exact for a given linking length and particle "
            "state, because friends-of-friends connectivity is order-independent; the "
            "label each fragment is numbered with follows the sorted particle order, so "
            "it too is fixed"
        ),
    },
    "spacesim_impact": {
        "state_advance": (
            "expected to replay bit for bit on the same device, driver, and Warp version, "
            "and observed to in the backend's checkpoint-replay test: every sum is a "
            "fixed-order loop over a neighbour list built in the grid's sorted order, and "
            "the only atomics are counters (clamped speeds, list overflow) that never feed "
            "the state. Not claimed across devices, drivers, or Warp versions, and the "
            "number of outer and inner steps a frame spends is fixed when the scene loads, "
            "from the resolution, the authored impact speed and the bond network's measured "
            "stability limit, so it does not follow the state"
        ),
        "diagnostics": (
            "reproducible to floating-point round-off only: the scalars are reductions over "
            "the state above, and the broken-bond count reads the bond rows"
        ),
        "collision_events": (
            "no collision stage runs; bodies meet through the implicit pressure solve, "
            "and break along bonds whose length passes its break length"
        ),
        "fragment_diagnostics": (
            "the fragment partition is exact for a given linking length and particle "
            "state, because friends-of-friends connectivity is order-independent; the "
            "label each fragment is numbered with follows the sorted particle order, so "
            "it too is fixed"
        ),
    },
}
WARMED_REPETITIONS = 5
# The per-backend facts (pipeline, modules, classes, family) live in `backend_registry` (ADR
# 0208). These names are the views of it this module and its tests have always read; each is
# derived, so there is no second place to edit.
EXPECTED_PIPELINES = {name: spec.pipeline for name, spec in BACKENDS.items()}
# The backends that integrate a sampled particle cloud as one solid. None of them supports force
# fields, emitters or pinned bodies: those act on, release, or hold the analytic point masses
# only the direct and tree solvers have (ADR 0045, 0047, 0158).
_PARTICLE_BACKENDS = frozenset(
    name for name, spec in BACKENDS.items() if spec.family is Family.PARTICLE
)
# The optional external-field step (ADR 0045). It exists in a plan exactly when the
# descriptor binds FIELD-role members, so it lives outside EXPECTED_PIPELINES and is
# spliced in by _expected_pipeline at the index the (family, node_id) plan sort puts it:
# "force.external_field" sorts before "force.newtonian_gravity" (direct) and
# "force.tree_newtonian_gravity" (tree) but after "force.auto_newtonian_gravity" (auto).
_EXTERNAL_FIELD_MODULE = "astro.force.external_field"
_EXTERNAL_FIELD_STEP = (StageFamily.FORCE, _EXTERNAL_FIELD_MODULE)
# The optional emitter-activation step (ADR 0047), spliced the same way and for the same
# reason. Its position is not a hard-coded index but the position of the collision step:
# both live in EVENTS, and the (family, node_id) plan sort puts "events.emitter_activate"
# before "events.point_collisions". Anchoring on the step rather than on a number is what
# keeps the two optional splices independent -- the field step lands at a lower index on
# every orbital backend, so inserting it first would silently shift a hard-coded emitter
# index by one.
_EMITTER_MODULE = "astro.emitter.activate"
_EMITTER_STEP = (StageFamily.EVENTS, _EMITTER_MODULE)
_EMITTER_ANCHOR = (StageFamily.EVENTS, "astro.collision.point")
# The optional shearing-box step (ADR 0202). It exists in a plan exactly when the System runs in
# a local shearing box, and it is a Direct-solver option only: the tree builds one open octree
# and has no images to sum. Its FORCE node sorts directly after the gravity node
# ("force.newtonian_gravity" < "force.shearing_box"), so it is spliced at that anchor. It
# carries the box (omega, box_x, box_y, images) as parameters, so editing the box moves
# `graph_hash`; the runtime reads it from here and nowhere else.
_SHEARING_BOX_MODULE = SHEARING_BOX_MODULE
_SHEARING_BOX_STEP = (StageFamily.FORCE, _SHEARING_BOX_MODULE)
_SHEARING_BOX_ANCHOR = (StageFamily.FORCE, "astro.gravity.newtonian")


def _expected_pipeline(
    backend: str,
    *,
    external_field: bool = False,
    emitter: bool = False,
    shearing_box: bool = False,
) -> tuple[tuple[StageFamily, str], ...]:
    """The exact executable pipeline for a backend, with or without the optional steps.

    The one place the factory knows what a fielded or emitting plan looks like:
    _validate_plan and validation_summary both call this, so the creation-time check and
    the bake-time re-check cannot drift apart. The Solid backends refuse both: a force field
    does not act on a Solid's particles (ADR 0045), and emission activates slots in an
    ``active`` array that the point-mass solvers own and the Solid lanes do not carry --
    which is the same reason those pipelines have no EVENTS family at all.
    """

    spec = backend_spec(backend)
    pipeline = spec.pipeline
    if external_field:
        index = spec.optional_steps.get("external_field")
        if index is None:
            raise ProductRuntimeError(
                f"the {backend!r} backend does not support external force fields"
            )
        pipeline = pipeline[:index] + (_EXTERNAL_FIELD_STEP,) + pipeline[index:]
    if emitter:
        if "emitter" not in spec.optional_steps:
            raise ProductRuntimeError(f"the {backend!r} backend does not support emitters")
        index = pipeline.index(_EMITTER_ANCHOR)
        pipeline = pipeline[:index] + (_EMITTER_STEP,) + pipeline[index:]
    if shearing_box:
        if "shearing_box" not in spec.optional_steps:
            raise ProductRuntimeError(
                f"the shearing box runs on the Direct solver only, but this plan runs on "
                f"{backend!r}. Set the System's Solver to Direct (or Auto) and recompile"
            )
        index = pipeline.index(_SHEARING_BOX_ANCHOR) + 1
        pipeline = pipeline[:index] + (_SHEARING_BOX_STEP,) + pipeline[index:]
    return pipeline


# What every executable pipeline has in common is the context it reads and the output it
# writes; everything else is a backend's own (`BackendSpec.modules`).
COMMON_RUNTIME_MODULES = REGISTRY_COMMON_RUNTIME_MODULES
# Its own set per backend. The two Solid backends share only the fragment diagnostic, which is
# why the membership test in `_validate_plan` asks for modules in another backend's set *and
# not in this one's*: a plan naming the fragment diagnostic is not borrowing a stage.
BACKEND_RUNTIME_MODULES = {name: spec.modules for name, spec in BACKENDS.items()}


def serves_resumable_frames(candidate: Any) -> bool:
    """Whether ``candidate`` carries the resumable-frame protocol.

    Deliberately usable on a *class* as well as an instance, and that is the whole reason it
    exists as a function rather than staying inline in
    :attr:`ProductRuntime.supports_resumable_frame`. Asking the question of a class is asking
    it without a device, a compiled descriptor, or a scene, which is what lets an Add-menu
    template and a unit test ask it at all. The property below now delegates here, so the
    device-free answer and the one the scheduler acts on cannot drift apart -- if they were
    two copies of this predicate, the copy nobody runs on a device would be the one that
    quietly went stale.
    """

    return bool(
        getattr(candidate, "frame_transaction_resumable", False)
        and callable(getattr(candidate, "begin_frame", None))
        and (
            callable(getattr(candidate, "advance_frame_primitive", None))
            or callable(getattr(candidate, "advance_frame_substep", None))
        )
        and callable(getattr(candidate, "finish_frame", None))
    )


# `auto` is both orbital runtimes: it resolves to one of them at create time.
BACKEND_RUNTIME_CLASSES: dict[str, tuple[type, ...]] = {
    name: spec.runtime_classes for name, spec in BACKENDS.items()
}
"""Which runtime class :meth:`ProductRuntimeFactory.create` can construct, per backend.

Kept beside the dispatch it mirrors so the two are read together. A backend absent from here
is a backend `create` refuses, and :func:`backend_serves_resumable_frames` says so rather
than guessing.
"""


def backend_serves_resumable_frames(backend: str) -> bool:
    """Whether every runtime ``backend`` can resolve to serves the resumable-frame protocol.

    The device-free form of the question the scheduler asks on a live scene. A backend whose
    runtime cannot yield after one solver primitive publishes no substep bound, and a frame
    with no bound cannot be admitted -- so a scene compiled onto such a backend can be baked
    but can never play live, whatever its author intended.

    Every candidate must serve it, not merely one: `auto` picks its solver from the scene, and
    a capability that holds for only one of the two choices is not a capability the caller can
    rely on. An unknown backend is False, because `create` refuses it outright.

    Read from the registered spec's capability bound (ADR 0208), which the backend conformance
    suite holds equal to :func:`serves_resumable_frames` over every runtime class the spec names.
    """

    spec = find_backend(str(backend).lower())
    return spec is not None and spec.capabilities.resumable_upper_bound


class ProductRuntimeError(RuntimeError):
    """Raised when a compiled product runtime cannot be created or advanced."""


@dataclass
class Checkpoint:
    checkpoint_id: str
    step: int
    snapshot: dict[str, Any]
    snapshot_sha256: str


@dataclass(frozen=True)
class ColorSourceAuthority:
    """Immutable host facts for one binding's exact pre-upload source range."""

    binding_id: str
    source_start: int
    source_stop: int
    role: str
    backend: str
    has_committed_velocity: bool
    has_damage_source: bool
    damage_active: bool
    damage_model: str
    has_material_registry: bool
    has_persistent_material_id: bool
    has_specific_internal_energy: bool


#: Which source fact says a bake must carry each raw colour channel. The keys are
#: ``color_by.RAW_COLOR_BAKE_CHANNELS``; a test holds the two together.
_BAKE_CHANNEL_FACTS = {
    "damage": "has_damage_source",
    "internal_energies": "has_specific_internal_energy",
    "material_ids": "has_material_registry",
    "velocities": "has_committed_velocity",
}


@dataclass(frozen=True)
class _BakeColorPlan:
    """Frozen D13 source contract shared by every frame in one bake revision."""

    provenance: ColorProvenance
    allocated_count: int
    carry_damage: bool
    carry_material_ids: bool
    expected_channels: tuple[str, ...]


def _bake_color_plan(
    descriptor: SimulationDescriptor,
    *,
    backend: str,
    authority: Mapping[str, ColorSourceAuthority],
    channels: Mapping[str, Any],
) -> _BakeColorPlan | None:
    """Preflight immutable binding facts and raw device channels for one revision.

    An empty authority mapping is the explicit legacy/synthetic path. Once any authority
    exists, every compiled binding and source range must match exactly; no staged value is
    allowed to manufacture availability that was not frozen before upload.
    """

    if not authority:
        return None
    bindings = tuple(descriptor.bindings)
    binding_ids = tuple(binding.binding_id for binding in bindings)
    if len(set(binding_ids)) != len(binding_ids) or set(authority) != set(binding_ids):
        raise ProductRuntimeError(
            "COLOR_BY_BAKE_UNAVAILABLE: per-binding source authority is missing or mismatched"
        )
    backend = str(backend).lower()
    if not backend:
        raise ProductRuntimeError("COLOR_BY_BAKE_UNAVAILABLE: runtime has no backend")

    facts_by_id: dict[str, ColorSourceAuthority] = {}
    output_bindings: list[tuple[BindingDescriptor, str]] = []
    expected_start = 0
    for binding in bindings:
        facts = authority[binding.binding_id]
        if not isinstance(facts, ColorSourceAuthority):
            raise ProductRuntimeError(
                "COLOR_BY_BAKE_UNAVAILABLE: per-binding source authority has an invalid "
                f"record for {binding.binding_id!r}"
            )
        expected_stop = expected_start + binding_body_count(binding)
        expected_role = infer_d13_role(binding, backend=backend)
        boolean_fields = (
            facts.has_committed_velocity,
            facts.has_damage_source,
            facts.damage_active,
            facts.has_material_registry,
            facts.has_persistent_material_id,
            facts.has_specific_internal_energy,
        )
        if (
            facts.binding_id != binding.binding_id
            or type(facts.source_start) is not int
            or facts.source_start != expected_start
            or type(facts.source_stop) is not int
            or facts.source_stop != expected_stop
            or facts.backend != backend
            or facts.role != expected_role
            or any(type(value) is not bool for value in boolean_fields)
            or facts.damage_model not in {"ACTIVE", "NONE"}
            or facts.damage_active != (facts.damage_model == "ACTIVE")
            or facts.has_persistent_material_id and not facts.has_material_registry
        ):
            raise ProductRuntimeError(
                "COLOR_BY_BAKE_UNAVAILABLE: per-binding source authority range or identity "
                f"mismatch for {binding.binding_id!r}"
            )
        facts_by_id[binding.binding_id] = facts
        if expected_role in GEOMETRY_BEARING_ROLES:
            output_bindings.append((binding, expected_role))
        expected_start = expected_stop

    allocated_count = sum(binding_body_count(binding) for binding in bindings)
    if expected_start != allocated_count:
        raise ProductRuntimeError(
            "COLOR_BY_BAKE_UNAVAILABLE: source authority ranges do not cover the allocated "
            "binding population"
        )
    if not output_bindings:
        return None

    expected_channels = {
        name: any(
            getattr(facts_by_id[binding.binding_id], _BAKE_CHANNEL_FACTS[name])
            for binding, _role in output_bindings
        )
        for name in RAW_COLOR_BAKE_CHANNELS
    }
    mismatched = sorted(
        name
        for name, expected in expected_channels.items()
        if (channels.get(name) is not None) != expected
    )
    if mismatched:
        raise ProductRuntimeError(
            "COLOR_BY_BAKE_UNAVAILABLE: raw runtime sources disagree with immutable binding "
            "authority: " + ", ".join(mismatched)
        )

    partitions: list[ColorPartitionProvenance] = []
    for binding, role in output_bindings:
        facts = facts_by_id[binding.binding_id]
        sources = PartitionSources(
            name=binding.object_name or binding.binding_id,
            role=role,
            backend=backend,
            has_committed_velocity=facts.has_committed_velocity,
            damage_active=facts.damage_active,
            damage_model=facts.damage_model,
            has_material_registry=facts.has_material_registry,
            has_persistent_material_id=facts.has_persistent_material_id,
            has_specific_internal_energy=facts.has_specific_internal_energy,
        )
        matrix = availability_matrix(sources)
        partitions.append(
            ColorPartitionProvenance(
                name=sources.name,
                role=sources.role,
                backend=sources.backend,
                available=tuple(
                    mode.value for mode, result in matrix.items() if result.available
                ),
                refusals={
                    mode.value: result.refusal
                    for mode, result in matrix.items()
                    if not result.available
                },
            )
        )

    provenance = ColorProvenance(partitions=tuple(partitions))
    available_by_partition = [set(partition.available) for partition in provenance.partitions]
    carry_damage = any("DAMAGE" in available for available in available_by_partition)
    carry_material_ids = any("MATERIAL" in available for available in available_by_partition)
    for mode, carry in (("DAMAGE", carry_damage), ("MATERIAL", carry_material_ids)):
        if carry and not all(mode in available for available in available_by_partition):
            raise ProductRuntimeError(
                f"COLOR_BY_BAKE_UNAVAILABLE: heterogeneous {mode.lower()} availability cannot "
                "be represented by the homogeneous immutable bake schema"
            )
    return _BakeColorPlan(
        provenance=provenance,
        allocated_count=allocated_count,
        carry_damage=carry_damage,
        carry_material_ids=carry_material_ids,
        expected_channels=tuple(
            sorted(name for name, expected in expected_channels.items() if expected)
        ),
    )


def _color_source_authority(
    bindings: tuple[BindingDescriptor, ...],
    bodies: BodySet,
    *,
    backend: str,
    materials: MaterialRegistry | None,
) -> Mapping[str, ColorSourceAuthority]:
    """Freeze D13 source truth from the exact host arrays that will be uploaded.

    This runs before ``load_bodies``. It therefore never inspects a staged sample or reads
    device state back, and its ranges describe the same concatenated binding order as
    :func:`expand_bindings`.
    """

    if len({binding.binding_id for binding in bindings}) != len(bindings):
        raise ProductRuntimeError("Color source authority requires unique binding IDs")
    expected_count = sum(binding_body_count(binding) for binding in bindings)
    if len(bodies) != expected_count:
        raise ProductRuntimeError(
            "Color source authority body count does not match compiled binding ranges"
        )
    material_ids = bodies.material_ids
    if material_ids is not None and len(material_ids) != expected_count:
        raise ProductRuntimeError(
            "Color source authority material assignments do not match body count"
        )
    if (materials is None) != (material_ids is None):
        raise ProductRuntimeError(
            "Color source authority requires material registry and assignments together"
        )

    # An unregistered backend carries no engine facts (and infer_d13_role refuses it anyway).
    registered = find_backend(backend)
    color_facts = registered.color if registered is not None else ColorFacts()
    authority: dict[str, ColorSourceAuthority] = {}
    start = 0
    for binding in bindings:
        stop = start + binding_body_count(binding)
        role = infer_d13_role(binding, backend=backend)
        assigned = (
            ()
            if material_ids is None
            else tuple(int(value) for value in material_ids[start:stop])
        )
        persistent = bool(assigned) and all(value >= 0 for value in assigned)
        has_damage_source = False
        damage_active = False
        if persistent:
            assert materials is not None
            try:
                for value in assigned:
                    materials.get(value)
            except KeyError as exc:
                raise ProductRuntimeError(
                    f"Color source authority binding {binding.binding_id!r} names an "
                    f"unknown material: {exc}"
                ) from exc
        if color_facts.publishes_damage and role != "FORCE" and stop > start:
            # The bond model's damage is the fraction of a particle's bonds that have broken.
            # It is the only damage number any engine publishes (a material's own damage model
            # drives nothing since the compressible engine was deleted), and the runtime
            # publishes it for every body it holds, so the authority states it for every body
            # too and the bake's channel check agrees.
            has_damage_source = True
            damage_active = True
        authority[binding.binding_id] = ColorSourceAuthority(
            binding_id=binding.binding_id,
            source_start=start,
            source_stop=stop,
            role=role,
            backend=backend,
            has_committed_velocity=role != "FORCE" and stop > start,
            has_damage_source=has_damage_source,
            damage_active=damage_active,
            damage_model="ACTIVE" if damage_active else "NONE",
            has_material_registry=materials is not None,
            has_persistent_material_id=persistent,
            has_specific_internal_energy=color_facts.internal_energy and role != "FORCE",
        )
        start = stop
    return MappingProxyType(authority)


@dataclass
class ProductRuntime:
    descriptor: SimulationDescriptor
    backend: str
    device_context: DeviceContext
    runtime: BackendRuntime
    calibration: dict[str, Any] = field(default_factory=dict)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    frame_index: int = 0
    closed: bool = False
    display: DisplayStage | None = None
    measured_solver_primitive_ms: float | None = None
    # ADR 0225: the largest calibrated duration of each named work slice, when the backend
    # declares a slice sequence. ``measured_solver_primitive_ms`` is the largest of these.
    measured_solver_phase_ms: Mapping[str, float] | None = None
    # SR-09: why the last calibration produced no measurement (exception text), else None.
    calibration_failure: str | None = None
    frame_substeps_completed: int = 0
    frame_open: bool = False
    frame_tainted: bool = False
    partial_frame_elapsed_ms: float = 0.0
    # A parameter edit that arrived while a frame was open (ADR 0165): the descriptor already
    # carries it, the backend takes it at the next frame boundary. Last edit wins.
    pending_parameters: SimulationDescriptor | None = None
    color_source_authority: Mapping[str, ColorSourceAuthority] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def step_index(self) -> int:
        return self.frame_index

    @property
    def last_step_ms(self) -> float:
        """Cost of the inner runtime's last `step`, which the bounded scheduler divides by.

        Delegated like `step_index`: the scheduler (`_service_context`) times its work through
        the `ProductRuntime` interface, and every backend runtime records this on its own
        `last_step_ms`. Without this property the first timer-driven service raised
        ``'ProductRuntime' object has no attribute 'last_step_ms'`` and faulted the context,
        which is what left the timeline advancing with a frozen simulation.
        """

        return float(getattr(self.runtime, "last_step_ms", 0.0))

    @property
    def frame_in_progress(self) -> bool:
        return self.frame_open

    @property
    def supports_resumable_frame(self) -> bool:
        """Whether this backend can yield after one synchronized solver primitive."""

        return serves_resumable_frames(self.runtime)

    @property
    def solver_work_sliced(self) -> bool:
        """Whether one authored substep contains multiple declared backend slices."""

        phases = getattr(self.runtime, "frame_primitive_calibration_phases", None)
        return (
            self.supports_resumable_frame
            and type(phases) is tuple
            and len(phases) > 1
        )

    @property
    def solver_primitive_adaptive(self) -> bool:
        """Whether one authored substep is an adaptive-length run of identical primitives.

        The complement of `solver_work_sliced` rather than a variant of it (ADR 0109). A
        work-sliced backend declares the ordered, unlike slices its substep is made of, so
        a bound covers a named sequence; an adaptive backend --- one sub-cycling a Courant
        step --- repeats one identical synchronized step as many times as the signal
        velocity asks for, so a bound covers one step and nothing bounds the count. No
        backend shipped today is adaptive (the one that was, the detailed impact engine,
        was deleted), and the property stays as the contract a future one would meet. Read
        off the backend because only the backend knows which shape its loop has, and gated
        on resumability because a backend that never yields has no primitive to describe.
        """

        return self.supports_resumable_frame and bool(
            getattr(self.runtime, "frame_primitive_sequence_adaptive", False)
        )

    @property
    def frame_substeps_total(self) -> int:
        """How many authored substeps must complete before `resume_frame` closes a frame.

        An adaptive backend answers one, and the reason is that ``substeps`` means something
        different to it (ADR 0109). For the gravity backends the field is a count of solver
        calls a frame is divided into, and each one is a thing `advance_frame_substep`
        performs. For a Courant sub-cycling backend it is a *resolution cap*, bounding how
        coarsely one Courant step may cross a frame, and the sub-cycle then runs as many steps
        as the signal velocity asks for and stops at the frame boundary. There is one
        authored substep in that frame, so counting to ``config.substeps`` would wait for
        completions that no solver call produces: the frame would close on the right instant
        anyway, but only after reporting substeps that did no work.
        """

        if self.solver_primitive_adaptive:
            return 1
        config = getattr(self.runtime, "config", None)
        return int(getattr(config, "substeps", self.descriptor.timeline.substeps))

    def _clear_partial_frame(self) -> None:
        self.frame_substeps_completed = 0
        self.frame_open = False
        self.frame_tainted = False
        self.partial_frame_elapsed_ms = 0.0

    def resume_frame(self) -> bool:
        """Run one synchronized primitive; return true only when its frame completed.

        The backend owns device launches and diagnostics, while this wrapper owns the
        partial-frame coordinate. A primitive-capable backend reports when its current
        authored substep is complete; the legacy substep surface is adapted as one complete
        substep. ``frame_index`` moves only after ``finish_frame`` succeeds, so no
        checkpoint/cache caller can observe half an authored frame as a completed one.
        """

        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        if not self.supports_resumable_frame:
            raise ProductRuntimeError(
                f"the {self.backend} backend does not support resumable authored frames"
            )
        if self.frame_tainted:
            raise ProductRuntimeError(
                "authored frame is tainted by a failed solver primitive; reset or restore it"
            )
        if not self.frame_open:
            # A parameter edit made mid-frame lands here, between two frames, so one frame is
            # never integrated with two different substep counts (ADR 0165).
            self._apply_pending_parameters()
            # Open before any backend call: begin/advance can enqueue or execute device work
            # before a synchronized diagnostic raises, so completed-substep count alone cannot
            # distinguish pristine state from an unpublishable partial frame.
            self.frame_open = True
            try:
                self.runtime.begin_frame()  # type: ignore[attr-defined]
            except Exception:
                self.frame_tainted = True
                raise
        try:
            advance_primitive = getattr(self.runtime, "advance_frame_primitive", None)
            if callable(advance_primitive):
                progress = advance_primitive()
                if not isinstance(progress, FramePrimitiveProgress):
                    raise ProductRuntimeError(
                        "backend work slice returned invalid primitive progress"
                    )
            else:
                self.runtime.advance_frame_substep()  # type: ignore[attr-defined]
                progress = FramePrimitiveProgress(
                    phase="solver_substep",
                    authored_substep_complete=True,
                )
        except Exception:
            self.frame_tainted = True
            raise
        # The backend's resumable primitive timer is intentionally distinct from
        # admission's conservative bound.  Accumulate the actual synchronized primitive
        # durations so the authored-frame diagnostic remains truthful even though no
        # backend ``step()`` call wrapped the complete frame.
        self.partial_frame_elapsed_ms += float(
            getattr(self.runtime, "last_solver_primitive_ms", 0.0)
        )
        if not progress.authored_substep_complete:
            return False
        self.frame_substeps_completed += 1
        if self.frame_substeps_completed < self.frame_substeps_total:
            return False
        try:
            self.runtime.finish_frame()  # type: ignore[attr-defined]
        except Exception:
            self.frame_tainted = True
            raise
        completed_frame_ms = self.partial_frame_elapsed_ms
        self._clear_partial_frame()
        self.runtime.last_step_ms = completed_frame_ms
        self.frame_index += 1
        return True

    def step(self, steps: int = 1) -> None:
        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        if steps < 1:
            raise ValueError("steps must be positive")
        if not self.supports_resumable_frame:
            if self.frame_in_progress:
                raise ProductRuntimeError(
                    "cannot use whole-frame stepping while a partial authored frame exists"
                )
            self.runtime.step(steps)
            self.frame_index += steps
            return
        for _ in range(steps):
            while not self.resume_frame():
                pass

    def ensure_display(
        self,
        cap: int = DEFAULT_DISPLAY_CAP,
        channel_spec: DisplayChannelSpec = POSITIONS_ONLY,
    ) -> DisplayStage:
        """The display stage for this runtime, rebuilt only when its shape changed.

        Reallocating per refresh would throw away the pinned staging whose reuse is the
        point of `adr/0008-display-interop-boundary.md`, so the cap is compared first and
        the buffers survive every refresh that does not move it. The channel spec is
        compared for the same reason and by the same rule: it decides which staging exists,
        so a look that asks for a channel already staged rebuilds nothing.
        """

        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        _, body_count, precision = self.runtime.display_source()
        stage = self.display
        if (
            stage is None
            or stage.closed
            or stage.cap != int(cap)
            or stage.body_count != int(body_count)
            or stage.precision != precision
            or stage.channel_spec != channel_spec
        ):
            if stage is not None:
                stage.close()
            stage = DisplayStage(
                self.device_context,
                body_count=body_count,
                precision=precision,
                cap=cap,
                channel_spec=channel_spec,
            )
            self.display = stage
        return stage

    def refresh_display(
        self,
        cap: int = DEFAULT_DISPLAY_CAP,
        channel_spec: DisplayChannelSpec = POSITIONS_ONLY,
        identity_ids: Any = None,
        ramp_bounds: tuple[float, float] | None = None,
        pinned_ids: Any = None,
    ) -> DisplaySample:
        """Stage a bounded display sample for the viewport (PRODUCT-009, ARCH-006).

        ``identity_ids`` asks for the retained identity sequence rather than the stride
        (ADR 0093); the scrub cache passes it above the presentation cap. ``pinned_ids`` are
        kept in that sequence whenever live (ADR 0239).

        ``display_channels()`` rather than ``display_source()`` for the arrays: the source
        accessor answers with positions only, and the optional channels read device handles
        the read accessors would route through a validated readback. Which of those handles
        the stage actually touches is the channel spec's business, so a runtime offering
        more than the look asks for costs nothing.

        A look asking for more than the lane offers --- a density ramp on a gravity-only
        runtime --- is trimmed rather than refused. The caller learns what it got from the
        sample's own channels; a raise here would take the viewport down over a look the
        control plane let a user pick.
        """

        channels = self.runtime.display_channels()
        stage = self.ensure_display(cap, channel_spec.supported_by(channels))
        return stage.refresh(
            channels,
            step_index=self.frame_index,
            identity_ids=identity_ids,
            ramp_bounds=ramp_bounds,
            pinned_ids=pinned_ids,
        )

    def display_report(self) -> dict[str, Any]:
        if self.display is None:
            return {"staged": False}
        return {"staged": True, **self.display.report()}

    def checkpoint(self, name: str = "") -> Checkpoint:
        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        if self.frame_in_progress:
            raise ProductRuntimeError("cannot checkpoint a partial authored frame")
        snapshot = {**self.runtime.snapshot(), "product_frame": self.frame_index}
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        checkpoint_id = name or f"checkpoint-{uuid.uuid5(uuid.NAMESPACE_URL, digest)}"
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            step=self.frame_index,
            snapshot=snapshot,
            snapshot_sha256=digest,
        )
        self.checkpoints[checkpoint_id] = checkpoint
        return checkpoint

    def restore(self, checkpoint_id: str) -> Checkpoint:
        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        checkpoint = self.checkpoints[checkpoint_id]
        try:
            self.runtime.restore_snapshot(checkpoint.snapshot)
        except Exception:
            # A backend restore may have copied only part of its authoritative arrays.
            # Keep the old coordinate unavailable and require another successful state
            # replacement instead of allowing simulation to resume from mixed state.
            self.frame_open = True
            self.frame_tainted = True
            raise
        self.frame_index = checkpoint.step
        self._clear_partial_frame()
        return checkpoint

    def measure_solver_primitive_cost_ms(
        self, *, warmup: int = 2, samples: int = 5
    ) -> float | None:
        """Calibrate the actual synchronized solver primitive used during playback.

        The scrub-cache admission (``scheduler.admit_frame_transaction``) needs a worst-case
        cost for one uninterruptible solver primitive before it lets Play cache a frame.
        Its device-independent floor (``scheduler.solver_call_bound``) is deliberately the
        weakest-supported-card bound, so on a real GPU it mis-estimates in both directions: it
        refuses a direct/auto frame that genuinely fits the slice and admits a tree frame that
        does not. Resumable Direct/Tree runtimes measure one separately synchronized captured
        substep. A bounded non-resumable orbital fallback may measure its whole-frame
        ``step(1)``. Emitter-bearing runtimes decline because early frames do not bound later
        release/finalizer work. Declining is no longer the same as being unplayable: since ADR
        0109 an adaptive backend yields between Courant steps, so its admission is bounded per
        step by the device-independent floor and reported as a notice rather than measured
        here (ADR 0090's cost notice).

        Best-effort and self-restoring: a primitive-progress backend declares the exact
        phase sequence for one authored substep. Calibration prewarms and then measures
        complete sequences, rejecting a missing, repeated, out-of-order, non-positive, or
        non-finite slice. Legacy backends retain their existing sample behavior. The
        ``initial`` checkpoint is restored in ``finally`` so the artist still starts at
        frame 0, and the largest separately synchronized observation is retained. An
        ordinary measurement failure leaves ``measured_solver_primitive_ms`` unset and
        returns ``None``. A rollback failure is different: device state may already be
        advanced or tainted, so the unsafe runtime is closed and ``ProductRuntimeError``
        is raised.
        """

        self.measured_solver_primitive_ms = None
        self.measured_solver_phase_ms = None
        self.calibration_failure = None
        if (
            self.closed
            or "initial" not in self.checkpoints
            or bool(getattr(getattr(self.runtime, "config", None), "emitters", ()))
        ):
            return None
        uses_primitive_progress = callable(
            getattr(self.runtime, "advance_frame_primitive", None)
        )
        if (
            uses_primitive_progress
            and getattr(self.runtime, "frame_primitive_admission_ready", None) is not True
        ):
            # Coarse diagnostic phases are executable before their minimum work
            # quanta have a complete structural/calibration proof. Presence of the
            # primitive method alone must therefore never manufacture admission.
            self.measured_solver_primitive_ms = None
            return None
        expected_phases: tuple[str, ...] | None = None
        if uses_primitive_progress:
            candidate = getattr(self.runtime, "frame_primitive_calibration_phases", None)
            if (
                type(candidate) is not tuple
                or not candidate
                or any(
                    not isinstance(phase, str)
                    or not phase.strip()
                    or phase != phase.strip()
                    for phase in candidate
                )
            ):
                self.measured_solver_primitive_ms = None
                return None
            expected_phases = candidate

        timings: list[float] = []
        phase_timings: dict[str, list[float]] = {}
        measurement_failed = False
        initial_coordinate: tuple[Any, ...] | None = None

        def calibration_coordinate() -> tuple[Any, ...]:
            backend_coordinate = tuple(
                (name, getattr(self.runtime, name))
                for name in (
                    "step_index",
                    "frame_primitive_cursor",
                    "phase_cursor",
                    "primitive_index",
                    "last_frame_primitive_phase",
                )
                if hasattr(self.runtime, name)
            )
            return (
                self.frame_index,
                self.frame_substeps_completed,
                self.frame_open,
                self.frame_tainted,
                self.partial_frame_elapsed_ms,
                backend_coordinate,
            )

        def run_complete_primitive_substep(*, record_timings: bool) -> None:
            assert expected_phases is not None
            backend_step_before = getattr(self.runtime, "step_index", None)
            if type(backend_step_before) is not int:
                raise ProductRuntimeError(
                    "backend work-slice calibration requires an integer step index"
                )
            frame_before = self.frame_index
            completed_before = self.frame_substeps_completed
            for index, expected_phase in enumerate(expected_phases):
                final_phase = index == len(expected_phases) - 1
                frame_complete = self.resume_frame()
                observed_phase = getattr(
                    self.runtime, "last_frame_primitive_phase", None
                )
                if observed_phase != expected_phase:
                    raise ProductRuntimeError(
                        "backend work-slice calibration phase order is incomplete or invalid"
                    )
                observed_ms = getattr(self.runtime, "last_solver_primitive_ms", None)
                if (
                    isinstance(observed_ms, bool)
                    or not isinstance(observed_ms, (int, float))
                    or not math.isfinite(float(observed_ms))
                    or float(observed_ms) <= 0.0
                ):
                    raise ProductRuntimeError(
                        "backend work-slice calibration published an invalid duration"
                    )
                backend_step_after = getattr(self.runtime, "step_index", None)
                if (
                    type(backend_step_after) is not int
                    or backend_step_after
                    != backend_step_before + int(final_phase)
                ):
                    raise ProductRuntimeError(
                        "backend work-slice calibration completed a substep at the wrong phase"
                    )
                if not final_phase:
                    valid_progress = (
                        frame_complete is False
                        and self.frame_index == frame_before
                        and self.frame_substeps_completed == completed_before
                        and self.frame_open
                        and not self.frame_tainted
                    )
                elif completed_before + 1 < self.frame_substeps_total:
                    valid_progress = (
                        frame_complete is False
                        and self.frame_index == frame_before
                        and self.frame_substeps_completed == completed_before + 1
                        and self.frame_open
                        and not self.frame_tainted
                    )
                else:
                    valid_progress = (
                        frame_complete is True
                        and self.frame_index == frame_before + 1
                        and self.frame_substeps_completed == 0
                        and not self.frame_open
                        and not self.frame_tainted
                    )
                if not valid_progress:
                    raise ProductRuntimeError(
                        "backend work-slice calibration published invalid substep progress"
                    )
                if record_timings:
                    timings.append(float(observed_ms))
                    phase_timings.setdefault(expected_phase, []).append(float(observed_ms))

        rollback_failure: BaseException | None = None
        try:
            self.restore("initial")
            initial_coordinate = calibration_coordinate()
            if uses_primitive_progress:
                warmup_count = max(0, int(warmup))
                for _ in range(warmup_count):
                    run_complete_primitive_substep(record_timings=False)
                if warmup_count:
                    self.restore("initial")
                    if calibration_coordinate() != initial_coordinate:
                        raise ProductRuntimeError(
                            "solver calibration warmup rollback did not restore progress"
                        )
            else:
                for _ in range(max(0, int(warmup))):
                    if self.supports_resumable_frame:
                        self.resume_frame()
                    else:
                        self.step(1)
            sample_count = max(1, int(samples))
            if self.supports_resumable_frame and not uses_primitive_progress:
                # A contiguous full frame visits every substep position, including the
                # first primitive that also drains ``begin_frame``'s diagnostic clear.
                # Sampling fewer and dividing a frame total would miss that real tail.
                sample_count = max(sample_count, self.frame_substeps_total)
            if uses_primitive_progress and getattr(
                self.runtime, "frame_primitive_substeps_differ", False
            ):
                # ADR 0144: only a frame's closing substep builds and walks the tree, so a
                # bound sampled from fewer substeps than a frame would never see it.
                sample_count = max(sample_count, self.frame_substeps_total)
            if uses_primitive_progress:
                for _ in range(sample_count):
                    run_complete_primitive_substep(record_timings=True)
            else:
                for _ in range(sample_count):
                    if self.supports_resumable_frame:
                        self.resume_frame()
                        timings.append(
                            float(getattr(self.runtime, "last_solver_primitive_ms", 0.0))
                        )
                    else:
                        self.step(1)
                        timings.append(float(self.last_step_ms))
        except Exception as exc:  # noqa: BLE001 - a successful rollback keeps this best-effort
            measurement_failed = True
            # SR-09: still best-effort, but the cause is kept for the caller to surface.
            self.calibration_failure = f"{type(exc).__name__}: {exc}"
        finally:
            # A failed primitive may leave partial metadata set; a successful restore
            # replaces the device state and clears it at the same boundary.
            try:
                self.restore("initial")
            except Exception as exc:
                rollback_failure = exc
            if (
                rollback_failure is None
                and initial_coordinate is not None
                and calibration_coordinate() != initial_coordinate
            ):
                rollback_failure = ProductRuntimeError(
                    "solver calibration rollback did not restore progress"
                )
            if rollback_failure is not None:
                try:
                    self.close()
                except BaseException as close_error:
                    # close() marks the wrapper terminal before fallible backend teardown.
                    rollback_failure.add_note(
                        f"terminal close also failed: {type(close_error).__name__}: {close_error}"
                    )
                raise ProductRuntimeError(
                    "solver calibration rollback failed; runtime state is unsafe and closed"
                ) from rollback_failure
        if measurement_failed:
            self.measured_solver_primitive_ms = None
            return None
        if not timings:
            self.measured_solver_primitive_ms = None
            return None
        measured = max(timings)
        if not (measured > 0.0) or not math.isfinite(measured):
            self.measured_solver_primitive_ms = None
            return None
        self.measured_solver_primitive_ms = float(measured)
        if expected_phases is not None and all(
            phase_timings.get(phase) for phase in expected_phases
        ):
            # Every declared slice was observed, each validated above, so each phase's worst
            # is a measurement of that slice's real graph (ADR 0225).
            self.measured_solver_phase_ms = MappingProxyType(
                {phase: max(phase_timings[phase]) for phase in expected_phases}
            )
        return self.measured_solver_primitive_ms

    def next_solver_primitive_measured_ms(self) -> float | None:
        """The calibrated worst of the work slice the next ``resume_frame`` will run (ADR 0225).

        ``None`` when the backend declares fewer than two slices (its one slice *is* the
        admitted figure), calibration did not observe every slice, the declared sequence is no
        longer the calibrated one, or the backend's cursor does not name a slice: the caller
        then plans on the frame's admitted bound, the largest slice, the conservative direction.
        """

        phases_ms = self.measured_solver_phase_ms
        if phases_ms is None or self.closed:
            return None
        phases = getattr(self.runtime, "frame_primitive_calibration_phases", None)
        cursor = getattr(self.runtime, "frame_primitive_cursor", None)
        if type(phases) is not tuple or type(cursor) is not int:
            return None
        if len(phases) < 2 or tuple(phases_ms) != phases or not 0 <= cursor < len(phases):
            return None
        return phases_ms.get(phases[cursor])

    def scrub_to_step(self, step: int) -> Checkpoint:
        if step < 0:
            raise ValueError("step must be non-negative")
        candidates = [cp for cp in self.checkpoints.values() if cp.step <= step]
        if not candidates:
            raise ProductRuntimeError("no checkpoint at or before requested step")
        checkpoint = max(candidates, key=lambda cp: cp.step)
        self.restore(checkpoint.checkpoint_id)
        replay = step - checkpoint.step
        if replay:
            self.step(replay)
        return checkpoint

    def retime(self, timeline: TimelineMapping) -> None:
        """Adopt a timeline edit in place, keeping solved state.

        Playback reads the mode and the frame range off this descriptor, so a live runtime
        has to take the edit or it stops matching the item that owns it. The fingerprint
        guard is what makes keeping the state safe: it refuses a timeline that also moves a
        solver input, because ``seconds_per_frame`` and ``substeps`` set the integration
        step and every already-solved step was advanced with the old one.
        """

        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        if self.frame_in_progress:
            raise ProductRuntimeError(
                "cannot retime a partial authored frame; restore or finish it first"
            )
        timeline.validate()
        candidate = replace(self.descriptor, timeline=timeline)
        if solved_state_fingerprint(candidate) != solved_state_fingerprint(self.descriptor):
            raise ProductRuntimeError(
                "timeline change moves a solver input; the runtime must be rebuilt"
            )
        self.descriptor = candidate

    def reparameterize(self, descriptor: SimulationDescriptor) -> int:
        """Adopt a prospective solver-parameter edit in place, keeping solved state.

        The counterpart to :meth:`retime` for inputs that do reach the solver but only
        parameterize the steps still to come. The guard is the retroactive fingerprint:
        everything a solved step actually read must be unchanged, so what moved can only
        reach the next step. Returns that step, which is what PRODUCT-007 records as the
        edit's earliest affected time.

        Raises ``ProductRuntimeError`` when the backend cannot take the edit in place, and
        the caller falls back to rebuilding. A backend that submits recorded device graphs
        replays the scalars captured when the graph was recorded, so its parameters are
        fixed at record time and rebinding a config would silently do nothing — which is
        why adoption is asked for rather than assumed.
        """

        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        if retroactive_state_fingerprint(descriptor) != retroactive_state_fingerprint(
            self.descriptor
        ):
            raise ProductRuntimeError(
                "edit moves an input a solved step already read; the runtime must be rebuilt"
            )
        if self.frame_in_progress and self.frame_tainted:
            raise ProductRuntimeError(
                "cannot reparameterize a tainted partial authored frame; "
                "restore or finish it first"
            )
        if getattr(self.runtime, "apply_parameters", None) is None:
            raise ProductRuntimeError(
                f"the {self.backend} backend cannot adopt solver parameters in place"
            )
        if self.frame_in_progress:
            # ADR 0165: hold the edit to the frame boundary. Validate now so a bad value is
            # refused here, to the caller, and not at the boundary inside the play timer.
            try:
                descriptor.validate()
            except ValueError as error:
                raise ProductRuntimeError(f"parameter edit refused: {error}") from error
            self.pending_parameters = descriptor
            self.descriptor = descriptor
            return self.step_index + 1
        self._apply_parameters_to_backend(descriptor)
        self.pending_parameters = None
        self.descriptor = descriptor
        return self.step_index + 1

    def _apply_parameters_to_backend(self, descriptor: SimulationDescriptor) -> None:
        kwargs: dict[str, Any] = {
            "gravitational_constant": descriptor.units.gravitational_constant,
            "softening": descriptor.units.softening,
            "dt": descriptor.timeline.seconds_per_frame,
            "substeps": descriptor.timeline.substeps,
        }
        # Force strengths only travel when the scene has Force members, so a backend that
        # predates ADR 0165 keeps the four-argument call it was written against.
        fields = specs_from_bindings(descriptor.bindings)
        if fields:
            kwargs["field_strengths"] = tuple(spec.strength for spec in fields)
        self.runtime.apply_parameters(**kwargs)  # type: ignore[attr-defined]

    def _apply_pending_parameters(self) -> None:
        pending = self.pending_parameters
        if pending is None:
            return
        # SR-08: clear only after the backend accepted it. On failure the edit stays pending
        # (still visibly unapplied) and the error reaches the caller.
        self._apply_parameters_to_backend(pending)
        if self.pending_parameters is pending:
            self.pending_parameters = None

    def environment(self, host: dict[str, Any] | None = None) -> dict[str, Any]:
        """What produced the numbers, as far as this layer can see it.

        Everything here can move the result: the Warp build compiles the kernels, the
        device runs them, and the required-device policy is what makes a CPU fallback a
        visible failure rather than a quietly different bake. ``host`` carries the facts
        this layer cannot reach — Blender's version, above all — because ``src`` must not
        import ``bpy``.
        """

        descriptor = self.device_context.descriptor
        environment = {
            "warp_version": self.device_context.warp_version,
            "device_name": self.device_context.device_name,
            "required_device": descriptor.required_device,
            "required_warp_version": descriptor.required_warp_version,
            "cuda_required": bool(descriptor.cuda_required),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
        if host:
            environment["host"] = dict(host)
        return environment

    def validation_summary(self) -> dict[str, Any]:
        """Re-run the input gates at bake time and record what they said.

        The runtime already passed these at creation, but a manifest that only repeats a
        creation-time claim proves nothing about the state being baked. Running them again
        costs microseconds and makes the summary a result rather than an assertion.
        """

        plan = _require_plan(self.descriptor)
        checks: dict[str, Any] = {}
        try:
            self.descriptor.validate()
            checks["descriptor_validate"] = "pass"
        except ValueError as exc:
            checks["descriptor_validate"] = f"fail: {exc}"
        try:
            ProductRuntimeFactory._validate_plan(self.descriptor, plan)
            checks["plan_validate"] = "pass"
        except ProductRuntimeError as exc:
            checks["plan_validate"] = f"fail: {exc}"
        modules = [step.module for step in plan.steps]
        checks["modules_allowed"] = sorted(set(modules) - ALLOWED_RUNTIME_MODULES) == []
        has_field_bindings = any(
            binding.role == BindingRole.FIELD for binding in self.descriptor.bindings
        )
        has_emitter_bindings = any(
            binding.role == BindingRole.EMITTER and binding.active
            for binding in self.descriptor.bindings
        )
        try:
            expected_pipeline = _expected_pipeline(
                plan.backend,
                external_field=has_field_bindings,
                emitter=has_emitter_bindings,
                shearing_box=_SHEARING_BOX_MODULE in modules,
            )
        except ProductRuntimeError:
            # Fields or emitters on a fluid backend: there is no valid pipeline to match.
            checks["pipeline_matches_backend"] = False
        else:
            checks["pipeline_matches_backend"] = (
                tuple((step.family, step.module) for step in plan.steps)
                == expected_pipeline
            )
        planned_versions = {step.module: step.module_version for step in plan.steps}
        checks["module_versions_installed"] = all(
            version == module_version(module)
            for module, version in planned_versions.items()
            if module in ALLOWED_RUNTIME_MODULES
        )
        return {
            "schema_version": self.descriptor.schema_version,
            "plan_id": plan.plan_id,
            "graph_hash": plan.graph_hash,
            "plan_backend": plan.backend,
            "executed_backend": self.backend,
            "modules": modules,
            "module_versions": planned_versions,
            "checks": checks,
            "passed": all(
                value is True or value == "pass" for value in checks.values()
            ),
        }

    def diagnostics_record(self) -> dict[str, Any]:
        """Diagnostics as plain JSON data, whichever backend produced them."""

        diagnostics = self.diagnostics()
        if is_dataclass(diagnostics) and not isinstance(diagnostics, type):
            return asdict(diagnostics)
        return dict(vars(diagnostics))

    def fragment_record(self) -> dict[str, Any] | None:
        """What the bodies are currently broken into, or ``None`` where that has no meaning.

        Asked of the runtime rather than of `self.backend`, because the question is whether
        the runtime holds a particle cloud with a particle spacing to set a linking length
        from - which the two Solid engines do. A gravitational backend of point masses has no
        connectivity to report and says so by not carrying the method.
        """

        report = getattr(self.runtime, "fragment_report", None)
        if report is None:
            return None
        record: dict[str, Any] = report()
        return record

    def bake(
        self,
        *,
        directory: Path,
        name: str,
        sample_count: int,
        steps_per_sample: int = 1,
        host: dict[str, Any] | None = None,
    ) -> tuple[BakeRevision, Path]:
        """Bake ``sample_count`` frames to disk in one blocking call.

        A thin driver over ``begin_bake()``/``IncrementalBake`` — draining every tick
        before finalizing — so the blocking path and the one a modal operator drives one
        tick at a time can never drift apart; they are the same code.
        """

        incremental = self.begin_bake(
            directory=directory,
            name=name,
            sample_count=sample_count,
            steps_per_sample=steps_per_sample,
            host=host,
        )
        while incremental.produce_next():
            pass
        return incremental.finalize()

    def _raw_bake_color_channels(self) -> Mapping[str, Any]:
        display_channels = getattr(self.runtime, "display_channels", None)
        if not callable(display_channels):
            raise ProductRuntimeError(
                "COLOR_BY_BAKE_UNAVAILABLE: runtime exposes no authoritative raw channels"
            )
        channels = display_channels()
        if not isinstance(channels, Mapping):
            raise ProductRuntimeError(
                "COLOR_BY_BAKE_UNAVAILABLE: runtime raw channels are not a mapping"
            )
        # Exactly the channels Color By reads (`color_by.RAW_COLOR_BAKE_CHANNELS`): the bake path
        # carries those and no others, and a channel the runtime does not serve reads as None.
        return {name: channels.get(name) for name in RAW_COLOR_BAKE_CHANNELS}

    @staticmethod
    def _bake_channel_values(
        channels: Mapping[str, Any], *, name: str, count: int
    ) -> tuple[Any, ...]:
        source = channels.get(name)
        if source is None:
            raise ProductRuntimeError(
                f"COLOR_BY_BAKE_UNAVAILABLE: required raw channel {name!r} is missing"
            )
        to_numpy = getattr(source, "numpy", None)
        try:
            values = to_numpy() if callable(to_numpy) else source
            result = tuple(values)
        except Exception as exc:
            raise ProductRuntimeError(
                f"COLOR_BY_BAKE_UNAVAILABLE: raw channel {name!r} cannot be read"
            ) from exc
        if len(result) != count:
            raise ProductRuntimeError(
                f"COLOR_BY_BAKE_UNAVAILABLE: raw channel {name!r} has {len(result)} values "
                f"for {count} allocated bodies"
            )
        return result

    def _prepare_bake_color_plan(self) -> _BakeColorPlan | None:
        if not self.color_source_authority:
            return None
        channels = self._raw_bake_color_channels()
        plan = _bake_color_plan(
            self.descriptor,
            backend=self.backend,
            authority=self.color_source_authority,
            channels=channels,
        )
        if plan is None:
            return None
        # The typed writer's first mutation happens only after every extra full-state source
        # has proved readable at the exact immutable allocation length.
        if plan.carry_damage:
            self._bake_channel_values(
                channels, name="damage", count=plan.allocated_count
            )
        if plan.carry_material_ids:
            self._bake_channel_values(
                channels, name="material_ids", count=plan.allocated_count
            )
        return plan

    def begin_bake(
        self,
        *,
        directory: Path,
        name: str,
        sample_count: int,
        steps_per_sample: int = 1,
        host: dict[str, Any] | None = None,
    ) -> IncrementalBake:
        """Start a bake driven one tick at a time instead of run to completion in one call.

        A long bake (STAB-003: the acceptance scale is 1e5 bodies over 250 frames) held a
        Blender operator's ``execute()`` for its whole duration with no way to report
        progress or cancel. ``IncrementalBake.produce_next()`` reproduces exactly one step
        of ``bake()``'s own frame/step interleaving per call, so a caller that drains it in
        a modal timer instead of a single loop still lands the identical revision.
        """

        if self.closed:
            raise ProductRuntimeError("runtime is closed")
        if self.frame_in_progress:
            raise ProductRuntimeError("cannot bake from a partial authored frame")
        validation = self.validation_summary()
        # Diagnostics consume the collision counters they report, so these two calls own
        # the counts for the state entering the bake and for the bake itself. Both land in
        # the manifest rather than being dropped on the floor.
        entry_diagnostics = self.diagnostics_record()
        entry_fragments = self.fragment_record()

        # STAB-005: frames are produced one at a time and written as they are produced, so
        # the host never holds the bake. The first frame is taken eagerly because the
        # revision's identity is keyed on the bodies it starts from.
        color_plan = self._prepare_bake_color_plan()
        first_sample = (
            self._bake_sample(color_plan=color_plan) if sample_count > 0 else None
        )
        if isinstance(first_sample, ColorBakeFrame):
            # Validate the entire first snapshot before BakeSession creates its staging file.
            carry_bake_frame(first_sample, provenance=color_plan.provenance)
        plan = _require_plan(self.descriptor)
        first_sample_dict = (
            first_sample.sample
            if isinstance(first_sample, ColorBakeFrame)
            else first_sample
        )
        stable_ids = (
            [int(body["stable_id"]) for body in first_sample_dict["bodies"]]
            if first_sample_dict is not None
            else []
        )
        session = BakeSession(
            directory=directory,
            context_id=self.descriptor.context_id,
            stable_ids=stable_ids,
            color_provenance=(None if color_plan is None else color_plan.provenance),
        )
        return IncrementalBake(
            runtime=self,
            session=session,
            first_sample=first_sample,
            color_plan=color_plan,
            sample_count=sample_count,
            steps_per_sample=steps_per_sample,
            entry_diagnostics=entry_diagnostics,
            entry_fragments=entry_fragments,
            finalize_kwargs={
                "name": name,
                "graph_hash": plan.graph_hash,
                "frame_start": (
                    int(first_sample_dict["frame"])
                    if first_sample_dict is not None
                    else self.descriptor.timeline.frame_start + self.frame_index
                ),
                "backend": self.backend,
                # Read from the plan rather than named here: the Solid pipelines integrate
                # under their own modules, and a manifest that always claimed the orbital
                # integrator would misreport what advanced the state.
                "integrator": plan_integrator_name(plan),
                "dt": self.descriptor.timeline.seconds_per_frame,
                "substeps": self.descriptor.timeline.substeps,
                "units": dict(self.descriptor.units.__dict__),
                # STAB-005: the product runtime writes the chunked typed binary payload so a
                # production-scale bake (1e5 bodies x 250 frames) is a byte-addressable
                # cache rather than tens of GB of JSON. The v1 canonical-JSON reader stays
                # in bake.py for pre-cache bakes (ADR 0025 validation-impact).
                "compatible_schema": BINARY_BAKE_SCHEMA,
                "provenance": {
                    "backend": self.backend,
                    # `bake_id` is a uuid5 over the manifest key, which includes provenance,
                    # so recording the lane here is what keeps two bakes that differ only in
                    # precision from collapsing onto the same immutable revision. The same
                    # argument covers environment: a different Warp build or device produces
                    # different numbers, so it produces a different revision.
                    "precision": self.descriptor.precision.value,
                    "determinism": BACKEND_DETERMINISM[self.backend],
                    # No kernel samples an RNG, so a run has no seed of its own. A sampled
                    # particle binding does: its cloud is drawn on the host from the
                    # authored seed mixed with the binding ID. Recording the seeds that were
                    # actually drawn from - and nothing when none were - keeps the manifest
                    # a statement about this bake rather than provenance theatre either way.
                    **_sampling_seed_provenance(self.descriptor),
                    # The encounter the setup predicted, on every backend rather than only
                    # the Solid ones: the conic is a statement about the authored bindings,
                    # and a two-body gravity bake is entitled to the same reading of them.
                    **_encounter_provenance(self.descriptor),
                    "environment": self.environment(host),
                    "validation": validation,
                    "runtime_plan": {
                        "plan_id": plan.plan_id,
                        "backend": plan.backend,
                        "capacities": plan.capacities,
                    },
                    # The module versions that actually executed, per step. They are already
                    # inside `graph_hash`, so this does not add identity; it makes the
                    # manifest readable without recompiling the graph to find out what ran
                    # (PRODUCT-008 c1).
                    "module_versions": {
                        step.module: step.module_version for step in plan.steps
                    },
                    "units": self.descriptor.units.__dict__,
                    "timeline": self.descriptor.timeline.__dict__,
                    "stable_ids": stable_ids,
                    "body_count": body_count(self.descriptor.bindings),
                    "calibration": self.calibration,
                },
            },
        )

    def _bake_sample(
        self, *, color_plan: _BakeColorPlan | None = None
    ) -> dict[str, Any] | ColorBakeFrame:
        sample = {
            **self.runtime.snapshot(),
            "frame": self.descriptor.timeline.frame_start + self.frame_index,
        }
        if color_plan is None:
            return sample
        bodies = sample.get("bodies")
        if not isinstance(bodies, list) or len(bodies) != color_plan.allocated_count:
            actual = len(bodies) if isinstance(bodies, list) else "non-list"
            raise ProductRuntimeError(
                "COLOR_BY_BAKE_UNAVAILABLE: snapshot body count does not match immutable "
                f"binding allocation ({actual} != {color_plan.allocated_count})"
            )
        channels = self._raw_bake_color_channels()
        present = tuple(
            sorted(name for name in RAW_COLOR_BAKE_CHANNELS if channels.get(name) is not None)
        )
        if present != color_plan.expected_channels:
            raise ProductRuntimeError(
                "COLOR_BY_BAKE_UNAVAILABLE: raw runtime sources changed during the bake"
            )
        damage = (
            self._bake_channel_values(
                channels, name="damage", count=color_plan.allocated_count
            )
            if color_plan.carry_damage
            else None
        )
        material_ids = (
            self._bake_channel_values(
                channels, name="material_ids", count=color_plan.allocated_count
            )
            if color_plan.carry_material_ids
            else None
        )
        return ColorBakeFrame(
            sample=sample,
            solid_damage=damage,
            material_ids=material_ids,
        )

    def diagnostics(self) -> Any:
        return self.runtime.diagnostics()

    def close(self) -> None:
        if self.closed:
            return
        # Disposal is itself the state-replacement boundary: no caller may observe or
        # checkpoint a half-authored frame once teardown has started, even if a backend
        # close subsequently reports an error.
        self.closed = True
        failure: BaseException | None = None
        closers = []
        if self.display is not None:
            closers.append(self.display.close)
        closers.extend((self.runtime.close, self.device_context.close))
        try:
            for closer in closers:
                try:
                    closer()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
        finally:
            self.display = None
            self.checkpoints.clear()  # SR-43: release the host snapshots with the runtime
            self._clear_partial_frame()
        if failure is not None:
            raise failure


class IncrementalBake:
    """Ticked counterpart to ``ProductRuntime.bake()``.

    ``produce_next()`` advances by exactly one iteration of ``bake()``'s own ``_frames()``
    generator — write the pending sample if one is due, step the runtime — so a caller that
    calls it until it returns ``False`` and then ``finalize()`` reproduces ``bake()``'s
    output exactly. A modal operator calls ``produce_next()`` from a timer instead of a
    tight loop and can call ``cancel()`` between ticks; nothing else about the bake differs.
    """

    def __init__(
        self,
        *,
        runtime: ProductRuntime,
        session: BakeSession,
        first_sample: dict[str, Any] | ColorBakeFrame | None,
        color_plan: _BakeColorPlan | None,
        sample_count: int,
        steps_per_sample: int,
        entry_diagnostics: dict[str, Any],
        entry_fragments: dict[str, Any] | None,
        finalize_kwargs: dict[str, Any],
    ) -> None:
        self._runtime = runtime
        self._session = session
        self._color_plan = color_plan
        self._steps_per_sample = steps_per_sample
        self._sample_count = sample_count
        self._entry_diagnostics = entry_diagnostics
        self._entry_fragments = entry_fragments
        self._finalize_kwargs = finalize_kwargs
        self._cancelled = False
        # ``_frames()`` calls ``step()`` once unconditionally right after ``first_sample``,
        # then once per remaining sample — ``sample_count`` steps in total — and yields
        # ``sample_count - 1`` samples beyond ``first_sample``.
        has_first = first_sample is not None
        self._pending_steps = sample_count if has_first else 0
        self._pending_samples = max(sample_count - 1, 0) if has_first else 0
        if has_first:
            self._session.add_sample(first_sample)

    @property
    def frame_count(self) -> int:
        return self._session.frame_count

    def produce_next(self) -> bool:
        """Advance by one tick and report whether another tick remains."""

        if self._cancelled or self._pending_steps <= 0:
            return False
        try:
            self._runtime.step(self._steps_per_sample)
            self._pending_steps -= 1
            if self._pending_samples > 0:
                self._session.add_sample(
                    self._runtime._bake_sample(color_plan=self._color_plan)
                )
                self._pending_samples -= 1
        except BaseException:
            self._cancelled = True
            self._session.cancel()
            raise
        return self._pending_steps > 0

    def cancel(self) -> None:
        """Drop everything staged so far; the bake leaves nothing committed on disk."""

        self._cancelled = True
        self._session.cancel()

    def finalize(self) -> tuple[BakeRevision, Path]:
        if self._cancelled:
            raise ProductRuntimeError("cannot finalize a cancelled bake")
        if self._pending_steps > 0:
            raise ProductRuntimeError(
                "cannot finalize a bake before produce_next() reports it complete"
            )

        def _observed() -> dict[str, Any]:
            # Read once the payload is written, because that is when the last frame has
            # been produced and the counters mean what the manifest claims they do.
            return _bake_observations(
                entry_diagnostics=self._entry_diagnostics,
                exit_diagnostics=self._runtime.diagnostics_record(),
                entry_fragments=self._entry_fragments,
                exit_fragments=self._runtime.fragment_record(),
                sample_count=self._sample_count,
                steps_per_sample=self._steps_per_sample,
            )

        return self._session.finalize(observations=_observed, **self._finalize_kwargs)


class ProductRuntimeFactory:
    """Create a concrete runtime from the compiled Astro System Graph."""

    def __init__(self, calibration_path: Path | None = None) -> None:
        self.calibration_path = calibration_path

    def create(self, descriptor: SimulationDescriptor) -> ProductRuntime:
        descriptor = self._compiled_descriptor(descriptor)
        plan = _require_plan(descriptor)
        # A plan saved by the deleted detailed impact engine is refused first, in plain words,
        # before the module and pipeline checks below would call it "unknown".
        _refuse_retired_engine(plan)
        self._validate_plan(descriptor, plan)
        backend = plan.backend
        # ADR 0195 (completion-audit C06). The graphics-memory pre-flight sits here for the same
        # reason: it costs one driver query and no allocation, and a scene that cannot fit is
        # refused in plain words before the runtime starts holding arrays it would have to give
        # back.
        report = admit_scene(descriptor, plan, error_type=ProductRuntimeError)
        # The forecast can be wrong or the memory taken between the read and the allocation, so
        # a device allocation failure while the runtime is built is mapped to the same refusal,
        # after closing whatever the half-built runtime allocated (ADR 0195).
        with allocation_failure_guard(
            ProductRuntimeError, report=report, closable_types=(ProductRuntime,)
        ):
            # Dispatch on how the registered spec says the backend is assembled (ADR 0208), not
            # on its name: a new particle backend is registered, not given a new branch here.
            spec = find_backend(backend)
            if spec is None:
                raise ProductRuntimeError(f"unsupported runtime backend {backend!r}")
            if spec.assembly is Assembly.DIRECT:
                product = self._create_direct(descriptor, backend=spec.name)
            elif spec.assembly is Assembly.TREE:
                product = self._create_tree(descriptor, backend=spec.name)
            elif spec.assembly is Assembly.AUTO:
                product = self._create_auto(descriptor)
            else:
                product = self._create_particle(spec, descriptor)
        return product

    def _compiled_descriptor(self, descriptor: SimulationDescriptor) -> SimulationDescriptor:
        try:
            descriptor.validate()
        except ValueError as exc:
            raise ProductRuntimeError(str(exc)) from exc
        if descriptor.runtime_plan is None:
            raise ProductRuntimeError("descriptor must be compiled before runtime initialization")
        compiled = replace(
            descriptor,
            bindings=tuple(sorted(descriptor.bindings, key=lambda item: item.binding_id)),
        )
        expected_hash = GraphCompiler().graph_hash(compiled.graph)
        plan = _require_plan(compiled)
        if plan.graph_hash != expected_hash:
            raise ProductRuntimeError("runtime_plan graph_hash does not match descriptor graph")
        if plan.plan_id != f"plan-{uuid.uuid5(uuid.NAMESPACE_URL, expected_hash)}":
            raise ProductRuntimeError("runtime_plan plan_id does not match graph_hash")
        return compiled

    @staticmethod
    def _validate_plan(descriptor: SimulationDescriptor, plan: RuntimePlan) -> None:
        if descriptor.schema_version != SCHEMA_VERSION:
            raise ProductRuntimeError(f"unsupported descriptor schema {descriptor.schema_version}")
        if not descriptor.bindings:
            raise ProductRuntimeError("runtime requires at least one binding")
        _refuse_retired_engine(plan)
        modules = [step.module for step in plan.steps]
        unknown = sorted(set(modules) - ALLOWED_RUNTIME_MODULES)
        if unknown:
            raise ProductRuntimeError(
                "runtime plan contains unknown modules: " + ", ".join(unknown)
            )
        # ADR 0045 coherence: FIELD bindings and the external-field step travel together.
        # Either alone means the bindings and the plan were edited out of sync, and
        # recompiling is the only honest repair.
        has_field_bindings = any(
            binding.role == BindingRole.FIELD for binding in descriptor.bindings
        )
        field_steps = modules.count(_EXTERNAL_FIELD_MODULE)
        # ADR 0209. The structural policy -- what the impact backends and the shearing box
        # cannot carry -- is one module the compiler asks as well; held here too for a plan
        # saved or edited some other way. Its first sentence is the one raised.
        policy_errors = plan_policy_errors(descriptor, plan.backend, plan.steps)
        if policy_errors:
            raise ProductRuntimeError(policy_errors[0])
        if has_field_bindings and field_steps != 1:
            raise ProductRuntimeError(
                "descriptor binds force fields but the runtime plan has no "
                f"{_EXTERNAL_FIELD_MODULE} step; recompile the context"
            )
        if field_steps and not has_field_bindings:
            raise ProductRuntimeError(
                f"runtime plan carries a {_EXTERNAL_FIELD_MODULE} step but the "
                "descriptor binds no force fields; recompile the context"
            )
        # ADR 0047 coherence, the same contract one family down: EMITTER bindings and the
        # activation step travel together. The asymmetry with FIELD is deliberate --- an
        # *inactive* emitter binding still occupies its slots, but it releases nothing and
        # the compiler emits no step for it, so the test is on active bindings alone.
        has_emitter_bindings = any(
            binding.role == BindingRole.EMITTER and binding.active
            for binding in descriptor.bindings
        )
        emitter_steps = modules.count(_EMITTER_MODULE)
        if has_emitter_bindings and emitter_steps != 1:
            raise ProductRuntimeError(
                "descriptor binds emitters but the runtime plan has no "
                f"{_EMITTER_MODULE} step; recompile the context"
            )
        if emitter_steps and not has_emitter_bindings:
            raise ProductRuntimeError(
                f"runtime plan carries a {_EMITTER_MODULE} step but the descriptor "
                "binds no active emitters; recompile the context"
            )
        # A plan persisted in a blend file outlives the add-on that compiled it. Without this
        # a saved plan would keep validating after a module changed meaning, and the runtime
        # would execute new kernels under the old plan's identity (PRODUCT-004 c2).
        for step in plan.steps:
            installed = module_version(step.module)
            if step.module_version != installed:
                pinned = step.module_version or "no version"
                raise ProductRuntimeError(
                    f"runtime plan step {step.index} pins {step.module} {pinned} but the "
                    f"installed module is {installed}; recompile the context"
                )
        expected_capacities = {
            "contexts": 1,
            "bindings": len(descriptor.bindings),
            "active_bodies": active_body_count(descriptor.bindings),
        }
        if plan.capacities != expected_capacities:
            raise ProductRuntimeError("runtime plan capacities do not match descriptor")
        if plan.backend not in EXPECTED_PIPELINES:
            raise ProductRuntimeError(f"unsupported runtime backend {plan.backend!r}")
        # ADR 0202. What the shearing box cannot carry was refused above by the policy, in the
        # artist's words, ahead of the pipeline comparison, which would only say the plan "is not
        # the exact pipeline". Its shape is an integrity matter and stays here.
        shearing_steps = modules.count(_SHEARING_BOX_MODULE)
        if shearing_steps > 1:
            raise ProductRuntimeError(
                "runtime plan carries more than one shearing box; recompile the context"
            )
        expected_pipeline = _expected_pipeline(
            plan.backend,
            external_field=has_field_bindings,
            emitter=has_emitter_bindings,
            shearing_box=bool(shearing_steps),
        )
        actual_pipeline = tuple((step.family, step.module) for step in plan.steps)
        if actual_pipeline != expected_pipeline:
            raise ProductRuntimeError(
                f"runtime plan is not the exact {plan.backend!r} executable pipeline"
            )
        spatial = plan.steps[2]
        if spatial.parameters.get("backend") != plan.backend:
            raise ProductRuntimeError("runtime plan spatial backend parameter is incoherent")
        if plan.backend == "auto":
            tolerance = spatial.parameters.get("accuracy_tolerance")
            if tolerance is not None and (
                not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or float(tolerance) < 0.0
            ):
                raise ProductRuntimeError("Auto accuracy_tolerance must be finite and non-negative")
        if tuple(step.index for step in plan.steps) != tuple(range(len(plan.steps))):
            raise ProductRuntimeError("runtime plan step indices are not contiguous")
        ordered_nodes = tuple(
            sorted(
                descriptor.graph.nodes,
                key=lambda node: (STAGE_ORDER[node.family], node.node_id),
            )
        )
        if len(ordered_nodes) != len(plan.steps):
            raise ProductRuntimeError("runtime plan step count does not match graph")
        for node, step in zip(ordered_nodes, plan.steps, strict=True):
            if (
                step.node_id != node.node_id
                or step.family != node.family
                or step.module != node.module
                or step.module_version != node.module_version
                or step.parameters != node.parameters
            ):
                raise ProductRuntimeError(
                    f"runtime plan step {step.index} does not match compiled graph node"
                )
        required = COMMON_RUNTIME_MODULES | BACKEND_RUNTIME_MODULES[plan.backend]
        missing = sorted(required - set(modules))
        if missing:
            raise ProductRuntimeError(
                f"runtime plan is missing required modules: {', '.join(missing)}"
            )
        for module in required:
            if modules.count(module) != 1:
                raise ProductRuntimeError(
                    f"runtime plan requires exactly one {module!r} module"
                )
        backend_modules = set().union(*BACKEND_RUNTIME_MODULES.values())
        unexpected_backend = sorted(
            (set(modules) & backend_modules) - BACKEND_RUNTIME_MODULES[plan.backend]
        )
        if unexpected_backend:
            raise ProductRuntimeError(
                "runtime plan backend modules do not match selected backend: "
                + ", ".join(unexpected_backend)
            )
        for step in plan.steps:
            if step.module == "astro.collision.point":
                try:
                    CollisionMode(str(step.parameters.get("mode", "ignore")))
                except ValueError as exc:
                    raise ProductRuntimeError(
                        f"unsupported collision mode {step.parameters.get('mode')!r}"
                    ) from exc
        assembly = backend_spec(plan.backend).assembly
        if assembly in {Assembly.TREE, Assembly.AUTO}:
            try:
                _tree_config(descriptor).validate()
            except ValueError as exc:
                raise ProductRuntimeError(str(exc)) from exc
        if assembly is Assembly.PARTICLE:
            try:
                _PARTICLE_ASSEMBLY[plan.backend].read_config(descriptor).validate()
            except ValueError as exc:
                raise ProductRuntimeError(str(exc)) from exc

    def _create_direct(
        self,
        descriptor: SimulationDescriptor,
        *,
        backend: str,
        bodies: BodySet | None = None,
    ) -> ProductRuntime:
        context = DeviceContext()
        config = _direct_config(descriptor)
        runtime = DirectNBodyRuntime(context, config)
        if bodies is None:
            bodies = expand_bindings(
                descriptor.bindings,
                gravitational_constant=config.gravitational_constant,
                softening=config.softening,
                # A shearing box measures velocity from the circular shear flow (ADR 0202);
                # zero for every other Direct plan, which adds nothing.
                shear_flow_omega=(
                    0.0 if config.shearing_box is None else config.shearing_box.omega
                ),
            )
        authority = _color_source_authority(
            descriptor.bindings, bodies, backend=backend, materials=None
        )
        runtime.load_bodies(bodies)
        product = ProductRuntime(
            descriptor,
            backend,
            context,
            runtime,
            color_source_authority=authority,
        )
        product.checkpoint("initial")
        return product

    def _create_tree(
        self,
        descriptor: SimulationDescriptor,
        *,
        backend: str,
        bodies: BodySet | None = None,
    ) -> ProductRuntime:
        context = DeviceContext()
        config = _tree_config(descriptor)
        runtime = BarnesHutRuntime(context, config)
        if bodies is None:
            bodies = expand_bindings(
                descriptor.bindings,
                gravitational_constant=config.gravitational_constant,
                softening=config.softening,
            )
        authority = _color_source_authority(
            descriptor.bindings, bodies, backend=backend, materials=None
        )
        runtime.load_bodies(bodies)
        product = ProductRuntime(
            descriptor,
            backend,
            context,
            runtime,
            color_source_authority=authority,
        )
        product.checkpoint("initial")
        return product

    def _create_particle(
        self, spec: BackendSpec, descriptor: SimulationDescriptor
    ) -> ProductRuntime:
        """The runtime of a particle backend: the giant-impact or the rock-impact engine.

        Giant impact (ADR 0094): two Solids meet and a moon forms. Rock impact (ADR 0193): bodies
        that crack, at a viewport's pace; its materials, fault strain and fault cell size are
        plan parameters, so moving any of them moves `graph_hash`. Both engines sample every
        body at one common particle spacing, advance a whole engine frame per primitive, publish
        the same display channels (the rock engine adds the bond model's damage) and speak the
        same resumable protocol, so nothing downstream of `ProductRuntime` branches on which one
        it holds. What differs -- reading the config, laying the bodies out -- is
        `_PARTICLE_ASSEMBLY`'s; the order of the steps is not.
        """

        assembly = _PARTICLE_ASSEMBLY[spec.name]
        context = DeviceContext()
        config = assembly.read_config(descriptor)
        runtime = spec.runtime_classes[0](context, config)
        layout = assembly.sample_layout(descriptor, config)
        authority = _color_source_authority(
            descriptor.bindings,
            layout.bodies,
            backend=spec.name,
            materials=config.materials,
        )
        runtime.load_layout(layout)
        product = ProductRuntime(
            descriptor,
            spec.name,
            context,
            runtime,
            color_source_authority=authority,
        )
        product.checkpoint("initial")
        return product

    def _create_auto(self, descriptor: SimulationDescriptor) -> ProductRuntime:
        started = time.perf_counter()
        direct_started = time.perf_counter()
        # ADR 0133: sample the scene once. Direct, Tree and the quarter/half count curve
        # all expand the same bindings with the same units, so they get the same bodies;
        # expanding per use cost a Galaxy Collision at 100,000 bodies four full samplings
        # before its first frame.
        bodies = expand_bindings(
            descriptor.bindings,
            gravitational_constant=descriptor.units.gravitational_constant,
            softening=descriptor.units.softening,
        )
        direct = self._create_direct(descriptor, backend="direct", bodies=bodies)
        direct_initialization_ms = (time.perf_counter() - direct_started) * 1000.0
        if len(bodies) <= AUTO_SMALL_SCENE_DIRECT_LIMIT:
            # AW-009 / ADR 0155: below this size the speed check costs seconds and has
            # never picked Tree for a reason other than timing noise.
            direct.calibration = _small_scene_calibration(
                descriptor, len(bodies), started, direct_initialization_ms
            )
            return direct
        fingerprint_inputs = _calibration_fingerprint_inputs(descriptor, direct)
        fingerprint = _json_sha256(fingerprint_inputs)
        calibration_path = self.calibration_path or _default_calibration_path(descriptor)
        cached, cache_error = _load_calibration(calibration_path, fingerprint)
        if cached is not None:
            cached = {
                **cached,
                "cache_status": "hit",
                "calibration_path": str(calibration_path),
            }
            if cached.get("selected") == "tree":
                try:
                    tree = self._create_tree(descriptor, backend="tree", bodies=bodies)
                except Exception as exc:  # noqa: BLE001 - cached Tree must fail closed
                    direct.calibration = {
                        **cached,
                        "selected": "direct",
                        "fallback_reason": f"cached_tree_unavailable: {exc}",
                    }
                    return direct
                direct.close()
                tree.calibration = cached
                return tree
            direct.calibration = cached
            return direct

        tree: ProductRuntime | None = None
        try:
            direct.step(1)
            direct_warmup_ms = float(direct.last_step_ms)
            direct.restore("initial")
            direct_samples = []
            for _ in range(WARMED_REPETITIONS):
                direct.restore("initial")
                direct.step(1)
                direct_samples.append(float(direct.last_step_ms))
            direct.restore("initial")

            tree_started = time.perf_counter()
            tree = self._create_tree(descriptor, backend="tree", bodies=bodies)
            tree_initialization_ms = (time.perf_counter() - tree_started) * 1000.0
            try:
                # This calibration path is tree-specific by construction, so it reaches
                # BarnesHut-only build/traversal/error probes the BackendRuntime seam does
                # not expose; narrow once to the concrete backend.
                tree_backend = cast(BarnesHutRuntime, tree.runtime)
                tree.step(1)
                tree_warmup_ms = float(tree.last_step_ms)
                tree.restore("initial")
                tree_build_samples = []
                tree_traversal_samples = []
                tree_total_samples = []
                for _ in range(WARMED_REPETITIONS):
                    tree.restore("initial")
                    tree.step(1)
                    tree_total_samples.append(float(tree.last_step_ms))
                    tree.restore("initial")
                    tree_backend.rebuild_tree()
                    tree_build_samples.append(float(tree_backend.last_build_ms))
                    tree_traversal_samples.append(
                        float(tree_backend.benchmark_acceleration(1))
                    )
                tree.restore("initial")
                tree.step(1)
                error = tree_backend.direct_error_sample()
                tree_diag = tree.diagnostics()
                tree.restore("initial")
            except Exception as exc:  # noqa: BLE001 - fail closed to direct
                tree.close()
                direct.calibration = _calibration(
                    descriptor=descriptor,
                    selected="direct",
                    fallback_reason=f"tree_calibration_failed: {exc}",
                    started=started,
                    fingerprint=fingerprint,
                    fingerprint_inputs=fingerprint_inputs,
                    direct_initialization_ms=direct_initialization_ms,
                    direct_warmup_ms=direct_warmup_ms,
                    direct_samples=direct_samples,
                    cache_error=cache_error,
                )
                _store_calibration_or_record_error(
                    calibration_path, fingerprint, direct.calibration
                )
                return direct

            policy_name, tolerance = _auto_accuracy_policy(descriptor)
            direct_ms = statistics.median(direct_samples)
            tree_ms = statistics.median(tree_total_samples)
            count_curve = _representative_count_curve(
                descriptor,
                bodies,
                {
                    "body_count": body_count(descriptor.bindings),
                    "direct_median_ms": direct_ms,
                    "tree_build_median_ms": statistics.median(tree_build_samples),
                    "tree_traversal_median_ms": statistics.median(
                        tree_traversal_samples
                    ),
                    "tree_total_median_ms": tree_ms,
                    "error_metrics": error,
                    "overflow": bool(tree_diag.overflow),
                },
            )
            choose_tree, fallback_reason = _select_auto_backend(
                policy_name=policy_name,
                tolerance=tolerance,
                direct_ms=direct_ms,
                tree_ms=tree_ms,
                error=error,
                overflow=bool(tree_diag.overflow),
            )
            calibration = _calibration(
                descriptor=descriptor,
                selected="tree" if choose_tree else "direct",
                fallback_reason=fallback_reason,
                started=started,
                fingerprint=fingerprint,
                fingerprint_inputs=fingerprint_inputs,
                direct_initialization_ms=direct_initialization_ms,
                direct_warmup_ms=direct_warmup_ms,
                direct_samples=direct_samples,
                tree_initialization_ms=tree_initialization_ms,
                tree_warmup_ms=tree_warmup_ms,
                tree_build_samples=tree_build_samples,
                tree_traversal_samples=tree_traversal_samples,
                tree_total_samples=tree_total_samples,
                error=error,
                cache_error=cache_error,
                count_curve=count_curve,
            )
            if not _store_calibration_or_record_error(
                calibration_path, fingerprint, calibration
            ):
                calibration["selected"] = "direct"
                calibration["fallback_reason"] = "calibration_persist_failed"
                tree.close()
                direct.calibration = calibration
                return direct
            if choose_tree:
                direct.close()
                tree.calibration = calibration
                return tree
            tree.close()
            direct.calibration = calibration
            return direct
        except Exception:
            direct.close()
            if tree is not None:
                tree.close()
            raise


def create_runtime_from_descriptor(descriptor: SimulationDescriptor) -> ProductRuntime:
    return ProductRuntimeFactory().create(descriptor)


def _require_plan(descriptor: SimulationDescriptor) -> RuntimePlan:
    if descriptor.runtime_plan is None:
        raise ProductRuntimeError("descriptor is not compiled")
    return descriptor.runtime_plan


def _refuse_retired_engine(plan: RuntimePlan) -> None:
    """Refuse a plan the deleted detailed impact engine compiled, in plain words.

    A System saved before the engine was deleted still carries its plan: a backend of
    ``impact``, ``sph`` or ``solids``, and steps naming the engine's modules. Left to the
    ordinary checks they would be refused as "unknown modules" or as an "unsupported runtime
    backend", which names nothing the artist can act on. Pressing Play compiles the authored
    System again, and the compile now picks one of the realtime Solid engines, so that is
    the remedy the sentence gives.
    """

    if plan.backend in RETIRED_ENGINE_BACKENDS or any(
        step.module in RETIRED_ENGINE_MODULES for step in plan.steps
    ):
        raise ProductRuntimeError(RETIRED_ENGINE_MESSAGE)


def _bake_observations(
    *,
    entry_diagnostics: dict[str, Any],
    exit_diagnostics: dict[str, Any],
    entry_fragments: dict[str, Any] | None,
    exit_fragments: dict[str, Any] | None,
    sample_count: int,
    steps_per_sample: int,
) -> dict[str, Any]:
    """What was measured while baking — outside the revision's identity."""

    return {
        "diagnostics": {
            "at_first_sample": entry_diagnostics,
            "at_last_sample": exit_diagnostics,
        },
        # Both ends for the same reason the diagnostics record both: the answer the impact
        # story wants is what the bodies broke *into*, which is only readable against what
        # they came in as. A backend with no connectivity to report omits the key rather
        # than writing an empty one.
        **(
            {
                "fragments": {
                    "at_first_sample": entry_fragments,
                    "at_last_sample": exit_fragments,
                }
            }
            if entry_fragments is not None and exit_fragments is not None
            else {}
        ),
        "sample_count": sample_count,
        "steps_per_sample": steps_per_sample,
    }


def plan_integrator_name(plan: RuntimePlan) -> str:
    """The integrating module's short name, as the bake manifest records it.

    Public because the host reads it too: a bake's compatibility axis asks whether the
    revision was integrated by the module the active plan compiles, and a host that
    named one integrator would call every bake of another engine incompatible.
    """

    for step in plan.steps:
        if step.family is StageFamily.INTEGRATION:
            return step.module.rsplit(".", 1)[-1]
    raise ProductRuntimeError("runtime plan has no integration step")


def _encounter_provenance(descriptor: SimulationDescriptor) -> dict[str, Any]:
    """The two-body encounter the authored setup predicts, before anything is simulated.

    Provenance rather than an observation because it is a statement about the inputs: the
    conic through the authored relative state is fixed the moment the bindings are, so a
    reader can compare what the setup promised against what the fragment report at the end
    of the bake actually found without re-deriving either.

    Exactly two bodies, because that is the question a conic answers. A one-body or an
    n-body scene records why there is no prediction rather than a prediction of nothing:
    an impact-oriented manifest that silently omitted the field would be indistinguishable
    from one written before the field existed.
    """

    bodies = [
        binding
        for binding in descriptor.bindings
        if binding.role == BindingRole.BODY and binding.active
    ]
    if len(bodies) != 2:
        return {
            "encounter": None,
            "encounter_reason": (
                f"{len(bodies)} active bodies; a two-body conic predicts an encounter "
                "between exactly two"
            ),
        }
    first, second = bodies
    try:
        prediction = predict_encounter(
            position_a=first.position,
            velocity_a=first.velocity,
            mass_a=first.mass,
            radius_a=first.radius,
            position_b=second.position,
            velocity_b=second.velocity,
            mass_b=second.mass,
            radius_b=second.radius,
            gravitational_constant=descriptor.units.gravitational_constant,
        )
    except ValueError as exc:
        # A setup the conic cannot describe - coincident centres, a zero radius - is a
        # statement about the scene, not a reason to refuse the bake. The manifest records
        # why there is no prediction and the numbers are written either way.
        return {"encounter": None, "encounter_reason": str(exc)}
    return {
        "encounter": prediction.to_dict(),
        "encounter_bodies": [first.binding_id, second.binding_id],
    }


def _sampling_seed_provenance(descriptor: SimulationDescriptor) -> dict[str, Any]:
    """The authored seeds a sampled binding drew its cloud from, or their absence."""

    seeds = {
        binding.binding_id: int(binding.sampling_seed)
        for binding in descriptor.bindings
        if binding.representation == BodyRepresentation.SAMPLED_PARTICLE
        and binding_body_count(binding) > 1
    }
    if not seeds:
        return {
            "seed": None,
            "seed_reason": "no compiled module draws random numbers",
        }
    return {
        "seed": seeds,
        "seed_reason": (
            "per-binding host sampling seed; each cloud is drawn from this seed mixed "
            "with its binding ID, so the same seed reproduces the same cloud"
        ),
    }


def _shearing_box_from_plan(plan: RuntimePlan) -> ShearingBoxSpec | None:
    """The box a plan's shearing step describes, or ``None`` when the plan has no such step.

    The runtime is built from the step's parameters and from nothing else (they are inside
    ``graph_hash``). What the box cannot carry is refused by ``plan_policy`` (ADR 0209).
    """

    return shearing_box_from_steps(plan.steps)


def _direct_config(descriptor: SimulationDescriptor) -> NBodyConfig:
    mode = "ignore"
    for step in _require_plan(descriptor).steps:
        if step.module == "astro.collision.point":
            mode = str(step.parameters.get("mode", "ignore"))
    bodies = body_count(descriptor.bindings)
    return NBodyConfig(
        capacity=bodies,
        count=bodies,
        gravitational_constant=descriptor.units.gravitational_constant,
        dt=descriptor.timeline.seconds_per_frame,
        substeps=descriptor.timeline.substeps,
        softening=descriptor.units.softening,
        collision_mode=CollisionMode(mode),
        precision=descriptor.precision,
        external_fields=specs_from_bindings(descriptor.bindings),
        # `count` above is the full body count, emitter slots included: every emitter
        # particle is drawn at load and held inactive, so the slots a spec names are
        # inside the population the runtime just allocated. That is the invariant
        # `validate_slot_range` re-checks against this same `count`.
        emitters=emitter_specs_from_bindings(descriptor.bindings, descriptor.timeline),
        # None for every plan without the step, which is every plan there was (ADR 0202).
        shearing_box=_shearing_box_from_plan(_require_plan(descriptor)),
    )


def _tree_config(descriptor: SimulationDescriptor) -> BarnesHutConfig:
    theta = DEFAULT_TREE_THETA
    max_depth = DEFAULT_TREE_MAX_DEPTH
    leaf_capacity = 8
    stack_capacity = DEFAULT_TREE_STACK_CAPACITY
    node_capacity = 0
    quadrupoles = True
    collision_mode = CollisionMode.IGNORE
    for step in _require_plan(descriptor).steps:
        if step.module in {"astro.gravity.barnes_hut_tree", "astro.gravity.auto_select"}:
            theta = float(step.parameters.get("theta", theta))
            max_depth = int(step.parameters.get("max_depth", max_depth))
            leaf_capacity = int(step.parameters.get("leaf_capacity", leaf_capacity))
            stack_capacity = int(step.parameters.get("stack_capacity", stack_capacity))
            node_capacity = int(step.parameters.get("node_capacity", node_capacity))
            quadrupoles = bool(step.parameters.get("quadrupoles", quadrupoles))
        elif step.module == "astro.collision.point":
            collision_mode = CollisionMode(str(step.parameters.get("mode", "ignore")))
    bodies = body_count(descriptor.bindings)
    return BarnesHutConfig(
        capacity=bodies,
        count=bodies,
        gravitational_constant=descriptor.units.gravitational_constant,
        dt=descriptor.timeline.seconds_per_frame,
        substeps=descriptor.timeline.substeps,
        softening=descriptor.units.softening,
        theta=theta,
        max_depth=max_depth,
        leaf_capacity=leaf_capacity,
        stack_capacity=stack_capacity,
        node_capacity=node_capacity,
        quadrupoles=quadrupoles,
        collision_mode=collision_mode,
        precision=descriptor.precision,
        external_fields=specs_from_bindings(descriptor.bindings),
        emitters=emitter_specs_from_bindings(descriptor.bindings, descriptor.timeline),
    )


def _layered_inputs(
    parameters: Mapping[str, Any],
) -> tuple[MaterialRegistry | None, tuple[tuple[str, tuple[tuple[float, int, float], ...]], ...]]:
    """The materials and per-binding layering a layered initial-state step carries.

    Shared by the two Solid backends, whose initial states are both differentiated bodies
    read from the same two plan parameters.
    """

    manifest = dict(parameters.get("materials", {}))
    materials = registry_from_manifest(manifest) if manifest else None
    shells = tuple(
        (
            str(binding_id),
            tuple((float(layer[0]), int(layer[1]), float(layer[2])) for layer in layers),
        )
        for binding_id, layers in sorted(dict(parameters.get("shells", {})).items())
    )
    return materials, shells


def _giant_impact_config(descriptor: SimulationDescriptor) -> GiantImpactConfig:
    """Read the giant-impact plan into the engine's config (ADR 0094).

    The authored substep count is a floor on the pressure substeps, not the count: the engine
    sizes its own from the resolution and the impact speed, so a scene run finer stays the
    same shape. The authored precision is carried through so that `validate` can refuse the
    float64 the engine does not have rather than publish a label its arrays do not match.
    """

    materials = None
    shells: tuple[tuple[str, tuple[tuple[float, int, float], ...]], ...] = ()
    linking_length_factor = 0.75
    moon_start = 20.0
    moon_end = 0.0
    for step in _require_plan(descriptor).steps:
        if step.module == "astro.inputs.giant_impact_bodies":
            materials, shells = _layered_inputs(step.parameters)
        elif step.module == "astro.giant_impact.moon_guide":
            moon_start = float(step.parameters.get("start", moon_start))
            moon_end = float(step.parameters.get("end", moon_end))
        elif step.module == "astro.diagnostics.fragments":
            linking_length_factor = float(
                step.parameters.get("linking_length_factor", linking_length_factor)
            )
    return GiantImpactConfig(
        count=body_count(descriptor.bindings),
        gravitational_constant=descriptor.units.gravitational_constant,
        seconds_per_frame=descriptor.timeline.seconds_per_frame,
        minimum_pressure_substeps=descriptor.timeline.substeps,
        precision=descriptor.precision,
        materials=materials,
        shells=shells,
        linking_length_factor=linking_length_factor,
        moon_start=moon_start,
        moon_end=moon_end,
    )


def _spacesim_impact_config(descriptor: SimulationDescriptor) -> SpaceSimImpactConfig:
    """Read the rock-impact plan into the engine's config (ADR 0193).

    The frame length is the authored one, unchanged: the engine sizes its own outer and inner
    step counts from it, from the resolution and from the stiffest material, and refuses a frame
    past the ceilings rather than lengthening it (ADR 0123). The authored substep count is not
    read: unlike the giant-impact engine's floor, nothing here could use one, because the
    engine's step counts are decided by the resolution and the stiffness and a frame is one
    primitive whatever they are. The authored precision is carried through so that `validate`
    can refuse the float64 the engine does not have rather than publish a label its arrays
    do not match.
    """

    materials = None
    shells: tuple[tuple[str, tuple[tuple[float, int, float], ...]], ...] = ()
    linking_length_factor = 0.75
    fault_strain = DEFAULT_FAULT_STRAIN
    fault_cell_particles = FAULT_CELL_PARTICLES
    for step in _require_plan(descriptor).steps:
        if step.module == "astro.inputs.spacesim_bodies":
            materials, shells = _layered_inputs(step.parameters)
            fault_strain = float(step.parameters.get("fault_strain", fault_strain))
            fault_cell_particles = int(
                step.parameters.get("fault_cell_particles", fault_cell_particles)
            )
        elif step.module == "astro.diagnostics.fragments":
            linking_length_factor = float(
                step.parameters.get("linking_length_factor", linking_length_factor)
            )
    return SpaceSimImpactConfig(
        count=body_count(descriptor.bindings),
        gravitational_constant=descriptor.units.gravitational_constant,
        seconds_per_frame=descriptor.timeline.seconds_per_frame,
        precision=descriptor.precision,
        materials=materials,
        shells=shells,
        linking_length_factor=linking_length_factor,
        fault_strain=fault_strain,
        fault_cell_particles=fault_cell_particles,
    )


@dataclass(frozen=True)
class _ParticleAssembly:
    """What differs between the particle backends when `create` builds one."""

    #: Read the plan into the engine's config (and `validate()` it).
    read_config: Callable[[SimulationDescriptor], Any]
    #: Lay the bound bodies out at the engine's one common spacing. The samplers are looked up
    #: when called, so a test that replaces one in this module's namespace reaches it.
    sample_layout: Callable[[SimulationDescriptor, Any], Any]


#: One entry per registered backend with `Assembly.PARTICLE`; a conformance test holds the two
#: equal, so a registered particle backend cannot be left with no way to be built.
_PARTICLE_ASSEMBLY: Mapping[str, _ParticleAssembly] = MappingProxyType(
    {
        "giant_impact": _ParticleAssembly(
            read_config=_giant_impact_config,
            sample_layout=lambda descriptor, config: sample_giant_impact_bodies(
                descriptor.bindings,
                config.shell_map or None,
                gravitational_constant=config.gravitational_constant,
            ),
        ),
        "spacesim_impact": _ParticleAssembly(
            read_config=_spacesim_impact_config,
            sample_layout=lambda descriptor, config: sample_spacesim_bodies(
                descriptor.bindings,
                config.shell_map or None,
                gravitational_constant=config.gravitational_constant,
                fault_cell_particles=config.fault_cell_particles,
            ),
        ),
    }
)


def _calibration(
    *,
    descriptor: SimulationDescriptor,
    selected: str,
    fallback_reason: str,
    started: float,
    fingerprint: str,
    fingerprint_inputs: dict[str, Any],
    direct_initialization_ms: float,
    direct_warmup_ms: float,
    direct_samples: list[float],
    tree_initialization_ms: float | None = None,
    tree_warmup_ms: float | None = None,
    tree_build_samples: list[float] | None = None,
    tree_traversal_samples: list[float] | None = None,
    tree_total_samples: list[float] | None = None,
    error: dict[str, float] | None = None,
    cache_error: str = "",
    count_curve: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_name, tolerance = _auto_accuracy_policy(descriptor)
    tree_build_samples = tree_build_samples or []
    tree_traversal_samples = tree_traversal_samples or []
    tree_total_samples = tree_total_samples or []
    return {
        "schema": CALIBRATION_SCHEMA,
        "fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "cache_status": "miss" if not cache_error else "invalid",
        "cache_error": cache_error,
        "methodology": {
            "description": (
                "initialization and first kernel compilation recorded separately; "
                "one untimed restore-bounded warm-up; median of five one-step "
                "restore-bounded warmed repetitions at matching state"
            ),
            "warmed_repetitions": WARMED_REPETITIONS,
            "tree_cost_includes": ["build", "traversal", "KDK step"],
            "tree_component_scope": (
                "separate synchronized build and traversal at the same restored state; "
                "selection uses the full KDK step total"
            ),
            "error_denominator_floor": 1.0e-12,
        },
        "selected": selected,
        "fallback_reason": fallback_reason,
        "accuracy_policy": {
            "name": policy_name,
            "tolerance": tolerance,
            "status": "set" if policy_name and tolerance is not None else "unset",
        },
        "direct": {
            "initialization_ms": direct_initialization_ms,
            "warmup_ms": direct_warmup_ms,
            "samples_ms": direct_samples,
            "median_ms": statistics.median(direct_samples),
        },
        "tree": {
            "initialization_ms": tree_initialization_ms,
            "warmup_ms": tree_warmup_ms,
            "build_samples_ms": tree_build_samples,
            "build_median_ms": (
                statistics.median(tree_build_samples) if tree_build_samples else None
            ),
            "traversal_samples_ms": tree_traversal_samples,
            "traversal_median_ms": (
                statistics.median(tree_traversal_samples)
                if tree_traversal_samples
                else None
            ),
            "total_samples_ms": tree_total_samples,
            "total_median_ms": (
                statistics.median(tree_total_samples) if tree_total_samples else None
            ),
        },
        "error_metrics": error or {},
        "representative_count_calibration": count_curve or {
            "records": [],
            "performance_crossover_count": None,
            "accuracy_approved_crossover_count": None,
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def _small_scene_calibration(
    descriptor: SimulationDescriptor,
    count: int,
    started: float,
    direct_initialization_ms: float,
) -> dict[str, Any]:
    """The calibration record of a scene Auto sent to Direct without timing either solver."""
    record = _calibration(
        descriptor=descriptor,
        selected="direct",
        fallback_reason="small_scene_direct",
        started=started,
        fingerprint="",
        fingerprint_inputs={
            "body_count": count,
            "small_scene_limit": AUTO_SMALL_SCENE_DIRECT_LIMIT,
        },
        direct_initialization_ms=direct_initialization_ms,
        direct_warmup_ms=0.0,
        direct_samples=[0.0],
    )
    record["cache_status"] = "skipped"
    record["direct"]["samples_ms"] = []
    record["direct"]["median_ms"] = None
    return record


def _auto_accuracy_policy(descriptor: SimulationDescriptor) -> tuple[str, float | None]:
    for step in _require_plan(descriptor).steps:
        if step.module == "astro.gravity.auto_select":
            name = str(step.parameters.get("accuracy_policy") or "")
            tolerance = step.parameters.get("accuracy_tolerance")
            if tolerance is None:
                return name, None
            value = float(tolerance)
            if not math.isfinite(value) or value < 0.0:
                raise ProductRuntimeError(
                    "Auto accuracy_tolerance must be a finite non-negative value"
                )
            return name, value
    raise ProductRuntimeError("Auto runtime plan has no accuracy policy configuration")


def _select_auto_backend(
    *,
    policy_name: str,
    tolerance: float | None,
    direct_ms: float,
    tree_ms: float,
    error: dict[str, float],
    overflow: bool,
) -> tuple[bool, str]:
    if not policy_name or tolerance is None:
        return False, "accuracy_policy_unset"
    if overflow:
        return False, "tree_overflow"
    # The RMS-normalised series, not the per-body one: the tolerance is a theta**2 bound from
    # adr/0006-accuracy-gate-thresholds.md, and that bound is measured against the system RMS
    # acceleration. Judging it with a per-body denominator compares two different quantities
    # and rejects an accurate tree whenever one body sits near a force null.
    if float(error["rms_normalized_max"]) > tolerance:
        return False, "accuracy_policy_failed"
    if tree_ms >= direct_ms:
        return False, "tree_not_faster"
    return True, ""


def _representative_counts(body_count: int) -> tuple[int, ...]:
    return tuple(
        sorted({max(1, body_count // 4), max(1, body_count // 2), body_count})
    )


def _representative_count_curve(
    descriptor: SimulationDescriptor,
    bodies: BodySet,
    actual_record: dict[str, Any],
) -> dict[str, Any]:
    policy_name, tolerance = _auto_accuracy_policy(descriptor)
    records = []
    actual_count = body_count(descriptor.bindings)
    for count in _representative_counts(actual_count):
        if count == actual_count:
            records.append(dict(actual_record))
            continue
        try:
            records.append(_benchmark_representative_count(descriptor, bodies, count))
        except Exception as exc:  # noqa: BLE001 - auxiliary points remain visible
            records.append({"body_count": count, "error": str(exc)})
    valid = [record for record in records if "error" not in record]
    performance = [
        record
        for record in valid
        if record["tree_total_median_ms"] < record["direct_median_ms"]
    ]
    approved = [
        record
        for record in performance
        if policy_name
        and tolerance is not None
        and not record["overflow"]
        # Same quantity the selection rule applies, not the per-body series: gating the
        # curve on "max" made the reported crossover disagree with the backend actually
        # chosen for the very same scene.
        and float(record["error_metrics"]["rms_normalized_max"]) <= tolerance
    ]
    return {
        "methodology": (
            "derived scene-relative quarter, half, and full counts; each count uses "
            "one warm-up plus five restore-bounded matching-state repetitions"
        ),
        "records": records,
        "performance_crossover_count": (
            min(record["body_count"] for record in performance)
            if performance
            else None
        ),
        "accuracy_approved_crossover_count": (
            min(record["body_count"] for record in approved) if approved else None
        ),
    }


def _benchmark_representative_count(
    descriptor: SimulationDescriptor, scene: BodySet, count: int
) -> dict[str, Any]:
    # Slice bodies, not bindings: one sampled binding can carry the whole scene, so
    # slicing the authored rows would jump straight from zero to the full count. The
    # scene is the caller's one expansion (ADR 0133), never re-sampled per count.
    bodies = scene.head(count)
    direct_context = DeviceContext()
    tree_context = DeviceContext()
    direct = DirectNBodyRuntime(
        direct_context,
        replace(_direct_config(descriptor), capacity=count, count=count),
    )
    tree = BarnesHutRuntime(
        tree_context,
        replace(_tree_config(descriptor), capacity=count, count=count),
    )
    try:
        direct.load_bodies(bodies)
        tree.load_bodies(bodies)
        direct_initial = direct.snapshot()
        tree_initial = tree.snapshot()
        direct.step(1)
        tree.step(1)
        direct.restore_snapshot(direct_initial)
        tree.restore_snapshot(tree_initial)
        direct_samples = []
        tree_build_samples = []
        tree_traversal_samples = []
        tree_total_samples = []
        for _ in range(WARMED_REPETITIONS):
            direct.restore_snapshot(direct_initial)
            direct.step(1)
            direct_samples.append(float(direct.last_step_ms))
            tree.restore_snapshot(tree_initial)
            tree.step(1)
            tree_total_samples.append(float(tree.last_step_ms))
            tree.restore_snapshot(tree_initial)
            tree_build_samples.append(float(tree.last_build_ms))
            tree_traversal_samples.append(float(tree.benchmark_acceleration(1)))
        tree.restore_snapshot(tree_initial)
        tree.step(1)
        error = tree.direct_error_sample()
        diagnostics = tree.diagnostics()
        return {
            "body_count": count,
            "direct_median_ms": statistics.median(direct_samples),
            "tree_build_median_ms": statistics.median(tree_build_samples),
            "tree_traversal_median_ms": statistics.median(tree_traversal_samples),
            "tree_total_median_ms": statistics.median(tree_total_samples),
            "error_metrics": error,
            "overflow": bool(
                diagnostics.overflow or diagnostics.collision_overflow
            ),
        }
    finally:
        direct.close()
        tree.close()
        direct_context.close()
        tree_context.close()


def _root_box_octave(descriptor: SimulationDescriptor) -> int:
    """Octaves the tree's root cube spends before it reaches the bodies.

    The tree's traversal cost is set by how much empty space the root cube covers
    relative to the volume the bodies actually occupy: on one 65536-body cloud,
    moving a single distant primary so the cube grows from 1.7x to 6.2x the cloud
    turns a 2.2x tree win into a 2.3x tree loss. Positions therefore have to key the
    calibration, or a scene recalls a verdict measured on a different geometry.

    Bucketing by octave is what keeps that affordable: nudging an object does not
    invalidate a calibration, while a structural move misses and re-measures.
    """

    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    occupied = 0.0
    for binding in descriptor.bindings:
        if not binding.active:
            continue
        # Sampled clouds span position +/- radius; an analytical body is a point of
        # that radius. Either way the binding occupies radius about its position.
        extent = abs(float(binding.radius))
        for axis in range(3):
            centre = float(binding.position[axis])
            lo[axis] = min(lo[axis], centre - extent)
            hi[axis] = max(hi[axis], centre + extent)
        occupied = max(occupied, extent)
    if occupied <= 0.0 or not math.isfinite(lo[0]):
        return 0
    root = max(hi[axis] - lo[axis] for axis in range(3)) / 2.0
    return max(0, int(math.floor(math.log2(max(root / occupied, 1.0)))))


def _calibration_fingerprint_inputs(
    descriptor: SimulationDescriptor, direct: ProductRuntime
) -> dict[str, Any]:
    plan = _require_plan(descriptor)
    tree = _tree_config(descriptor)
    device = direct.device_context.device
    return {
        "schema": CALIBRATION_SCHEMA,
        "graph_hash": plan.graph_hash,
        "warp_version": direct.device_context.warp_version,
        "device_alias": str(device),
        "device_name": str(getattr(device, "name", str(device))),
        "device_arch": str(getattr(device, "arch", "")),
        "body_count": body_count(descriptor.bindings),
        "active_body_count": active_body_count(descriptor.bindings),
        "solved_state_fingerprint": solved_state_fingerprint(descriptor),
        "root_box_octave": _root_box_octave(descriptor),
        "precision": descriptor.precision.value,
        "dt": descriptor.timeline.seconds_per_frame,
        "substeps": descriptor.timeline.substeps,
        "softening": descriptor.units.softening,
        "theta": tree.theta,
        "max_depth": tree.max_depth,
        "leaf_capacity": tree.leaf_capacity,
        "stack_capacity": tree.stack_capacity,
        "node_capacity": tree.node_capacity
        or sparse_node_capacity(tree.count, tree.max_depth),
        # Part of the fingerprint, not decoration: the multipole order sets both the
        # traversal's per-cell cost and its accuracy, so a calibration measured against
        # one expansion cannot be reused to choose a backend for the other.
        "quadrupoles": tree.quadrupoles,
        "collision_mode": tree.collision_mode.value,
        "event_capacity": tree.event_capacity,
        "warmed_repetitions": WARMED_REPETITIONS,
        "representative_counts": _representative_counts(body_count(descriptor.bindings)),
    }


def _default_calibration_path(descriptor: SimulationDescriptor) -> Path:
    if descriptor.cache_root and not descriptor.cache_root.startswith("//"):
        return Path(descriptor.cache_root) / "auto-calibration.json"
    configured = os.environ.get("ASTRO_WARP_CALIBRATION_DIR")
    if configured:
        return Path(configured) / "auto-calibration.json"
    blender_data = os.environ.get("BLENDER_USER_DATAFILES")
    if blender_data:
        return Path(blender_data) / "astro_warp_phase0" / "auto-calibration.json"
    local_data = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(local_data) / "AstroWarp" / "auto-calibration.json"


def _load_calibration(
    path: Path, fingerprint: str
) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != CALIBRATION_SCHEMA:
            return None, "calibration_schema_mismatch"
        record = (payload.get("records") or {}).get(fingerprint)
        if record is None:
            return None, "calibration_fingerprint_miss"
        if record.get("fingerprint") != fingerprint:
            return None, "calibration_fingerprint_mismatch"
        return dict(record), ""
    except (OSError, ValueError, TypeError) as exc:
        return None, f"calibration_read_failed: {exc}"


def _store_calibration(path: Path, fingerprint: str, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") == CALIBRATION_SCHEMA:
                records = dict(payload.get("records") or {})
        except (OSError, ValueError, TypeError):
            records = {}
    records[fingerprint] = record
    payload = {"schema": CALIBRATION_SCHEMA, "records": records}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _store_calibration_or_record_error(
    path: Path, fingerprint: str, record: dict[str, Any]
) -> bool:
    try:
        _store_calibration(path, fingerprint, record)
        record["calibration_path"] = str(path)
        return True
    except OSError as exc:
        record["persistence_error"] = str(exc)
        return False


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
