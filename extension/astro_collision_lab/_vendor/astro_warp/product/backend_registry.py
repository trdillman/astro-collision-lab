"""One ``BackendSpec`` per engine: what a backend *is*, in one place (ADR 0208).

What an engine is used to be written in parallel tables inside ``runtime_factory.py`` (its
pipeline, its allowed modules, its runtime classes, whether it is a particle solver) and, further
out, in the colour and memory consumers. Adding or changing a backend meant editing all of them
and hoping nothing drifted. This module holds the facts once, as frozen data validated when it is
built; ``runtime_factory`` derives its public tables from it and keeps their names and values.

Device-free and Warp-free at import, and it never imports ``runtime_factory`` (the factory
imports this). It imports the runtime classes themselves because naming them is part of what a
backend is, and those modules are Warp-free at import too.

The spec is deliberately data-only for now. How a backend reads its config, sizes its device
memory and lays out its sample move onto it in later steps of ADR 0208, each with its own
conformance test; the dataclass is a plain frozen record so those fields can be added without
touching the lookups.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .giant_impact_runtime import GiantImpactRuntime
from .memory_admission import (
    auto_memory_parts,
    direct_memory_parts,
    giant_impact_memory_parts,
    spacesim_impact_memory_parts,
    tree_memory_parts,
)
from .model import RuntimePlan, StageFamily
from .nbody import DirectNBodyRuntime
from .spacesim_impact_runtime import SpaceSimImpactRuntime
from .tree import BarnesHutRuntime

Pipeline = tuple[tuple[StageFamily, str], ...]

#: ``(body_count, scalar_bytes, plan, anchors_pinned) -> [(part name, bytes)]``: the device arrays
#: a backend's runtime allocates for the scene, before the parts every backend shares (the fixed
#: workspace and the display staging, which ``memory_admission`` adds).
MemoryParts = Callable[[int, int, RuntimePlan, bool], list[tuple[str, int]]]

#: What every executable pipeline has in common: the context it reads and the output it writes.
#: Everything else in a pipeline is a solver's own, which is why ``BackendSpec.modules`` excludes
#: these two.
COMMON_RUNTIME_MODULES = frozenset({"astro.context.units", "astro.outputs.immutable_bake"})

#: The optional steps a plan can splice into a backend's pipeline, and where each lands. The
#: anchors are the facts ``runtime_factory._expected_pipeline`` splices against.
OPTIONAL_STEP_NAMES = frozenset({"external_field", "emitter", "shearing_box"})
EMITTER_ANCHOR = (StageFamily.EVENTS, "astro.collision.point")
SHEARING_BOX_ANCHOR = (StageFamily.FORCE, "astro.gravity.newtonian")


class Family(StrEnum):
    """Which kind of state a backend integrates."""

    #: Analytic point masses (direct, tree, auto): force fields, emitters and pinning act on them.
    ORBITAL = "orbital"
    #: A sampled particle cloud integrated as one solid (the two impact engines). None of the
    #: optional steps applies to them (ADR 0045, 0047, 0158).
    PARTICLE = "particle"


class Assembly(StrEnum):
    """How ``ProductRuntimeFactory.create`` builds a backend's runtime.

    The factory dispatches on this, not on the backend's name, so a new particle backend is
    assembled by registering it (and its assembler in ``runtime_factory``) with no new branch.
    """

    DIRECT = "direct"
    TREE = "tree"
    #: Calibrates a Direct and a Tree runtime against each other and keeps one.
    AUTO = "auto"
    #: Samples a layered particle layout and loads it into the engine (the two impact runtimes).
    PARTICLE = "particle"


@dataclass(frozen=True)
class ColorFacts:
    """What the Color By modes may read from a backend (ADR 0208 A3 derives consumers from this)."""

    #: A D13 ``SOLID`` role exists on this backend.
    d13_solid: bool = False
    #: It publishes a specific internal energy channel.
    internal_energy: bool = False
    #: It publishes a per-particle damage channel.
    publishes_damage: bool = False
    #: It carries a persistent material id per particle.
    persistent_material_ids: bool = False


@dataclass(frozen=True)
class Capabilities:
    """Static upper bounds on what a backend's runtime offers.

    Upper bounds because the answer for a live instance can be narrower (the orbital runtimes
    stop being resumable when an emitter is bound); the bound is what is true of the class, and
    for ``auto`` it is the intersection of the two classes it can resolve to.
    """

    #: Offers ``apply_parameters`` (in-place solver-parameter adoption, ``Reparameterizable``).
    reparameterizable: bool = False
    #: Offers ``fragment_report``.
    reports_fragments: bool = False
    #: The channels ``display_channels()`` can serve, in order.
    display_channel_names: tuple[str, ...] = ("positions",)
    #: Every runtime class serves the resumable-frame protocol (``serves_resumable_frames``).
    resumable_upper_bound: bool = False

    def __post_init__(self) -> None:
        if "positions" not in self.display_channel_names:
            raise ValueError("a backend must serve a positions display channel")
        if len(set(self.display_channel_names)) != len(self.display_channel_names):
            raise ValueError("display channel names must be unique")


@dataclass(frozen=True)
class BackendSpec:
    """The static facts about one engine."""

    name: str
    family: Family
    #: The exact executable pipeline, with none of the optional steps.
    pipeline: Pipeline
    #: The modules that are this backend's own: the pipeline's modules minus the common two.
    modules: frozenset[str]
    #: Optional steps this backend accepts. ``external_field`` maps to the index it is spliced at;
    #: ``emitter`` and ``shearing_box`` map to None, because they are spliced relative to an anchor
    #: step (``EMITTER_ANCHOR``, ``SHEARING_BOX_ANCHOR``) rather than at a number. A step that is
    #: absent is unsupported, and the factory refuses a plan that asks for it.
    optional_steps: Mapping[str, int | None] = field(hash=False)
    #: The runtime classes ``create`` can construct (both orbital ones for ``auto``).
    runtime_classes: tuple[type, ...]
    #: How ``create`` builds it. A particle backend is exactly one with ``Assembly.PARTICLE``.
    assembly: Assembly
    color: ColorFacts
    capabilities: Capabilities
    #: What the backend's runtime allocates on the device, by part. Required: a registered backend
    #: with no memory model would be admitted by the pre-flight at any size (ADR 0195, 0208 A4).
    memory_parts: MemoryParts

    def __post_init__(self) -> None:
        if not callable(self.memory_parts):
            raise ValueError(f"{self.name!r} needs a memory model (memory_parts)")
        if not self.name or self.name != self.name.lower():
            raise ValueError(f"backend name {self.name!r} must be non-empty and lower case")
        if not self.pipeline:
            raise ValueError(f"{self.name!r} has an empty pipeline")
        if len(set(self.pipeline)) != len(self.pipeline):
            raise ValueError(f"{self.name!r} pipeline repeats a step")
        pipeline_modules = {module for _, module in self.pipeline}
        if not pipeline_modules >= COMMON_RUNTIME_MODULES:
            raise ValueError(f"{self.name!r} pipeline lacks the common context/output modules")
        if frozenset(self.modules) != pipeline_modules - COMMON_RUNTIME_MODULES:
            raise ValueError(f"{self.name!r} modules are not its pipeline's modules")
        if not self.runtime_classes or not all(isinstance(c, type) for c in self.runtime_classes):
            raise ValueError(f"{self.name!r} needs at least one runtime class")
        steps = dict(self.optional_steps)
        unknown = sorted(set(steps) - OPTIONAL_STEP_NAMES)
        if unknown:
            raise ValueError(f"{self.name!r} names unknown optional steps: {', '.join(unknown)}")
        if (self.assembly is Assembly.PARTICLE) != (self.family is Family.PARTICLE):
            raise ValueError(f"{self.name!r} assembly {self.assembly} does not fit its family")
        if self.family is Family.PARTICLE and steps:
            raise ValueError(f"{self.name!r} is a particle backend and takes no optional steps")
        index = steps.get("external_field")
        if "external_field" in steps and not (
            isinstance(index, int) and 0 <= index <= len(self.pipeline)
        ):
            raise ValueError(f"{self.name!r} splices external_field outside its pipeline")
        for step, anchor in (("emitter", EMITTER_ANCHOR), ("shearing_box", SHEARING_BOX_ANCHOR)):
            if step in steps and (steps[step] is not None or anchor not in self.pipeline):
                raise ValueError(f"{self.name!r} cannot splice {step}: its anchor step is absent")
        object.__setattr__(self, "modules", frozenset(self.modules))
        object.__setattr__(self, "optional_steps", MappingProxyType(steps))
        object.__setattr__(self, "pipeline", tuple(self.pipeline))
        object.__setattr__(self, "runtime_classes", tuple(self.runtime_classes))


_CONTEXT = (StageFamily.CONTEXT, "astro.context.units")
_BAKE = (StageFamily.OUTPUT, "astro.outputs.immutable_bake")
_ORBITAL_HEAD: Pipeline = (_CONTEXT, (StageFamily.INITIAL_STATE, "astro.inputs.analytical_bodies"))
_ORBITAL_TAIL: Pipeline = (
    (StageFamily.INTEGRATION, "astro.integrators.kdk_leapfrog"),
    (StageFamily.EVENTS, "astro.collision.point"),
    (StageFamily.DIAGNOSTICS, "astro.diagnostics.orbital_scalars"),
    _BAKE,
)
#: The point-mass solvers' display channels. Direct and tree serve the same keys on purpose: Auto
#: swaps between them on body count alone, so a channel only one offered would show the swap.
_ORBITAL_CHANNELS = ("positions", "velocities", "masses", "radii", "active", "stable_ids")
_ORBITAL_CAPABILITIES = Capabilities(
    reparameterizable=True,
    reports_fragments=False,
    display_channel_names=_ORBITAL_CHANNELS,
    resumable_upper_bound=True,
)


def _orbital(
    name: str,
    spatial: str,
    force: str,
    classes: tuple[type, ...],
    assembly: Assembly,
    memory_parts: MemoryParts,
    *,
    field_index: int,
    shearing_box: bool = False,
) -> BackendSpec:
    pipeline = (
        _ORBITAL_HEAD
        + (
            (StageFamily.SPATIAL_SERVICE, spatial),
            (StageFamily.FORCE, force),
        )
        + _ORBITAL_TAIL
    )
    optional: dict[str, int | None] = {"external_field": field_index, "emitter": None}
    if shearing_box:
        optional["shearing_box"] = None
    return BackendSpec(
        name=name,
        family=Family.ORBITAL,
        pipeline=pipeline,
        modules=frozenset({module for _, module in pipeline}) - COMMON_RUNTIME_MODULES,
        optional_steps=optional,
        runtime_classes=classes,
        assembly=assembly,
        color=ColorFacts(),
        capabilities=_ORBITAL_CAPABILITIES,
        memory_parts=memory_parts,
    )


_DIRECT = _orbital(
    "direct",
    "astro.gravity.direct_all_pairs",
    "astro.gravity.newtonian",
    (DirectNBodyRuntime,),
    Assembly.DIRECT,
    direct_memory_parts,
    field_index=3,
    shearing_box=True,
)
_TREE = _orbital(
    "tree",
    "astro.gravity.barnes_hut_tree",
    "astro.gravity.tree_newtonian",
    (BarnesHutRuntime,),
    Assembly.TREE,
    tree_memory_parts,
    field_index=3,
)
#: ``auto`` resolves to one of the two orbital runtimes at create time, so it is both of them
#: here: a capability claimed for ``auto`` is only true if it is true of whichever one the
#: selector picks. Its field step sorts after the force-sorted ``auto_newtonian`` node (index 4).
_AUTO = _orbital(
    "auto",
    "astro.gravity.auto_select",
    "astro.gravity.auto_newtonian",
    (DirectNBodyRuntime, BarnesHutRuntime),
    Assembly.AUTO,
    auto_memory_parts,
    field_index=4,
)

# Three FORCE steps in the order a frame runs them (ADR 0094): the pressure substeps, gravity
# once around them, and the moon guide over the debris gravity has moved. No EVENTS step: bodies
# meet through the pressure constraint, not a point collision.
_GIANT_IMPACT_PIPELINE: Pipeline = (
    _CONTEXT,
    (StageFamily.INITIAL_STATE, "astro.inputs.giant_impact_bodies"),
    (StageFamily.SPATIAL_SERVICE, "astro.giant_impact.neighbor_grid"),
    (StageFamily.FORCE, "astro.giant_impact.pbf_pressure"),
    (StageFamily.FORCE, "astro.giant_impact.tiled_gravity"),
    (StageFamily.FORCE, "astro.giant_impact.moon_guide"),
    (StageFamily.INTEGRATION, "astro.integrators.giant_impact_frame"),
    (StageFamily.DIAGNOSTICS, "astro.diagnostics.giant_impact_scalars"),
    (StageFamily.DIAGNOSTICS, "astro.diagnostics.fragments"),
    _BAKE,
)
# Three FORCE steps in the order a frame runs them (ADR 0193): pressure once per outer step, the
# bonds that hold the bodies together, and gravity once around the frame. No EVENTS step.
_SPACESIM_IMPACT_PIPELINE: Pipeline = (
    _CONTEXT,
    (StageFamily.INITIAL_STATE, "astro.inputs.spacesim_bodies"),
    (StageFamily.SPATIAL_SERVICE, "astro.spacesim.neighbor_grid"),
    (StageFamily.FORCE, "astro.spacesim.iisph_pressure"),
    (StageFamily.FORCE, "astro.spacesim.bond_strength"),
    (StageFamily.FORCE, "astro.spacesim.tiled_gravity"),
    (StageFamily.INTEGRATION, "astro.integrators.spacesim_two_rate"),
    (StageFamily.DIAGNOSTICS, "astro.diagnostics.spacesim_scalars"),
    (StageFamily.DIAGNOSTICS, "astro.diagnostics.fragments"),
    _BAKE,
)


def _modules_of(pipeline: Pipeline) -> frozenset[str]:
    return frozenset({module for _, module in pipeline}) - COMMON_RUNTIME_MODULES


_GIANT_IMPACT = BackendSpec(
    name="giant_impact",
    family=Family.PARTICLE,
    pipeline=_GIANT_IMPACT_PIPELINE,
    modules=_modules_of(_GIANT_IMPACT_PIPELINE),
    optional_steps={},
    runtime_classes=(GiantImpactRuntime,),
    assembly=Assembly.PARTICLE,
    color=ColorFacts(d13_solid=True, internal_energy=True, persistent_material_ids=True),
    capabilities=Capabilities(
        reparameterizable=False,
        reports_fragments=True,
        display_channel_names=GiantImpactRuntime._DISPLAY_CHANNELS,
        resumable_upper_bound=True,
    ),
    memory_parts=giant_impact_memory_parts,
)
_SPACESIM_IMPACT = BackendSpec(
    name="spacesim_impact",
    family=Family.PARTICLE,
    pipeline=_SPACESIM_IMPACT_PIPELINE,
    modules=_modules_of(_SPACESIM_IMPACT_PIPELINE),
    optional_steps={},
    runtime_classes=(SpaceSimImpactRuntime,),
    assembly=Assembly.PARTICLE,
    color=ColorFacts(
        d13_solid=True, internal_energy=True, publishes_damage=True, persistent_material_ids=True
    ),
    capabilities=Capabilities(
        reparameterizable=False,
        reports_fragments=True,
        display_channel_names=SpaceSimImpactRuntime._DISPLAY_CHANNELS,
        resumable_upper_bound=True,
    ),
    memory_parts=spacesim_impact_memory_parts,
)

#: Every backend ``ProductRuntimeFactory.create`` can build, in the order the factory's tables
#: have always listed them.
BACKENDS: Mapping[str, BackendSpec] = MappingProxyType(
    {spec.name: spec for spec in (_DIRECT, _TREE, _AUTO, _GIANT_IMPACT, _SPACESIM_IMPACT)}
)


def backend_spec(name: str) -> BackendSpec:
    """The spec for ``name``; ``KeyError`` for a backend that is not registered.

    The factory wraps the error as its own ``ProductRuntimeError``, in the artist's words.
    """

    return BACKENDS[name]


def find_backend(name: str) -> BackendSpec | None:
    """The spec for ``name``, or None: for callers where an unknown backend is not an error."""

    return BACKENDS.get(name)
