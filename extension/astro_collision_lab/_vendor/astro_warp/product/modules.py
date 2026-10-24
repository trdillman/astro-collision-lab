"""Versioned registry of the graph modules a compiled plan may execute.

A graph node names a module; this registry owns that module's version. The version is
not a decoration: `GraphCompiler` resolves it, stamps it onto the compiled node and the
runtime step, and hashes it, so `graph_hash` — and therefore `plan_id` and every bake's
currentness axis — moves when a module's meaning moves. Before this existed a kernel
could be rewritten and every previously baked revision would still report itself current
(gap D8, PRODUCT-004 c2 and PRODUCT-008 c1).

The version is declared, not derived from source bytes. A content hash would mint a new
version for a comment edit and silently retire every bake, which is the opposite of the
requirement's "upgrade explicitly": the point of a declared version is that a human states
that the meaning changed. `implemented_by` names where to look when deciding.

Bump `version` when the module's observable behaviour changes:

- the numbers it produces for the same inputs, beyond the determinism its backend already
  declares in `BACKEND_DETERMINISM`,
- the parameters it reads, or how it interprets them,
- the contract of the outputs it publishes.

Do not bump for a refactor, a comment, a rename, or a performance change that leaves the
produced values and the read parameters identical.

A bump is a breaking change by construction. Every persisted plan pinned to the old
version is rejected by `ProductRuntimeFactory._validate_plan` until the graph is
recompiled or explicitly upgraded through `upgrade_graph_modules`, and every bake written
under the old plan hash reports HISTORICAL rather than current. That is the intended
blast radius; nothing is repinned silently.

Every module here is at its first declared version. Versioning starts with this registry,
so there is no earlier history to encode: `1.0.0` means "the behaviour these files had
when module versioning was introduced", not "unchanged since Phase 0".
"""

from __future__ import annotations

from dataclasses import dataclass

_NBODY = "src/astro_warp/product/nbody.py"
_FORCE_FIELDS = "src/astro_warp/product/force_fields.py"
_EMITTERS = "src/astro_warp/product/emitters.py"
_TREE = "src/astro_warp/product/tree.py"
_MODEL = "src/astro_warp/product/model.py"
_FACTORY = "src/astro_warp/product/runtime_factory.py"
_SAMPLING = "src/astro_warp/product/sampling.py"
_MATERIALS = "src/astro_warp/product/materials.py"
_IMPACT = "src/astro_warp/product/impact.py"
_GIANT_IMPACT = "src/astro_warp/product/giant_impact.py"
_GIANT_IMPACT_RUNTIME = "src/astro_warp/product/giant_impact_runtime.py"
_SPACESIM_IMPACT = "src/astro_warp/product/spacesim_impact.py"
_SPACESIM_IMPACT_RUNTIME = "src/astro_warp/product/spacesim_impact_runtime.py"
_SHEARING_BOX = "src/astro_warp/product/shearing_box.py"


@dataclass(frozen=True)
class ModuleSpec:
    """A graph module's declared version and the surfaces that decide it."""

    version: str
    implemented_by: tuple[str, ...]


MODULE_REGISTRY: dict[str, ModuleSpec] = {
    "astro.context.units": ModuleSpec("1.0.0", (_MODEL,)),
    "astro.inputs.bindings": ModuleSpec("1.0.0", (_MODEL,)),
    "astro.inputs.analytical_bodies": ModuleSpec(
        "1.0.0", ("src/astro_warp/product/orbit.py", _FACTORY)
    ),
    "astro.gravity.direct_all_pairs": ModuleSpec("1.0.0", (_NBODY,)),
    "astro.gravity.newtonian": ModuleSpec("1.0.0", (_NBODY,)),
    # 3.0.0 adds the quadrupole term to every accepted cell (ADR 0026). The opening-angle
    # test, the interaction set and the traversal order are untouched, so this is not a
    # different walk - but the accelerations it produces are different numbers for the same
    # inputs, and `astro.gravity.barnes_hut_tree` additionally reads a new `quadrupoles`
    # parameter. Both are bump conditions. `tree_newtonian` and `auto_newtonian` publish
    # those accelerations, so they move with it; `auto_select` reads the new parameter to
    # build the tree config. `astro.diagnostics.orbital_scalars` does not move: it computes
    # the same scalars from whatever state it is handed.
    "astro.gravity.barnes_hut_tree": ModuleSpec("3.0.0", (_TREE,)),
    "astro.gravity.tree_newtonian": ModuleSpec("2.0.0", (_TREE,)),
    "astro.gravity.auto_select": ModuleSpec("3.0.0", (_FACTORY,)),
    "astro.gravity.auto_newtonian": ModuleSpec("3.0.0", (_FACTORY, _NBODY, _TREE)),
    # The optional external-field stage (ADR 0045): FIELD-role bindings summed into the
    # same accelerations accumulator gravity writes. Parameter-free at the graph level --
    # the field physics lives on the bindings -- so its version moves only when the kernel
    # semantics move. Launched by both orbital backends, hence three deciding surfaces.
    "astro.force.external_field": ModuleSpec("1.0.0", (_FORCE_FIELDS, _NBODY, _TREE)),
    # The optional local shearing box (ADR 0202): Hill's equations in a frame turning with a
    # patch of a disk, shearing-periodic boundaries, and the mutual gravity summed over the box's
    # nearest images. A frame option of the Direct solver, spliced into its pipeline only when a
    # System asks for it, so the plan hash of every other System is untouched. It is a module of
    # its own, not a new parameter on `astro.integrators.kdk_leapfrog`, because what it changes is
    # the drift, the boundary and the force law together: naming kdk_leapfrog for that would make
    # the plan hash claim a box and an open space ran the same integrator. It does not bump
    # `astro.gravity.newtonian` or `astro.integrators.kdk_leapfrog`: they are not changed, and a
    # plan without this step runs exactly the kernels it always did.
    "astro.frame.shearing_box": ModuleSpec("1.0.0", (_SHEARING_BOX, _NBODY)),
    "astro.integrators.kdk_leapfrog": ModuleSpec("1.0.0", (_NBODY,)),
    "astro.collision.point": ModuleSpec("1.0.0", (_NBODY,)),
    # The optional emitter-activation stage (ADR 0037 section 3.6). Parameter-free at the
    # graph level for the same reason the external field is: the release schedule lives on
    # EMITTER-role bindings, so authoring or retiming an Emitter never moves the plan hash.
    # An EVENTS-family stage, and deliberately the sibling of `astro.collision.point`: both
    # are population changes applied after integration, one growing the live set and one
    # shrinking it. Launched by both orbital backends, hence three deciding surfaces.
    "astro.emitter.activate": ModuleSpec("1.0.0", (_EMITTERS, _NBODY, _TREE)),
    "astro.diagnostics.orbital_scalars": ModuleSpec("1.0.0", (_NBODY, _TREE)),
    # What did the bodies break into: friends-of-friends connectivity over the particle state.
    # Shared by the two realtime Solid pipelines below. The connectivity itself is
    # `impact.cluster_fragments`, so that file stays a deciding surface.
    "astro.diagnostics.fragments": ModuleSpec(
        "1.0.0", (_IMPACT, _GIANT_IMPACT_RUNTIME, _SPACESIM_IMPACT_RUNTIME)
    ),
    # The giant-impact pipeline (ADR 0094). New names throughout: its pressure is a
    # position-based constraint, not an equation of state; its gravity is a direct tiled sum;
    # and it advances a whole frame at a time, not at a Courant step. Naming another
    # pipeline's modules for those would make the plan hash claim a crater and a moon ran the
    # same physics. It uses `astro.diagnostics.fragments`, whose question - what did the body
    # break into - and whose answer are the same.
    "astro.inputs.giant_impact_bodies": ModuleSpec(
        "1.0.0", (_GIANT_IMPACT_RUNTIME, _SAMPLING)
    ),
    "astro.giant_impact.neighbor_grid": ModuleSpec("1.0.0", (_GIANT_IMPACT,)),
    "astro.giant_impact.pbf_pressure": ModuleSpec("1.0.0", (_GIANT_IMPACT,)),
    "astro.giant_impact.tiled_gravity": ModuleSpec("1.0.0", (_GIANT_IMPACT,)),
    "astro.giant_impact.moon_guide": ModuleSpec("1.0.0", (_GIANT_IMPACT,)),
    "astro.integrators.giant_impact_frame": ModuleSpec(
        "1.0.0", (_GIANT_IMPACT, _GIANT_IMPACT_RUNTIME)
    ),
    "astro.diagnostics.giant_impact_scalars": ModuleSpec("1.0.0", (_GIANT_IMPACT_RUNTIME,)),
    # The realtime rock-impact pipeline (ADR 0193). New names throughout, for the reason the
    # giant-impact comment gives: its pressure is SpaceSim's implicit incompressible solve, its
    # strength is a network of reference bonds that break, and its clock has two rates (outer
    # pressure steps, inner bond steps). Naming the giant-impact modules for those would
    # make the plan hash claim a rock and a planet ran the same physics. It reuses
    # `astro.diagnostics.fragments`, whose question and answer are the same.
    "astro.inputs.spacesim_bodies": ModuleSpec(
        "1.0.0", (_SPACESIM_IMPACT_RUNTIME, _SAMPLING)
    ),
    "astro.spacesim.neighbor_grid": ModuleSpec("1.0.0", (_SPACESIM_IMPACT,)),
    "astro.spacesim.iisph_pressure": ModuleSpec("1.0.0", (_SPACESIM_IMPACT,)),
    # 2.0.0 (ADR 0204): bonds yield instead of breaking at the yield strain, soften with shock
    # heat, and are cut past the critical strain on their current length or 2 h. Same inputs,
    # different bodies: every bake of the 1.x model is HISTORICAL. 3.0.0 (ADR 0204 amended): the
    # energy a yielding or cut bond gives up is heat the yield softens with, kept, not a glow.
    "astro.spacesim.bond_strength": ModuleSpec("3.0.0", (_SPACESIM_IMPACT, _MATERIALS)),
    "astro.spacesim.tiled_gravity": ModuleSpec("1.0.0", (_SPACESIM_IMPACT,)),
    # 2.0.0 (ADR 0204 amended): a frame's outer count is the planned one or more, from the fastest
    # particle at its start (SpaceSim's auto_substep).
    "astro.integrators.spacesim_two_rate": ModuleSpec(
        "2.0.0", (_SPACESIM_IMPACT, _SPACESIM_IMPACT_RUNTIME)
    ),
    "astro.diagnostics.spacesim_scalars": ModuleSpec("1.0.0", (_SPACESIM_IMPACT_RUNTIME,)),
    "astro.outputs.contract": ModuleSpec("1.0.0", (_MODEL,)),
    "astro.outputs.immutable_bake": ModuleSpec("1.0.0", ("src/astro_warp/product/bake.py",)),
}


#: The graph modules of the detailed impact engine: compressible particle hydrodynamics with the
#: solid constitutive stages inside it, and the layered initial state it started from. The
#: engine was deleted on the owner's order of 2026-10-13, so no build publishes these any more.
#: They are listed, rather than simply left unknown, because a System saved before the deletion
#: still names them, and "unknown graph module 'astro.sph.hydro_forces'" tells its author
#: nothing they can act on (see :data:`RETIRED_ENGINE_MESSAGE`).
RETIRED_ENGINE_MODULES = frozenset(
    {
        "astro.inputs.sph_particles",
        "astro.sph.neighbor_hash_grid",
        "astro.sph.hydro_forces",
        "astro.sph.self_gravity",
        "astro.integrators.sph_kdk_courant",
        "astro.diagnostics.sph_scalars",
        "astro.inputs.layered_bodies",
        "astro.solids.stress_rate",
        "astro.solids.plasticity",
        "astro.solids.damage",
        "astro.solids.porosity",
        "astro.solids.condensed_eos",
        "astro.solids.stress_force",
    }
)

#: The runtime backends of the same engine. A plan compiled before the deletion names one of
#: these in its ``backend``, and the factory refuses it with the same sentence.
RETIRED_ENGINE_BACKENDS = frozenset({"sph", "impact", "solids"})

#: What an author is told when a saved System or plan names the deleted engine. Plain words,
#: and the remedy is the one that works: pressing Play compiles the authored System again, and
#: the compile now picks one of the realtime Solid engines.
RETIRED_ENGINE_MESSAGE = (
    "This System was made with the retired detailed impact engine; "
    "press Play to rebuild it on the realtime engines"
)


class UnknownModuleError(ValueError):
    """Raised when a graph names a module this build does not publish."""


class RetiredModuleError(UnknownModuleError):
    """Raised when a graph names a module of the deleted detailed impact engine."""


def module_version(module: str) -> str:
    """The installed version of ``module``, or raise if this build has no such module."""

    spec = MODULE_REGISTRY.get(module)
    if spec is None:
        if module in RETIRED_ENGINE_MODULES:
            raise RetiredModuleError(RETIRED_ENGINE_MESSAGE)
        raise UnknownModuleError(f"unknown graph module {module!r}")
    return spec.version


def module_versions() -> dict[str, str]:
    """Every published module and its installed version."""

    return {module: spec.version for module, spec in MODULE_REGISTRY.items()}
