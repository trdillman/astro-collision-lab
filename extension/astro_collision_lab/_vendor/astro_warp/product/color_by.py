"""The D13 Color By availability matrix and its exact display formulas.

Color By picks which scalar a System's published points carry as ``astro_color_value``. The
temptation it creates is the reason this module exists: five modes in a dropdown look like
five equally real physical quantities, and four of them are only real on some roles and some
backends. A dropdown that always produces *a number* will happily produce a wrong one --
Speed substituted for a missing Damage, zero for an absent channel, a smooth gradient that
came from an index rather than from physics. Every one of those reads as a working feature.

So this module answers two questions and refuses to blur them:

  availability   given a role, a backend, and what state that output partition actually
                 committed, is this mode derivable *at all*? The answer is a value carrying
                 either the formula's name or an actionable refusal naming role, backend,
                 mode, and the missing source.
  value          given sources that are genuinely present, what exactly is the number?

What this module deliberately cannot do is as important as what it does. There is no code
path that produces a Temperature value, no fallback that substitutes Speed, no default of
zero, and no way to derive Material from a table row index. Those are not tested-against
behaviours; they are absent, so a later change has to add them deliberately rather than
reach them by accident.

**Color By is display-only.** D13: it is never part of ``SimulationDescriptor``, the solver
fingerprint, invalidation classification, ScrubCache identity, or BakeRevision identity.
Changing it performs no compile, initialize, step, restore, replay, target scheduling, cache
rewrite, or bake mutation. Nothing here imports a runtime or a cache, which is the cheapest
way to keep that true.

Pure Python and pure arithmetic: no ``bpy``, no Warp, no device. The live path derives the
same scalars on ``cuda:0`` at the already bounded stable sample indices; this is the
statement of *what* it derives, provable on any machine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .backend_registry import BACKENDS, Family
from .model import BindingDescriptor, BindingRole, BodyRepresentation


class ColorByMode(StrEnum):
    """D13's five signed entries. All five are offered; not all five are ever available.

    A sixth, ``MASS``, was offered by the shipped enum from Slice 3 until 2026-08-28 and is
    retired by ``adr/0044-color-by-enum-retirement-of-mass.md``. It appears in no accepted
    matrix and never coloured anything; the ADR records why it was neither promoted nor kept.
    """

    SPEED = "SPEED"
    DAMAGE = "DAMAGE"
    MATERIAL = "MATERIAL"
    TEMPERATURE = "TEMPERATURE"
    ENERGY = "ENERGY"


#: The name published on the point domain, exactly. Shipped node groups and materials consume
#: this attribute and nothing else, so the string is a contract rather than a label.
COLOR_ATTRIBUTE = "astro_color_value"

#: Backends that carry a specific internal energy, and therefore take D13's second Energy
#: definition. Direct and Tree are gravity-only: they have no internal energy to add, and
#: adding a zero for one would silently redefine the quantity across backends.
INTERNAL_ENERGY_BACKENDS = frozenset(
    name for name, spec in BACKENDS.items() if spec.color.internal_energy
)

#: Backends whose bodies are D13 ``SOLID`` members (the two impact engines), in registry order.
SOLID_BACKENDS = tuple(name for name, spec in BACKENDS.items() if spec.color.d13_solid)

#: Backend strings D13's table calls "Direct/Tree".
#:
#: ``auto`` is in this set and is not an extension of the table. The role System's Solver enum
#: is ``Auto / Direct / Barnes-Hut``, ``Auto`` is its **shipped default**, and it compiles
#: ``auto_orbital_graph`` -- an orbital graph with exactly the committed velocity Direct and
#: Tree have. Reading D13's "Direct/Tree" as the two literal strings would make every
#: default-solver System refuse every mode, which is plainly not what the table describes.
ORBITAL_BACKENDS = tuple(name for name, spec in BACKENDS.items() if spec.family is Family.ORBITAL)

#: The raw runtime channels (names a backend's ``display_channels()`` serves) that Color By reads
#: when a bake carries colour: velocity for Speed and Energy, damage, the persistent material id,
#: and the specific internal energy. Sorted, because the bake compares them as a sorted tuple.
#: The bake path (``runtime_factory``) carries exactly these and no others, so a source added here
#: without a way to carry it fails ``test_raw_color_bake_channels`` instead of baking without it.
RAW_COLOR_BAKE_CHANNELS = ("damage", "internal_energies", "material_ids", "velocities")

#: What Energy is *not*, stated once so the tooltip, the proof, and the exported attribute
#: metadata quote the same words. D13 calls Energy "resolved specific energy for display";
#: each exclusion below is a quantity a reader could otherwise reasonably assume was included.
ENERGY_EXCLUSIONS = (
    "not total System energy",
    "not per-particle gravitational potential energy",
    "not elastic-strain energy",
)


@dataclass(frozen=True)
class PartitionSources:
    """What one geometry-bearing output partition can *authoritatively* offer.

    Every field is a statement about committed state, never about intent. ``damage_active``
    is whether the damage source is running, not whether the user asked for damage; the
    distinction is the whole point, because a mode is available exactly when the number
    behind it exists.

    The two material fields are separate on purpose, and it is the subtlety most likely to
    be collapsed by a later edit. A registry can be validated while an individual particle
    still has no persistent ID -- and the ID that particle *does* have in a transient
    material table is a row index that changes when the table is rebuilt. Colouring by it
    produces a plausible, stable-looking image that means nothing, so it cannot be reachable
    from here.
    """

    role: str
    backend: str
    #: A committed velocity vector exists for these points. Speed and Energy both need it.
    has_committed_velocity: bool = False
    #: The engine publishes a damage number for these points.
    damage_active: bool = False
    #: The damage source is active (`ACTIVE`), or `NONE` where the engine publishes none.
    damage_model: str = "NONE"
    #: The material registry validated for this partition.
    has_material_registry: bool = False
    #: Each particle carries a persistent registry/shell ID -- never a transient table row.
    has_persistent_material_id: bool = False
    #: A committed specific internal energy exists (the giant and rock Solid engines only).
    has_specific_internal_energy: bool = False
    #: Names this partition in a refusal, so a multi-partition message is actionable.
    name: str = "output"


@dataclass(frozen=True)
class Availability:
    """Whether a mode is derivable here, and either how or why not.

    `formula` and `refusal` are mutually exclusive and one is always set, so there is no
    third state in which a caller might reasonably invent a value.
    """

    mode: ColorByMode
    available: bool
    formula: str | None = None
    refusal: str | None = None

    def __post_init__(self) -> None:
        if self.available and (self.formula is None or self.refusal is not None):
            raise ValueError("an available mode must carry a formula and no refusal")
        if not self.available and (self.refusal is None or self.formula is not None):
            raise ValueError("an unavailable mode must carry a refusal and no formula")


#: Role/backend pairs each mode is defined on, before per-partition source checks. A pair
#: absent from a row is refused on role/backend grounds alone; a pair present still has to
#: prove it committed the state the formula reads.
#
#: A giant-impact Solid (ADR 0094) takes Speed, Energy and Material but not Damage: it carries
#: a velocity, a specific internal energy and a persistent shell ID, and it runs no damage
#: model, so there is no damage number for Damage to show. A rock-impact Solid (ADR 0193) takes
#: all four: its bonds break, and the fraction of a particle's bonds that have broken is the
#: damage number; its internal energy is the viscous shock heat.
_SUPPORTED_PAIRS: dict[ColorByMode, frozenset[tuple[str, str]]] = {
    ColorByMode.SPEED: frozenset(
        {(role, backend) for role in ("BODY", "CLOUD") for backend in ORBITAL_BACKENDS}
        | {("SOLID", backend) for backend in SOLID_BACKENDS}
    ),
    ColorByMode.ENERGY: frozenset(
        {(role, backend) for role in ("BODY", "CLOUD") for backend in ORBITAL_BACKENDS}
        | {("SOLID", backend) for backend in SOLID_BACKENDS}
    ),
    ColorByMode.DAMAGE: frozenset(
        {("SOLID", name) for name, spec in BACKENDS.items() if spec.color.publishes_damage}
    ),
    ColorByMode.MATERIAL: frozenset(
        {("SOLID", name) for name, spec in BACKENDS.items() if spec.color.persistent_material_ids}
    ),
    # Deliberately empty rather than absent: Temperature is offered in the enum and is
    # available nowhere, and an empty set says that in the same vocabulary as the other rows.
    ColorByMode.TEMPERATURE: frozenset(),
}

#: The one refusal that is about the product rather than about a partition. D13: no accepted
#: temperature state and no EOS or heat-capacity mapping exists in v1, and authored initial
#: Temperature and internal energy are *not* relabelled as temperature. Naming the alias
#: explicitly is the point -- internal energy is right there, in the same record, on the two
#: backends where someone would most plausibly reach for it.
TEMPERATURE_REFUSAL = (
    "Temperature is unavailable in v1 on every role and backend: no accepted temperature "
    "state exists, and no EOS or heat-capacity mapping to one exists. Authored initial "
    "Temperature and specific internal energy are different quantities and are not "
    "relabelled as temperature."
)


def _pair_refusal(mode: ColorByMode, sources: PartitionSources) -> str:
    """The role/backend refusal, naming the partition it is about.

    The partition name is not decoration. Two outputs on the same role and backend produce
    byte-identical messages without it, so a multi-partition System's HUD remedy would list
    the same sentence twice and leave the user unable to tell which output to fix -- which
    is the one situation the transactional refusal exists to make legible.
    """

    return (
        f"{sources.name}: {mode.value} is not available for role {sources.role} on the "
        f"{sources.backend} backend."
    )


def availability(mode: ColorByMode, sources: PartitionSources) -> Availability:
    """Whether `mode` is derivable for one output partition, and how or why not.

    Checked in two stages, and the order carries the better message: role/backend first,
    because "Damage on a Cloud" is a category error and saying "no damage source" about it
    would send the user looking for a switch that does not apply to them; then the specific
    committed source, which is the case where there *is* a switch.
    """

    mode = ColorByMode(mode)

    if mode is ColorByMode.TEMPERATURE:
        return Availability(mode, available=False, refusal=TEMPERATURE_REFUSAL)

    pair = (sources.role, sources.backend)
    if pair not in _SUPPORTED_PAIRS[mode]:
        return Availability(mode, available=False, refusal=_pair_refusal(mode, sources))

    if mode is ColorByMode.SPEED:
        if not sources.has_committed_velocity:
            return Availability(
                mode,
                available=False,
                refusal=(
                    f"Speed needs a committed velocity vector, and {sources.name} has none "
                    f"({sources.role} on {sources.backend})."
                ),
            )
        return Availability(mode, available=True, formula="sqrt(vx^2 + vy^2 + vz^2)")

    if mode is ColorByMode.ENERGY:
        if not sources.has_committed_velocity:
            return Availability(
                mode,
                available=False,
                refusal=(
                    f"Energy needs a committed velocity vector, and {sources.name} has none "
                    f"({sources.role} on {sources.backend})."
                ),
            )
        if sources.backend in INTERNAL_ENERGY_BACKENDS:
            if not sources.has_specific_internal_energy:
                # Refused rather than silently degraded to the kinetic-only definition: the
                # same mode would then mean two different quantities depending on state the
                # user cannot see, and the discriminator is that a Solid's Energy differs from
                # kinetic-only by exactly the carried internal energy.
                return Availability(
                    mode,
                    available=False,
                    refusal=(
                        f"Energy on the {sources.backend} backend is kinetic plus specific "
                        f"internal energy, and {sources.name} carries no committed internal "
                        "energy. It is not silently reduced to the kinetic-only definition."
                    ),
                )
            return Availability(
                mode, available=True, formula="0.5 * dot(v, v) + specific_internal_energy"
            )
        return Availability(mode, available=True, formula="0.5 * dot(v, v)")

    if mode is ColorByMode.DAMAGE:
        if not sources.damage_active:
            return Availability(
                mode,
                available=False,
                refusal=(
                    f"Damage needs an active damage source on the impact graph, and "
                    f"{sources.name} has none."
                ),
            )
        if sources.damage_model.upper() == "NONE":
            return Availability(
                mode,
                available=False,
                refusal=(
                    f"Damage needs the selected material to have a damage model other than "
                    f"NONE, and {sources.name}'s material has NONE."
                ),
            )
        return Availability(mode, available=True, formula="committed damage scalar")

    if not sources.has_material_registry:
        return Availability(
            mode,
            available=False,
            refusal=(
                f"Material needs a validated material registry, and {sources.name} has none."
            ),
        )
    if not sources.has_persistent_material_id:
        return Availability(
            mode,
            available=False,
            refusal=(
                f"Material needs a persistent registry/shell ID per particle, and "
                f"{sources.name} has none. A transient material-table row index is not a "
                "substitute: it changes when the table is rebuilt, so the colouring would "
                "be stable-looking and meaningless."
            ),
        )
    return Availability(mode, available=True, formula="float(persistent material_id)")


def speed(velocity: tuple[float, float, float]) -> float:
    """`sqrt(vx^2 + vy^2 + vz^2)` from the committed velocity vector."""
    vx, vy, vz = (float(component) for component in velocity)
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def specific_energy(
    velocity: tuple[float, float, float],
    *,
    specific_internal_energy: float | None = None,
) -> float:
    """D13's two Energy definitions, chosen by whether internal energy is carried.

    `specific_internal_energy=None` is the Direct/Tree definition; a number is the Solid
    engines' one. `None` and `0.0` are different arguments on purpose -- a backend that carries
    internal energy and happens to be at zero is the kinetic value *by measurement*, whereas a
    backend with no internal energy at all is a different formula. Collapsing them would make
    the discriminator unprovable, since it rests on the two differing.
    """

    vx, vy, vz = (float(component) for component in velocity)
    kinetic = 0.5 * (vx * vx + vy * vy + vz * vz)
    if specific_internal_energy is None:
        return kinetic
    return kinetic + float(specific_internal_energy)


def damage_value(committed_damage: float) -> float:
    """The committed damage scalar, unchanged.

    A pass-through with a name, because "unchanged" is the requirement: no clamp, no
    normalisation to 0..1, no remap to make the viewport prettier. Any of those would make
    the displayed number disagree with the solver's, and the installed proof compares them.
    """

    return float(committed_damage)


def material_value(persistent_material_id: int) -> float:
    """`float(material_id)` for the *persistent* registry or shell ID.

    Rejects a negative ID because that is how an absent persistent ID reads in the runtime's
    arrays, and colouring by "no material" as if it were material -1 is precisely the
    fabricated physics D13 forbids.
    """

    if isinstance(persistent_material_id, bool) or not isinstance(persistent_material_id, int):
        raise TypeError(
            "persistent_material_id must be an int registry/shell ID, not "
            f"{type(persistent_material_id).__name__}"
        )
    if persistent_material_id < 0:
        raise ValueError(
            f"persistent_material_id {persistent_material_id} is not a valid ID; a particle "
            "with no persistent material must refuse, not colour as a negative material"
        )
    return float(persistent_material_id)


@dataclass(frozen=True)
class RecolorPlan:
    """The transactional decision for a whole System, across every output partition.

    One object rather than a per-partition list because the decision is not per-partition:
    D13 requires that a System with multiple geometry-bearing outputs validates all of them
    before updating any, and that one unsupported partition refuses the *whole* recolor with
    the previous ``astro_color_value`` preserved. A caller handed per-partition answers would
    have to remember to do that, and the first caller to forget produces a System where some
    outputs show the new mode and some silently show the old one -- with no way for the user
    to tell which is which.
    """

    mode: ColorByMode
    accepted: bool
    formula: str | None
    #: Every refusal, in partition order, so the HUD remedy names all of them rather than
    #: sending the user round the loop once per broken partition.
    refusals: tuple[str, ...] = ()

    def hud_remedy(self) -> str:
        """The persistent HUD message. Empty when the recolor was accepted."""
        if self.accepted:
            return ""
        return (
            f"Color By {self.mode.value} refused; the previous {COLOR_ATTRIBUTE} is unchanged. "
            + " ".join(self.refusals)
        )


def recolor_plan(
    mode: ColorByMode, partitions: tuple[PartitionSources, ...] | list[PartitionSources]
) -> RecolorPlan:
    """Validate every partition, then accept or refuse the recolor as one transaction."""

    mode = ColorByMode(mode)
    ordered = tuple(partitions)
    if not ordered:
        # Not an empty success. A System with no geometry-bearing output has nothing to
        # recolor, and reporting that as accepted would let a caller believe it published an
        # attribute it never wrote.
        return RecolorPlan(
            mode,
            accepted=False,
            formula=None,
            refusals=("no geometry-bearing output partition to recolor.",),
        )

    results = [availability(mode, partition) for partition in ordered]
    refusals = tuple(result.refusal for result in results if result.refusal is not None)
    if refusals:
        return RecolorPlan(mode, accepted=False, formula=None, refusals=refusals)

    formulas = {result.formula for result in results}
    if len(formulas) > 1:
        # Reachable only if a future edit lets one mode mean two things inside one System --
        # which is exactly the Energy-definition split, so it is worth failing loudly here
        # rather than publishing one attribute holding two incompatible quantities.
        return RecolorPlan(
            mode,
            accepted=False,
            formula=None,
            refusals=(
                f"{mode.value} resolves to more than one formula across this System's "
                "partitions (" + ", ".join(sorted(formulas)) + "); one attribute cannot hold "
                "two definitions.",
            ),
        )
    return RecolorPlan(mode, accepted=True, formula=results[0].formula)


def availability_matrix(sources: PartitionSources) -> dict[ColorByMode, Availability]:
    """Every mode's answer for one partition -- what a UI needs to grey out the dropdown."""
    return {mode: availability(mode, sources) for mode in ColorByMode}


#: Roles that publish points. An Emitter owns sampled output slots even though D13 deliberately
#: supports no Color By mode on them. Force remains a non-output contributor.
GEOMETRY_BEARING_ROLES = frozenset({"BODY", "CLOUD", "SOLID", "EMITTER"})

#: The role System's Solver enum, mapped to the backend string a compile selects. Mirrors
#: ``product_registry._ROLE_SOLVER_BACKENDS``; ``tests/unit/test_color_by_wiring.py`` asserts
#: the two agree, so this copy cannot quietly become a second opinion about which backend a
#: System runs. It exists here so availability is answerable without importing ``bpy``.
SOLVER_BACKENDS = {"AUTO": "auto", "DIRECT": "direct", "TREE": "tree"}


def infer_d13_role(binding: BindingDescriptor, *, backend: str) -> str:
    """Infer one binding's canonical D13 role from its compiled descriptor facts.

    The ordering is intentional: explicit non-body roles keep their authored meaning even in
    an impact graph. An Emitter's reserved slots are geometry, but they are not Solid damage
    output; a Field contributes force and is never output geometry. Body bindings then follow
    the graph and representation: impact bodies are Solids, particle sets are Clouds, and a
    single analytical body stays a Body for orbital graphs.
    """

    if not isinstance(binding, BindingDescriptor):
        raise TypeError(f"binding must be a BindingDescriptor, got {type(binding).__name__}")
    backend_name = str(backend).lower()
    if binding.role is BindingRole.EMITTER:
        return "EMITTER"
    if binding.role is BindingRole.FIELD:
        return "FORCE"
    if binding.role is not BindingRole.BODY:
        raise ValueError(f"binding role {binding.role.value!r} has no D13 output role")
    if backend_name in SOLID_BACKENDS:
        return "SOLID"
    if binding.representation in (
        BodyRepresentation.SAMPLED_PARTICLE,
        BodyRepresentation.IMPORTED_PARTICLE_SET,
    ):
        return "CLOUD"
    if backend_name in ORBITAL_BACKENDS:
        return "BODY"
    raise ValueError(
        f"backend {backend!r} and representation {binding.representation.value!r} have no "
        "D13 output role"
    )


def resolve_backend(
    solver: str, *, has_solid_member: bool, solid_backend: str = "spacesim_impact"
) -> str:
    """Which backend a role System compiles, for availability purposes.

    The Solid override is not a preference: a Solid member *is* the differentiated impact
    body (D7), and the orbital graphs hold no materials or shells, so once any member is a
    Solid an impact graph is the only one that can represent the System -- whatever the
    Solver enum says. Availability has to follow the same rule, or a Solid System would be
    asked about Damage under ``auto`` and told no.

    Which impact graph is the caller's to say, through ``solid_backend``: the choice between
    ``giant_impact`` (ADR 0094) and ``spacesim_impact`` (ADR 0193, the default every collision
    that is not a giant impact plays on, ADR 0201) is made from the System's explicit model
    selector and, under Auto, the members' masses, radii and approach, none of which this
    function sees.
    """

    if solid_backend not in SOLID_BACKENDS:
        options = " or ".join(repr(name) for name in SOLID_BACKENDS)
        raise ValueError(f"solid_backend must be {options}, got {solid_backend!r}")
    if has_solid_member:
        return solid_backend
    return SOLVER_BACKENDS.get(str(solver).upper(), "direct")


def system_partitions(
    members: Iterable[tuple[str, str, dict | None]],
    *,
    solver: str,
    solid_backend: str = "spacesim_impact",
) -> tuple[PartitionSources, ...]:
    """One :class:`PartitionSources` per geometry-bearing member of a role System.

    ``members`` is an iterable of ``(name, role, sources_kwargs)``: the caller knows what its
    members committed, this knows what the System's backend is. Members whose role bears no
    geometry are dropped rather than refused -- a Force in the System is not a partition that
    failed to recolor, it is not a partition at all, and reporting it would bury the real
    refusal among noise.
    """

    entries = [
        (str(name), str(role), dict(kwargs or {}))
        for name, role, kwargs in members
        if str(role) in GEOMETRY_BEARING_ROLES
    ]
    backend = resolve_backend(
        solver,
        has_solid_member=any(role == "SOLID" for _, role, _ in entries),
        solid_backend=solid_backend,
    )
    return tuple(
        PartitionSources(role=role, backend=backend, name=name, **kwargs)
        for name, role, kwargs in entries
    )
