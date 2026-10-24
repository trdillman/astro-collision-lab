"""Which Systems cannot be played on which solver: the structural plan policy (ADR 0209).

Five rules say "this cannot run here" and depend on nothing but the descriptor's bindings, the
backend the graph names and the steps of the plan. They used to be written only in
``ProductRuntimeFactory._validate_plan``, so a System they refuse compiled to READY and was
turned away at Play. They live here once, and both the compiler (``GraphCompiler._errors``) and
the factory (defence in depth, for a plan saved or edited some other way) ask them.

The rules:

* the impact (particle) backends integrate a sampled cloud as one solid, so a force field, an
  active emitter or a pinned body has nothing to act on, release or hold (ADR 0045, 0047, 0158);
* the local shearing box (ADR 0202) is a Direct-solver frame option, needs a usable box, and
  cannot carry force fields, emitters, pinned bodies or a collision mode other than Ignore.

A rule belongs here only if it is **frame-independent**: it may not read the timeline, a frame
duration, a calibration or a particle count. ADR 0108 keeps those in ``compile_role_system``
because compile chooses the frame and runs once before it has. Everything that checks a *saved*
plan for integrity (version pins, capacities, the exact pipeline, bindings agreeing with steps)
stays in the factory.

Device-free: it imports neither ``runtime_factory`` nor Warp.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from .backend_registry import Family, find_backend
from .model import BindingRole, SimulationDescriptor
from .nbody import CollisionMode
from .shearing_box import ShearingBoxSpec

#: The shearing-box step (ADR 0202). The factory splices and counts it by this name too.
SHEARING_BOX_MODULE = "astro.frame.shearing_box"
_COLLISION_MODULE = "astro.collision.point"


class PolicyStep(Protocol):
    """What a rule reads of a step: a runtime-plan step and a graph node both have these."""

    @property
    def module(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, Any]: ...


def shearing_box_from_steps(steps: Iterable[PolicyStep]) -> ShearingBoxSpec | None:
    """The box the steps describe, or ``None`` when none is a shearing-box step.

    The step's parameters are the one place a box is written down (they are inside
    ``graph_hash``). A step whose numbers are missing or not usable raises ``ValueError`` in the
    artist's words.
    """

    for step in steps:
        if step.module == SHEARING_BOX_MODULE:
            return ShearingBoxSpec.from_parameters(dict(step.parameters))
    return None


def plan_policy_errors(
    descriptor: SimulationDescriptor, backend: str, steps: Iterable[PolicyStep]
) -> list[str]:
    """Every structural reason this System cannot be played on ``backend``, in plain words.

    The first entry is the one the factory raises. An empty list means the policy has no
    objection; it does not mean the plan is sound (the factory's integrity checks still run).
    """

    steps = tuple(steps)
    bindings = descriptor.bindings
    has_field_bindings = any(binding.role == BindingRole.FIELD for binding in bindings)
    # An *inactive* emitter binding still occupies its slots but releases nothing, so the test is
    # on active bindings alone (ADR 0047).
    has_emitter_bindings = any(
        binding.role == BindingRole.EMITTER and binding.active for binding in bindings
    )
    has_pinned_bodies = any(binding.pinned for binding in bindings)
    errors: list[str] = []

    spec = find_backend(backend)
    if spec is not None and spec.family is Family.PARTICLE:
        if has_field_bindings:
            errors.append(
                f"the {backend!r} backend does not support external force fields; "
                "remove the Force members or use an orbital solver"
            )
        if has_emitter_bindings:
            errors.append(
                f"the {backend!r} backend does not support emitters; "
                "remove the Emitter members or use an orbital solver"
            )
        # ADR 0158. Pinned holds an analytic point mass at its anchor, which only the direct and
        # tree solvers do. The particle backends integrate a sampled body as a cloud with no
        # single position to hold, so a pinned binding reaching one would be silently ignored --
        # the exact "shown but not simulated" failure this flag exists to end.
        if has_pinned_bodies:
            errors.append(
                f"the {backend!r} backend does not support pinned bodies; "
                "turn Pinned off or use an orbital solver"
            )

    if any(step.module == SHEARING_BOX_MODULE for step in steps):
        errors.extend(
            _shearing_box_errors(
                backend,
                steps,
                has_field_bindings=has_field_bindings,
                has_emitter_bindings=has_emitter_bindings,
                has_pinned_bodies=has_pinned_bodies,
            )
        )
    return errors


def _shearing_box_errors(
    backend: str,
    steps: tuple[PolicyStep, ...],
    *,
    has_field_bindings: bool,
    has_emitter_bindings: bool,
    has_pinned_bodies: bool,
) -> list[str]:
    """Everything a shearing box cannot honour (ADR 0202), so the artist reads which setting to
    change instead of "not the exact pipeline"."""

    errors: list[str] = []
    if backend != "direct":
        errors.append(
            f"the shearing box runs on the Direct solver only, but this plan runs on "
            f"{backend!r}. Set the System's Solver to Direct (or Auto) and recompile"
        )
    try:
        shearing_box_from_steps(steps)
    except ValueError as exc:
        errors.append(str(exc))
    if has_field_bindings:
        errors.append(
            "a shearing box cannot carry force fields: a field's position means nothing in a "
            "box that repeats. Remove the Force members"
        )
    if has_emitter_bindings:
        errors.append(
            "a shearing box cannot carry emitters: the release schedule is written for an open "
            "space. Remove the Emitter members"
        )
    if has_pinned_bodies:
        errors.append(
            "a shearing box cannot hold a body still: the frame turns and the box repeats, so "
            "there is no fixed place to pin it. Turn Pinned off"
        )
    for step in steps:
        mode = str(step.parameters.get("mode", "ignore"))
        if step.module == _COLLISION_MODULE and mode != CollisionMode.IGNORE.value:
            errors.append(
                "a shearing box cannot resolve collisions: a body that leaves through one side "
                "re-enters through the other, so the collision stage's neighbours are wrong. "
                "Set the System's Collision to Ignore"
            )
    return errors
