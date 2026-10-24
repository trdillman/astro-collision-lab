"""Guided first-simulation checklist derived from observed state (PRODUCT-013 c2).

The clause the spec states is "verifies actual completed actions, not merely clicked
buttons", so nothing here records that an operator ran. Every step is a question asked of
the state that survives: the stored descriptor, and a small set of host facts the
descriptor cannot see (does the controller Empty still exist, is a runtime live, was the
selected bake inspected). A user who deletes the controller Empty, edits a unit, or lets a
recompile invalidate the plan sees the step go back to not-done, which is the whole point.

Keeping the derivation here rather than in the panel is what makes it testable: the host
side collects facts and draws rows, and this module owns every judgement about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import BindingRole, ContextState, SimulationDescriptor
from .sampling import active_body_count

# The two states a user can act on and continue from. Nothing else belongs here: DIRTY and
# UNLOADED are ordinary, and the checklist steps already name what each of them needs.
RECOVERY_ACTIONS: dict[ContextState, str] = {
    ContextState.FAULTED: "reset the context to clear the fault, or close and reopen it",
    ContextState.MISSING_CACHE: (
        "repoint the cache root at the bake directory, or bake again; "
        "the descriptor was not reset"
    ),
}

# An orbital preview needs a second body to orbit or be orbited, so one bound body is a
# configuration the user is not finished with rather than an invalid descriptor.
MINIMUM_ACTIVE_BODIES = 2


@dataclass(frozen=True)
class ChecklistEvidence:
    """Host-observed facts the stored descriptor cannot answer for itself.

    Defaults describe a context that has just been created: nothing initialized, nothing
    inspected. Every field is a fact about current state, never a record that an action
    was taken.
    """

    controller_present: bool = False
    # None means no live runtime at all; an int is the step a live runtime sits at.
    preview_step: int | None = None
    # False for a live runtime whose descriptor no longer matches the compiled plan.
    preview_matches_plan: bool = False
    checkpoint_steps: tuple[int, ...] = ()
    diagnostics_backend: str = ""
    diagnostics_metrics: tuple[str, ...] = ()
    # The bake the context currently points at. This is a host fact, not a descriptor one: a
    # stored descriptor is the snapshot compiling took, while selecting a bake happens after
    # that and must not rewrite it, because rewriting a descriptor recompiles the plan.
    selected_bake_id: str = ""
    inspected_bake_id: str = ""
    bake_integrity: bool = False
    bake_currentness: bool = False
    bake_compatibility: bool = False
    bake_errors: tuple[str, ...] = ()
    output_datablocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChecklistStep:
    """One checklist row: what it is, whether state says it happened, and what is next."""

    key: str
    label: str
    done: bool
    detail: str
    action: str = ""


@dataclass(frozen=True)
class ChecklistReport:
    steps: tuple[ChecklistStep, ...] = ()
    faults: tuple[str, ...] = ()
    next_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _context_step(descriptor: SimulationDescriptor | None, evidence: ChecklistEvidence):
    if descriptor is None:
        return False, "no context descriptor is stored", "create an Astro Simulation Context"
    if not evidence.controller_present:
        name = descriptor.controller_object_name or "<unnamed>"
        return (
            False,
            f"controller Empty {name!r} is missing from the scene",
            "recreate the context so it owns a selectable controller Empty",
        )
    return True, f"{descriptor.name!r} controlled by {descriptor.controller_object_name!r}", ""


def _units_step(descriptor: SimulationDescriptor | None):
    if descriptor is None:
        return False, "no context descriptor is stored", "create a context first"
    units = descriptor.units
    try:
        units.validate()
    except ValueError as exc:
        return False, str(exc), "correct the unit names and gravitational constant"
    return (
        True,
        f"{units.length}/{units.mass}/{units.time}, G={units.gravitational_constant:g}",
        "",
    )


def _bindings_step(descriptor: SimulationDescriptor | None):
    if descriptor is None:
        return False, "no context descriptor is stored", "create a context first"
    try:
        descriptor.validate()
    except ValueError as exc:
        return False, str(exc), "correct the invalid binding"
    # Counted in bodies, not bindings: one sampled particle binding is a whole
    # gravitating system on its own, and the rule is about what the solver integrates.
    bodies = active_body_count(
        tuple(binding for binding in descriptor.bindings if binding.role == BindingRole.BODY)
    )
    if bodies < MINIMUM_ACTIVE_BODIES:
        return (
            False,
            f"{bodies} active body/bodies; an orbital system needs {MINIMUM_ACTIVE_BODIES}",
            "select the objects to simulate and bind them to this context",
        )
    return True, f"{bodies} active bodies validate", ""


def _graph_step(descriptor: SimulationDescriptor | None):
    if descriptor is None:
        return False, "no context descriptor is stored", "create a context first"
    plan = descriptor.runtime_plan
    if plan is None:
        # Every solver-input edit clears the stored plan, so its presence is the standing
        # evidence that the current inputs were compiled and have not moved since.
        return (
            False,
            "no compiled plan is stored for the current inputs",
            "compile the context",
        )
    return (
        True,
        f"{len(plan.steps)} stages on backend {plan.backend}, graph {plan.graph_hash[:12]}",
        "",
    )


def _preview_step(descriptor: SimulationDescriptor | None, evidence: ChecklistEvidence):
    if descriptor is None:
        return False, "no context descriptor is stored", "create a context first"
    if evidence.preview_step is None:
        return False, "no live runtime holds this context", "initialize the preview runtime"
    if not evidence.preview_matches_plan:
        return (
            False,
            f"live runtime at step {evidence.preview_step} no longer matches the stored plan",
            "initialize the context again to rebuild its runtime",
        )
    seconds = evidence.preview_step * float(descriptor.timeline.seconds_per_frame)
    return True, f"live runtime at step {evidence.preview_step} (t={seconds:g}s)", ""


def _diagnostics_step(evidence: ChecklistEvidence):
    if not evidence.diagnostics_metrics:
        return (
            False,
            "no diagnostics have been read from the runtime",
            "refresh diagnostics",
        )
    return (
        True,
        f"{evidence.diagnostics_backend or 'runtime'} reported "
        f"{len(evidence.diagnostics_metrics)} metrics",
        "",
    )


def _checkpoint_step(evidence: ChecklistEvidence):
    if not evidence.checkpoint_steps:
        return (
            False,
            "no checkpoint is held, so backward scrub has nothing to restore",
            "create a checkpoint before scrubbing backward",
        )
    return (
        True,
        f"{len(evidence.checkpoint_steps)} checkpoint(s), earliest at step "
        f"{min(evidence.checkpoint_steps)}",
        "",
    )


def _bake_step(evidence: ChecklistEvidence):
    selected = evidence.selected_bake_id
    if not selected:
        return False, "no bake revision is selected", "bake the context"
    if evidence.inspected_bake_id != selected:
        return (
            False,
            f"bake {selected} has not been inspected since it was selected",
            "validate the selected bake",
        )
    if not evidence.bake_integrity:
        return (
            False,
            f"bake {selected} failed integrity: {', '.join(evidence.bake_errors) or 'unknown'}",
            "bake again; the selected revision cannot be trusted",
        )
    currentness = "CURRENT" if evidence.bake_currentness else "HISTORICAL"
    return True, f"bake {selected} integrity VALID, {currentness}", ""


def _outputs_step(evidence: ChecklistEvidence):
    if not evidence.output_datablocks:
        return (
            False,
            "no output datablock has been built from a bake",
            "build outputs from the selected bake",
        )
    return True, "; ".join(evidence.output_datablocks), ""


def _validation_step(evidence: ChecklistEvidence):
    selected = evidence.selected_bake_id
    if not selected or evidence.inspected_bake_id != selected:
        return (
            False,
            "render and export are blocked until a selected bake is validated",
            "validate the selected bake",
        )
    blocked = [
        name
        for name, passed in (
            ("integrity", evidence.bake_integrity),
            ("compatibility", evidence.bake_compatibility),
        )
        if not passed
    ]
    if blocked:
        return (
            False,
            f"selected bake fails {', '.join(blocked)}: "
            f"{', '.join(evidence.bake_errors) or 'unknown'}",
            "select or produce an integrity-valid compatible bake before rendering",
        )
    # Currentness is recorded and deliberately not required: the spec keeps a HISTORICAL
    # revision renderable when it is explicitly selected.
    return True, f"bake {selected} is renderable without running the solver", ""


def first_simulation_checklist(
    descriptor: SimulationDescriptor | None,
    evidence: ChecklistEvidence | None = None,
) -> tuple[ChecklistStep, ...]:
    """The ten guided-onboarding steps, each answered from state rather than from clicks."""

    evidence = evidence or ChecklistEvidence()
    rows = (
        ("context", "context created", _context_step(descriptor, evidence)),
        ("units", "units selected", _units_step(descriptor)),
        ("bindings", "bindings valid", _bindings_step(descriptor)),
        ("graph", "graph compiled", _graph_step(descriptor)),
        ("preview", "preview initialized", _preview_step(descriptor, evidence)),
        ("diagnostics", "diagnostics available", _diagnostics_step(evidence)),
        ("checkpoint", "checkpoint policy active", _checkpoint_step(evidence)),
        ("bake", "bake valid", _bake_step(evidence)),
        ("outputs", "outputs integrated", _outputs_step(evidence)),
        ("validation", "render/export validation complete", _validation_step(evidence)),
    )
    return tuple(
        ChecklistStep(key=key, label=label, done=done, detail=detail, action=action)
        for key, label, (done, detail, action) in rows
    )


def next_required_action(steps: tuple[ChecklistStep, ...]) -> str:
    """The action for the first step state says is unfinished; empty when all are done."""

    for step in steps:
        if not step.done:
            return step.action or f"complete: {step.label}"
    return ""


def blocking_action(steps: tuple[ChecklistStep, ...], required_key: str) -> str:
    """The action blocking a control that itself depends on `required_key`.

    Considers only steps up to and including `required_key`, in order, and returns the
    first not-done one's action -- empty once everything through `required_key` is done.
    One shared guard (STAB-008): every control names the checklist step it needs and gets
    back the single thing still missing for *that* dependency, never a later step it does
    not itself need.
    """

    for step in steps:
        if not step.done:
            return step.action or f"complete: {step.label}"
        if step.key == required_key:
            return ""
    return ""


def recoverable_faults(state: str, fault_text: str = "") -> tuple[str, ...]:
    """Faults a user can act on, each paired with the action that clears it.

    A healthy state reports nothing even when a message is attached to it, because most
    messages are ordinary progress ("step 12") rather than faults. An unrecognised state is
    reported: it means the stored document disagrees with this build.
    """

    try:
        context_state = ContextState(state)
    except ValueError:
        return (f"unknown context state {state!r}: reopen the context",) if state else ()
    recovery = RECOVERY_ACTIONS.get(context_state)
    if recovery is None:
        return ()
    detail = f"{context_state.value}: {fault_text}" if fault_text else context_state.value
    return (f"{detail} — {recovery}",)


def checklist_report(
    descriptor: SimulationDescriptor | None,
    evidence: ChecklistEvidence | None = None,
    *,
    state: str = "",
    fault_text: str = "",
) -> ChecklistReport:
    """Everything the guided-onboarding surface draws, in one derived value."""

    steps = first_simulation_checklist(descriptor, evidence)
    return ChecklistReport(
        steps=steps,
        faults=recoverable_faults(state, fault_text),
        next_action=next_required_action(steps),
        metadata={
            "complete": sum(1 for step in steps if step.done),
            "total": len(steps),
        },
    )
