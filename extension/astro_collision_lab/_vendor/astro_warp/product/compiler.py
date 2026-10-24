"""Deterministic compiler for typed Astro System Graph descriptors."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace

from .diagnostics import Finding, Severity
from .model import (
    STAGE_ORDER,
    AstroSystemGraph,
    BindingRole,
    ContextState,
    GraphNode,
    Precision,
    RuntimePlan,
    RuntimeStep,
    SimulationDescriptor,
    StageFamily,
)
from .modules import MODULE_REGISTRY, RetiredModuleError, UnknownModuleError, module_version
from .plan_policy import plan_policy_errors
from .sampling import active_body_count, binding_body_count
from .timeline_stages import inert_stage_boundaries


class ValidationError(ValueError):
    """Raised when graph semantics reject a descriptor."""


def upgrade_graph_modules(graph: AstroSystemGraph) -> AstroSystemGraph:
    """Repin every node to the installed module version.

    This is the explicit upgrade the compiler refuses to perform for you (PRODUCT-004 c2).
    The result compiles to a different ``graph_hash``, so plans built from the old graph
    stop validating and bakes written under them report HISTORICAL — which is the whole
    point of asking rather than repinning behind the caller's back.
    """

    return replace(
        graph,
        nodes=tuple(
            replace(node, module_version=module_version(node.module)) for node in graph.nodes
        ),
    )


def _resolved_nodes(nodes: tuple[GraphNode, ...]) -> tuple[GraphNode, ...]:
    return tuple(
        node
        if node.module_version == MODULE_REGISTRY[node.module].version
        else replace(node, module_version=MODULE_REGISTRY[node.module].version)
        for node in nodes
    )


class GraphCompiler:
    """Validate a typed graph and produce the runtime plan used for execution."""

    REQUIRED_FAMILIES = {
        StageFamily.CONTEXT,
        StageFamily.INITIAL_STATE,
        StageFamily.OUTPUT,
    }

    # ADR 0037 section 3.4: above this heaviest-to-lightest per-body mass ratio, the
    # float32 lane loses the light bodies' force contributions to truncation.
    MASS_CONTRAST_THRESHOLD = 1e6

    def validate(self, descriptor: SimulationDescriptor) -> list[str]:
        """Error strings only. Preserved for existing callers; new code uses inspect()."""
        return [f.message for f in self.inspect(descriptor) if f.severity is Severity.ERROR]

    def advisories(self, descriptor: SimulationDescriptor) -> list[str]:
        """Advisory strings only -- the mirror of ``validate``, and the surfaced channel.

        ``validate`` filters the advisory channel out by construction, and every addon
        compile path reached the compiler through ``compile`` -> ``validate``. So until
        this method existed, ``inspect`` had no caller outside the tests and *nothing*
        raised on the ADVISORY channel had ever been shown to a user -- including the
        float32 mass-contrast warning, despite ``diagnostics.py`` describing this channel
        as one the HUD reads. ADR 0086 records that finding. This is the read side; the
        Compile Context operator is the surface that reports these to the user.
        """

        return [f.message for f in self.inspect(descriptor) if f.severity is Severity.ADVISORY]

    def inspect(self, descriptor: SimulationDescriptor) -> list[Finding]:
        """Every finding for a descriptor: ERROR blocks compilation, ADVISORY never does."""
        findings = [
            Finding(Severity.ERROR, "descriptor-invalid", message)
            for message in self._errors(descriptor)
        ]
        findings.extend(self._advisories(descriptor))
        return findings

    def _advisories(self, descriptor: SimulationDescriptor) -> list[Finding]:
        findings: list[Finding] = []
        if descriptor.precision is Precision.FLOAT32:
            # Contrast is over the masses the solver actually holds: a SAMPLED_PARTICLE
            # binding spreads its mass across particle_count bodies, so its per-body
            # mass — not the binding mass — is what float32 accumulation sees. For
            # layered clouds (_sample_shells) per-particle masses are rescaled per
            # layer, so the uniform split here is an estimate — a lower bound on the
            # true contrast.
            per_body = [
                binding.mass / binding_body_count(binding)
                for binding in descriptor.bindings
                # mass > 0 drops non-BODY-role bindings permitted at mass <= 0
                # (model.py:202) and guards the ratio against zero/negative masses.
                if binding.active and binding.mass > 0.0
            ]
            if len(per_body) >= 2:
                ratio = max(per_body) / min(per_body)
                if ratio > self.MASS_CONTRAST_THRESHOLD:
                    findings.append(
                        Finding(
                            Severity.ADVISORY,
                            "mass-contrast-float32",
                            f"per-body mass contrast {ratio:.0e} exceeds "
                            f"{self.MASS_CONTRAST_THRESHOLD:.0e} on the float32 lane; "
                            "forces from the lightest bodies may truncate to zero — "
                            "switch the context precision to float64",
                        )
                    )

        # A stage boundary nothing reads is worse than no boundary at all: it is legal, it
        # persists through the authoring document, and it sits in both state fingerprints,
        # so editing one forces a full re-solve and changes nothing about the result. No
        # engine reads a stage boundary any more (the one that did, the detailed impact
        # engine, was deleted), so a stage name is a label on every engine: it marks a moment
        # on the timeline and nothing checks or changes anything there. Advisory rather than
        # error because the authoring is well-formed; the user is told what it does and does
        # not do, and is not stopped.
        backend = self._backend(descriptor.graph)
        for boundary in inert_stage_boundaries(
            descriptor.stages, backend=None if backend == "none" else backend
        ):
            where = (
                "at the start of the timeline"
                if boundary.boundary_frame is None
                else f"at frame {boundary.boundary_frame}"
            )
            findings.append(
                Finding(
                    Severity.ADVISORY,
                    "stage-boundary-inert",
                    f"the {boundary.from_kind.value} to {boundary.to_kind.value} stage change "
                    f"{where} is a label only: it marks the moment on the timeline, but "
                    "nothing in the simulation reads it, and moving it re-solves the scene "
                    "from the first frame for no change in the result",
                )
            )
        return findings

    def _errors(self, descriptor: SimulationDescriptor) -> list[str]:
        errors: list[str] = []
        try:
            descriptor.validate()
        except ValueError as exc:
            errors.append(str(exc))

        # A context with nothing to solve compiles to a perfectly valid plan, initializes,
        # and plays to a viewport that never moves -- a silent failure indistinguishable
        # from a physics bug. The refusal lives here, not in ``SimulationDescriptor``:
        # ``validate`` runs on every ``ContextRegistry.create`` and every ``from_dict``
        # decode, where an unbound descriptor is the legitimate state authoring starts in.
        # Compiling one never is, and every compile path reaches this function.
        if not descriptor.bindings:
            errors.append(
                "this context has no bodies to solve -- bind at least one object to it "
                "before compiling"
            )
        elif not any(binding.active for binding in descriptor.bindings):
            # Distinguished from the empty case on purpose: the remedy is to re-enable what
            # is already here, not to author something new.
            errors.append(
                "every object in this context is disabled, so it has no bodies to solve "
                "-- re-enable at least one"
            )
        elif active_body_count(descriptor.bindings) == 0:
            # The third silent-viewport shape (ADR 0045): active bindings exist, but every
            # one of them is a FIELD, which expands to zero solver bodies. Same refusal,
            # different remedy -- naming the fields, so an artist with several members
            # can see exactly which objects this is about.
            field_names = ", ".join(
                f"'{binding.object_name}'"
                for binding in descriptor.bindings
                if binding.active
            )
            errors.append(
                f"this context only binds force fields ({field_names}), so there is "
                "nothing for them to act on -- bind at least one Body, Cloud, or "
                "Solid as well"
            )

        graph = descriptor.graph
        if graph.schema_version != descriptor.schema_version:
            errors.append("graph schema_version must match descriptor schema_version")
        if not graph.nodes:
            errors.append("graph must contain nodes")

        node_ids: set[str] = set()
        families: set[StageFamily] = set()
        last_order = -1
        retired_engine = False
        for node in graph.nodes:
            if not node.node_id:
                errors.append("graph node_id is required")
            if node.node_id in node_ids:
                errors.append(f"duplicate graph node_id {node.node_id}")
            node_ids.add(node.node_id)
            families.add(node.family)
            try:
                installed = module_version(node.module)
            except RetiredModuleError as exc:
                # One sentence for the whole graph, not one per retired node: every node of
                # the deleted engine would otherwise repeat it.
                if not retired_engine:
                    errors.append(str(exc))
                retired_engine = True
            except UnknownModuleError as exc:
                errors.append(f"node {node.node_id}: {exc}")
            else:
                if node.module_version and node.module_version != installed:
                    errors.append(
                        f"node {node.node_id} pins {node.module} "
                        f"{node.module_version} but the installed module is {installed}; "
                        "upgrade the graph explicitly"
                    )
            order = STAGE_ORDER[node.family]
            if order < last_order:
                errors.append(f"node {node.node_id} is out of staged order")
            last_order = order

        missing = sorted(family.value for family in self.REQUIRED_FAMILIES - families)
        if missing:
            errors.append(f"missing required graph families: {', '.join(missing)}")
        if StageFamily.INTEGRATION in families and StageFamily.FORCE not in families:
            errors.append("integration stage requires a force stage")
        if StageFamily.FORCE in families and StageFamily.SPATIAL_SERVICE not in families:
            errors.append("force stage requires a spatial_service stage")
        if StageFamily.EVENTS in families and StageFamily.INTEGRATION not in families:
            errors.append("events stage requires an integration stage")

        # ADR 0037 section 3.6: an Emitter's declared budget — every particle it could ever
        # release — must fit inside its declared capacity. Checked here at compile so an
        # over-budget scene is rejected before a bake starts, not faulted at frame N of a long
        # run (the failure mode ADR D4 exists to prevent). Only Emitter-role bindings carry
        # these fields meaningfully; every other role keeps its defaults and is untouched.
        for binding in descriptor.bindings:
            if binding.role is not BindingRole.EMITTER:
                continue
            start, end = binding.active_range
            budget = binding.rate * (end - start + 1)
            if budget > binding.emitter_capacity:
                # Render an integer-valued budget with no decimals (1000000, not 1000000.0)
                # but keep a fractional budget's real value: `int(budget)` alone would print
                # "budget=1" for a 1.5-particle overflow against a capacity of 1, a
                # self-contradictory message (1.5 > 1 fired the check).
                budget_text = int(budget) if budget == int(budget) else budget
                errors.append(
                    f"emitter binding {binding.binding_id} budget={budget_text} "
                    f"exceeds emitter_capacity={binding.emitter_capacity}"
                )

        # ADR 0209. What Play would refuse because of the solver the graph names -- a force
        # field, an emitter or a pinned body on an impact engine, a shearing box on the wrong
        # solver or carrying what it cannot -- is refused here, in the same words, instead of
        # compiling to READY. Only rules that read the bindings, the backend and the nodes live
        # in that module; the frame-dependent ones stay in `compile_role_system` (ADR 0108).
        errors.extend(plan_policy_errors(descriptor, self._backend(graph), graph.nodes))
        return errors

    def compile(self, descriptor: SimulationDescriptor) -> SimulationDescriptor:
        errors = self.validate(descriptor)
        if errors:
            raise ValidationError("; ".join(errors))

        graph = descriptor.graph
        # Unpinned nodes take the installed module version here, so the compiled graph, the
        # plan hash, and every runtime step state the exact module build that will run.
        ordered = _resolved_nodes(
            tuple(sorted(graph.nodes, key=lambda node: (STAGE_ORDER[node.family], node.node_id)))
        )
        graph_hash = self.graph_hash(replace(graph, nodes=ordered))
        backend = self._backend(graph)
        plan = RuntimePlan(
            plan_id=f"plan-{uuid.uuid5(uuid.NAMESPACE_URL, graph_hash)}",
            graph_hash=graph_hash,
            backend=backend,
            steps=tuple(
                RuntimeStep(
                    index=index,
                    node_id=node.node_id,
                    family=node.family,
                    module=node.module,
                    module_version=node.module_version,
                    parameters=dict(node.parameters),
                )
                for index, node in enumerate(ordered)
            ),
            capacities={
                "contexts": 1,
                "bindings": len(descriptor.bindings),
                # Bindings are authored rows; bodies are what the solver allocates. A
                # sampled particle binding is one row and many bodies, so the two counts
                # are no longer the same number.
                "active_bodies": active_body_count(descriptor.bindings),
            },
            diagnostics=("graph_hash", "binding_count", "active_body_count"),
        )
        return replace(
            descriptor,
            graph=replace(graph, nodes=ordered),
            runtime_plan=plan,
            state=ContextState.READY,
            fault_text="",
        )

    def graph_hash(self, graph: AstroSystemGraph) -> str:
        payload = {
            "schema_version": graph.schema_version,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "family": node.family.value,
                    "module": node.module,
                    # A module version bump is a different plan, and therefore a different
                    # plan_id and a currentness break for every bake under the old hash.
                    "module_version": node.module_version,
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "parameters": node.parameters,
                }
                for node in graph.nodes
            ],
            "manifests": graph.manifests,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _backend(self, graph: AstroSystemGraph) -> str:
        for node in graph.nodes:
            backend = node.parameters.get("backend")
            if backend:
                return str(backend)
        return "none"
