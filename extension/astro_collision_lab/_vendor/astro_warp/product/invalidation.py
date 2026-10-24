"""What solved state depends on, and where an invalidation starts.

An edit to a context falls in one of three places, and PRODUCT-007 requires the boundary
between them to be stated once here rather than re-derived at each edit site:

* It reaches no solver input. Solved state stands and a live runtime adopts the edit in
  place — display scale, output object names, and frame range.
* It parameterizes the steps still to come, but no step that has already run read it.
  ``G``, softening, and the substeps a step is subdivided into all enter the solver as
  kernel launch arguments sized at launch, so every solved step is still a correct
  integration of the system as it was parameterized when it ran, at the simulation time it
  has always sat at. The solved state stands, a live runtime adopts the new values, and
  the edit's earliest affected step is the next one. This is a parameter that varies with
  time, which is a physical thing to author; the conservation diagnostics will show the
  change rather than hide it.
* It reaches an input a solved step already read — a mass, a body count, a seed, the
  precision the arrays were built at, the seconds a frame advances by. Every solved step
  was computed from inputs that no longer hold, or now reports a simulation time it was
  never integrated at, so the solved state goes and the earliest affected step is zero.

Both narrower sets are by declaration. A descriptor field is retroactive unless this
module names it excluded or prospective, so a field added later invalidates solved state
until someone decides otherwise — the safe direction to be wrong in. Every name is popped
without a default, so renaming a descriptor field fails here loudly instead of silently
widening the kept set and keeping state that a retroactive input just moved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .model import SimulationDescriptor

# Descriptor fields that no solver input reads: identity and presentation
# (``name``, ``display_scale``), lifecycle bookkeeping (``state``, ``fault_text``),
# output-side selection (``cache_root``, ``selected_bake_id``, ``controller_object_name``),
# and ``runtime_plan``, which is derived from ``graph`` — hashing the graph covers it and
# hashing the plan as well would make a recompile of identical inputs read as a change.
EXCLUDED_DESCRIPTOR_FIELDS: tuple[str, ...] = (
    "cache_root",
    "controller_object_name",
    "display_scale",
    "fault_text",
    "name",
    "runtime_plan",
    "selected_bake_id",
    "state",
)

# Timeline fields that only decide which frames are played and who drives them.
# ``seconds_per_frame`` and ``substeps`` are absent on purpose: they set the integration
# step, so they are solver inputs — ``substeps`` a prospective one, declared below, and
# ``seconds_per_frame`` a retroactive one for the reason ADR 0021 records.
EXCLUDED_TIMELINE_FIELDS: tuple[str, ...] = (
    "frame_end",
    "frame_start",
)

# Solver inputs a live runtime can adopt going forward (see the module docstring). Each is
# read fresh when the next step launches — ``gravitational_constant`` and ``softening`` are
# arguments to the acceleration kernel, and ``substeps`` subdivides the second the next
# step advances by — so none of them is reachable by a step that already ran.
#
# ``seconds_per_frame`` is deliberately absent even though the kernel reads it just as
# late. It is the one prospective candidate whose own edit breaks the report PRODUCT-007
# requires: `invalidation_text` places step ``k`` at ``k * seconds_per_frame``, so a
# mid-run change would restate every solved step's simulation time in units it was never
# integrated at. Keeping it retroactive costs a recompile and keeps the timeline honest
# (ADR 0021).
#
# Anything that sizes an allocation is also absent: ``precision`` sets the array dtypes,
# and ``collision_mode`` is written into every checkpoint and checked on restore, so
# moving either has to rebuild rather than reparameterize.
# ``gravitational_constant`` was prospective until ADR 0040. It left that set because
# VIRIALIZE and KEPLERIAN consume G **once, during expansion**, to derive per-particle
# initial velocities -- so a step that already ran did read it, and a live runtime cannot
# adopt a new G going forward. Treating it as prospective after ADR 0040 would present a
# bake whose initial condition came from the old G as still valid.
#
# ``softening`` stays prospective: it is an argument to the acceleration kernel, read
# fresh when the next step launches, and no already-run step is reachable by changing it.
PROSPECTIVE_UNITS_FIELDS: tuple[str, ...] = ("softening",)
PROSPECTIVE_TIMELINE_FIELDS: tuple[str, ...] = ("substeps",)

# A Force member's ``strength`` (ADR 0165). Its device array (``field_strengths``) is read on
# every launch, so no solved step is reachable by changing it and a live runtime can adopt it
# going forward. Prospective for FIELD bindings only: for every other role the field is not a
# solver input at all, so it stays in the digest (nothing else may hide behind this).
PROSPECTIVE_FIELD_BINDING_FIELDS: tuple[str, ...] = ("strength",)
_FIELD_ROLE = "Field"

# Per-binding output-contract fields: ``render_object_name`` names the datablock an output
# build writes into and reaches no solver input, and ``spin_rest`` (ADR 0158) is the object's
# own rotation at compile time, which a Body Spin turns *from* when the write-back poses the
# object -- the solver never reads it, so turning the object by hand must not throw away a
# solved run.
EXCLUDED_BINDING_FIELDS: tuple[str, ...] = ("render_object_name", "spin_rest")

# Solver inputs added after solved state was already being saved (ADR 0158). Each is a real
# input -- a Pinned body changes every position, a Solid's Spin every starting velocity --
# but its default is exactly the behaviour before the field existed, so it is left out of
# the digest while it sits at that default. A scene that never uses it keeps the fingerprint
# it was saved under; the moment it is used the key appears and the digest moves.
#
# ADR 0202 adds the Expansion model's two numbers on the same terms: they are read only by an
# Expansion cloud, so a descriptor that never chose that model carries both at their defaults
# and keeps the fingerprint it had.
OMITTED_AT_DEFAULT_BINDING_FIELDS: dict[str, Any] = {
    "spin": [0.0, 0.0, 0.0],
    "pinned": False,
    "expansion_speed": 0.0,
    "expansion_proportional": False,
}


def solved_state_inputs(descriptor: SimulationDescriptor) -> dict[str, Any]:
    """The part of ``descriptor`` that solved state was computed from."""

    payload = descriptor.to_dict()
    for name in EXCLUDED_DESCRIPTOR_FIELDS:
        payload.pop(name)
    timeline = payload["timeline"]
    for name in EXCLUDED_TIMELINE_FIELDS:
        timeline.pop(name)
    for binding in payload["bindings"]:
        for name in EXCLUDED_BINDING_FIELDS:
            binding.pop(name)
        for name, default in OMITTED_AT_DEFAULT_BINDING_FIELDS.items():
            value = binding[name]
            if (list(value) if isinstance(value, (list, tuple)) else value) == default:
                binding.pop(name)
    return payload


def solved_state_fingerprint(descriptor: SimulationDescriptor) -> str:
    """Digest of the solver inputs: equal digests mean solved state survives the edit."""

    payload = json.dumps(solved_state_inputs(descriptor), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def retroactive_state_inputs(descriptor: SimulationDescriptor) -> dict[str, Any]:
    """The part of ``descriptor`` that a step which has already run actually read."""

    payload = solved_state_inputs(descriptor)
    units = payload["units"]
    for name in PROSPECTIVE_UNITS_FIELDS:
        units.pop(name)
    timeline = payload["timeline"]
    for name in PROSPECTIVE_TIMELINE_FIELDS:
        timeline.pop(name)
    for binding in payload["bindings"]:
        if str(binding["role"]) == _FIELD_ROLE:
            for name in PROSPECTIVE_FIELD_BINDING_FIELDS:
                binding.pop(name)
    return payload


def retroactive_state_fingerprint(descriptor: SimulationDescriptor) -> str:
    """Digest of the inputs solved steps read: equal digests mean the state stays solved.

    Weaker than :func:`solved_state_fingerprint`, and the difference is the point. Equal
    solved-state digests mean nothing the solver reads has moved at all, so a live runtime
    needs no work. Equal retroactive digests allow the prospective parameters to have
    moved, which is exactly the case where solved state survives but the runtime has to be
    told the new values before it takes another step.
    """

    payload = json.dumps(
        retroactive_state_inputs(descriptor), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def invalidation_text(earliest_step: int, *, frame_start: int, seconds_per_frame: float) -> str:
    """Name the earliest simulation time an invalidation affects (PRODUCT-007).

    One product step advances one frame by ``seconds_per_frame``, so step ``k`` sits at
    frame ``frame_start + k`` and simulation time ``k * seconds_per_frame``. The two
    timeline values are taken as scalars, and keyword-only, so a caller holding a Blender
    property group can answer without first building a ``TimelineMapping`` and so
    ``earliest_step`` can never be swapped with ``frame_start``.
    """

    if earliest_step < 0:
        raise ValueError("earliest_step must be non-negative")
    frame = int(frame_start) + earliest_step
    seconds = earliest_step * float(seconds_per_frame)
    where = f"step {earliest_step} (frame {frame}, t={seconds:g}s)"
    if earliest_step == 0:
        return f"invalidated from {where}: no solved state is kept"
    return f"invalidated from {where}: solved state before it is unaffected"
