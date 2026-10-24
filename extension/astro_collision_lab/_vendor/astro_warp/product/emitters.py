"""Emitter release schedules: the ``astro.emitter.activate`` module surface (ADR 0037 3.6).

An Emitter member of a System compiles to a ``BindingRole.EMITTER`` binding carrying a
``SAMPLED_PARTICLE`` cloud of exactly ``emitter_capacity`` particles. Every one of those
particles is drawn once, at load, by the ordinary cloud expansion --- the same positions,
masses and velocities any other cloud of that shape would get --- and then held *inactive*
until its frame arrives. Emission is therefore not a spawn: it is the activation of a slot
the solver already owns, which is why it costs no allocation, no reload, and no change to
the capacity the plan was admitted against.

The schedule is a pure function of the frame:

    released(frame) = floor(rate * frames_elapsed), clamped to the emitter's slot count

with ``frames_elapsed`` counted inclusively from the first frame of ``active_range`` and
frozen at its last. Purity is the whole design. A scrub backwards restores a checkpoint,
which restores the ``active`` array along with everything else, and re-stepping forward
recomputes the identical release from the restored step index --- there is no emitted
counter to drift, and no frame at which the population depends on how the transport got
there. The same property makes a bake and a live play agree by construction.

``floor(rate * elapsed)`` is the definition, evaluated in the same float arithmetic the
compiler's budget check uses (``rate * (end - start + 1)``, `GraphCompiler.validate`), so
the two cannot disagree about how many particles an emitter releases. A rate with no exact
binary representation releases what that expression says, deterministically, which is the
number the budget was checked against.

The device tier is deliberately thin: one launch per emitter per frame, sized to the
*delta* between the previous frame's release and this one's, skipped entirely when the
delta is zero. It lives outside the captured substep graph because emission is a per-frame
event, not a per-substep one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .model import BindingDescriptor, BindingRole, TimelineMapping
from .sampling import binding_body_count


@dataclass(frozen=True)
class EmitterSpec:
    """One EMITTER binding resolved for a runtime: which slots, how many per frame, when.

    ``begin_index`` and ``slots`` locate the binding's block inside the concatenated body
    arrays ``expand_bindings`` produces, so the runtime never has to re-derive the binding
    layout it was loaded with.

    ``first_frame`` is runtime-relative --- frame 0 is the timeline's ``frame_start`` --- and
    is allowed to be negative. An emitter whose authored range lies wholly before the
    timeline starts has already finished releasing by the time the run begins, and a
    negative first frame is what makes ``released_at_frame`` say so instead of replaying a
    past emission from step zero.
    """

    binding_id: str
    begin_index: int
    slots: int
    rate: float
    first_frame: int
    frame_span: int

    def validate(self) -> None:
        if self.begin_index < 0:
            raise ValueError(f"emitter begin_index must be non-negative, got {self.begin_index!r}")
        if self.slots < 1:
            raise ValueError(f"emitter slots must be positive, got {self.slots!r}")
        if not math.isfinite(self.rate) or self.rate < 0.0:
            raise ValueError(f"emitter rate must be finite and non-negative, got {self.rate!r}")
        if self.frame_span < 1:
            raise ValueError(f"emitter frame_span must be positive, got {self.frame_span!r}")


def validate_slot_range(spec: EmitterSpec, count: int) -> None:
    """Refuse a spec whose slots fall outside the ``count`` bodies a runtime loaded.

    ``EmitterSpec.validate`` can only check the spec against itself; this is the half that
    needs the runtime's population, so both orbital configs call it from their own
    ``validate``. The activation kernel indexes ``active`` without a bound test --- it is
    launched at exactly the delta width and trusts the block --- so an out-of-range spec
    would be an out-of-bounds device write rather than a host error.
    """

    if spec.begin_index + spec.slots > count:
        raise ValueError(
            f"emitter {spec.binding_id!r} names slots "
            f"[{spec.begin_index}, {spec.begin_index + spec.slots}) but the runtime "
            f"loaded {count} bodies"
        )


def released_at_frame(spec: EmitterSpec, frame: int) -> int:
    """How many of ``spec``'s slots are live at runtime-relative ``frame``.

    Monotone non-decreasing in ``frame`` by construction, which is what lets the runtime
    activate only the delta between two frames and never have to deactivate anything: a
    slot the collision kernels killed stays dead, because emission only ever advances the
    watermark and never rewrites the slots below it.
    """

    if frame < spec.first_frame:
        return 0
    last_frame = spec.first_frame + spec.frame_span - 1
    elapsed = min(frame, last_frame) - spec.first_frame + 1
    return min(spec.slots, int(math.floor(spec.rate * elapsed)))


def specs_from_bindings(
    bindings: tuple[BindingDescriptor, ...],
    timeline: TimelineMapping,
) -> tuple[EmitterSpec, ...]:
    """Every active EMITTER binding as a runtime spec, in binding order.

    The block offset is accumulated across *all* bindings, active or not, because that is
    exactly how :func:`~astro_warp.product.sampling.expand_bindings` lays the body arrays
    out: an inactive binding still occupies its slots, holding them at zero in ``active``.

    A binding whose slot count disagrees with its declared ``emitter_capacity`` is refused
    here rather than silently released against the wrong ceiling. The compiler checked the
    authored budget against ``emitter_capacity``; the runtime can only release the slots it
    actually allocated, so the two numbers being equal is the condition that makes the
    compile-time check mean anything at run time.
    """

    specs: list[EmitterSpec] = []
    offset = 0
    for binding in bindings:
        count = binding_body_count(binding)
        if binding.role != BindingRole.EMITTER or not binding.active:
            offset += count
            continue
        if count != int(binding.emitter_capacity):
            raise ValueError(
                f"emitter binding {binding.binding_id!r} expands to {count} solver "
                f"bodies but declares emitter_capacity={binding.emitter_capacity}; the "
                "compile-time particle budget was checked against a ceiling the runtime "
                "has no slots for"
            )
        start, end = binding.active_range
        spec = EmitterSpec(
            binding_id=binding.binding_id,
            begin_index=offset,
            slots=count,
            rate=float(binding.rate),
            first_frame=int(start) - int(timeline.frame_start),
            frame_span=int(end) - int(start) + 1,
        )
        spec.validate()
        specs.append(spec)
        offset += count
    return tuple(specs)


def emitter_kernels(wp: Any) -> Any:
    """Compile the slot-activation kernel.

    Precision-free, unlike the other kernel factories in this package: activation writes
    the ``int32`` ``active`` flags and reads no state array, so one compilation serves both
    precision lanes and nothing here has to know which one is running.

    It also needs no ``_registrars`` resolution. That helper exists because ``scalar`` and
    ``vector`` name types that a stringized annotation cannot always resolve --- they are
    closure variables only for the kernels whose *bodies* mention them. The one name this
    signature uses is ``wp``, which the body references through ``wp.tid()``, so the
    closure the annotations are evaluated against carries it by construction.
    """

    @wp.kernel
    def activate_slots(active: wp.array(dtype=wp.int32), first_slot: int):
        # One thread per newly released slot, launched at ``dim=delta``. The kernel is
        # never told the watermark and so cannot lower one: it writes 1 into the delta
        # block and touches nothing below it. That is what lets a merge kill an emitted
        # particle permanently --- a kernel that re-asserted ``active[i] = 1 if i <
        # released`` over the whole block every frame would resurrect the dead.
        active[first_slot + wp.tid()] = 1

    @dataclass(frozen=True)
    class EmitterKernels:
        activate_slots: Any

    return EmitterKernels(activate_slots=activate_slots)
