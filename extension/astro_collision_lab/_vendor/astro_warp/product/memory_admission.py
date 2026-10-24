"""Graphics-memory pre-flight for the product runtime (ADR 0195, completion-audit item C06).

``scheduler.py`` admits a frame on time only. Before this module nothing asked whether the
graphics card could hold a scene before the runtime started allocating it, so a particle count
too large for the free memory ended as a CUDA allocation failure half-way through building the
runtime, with the arrays allocated so far still held. This module answers the question first and
turns the failure, when it happens anyway, into the same plain refusal.

It has three jobs, all small:

* :func:`estimate_device_bytes` adds up the device arrays a compiled descriptor will make the
  runtime allocate: the per-body state, the Barnes-Hut tree, the neighbour, pressure and bond
  tables of the two Solid engines, and the display staging the device holds. It reads only the
  descriptor and its plan, so it runs before any device object exists.
* :func:`check_admission` compares that with the free memory and refuses, in plain words, a scene
  that does not fit with room to spare. :func:`admit_scene` is the whole sequence the factory
  runs: estimate, read, decide, record.
* :func:`allocation_failure_guard` wraps runtime construction. An allocation failure that gets
  past the estimate (an estimate is a forecast, and another application can take memory between
  the read and the allocation) is closed down and re-raised as the same plain refusal, with
  everything the half-built runtime had allocated released.

The estimate is a model of the allocation sites in ``nbody.py``, ``tree.py``,
``giant_impact.py`` and ``spacesim_impact.py``, and ``tests/unit/test_memory_admission.py``
holds it to the bytes a real device reports for a scene of every backend, so a new array that
nobody added here fails a test instead of passing admission unnoticed. It estimates device
memory only. The snapshots the runtime keeps and the scrub cache live in host memory, and the
scrub cache has its own fixed ceiling (``scrub_cache.SCRUB_CACHE_CAPACITY_BYTES``).

Warp-free at import, like the factory that calls it: only the default free-memory reader touches
a device, and only through :class:`DeviceContext`.
"""

from __future__ import annotations

import contextlib
import gc
import math
import re
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .broadphase import BroadphaseConfig
from .display import DEFAULT_DISPLAY_CAP, display_count_for
from .model import (
    AUTO_SMALL_SCENE_DIRECT_LIMIT,
    DEFAULT_TREE_MAX_DEPTH,
    BindingRole,
    BodyRepresentation,
    Precision,
    RuntimePlan,
    SimulationDescriptor,
)
from .nbody import CollisionMode
from .runtime_foundation import BackendRuntime, DeviceContext
from .sampling import body_count
from .tree import sparse_node_capacity

MIB = 1 << 20

#: The share of the free graphics memory one scene may claim. The remainder is not slack for the
#: estimate's error (the estimate is held to the bytes a device reports); it is room for what the
#: estimate cannot see: Blender's own viewport buffers, other contexts built after this read, and
#: the driver's bookkeeping.
DEFAULT_HEADROOM = 0.8

#: ``model.AUTO_SMALL_SCENE_DIRECT_LIMIT``, the factory's own limit (ADR 0208). Auto builds Direct
#: and Tree side by side above it (calibration measures one against the other) and builds Direct
#: alone at or below it.
#: Bytes every runtime allocates once whatever the scene: the reduction and diagnostic words, the
#: collision event lists, the force-field and material tables, and the CUDA graphs' bookkeeping.
#: Measured in kilobytes; a megabyte is the round figure that cannot be an underestimate.
FIXED_WORKSPACE_BYTES = MIB

#: Per-body allowances for the two engines that own their allocations and import Warp at module
#: scope, so this module cannot read them. Each is the sum of the engine's arrays plus the
#: runtime's published channels, rounded up to a figure that survives an added array or two:
#: giant impact holds 204 bytes a particle (seven position-sized vectors, the pressure and density
#: scalars, the sort keys and the 60 published bytes); the rock engine holds 1,808 (its 64-slot
#: neighbour, bond and bond-data rows alone are 64 x (4 + 4 + 16) = 1,536). A device test pins both
#: allowances at or above what the engines really allocate.
GIANT_IMPACT_BYTES_PER_PARTICLE = 256
SPACESIM_IMPACT_BYTES_PER_PARTICLE = 2048

#: ``giant_impact.GRID_DIM``, mirrored (the module imports Warp). Both engines build their
#: neighbour lists on a fixed grid of this many cells a side holding a start and an end offset
#: per cell, however few particles there are: 2 x 128^3 x 4 bytes = 16 MiB. A test reads the
#: constant out of the engine's source and pins them equal.
_ENGINE_GRID_DIM = 128
ENGINE_GRID_BYTES = 2 * _ENGINE_GRID_DIM**3 * 4

#: The device side of ``display.DisplayStage`` at its widest: per slot the sample plan (4), the
#: staged position (12) and every optional channel the look can ask for (velocity 12, size 4,
#: stable id 8, damage 4, material id 8, internal energy 4, colour 16, ramp scalar 4); per body the
#: four int32 scan and identity-selection arrays. The pinned host copies are host memory.
DISPLAY_BYTES_PER_SLOT = 76
DISPLAY_BYTES_PER_BODY = 16

_POINT_COLLISION_MODULE = "astro.collision.point"
_TREE_MODULES = ("astro.gravity.barnes_hut_tree", "astro.gravity.auto_select")

# A Warp device-memory failure, in the two ways it surfaces. The allocator raises "Failed to
# allocate N bytes on device 'cuda:0'" (``warp/_src/context.py``, the CUDA allocators); a failed
# CUDA call reports the driver's own words. The CPU allocator raises the same sentence for
# 'cpu', which is host memory and not what this module's wording is about, so it is excluded.
_DEVICE_ALLOCATION_FAILURE = re.compile(
    r"Failed to allocate \d+ bytes\b.*\bon device '(?!cpu\b)"
    r"|out of memory|cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY",
    re.IGNORECASE,
)


# -- the estimate ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryEstimate:
    """The device bytes one compiled scene will make its runtime allocate, by part."""

    backend: str
    body_count: int
    precision: str
    components: tuple[tuple[str, int], ...]

    @property
    def total_bytes(self) -> int:
        return sum(size for _name, size in self.components)

    def component(self, name: str) -> int:
        """The bytes of the named part, or zero when this scene has no such part."""

        return sum(size for part, size in self.components if part == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "body_count": self.body_count,
            "precision": self.precision,
            "total_bytes": self.total_bytes,
            "components": {name: size for name, size in self.components},
        }


def estimate_device_bytes(
    descriptor: SimulationDescriptor,
    plan: RuntimePlan | None = None,
    *,
    display_cap: int = DEFAULT_DISPLAY_CAP,
) -> MemoryEstimate:
    """Add up the device arrays ``descriptor`` makes its backend runtime allocate.

    ``plan`` defaults to the descriptor's own. The count is ``body_count(bindings)``, the
    allocation the runtimes size every array from, so emitter slots and idle slots are paid for
    exactly as the runtime pays for them. The two particle engines that keep only the live
    particles on the device are charged for all slots, which can only over-estimate.

    Raises ``ValueError`` for a descriptor that has no plan; the caller decides what an
    unreadable scene means (:func:`admit_scene` admits it and says so).
    """

    plan = plan if plan is not None else descriptor.runtime_plan
    if plan is None:
        raise ValueError("descriptor must be compiled before its memory can be estimated")
    count = body_count(descriptor.bindings)
    backend = plan.backend
    scalar = 8 if descriptor.precision == Precision.FLOAT64 else 4
    # The anchor arrays exist only when an analytic Body is pinned (``expand_bindings`` flags no
    # other binding), however many bindings carry the flag.
    pinned = any(
        binding.pinned
        and binding.role == BindingRole.BODY
        and binding.representation == BodyRepresentation.ANALYTICAL
        for binding in descriptor.bindings
    )
    parts: list[tuple[str, int]] = []
    # The model of each backend is on its registered spec (``BackendSpec.memory_parts``), and a
    # spec cannot be built without one. A name nobody registered (or a plan compiled by the
    # deleted detailed impact engine) has no spec and so gets an empty estimate, which admits;
    # the factory refuses those backends on its own account. Imported late because the registry
    # imports this module for the sizing functions.
    from .backend_registry import find_backend

    spec = find_backend(backend)
    if count >= 1 and spec is not None:
        parts.extend(spec.memory_parts(count, scalar, plan, pinned))
        parts.append(("workspace", FIXED_WORKSPACE_BYTES))
        slots = display_count_for(count, display_cap)
        parts.append(
            (
                "display staging",
                slots * DISPLAY_BYTES_PER_SLOT + count * DISPLAY_BYTES_PER_BODY,
            )
        )
    return MemoryEstimate(
        backend=backend,
        body_count=count,
        precision=str(descriptor.precision.value),
        components=tuple(parts),
    )


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _plan_parameters(plan: RuntimePlan, *modules: str) -> Mapping[str, Any]:
    """The parameters of the plan's step for one of ``modules``, or none."""

    for step in plan.steps:
        if step.module in modules:
            return step.parameters
    return {}


def _collision_mode(plan: RuntimePlan) -> CollisionMode:
    return CollisionMode(str(_plan_parameters(plan, _POINT_COLLISION_MODULE).get("mode", "ignore")))


def _collision_grid_bytes(count: int) -> int:
    """The collision broad phase's cell counts and cell items (``broadphase.py``)."""

    grid = BroadphaseConfig.for_capacity(count)
    return 4 * (grid.cell_capacity + grid.item_capacity)


def direct_memory_parts(
    count: int, scalar: int, plan: RuntimePlan, pinned: bool
) -> list[tuple[str, int]]:
    """``DirectNBodyRuntime.load_bodies``: eleven scalars and twenty bytes of words a body."""

    # Positions, velocities and accelerations are three scalars each, masses and radii one each;
    # then stable ids (8), active (4), body flags (4) and the claim counters (4).
    parts = [("body state", count * (11 * scalar + 20))]
    mode = _collision_mode(plan)
    if mode is not CollisionMode.IGNORE:
        parts.append(("collision grid", _collision_grid_bytes(count)))
    if mode is CollisionMode.ABSORB:
        # target (4), mass (8), momentum (24), position (24) and r^3 (8) for each body.
        parts.append(("absorption scratch", count * 68))
    if pinned:
        parts.append(("pinned anchors", count * (4 + 3 * scalar)))
    return parts


def tree_memory_parts(
    count: int, scalar: int, plan: RuntimePlan, pinned: bool
) -> list[tuple[str, int]]:
    """``BarnesHutRuntime.load_bodies``: the body state, the walk arrays and the node arrays."""

    parameters = _plan_parameters(plan, *_TREE_MODULES)
    depth = int(parameters.get("max_depth", DEFAULT_TREE_MAX_DEPTH))
    nodes = int(parameters.get("node_capacity", 0)) or sparse_node_capacity(count, depth)
    # Twenty-six scalars a body: six vectors of three (positions, velocities, accelerations and
    # the direct, slow and total accelerations), the walk's body quad of four, and masses, radii,
    # relative errors and tree masses. Then 36 bytes of words (stable id 8, sort key 8, active,
    # claim, walk first level, walk node count and walk offset 4 each), the per-level slot nodes,
    # and the two sort buffers (12 bytes a slot) padded to a power of two.
    body = count * (26 * scalar + 36 + 4 * (depth + 1)) + 12 * _next_power_of_two(count)
    parts = [
        ("tree body state", body),
        # Per node: levels (4), prefix (8), count, start and end (12), the walk links (16) and 21
        # scalars (mass, moment, quadrupole, centre quad, reach and the two diagonals).
        ("tree nodes", nodes * (40 + 21 * scalar)),
    ]
    if _collision_mode(plan) is not CollisionMode.IGNORE:
        parts.append(("collision grid", _collision_grid_bytes(count)))
    if pinned:
        parts.append(("pinned anchors", count * (4 + 3 * scalar)))
    return parts


def auto_memory_parts(
    count: int, scalar: int, plan: RuntimePlan, pinned: bool
) -> list[tuple[str, int]]:
    """``auto``: calibration builds Direct first and Tree beside it and keeps both until the
    choice is made; at or below the small-scene limit it never builds the Tree."""

    parts = direct_memory_parts(count, scalar, plan, pinned)
    if count > AUTO_SMALL_SCENE_DIRECT_LIMIT:
        parts.extend(tree_memory_parts(count, scalar, plan, pinned))
    return parts


def giant_impact_memory_parts(
    count: int, scalar: int, plan: RuntimePlan, pinned: bool
) -> list[tuple[str, int]]:
    """The moon-birth engine: neighbour and pressure tables, and the neighbour grid."""

    return [
        ("neighbour and pressure tables", count * GIANT_IMPACT_BYTES_PER_PARTICLE),
        ("neighbour grid", ENGINE_GRID_BYTES),
    ]


def spacesim_impact_memory_parts(
    count: int, scalar: int, plan: RuntimePlan, pinned: bool
) -> list[tuple[str, int]]:
    """The rock-impact engine: neighbour, pressure and bond tables, and the neighbour grid."""

    return [
        ("neighbour, pressure and bond tables", count * SPACESIM_IMPACT_BYTES_PER_PARTICLE),
        ("neighbour grid", ENGINE_GRID_BYTES),
    ]


# -- the decision ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionReport:
    """What the pre-flight decided for one scene, and why, in a form a diagnostic can print.

    ``checked`` is False when the scene was admitted without a comparison: the free memory could
    not be read, or the scene could not be sized. ``note`` says which, in words.
    """

    backend: str
    body_count: int
    needed_bytes: int
    free_bytes: int | None
    usable_bytes: int | None
    headroom: float
    checked: bool
    admitted: bool
    note: str
    components: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "body_count": self.body_count,
            "needed_bytes": self.needed_bytes,
            "free_bytes": self.free_bytes,
            "usable_bytes": self.usable_bytes,
            "headroom": self.headroom,
            "checked": self.checked,
            "admitted": self.admitted,
            "note": self.note,
            "components": {name: size for name, size in self.components},
        }


def _factory_error() -> type[Exception]:
    """The factory's own error type, imported late because the factory imports this module."""

    from .runtime_factory import ProductRuntimeError

    return ProductRuntimeError


def _mib_up(size: int) -> int:
    return max(1, math.ceil(size / MIB))


def refusal_message(needed_bytes: int, free_bytes: int, headroom: float) -> str:
    """The sentence an artist reads when a scene is too large for the graphics card."""

    reserve = round((1.0 - headroom) * 100)
    return (
        f"This scene needs about {_mib_up(needed_bytes):,} MiB of graphics memory and only "
        f"{free_bytes // MIB:,} MiB is free ({reserve}% of it is kept in reserve); "
        "lower the particle count."
    )


def allocation_failure_message(report: AdmissionReport | None) -> str:
    """The sentence for a failure the forecast did not see coming."""

    detail = ""
    if report is not None and report.needed_bytes > 0:
        detail = f" (it needs about {_mib_up(report.needed_bytes):,} MiB"
        if report.free_bytes is not None:
            detail += f"; {report.free_bytes // MIB:,} MiB was free when it started"
        detail += ")"
    return (
        "The graphics card ran out of memory while this scene was being set up"
        f"{detail}. Everything the scene had allocated has been released; lower the particle "
        "count and try again."
    )


def check_admission(
    estimate: MemoryEstimate,
    free_bytes: int | None,
    headroom: float = DEFAULT_HEADROOM,
    *,
    error_type: type[Exception] | None = None,
) -> AdmissionReport:
    """Admit the scene, or raise the factory's error with a plain reason it does not fit.

    A scene fits when its estimate is at most ``headroom`` of ``free_bytes``. ``free_bytes`` of
    None means the free memory could not be read, and the scene is admitted without a check:
    refusing a scene on a reading that never happened would stop every machine whose driver
    does not answer, and a scene that genuinely does not fit is caught by the allocation guard.
    The refusal carries its :class:`AdmissionReport` as ``report``.
    """

    if not 0.0 < headroom <= 1.0:
        raise ValueError("headroom must be in (0, 1]")
    needed = estimate.total_bytes
    if free_bytes is None:
        return AdmissionReport(
            backend=estimate.backend,
            body_count=estimate.body_count,
            needed_bytes=needed,
            free_bytes=None,
            usable_bytes=None,
            headroom=headroom,
            checked=False,
            admitted=True,
            note=(
                "free graphics memory could not be read, so the scene was admitted without "
                "a memory check"
            ),
            components=estimate.components,
        )
    usable = int(free_bytes * headroom)
    admitted = needed <= usable
    report = AdmissionReport(
        backend=estimate.backend,
        body_count=estimate.body_count,
        needed_bytes=needed,
        free_bytes=int(free_bytes),
        usable_bytes=usable,
        headroom=headroom,
        checked=True,
        admitted=admitted,
        note="" if admitted else refusal_message(needed, int(free_bytes), headroom),
        components=estimate.components,
    )
    if not admitted:
        refusal = (error_type or _factory_error())(report.note)
        refusal.report = report  # type: ignore[attr-defined]
        raise refusal
    return report


def read_free_device_bytes(context: DeviceContext | None = None) -> int | None:
    """The device's free memory in bytes, or None when it cannot be read.

    ``DeviceContext.initialize`` selects the device and allocates nothing, and Warp's
    ``free_memory`` is one driver query, so the read costs no graphics memory. The device is
    synchronised first, because Warp frees through a memory pool and a pool hands freed blocks
    back to the driver at a synchronisation: a read taken before one would count a runtime that
    was just closed as still in use, and refuse a scene that replaces it. Nothing raised here
    is allowed to escape: a machine with no usable driver is the factory's to report, with its
    own wording, when it builds the runtime. Zero is treated as unreadable because Warp leaves
    its output at zero when the driver call fails, and a card that is truly full is caught by
    :func:`allocation_failure_guard` at the first allocation.
    """

    owned = context is None
    device_context = context
    try:
        if device_context is None:
            device_context = DeviceContext()
        device_context.initialize()
        with contextlib.suppress(Exception):
            device_context.wp.synchronize_device(device_context.device)
        free = int(device_context.device.free_memory)
    except Exception:  # noqa: BLE001 - an unreadable query admits; it must never refuse
        return None
    finally:
        if owned and device_context is not None:
            # Closing is bookkeeping on the way out; it must not turn a read into a raise.
            with contextlib.suppress(Exception):
                device_context.close()
    return free if free > 0 else None


_last_admission: AdmissionReport | None = None


def last_admission() -> AdmissionReport | None:
    """The most recent decision, admitted or refused, for a diagnostic to print."""

    return _last_admission


def _record(report: AdmissionReport) -> AdmissionReport:
    global _last_admission
    _last_admission = report
    return report


def admit_scene(
    descriptor: SimulationDescriptor,
    plan: RuntimePlan | None = None,
    *,
    free_reader: Callable[[], int | None] | None = None,
    headroom: float = DEFAULT_HEADROOM,
    error_type: type[Exception] | None = None,
) -> AdmissionReport:
    """Estimate, read the free memory, decide and record: what the factory runs before it builds.

    ``free_reader`` is the injected seam: a callable returning the free bytes, or None. It
    defaults to :func:`read_free_device_bytes`, looked up when called so a test can replace the
    module's reader. Neither an estimate that cannot be made nor a reader that raises refuses
    anything; both admit and the report's note says why.
    """

    try:
        estimate = estimate_device_bytes(descriptor, plan)
    except Exception as exc:  # noqa: BLE001 - a forecast that fails must not stop a build
        resolved = plan if plan is not None else descriptor.runtime_plan
        return _record(
            AdmissionReport(
                backend=resolved.backend if resolved is not None else "unknown",
                body_count=0,
                needed_bytes=0,
                free_bytes=None,
                usable_bytes=None,
                headroom=headroom,
                checked=False,
                admitted=True,
                note=(
                    "the scene's memory could not be estimated "
                    f"({type(exc).__name__}: {exc}), so it was admitted without a memory check"
                ),
            )
        )
    reader = free_reader if free_reader is not None else read_free_device_bytes
    try:
        free = reader()
    except Exception:  # noqa: BLE001 - see read_free_device_bytes
        free = None
    try:
        report = check_admission(estimate, free, headroom, error_type=error_type)
    except Exception as refusal:
        refused = getattr(refusal, "report", None)
        if refused is not None:
            _record(refused)
        raise
    return _record(report)


# -- the allocation that got past the forecast -----------------------------------------------


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Visit every retained cause/context once, even when traceback display is suppressed."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)


def is_device_allocation_failure(error: BaseException) -> bool:
    """Whether ``error``, or a retained cause/context, is a device allocation failure.

    Narrow on purpose. Host-memory exhaustion (``MemoryError``, a failed allocation on 'cpu')
    is a different problem with a different remedy, and wording about graphics memory over it
    would send an artist after the wrong fix. The factory's own errors are not matched by their
    text: this module's refusals say "ran out of memory" themselves, and must never be read as a
    new failure by a second guard. Wrappers still match through their retained exceptions.
    """

    if getattr(error, "_astro_warp_allocation_refusal", False):
        return False
    factory_error = _factory_error()
    for current in _exception_chain(error):
        if (
            isinstance(current, RuntimeError)
            and not isinstance(current, factory_error)
            and _DEVICE_ALLOCATION_FAILURE.search(str(current))
        ):
            return True
    return False


def release_partial_runtimes(
    error: BaseException, extra_types: tuple[type, ...] = ()
) -> int:
    """Close every runtime a failed construction left reachable, then drop what it held.

    A runtime whose ``load_bodies`` raised is not returned to anyone, and its arrays are held by
    nothing but the frames in the exception's traceback (the caller's locals ``runtime`` and
    ``context``, and ``self`` in the frames below). Closing the runtimes found in those frames is
    the explicit release; clearing the frames then drops the locals that are not runtimes (a
    stage that builds its sub-runtimes in locals and assigns them only when the load finishes
    is held by nothing but the frame); the collection breaks any cycle left.
    ``extra_types`` names further closable classes to look for, which the factory uses for
    ``ProductRuntime``. Returns how many runtimes were closed. A failing close is ignored: this
    runs because something already failed, and must not replace that failure with another.
    """

    kinds = (BackendRuntime, DeviceContext, *extra_types)
    failures = tuple(_exception_chain(error))
    closed = 0
    seen: set[int] = set()
    for failure in failures:
        frame_entry = failure.__traceback__
        while frame_entry is not None:
            for value in tuple(frame_entry.tb_frame.f_locals.values()):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                # One odd local must not stop release, nor may a failing close replace the error.
                with contextlib.suppress(Exception):
                    if not isinstance(value, type) and isinstance(value, kinds):
                        value.close()
                        closed += 1
            frame_entry = frame_entry.tb_next
    for failure in failures:
        traceback.clear_frames(failure.__traceback__)
    gc.collect()
    return closed


@contextlib.contextmanager
def allocation_failure_guard(
    error_type: type[Exception] | None = None,
    *,
    report: AdmissionReport | None = None,
    closable_types: tuple[type, ...] = (),
) -> Iterator[None]:
    """Turn a device allocation failure inside the block into the factory's plain refusal.

    Anything else, including the factory's own refusals, passes through untouched. On a device
    allocation failure the half-built runtimes are closed and their frames cleared
    (:func:`release_partial_runtimes`) before the refusal is raised, so by the time the artist
    reads it the memory is back, a second context builds, and a smaller scene can be tried at
    once. The refusal suppresses the original traceback display; its retained chain's frames
    are cleared so they cannot keep allocations alive. Diagnostic text travels as notes.
    """

    try:
        yield
    except Exception as failure:
        if not is_device_allocation_failure(failure):
            raise
        reported = tuple(str(current) for current in _exception_chain(failure))
        release_partial_runtimes(failure, closable_types)
        friendly = (error_type or _factory_error())(allocation_failure_message(report))
        # A retained allocation context must not turn this completed refusal into a second one.
        friendly._astro_warp_allocation_refusal = True
        for message in reported:
            friendly.add_note(f"Warp reported: {message}")
    else:
        return
    raise friendly from None
