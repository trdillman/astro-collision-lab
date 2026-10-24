"""Phase 1 runtime ownership primitives."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .model import ContextState, DeviceDescriptor, Precision, SimulationDescriptor
from .sampling import body_count


class DeviceRequirementError(RuntimeError):
    """Raised when Warp/CUDA requirements are unmet."""


@runtime_checkable
class BackendRuntime(Protocol):
    """The surface every backend solver runtime presents to ``ProductRuntime``.

    ``DirectNBodyRuntime`` (direct), ``BarnesHutRuntime`` (tree/auto),
    ``GiantImpactRuntime`` (giant_impact) and ``SpaceSimImpactRuntime`` (spacesim_impact)
    each implement this independently; the factory
    binds exactly one of them into ``ProductRuntime.runtime`` and drives playback,
    checkpointing, and display staging through these methods alone. Typing the seam
    here means a slice-4/5 backend extends one contract instead of the factory
    re-learning each backend's quirks — the duplication the 2026-08-18 review (H2)
    flagged.

    Two capabilities are deliberately *absent* from the required surface because only
    some backends offer them and ``ProductRuntime`` reaches them defensively:

    * ``last_step_ms`` — a ``float`` the bounded scheduler reads via
      ``getattr(runtime, "last_step_ms", 0.0)``; present on all three backends today
      but read through a default so a future backend need not carry it.
    * ``apply_parameters`` — the in-place solver-parameter adoption path, split into
      :class:`Reparameterizable` because the Solid runtimes (which replay recorded
      device graphs) intentionally do not offer it and ``reparameterize`` probes for
      it with ``getattr``.
    """

    def step(self, steps: int = 1) -> None: ...

    def diagnostics(self) -> Any: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    def display_source(self) -> tuple[Any, int, Precision]: ...

    def display_channels(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class ResumableFrameRuntime(Protocol):
    """Optional backend capability for yielding between authored substeps.

    Each ``advance_frame_substep`` call must synchronize the device and surface solver
    errors before it returns. ``ProductRuntime`` owns the partial-frame counter so a
    checkpoint restore or runtime replacement can clear progress together with replacing
    the authoritative device state.
    """

    def begin_frame(self) -> None: ...

    def advance_frame_substep(self) -> None: ...

    def finish_frame(self) -> None: ...


@dataclass(frozen=True)
class FramePrimitiveProgress:
    """Immutable outcome of one synchronized backend-owned work slice."""

    phase: str
    authored_substep_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("primitive progress phase must be non-empty")
        if type(self.authored_substep_complete) is not bool:
            raise TypeError("authored_substep_complete must be a bool")


@runtime_checkable
class PrimitiveFrameRuntime(Protocol):
    """Optional frame surface for substeps split across bounded work slices."""

    def begin_frame(self) -> None: ...

    def advance_frame_primitive(self) -> FramePrimitiveProgress: ...

    def finish_frame(self) -> None: ...


@runtime_checkable
class Reparameterizable(Protocol):
    """A backend that can adopt a solver-parameter edit in place (PRODUCT-007).

    ``ProductRuntime.reparameterize`` probes for this with ``getattr`` so a backend
    that replays recorded device graphs (the Solid engines) can decline and force a rebuild.
    ``DirectNBodyRuntime`` and ``BarnesHutRuntime`` implement it; the Solid runtimes do not.
    """

    def apply_parameters(
        self,
        *,
        gravitational_constant: float,
        softening: float,
        dt: float,
        substeps: int,
    ) -> None: ...


def warp_version_compatible(found: str, required: str) -> bool:
    """Is an installed Warp acceptable against the version the product was built for?

    A floor, not an exact match (ADR 0024). Same major, minor at or above the required
    minor. Blender ships its own Warp and it wins the import, so an exact comparison
    refused a host one patch release ahead for no reason an artist could act on. A
    different major is still refused: Warp's kernel API is not stable across one, and
    accepting it would mean crashing mid-shot rather than saying so before the first
    step. Anything that does not read as ``major.minor`` with both parts numeric is
    refused rather than guessed at.

    A near-copy lives in ``phase0.contracts``, deliberately: ``phase0`` stands alone and
    this layer imports nothing from it. ``test_warp_version_acceptance`` runs one table
    against both so the copies cannot drift apart in silence.
    """

    def _floor(version: str) -> tuple[int, int] | None:
        parts = version.split(".")
        if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            return None
        return int(parts[0]), int(parts[1])

    found_floor = _floor(found)
    required_floor = _floor(required)
    if found_floor is None or required_floor is None:
        return False
    return found_floor[0] == required_floor[0] and found_floor[1] >= required_floor[1]


@dataclass
class DeviceContext:
    """Owns the required Warp CUDA device for product runtimes."""

    descriptor: DeviceDescriptor = field(default_factory=DeviceDescriptor)
    state: ContextState = ContextState.UNLOADED
    warp_version: str = "unprobed"
    device_name: str = "unprobed"
    _wp: Any | None = field(default=None, init=False, repr=False)
    _device: Any | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        if self.state == ContextState.READY:
            return
        try:
            wp = importlib.import_module("warp")
        except Exception as exc:  # noqa: BLE001 - visible requirement failure
            self.state = ContextState.FAULTED
            raise DeviceRequirementError(f"failed to import Warp: {exc}") from exc

        self.warp_version = str(getattr(wp, "__version__", "unknown"))
        required = self.descriptor.required_warp_version
        if not warp_version_compatible(self.warp_version, required):
            self.state = ContextState.FAULTED
            raise DeviceRequirementError(
                f"Astro Warp requires Warp {required} or newer, in the same major "
                f"version; found {self.warp_version}"
            )
        if self.descriptor.cuda_required and not bool(wp.is_cuda_available()):
            self.state = ContextState.FAULTED
            raise DeviceRequirementError("CUDA is not available; product runtime requires cuda:0")
        try:
            device = wp.get_device(self.descriptor.required_device)
        except Exception as exc:  # noqa: BLE001 - device failure must stay visible
            self.state = ContextState.FAULTED
            raise DeviceRequirementError(
                f"failed to select {self.descriptor.required_device}: {exc}"
            ) from exc

        self._wp = wp
        self._device = device
        self.device_name = str(device)
        self.state = ContextState.READY

    @property
    def wp(self):
        if self._wp is None:
            raise DeviceRequirementError("DeviceContext is not initialized")
        return self._wp

    @property
    def device(self):
        if self._device is None:
            raise DeviceRequirementError("DeviceContext is not initialized")
        return self._device

    def close(self) -> None:
        self._device = None
        self._wp = None
        self.state = ContextState.CLOSED


@dataclass
class SchedulerState:
    submitted_steps: int = 0
    checkpoints_written: int = 0
    restored_checkpoint_index: int = -1


@dataclass
class SimulationContext:
    """Owns one compiled simulation descriptor and its persistent GPU arena."""

    descriptor: SimulationDescriptor
    device_context: DeviceContext
    state: ContextState = ContextState.UNLOADED
    scheduler: SchedulerState = field(default_factory=SchedulerState)
    arena_capacity: int = 0
    checkpoint_indices: list[int] = field(default_factory=list)
    _arena: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def initialize(self) -> None:
        if self.state == ContextState.READY:
            return
        descriptor = self.descriptor
        if descriptor.runtime_plan is None:
            self.state = ContextState.FAULTED
            raise DeviceRequirementError(
                "descriptor must be compiled before context initialization"
            )
        self.device_context.initialize()
        self._allocate_arena(max(1, body_count(descriptor.bindings)))
        self.state = ContextState.READY

    def _allocate_arena(self, capacity: int) -> None:
        wp = self.device_context.wp
        device = self.device_context.device
        if self.arena_capacity == capacity and self._arena:
            return
        self._arena = {
            "stable_ids": wp.empty(capacity, dtype=wp.int64, device=device),
            "active": wp.empty(capacity, dtype=wp.int32, device=device),
        }
        self.arena_capacity = capacity

    def write_checkpoint_marker(self) -> int:
        self.scheduler.checkpoints_written += 1
        self.checkpoint_indices.append(self.scheduler.submitted_steps)
        return self.scheduler.checkpoints_written - 1

    def restore_checkpoint_marker(self, index: int) -> None:
        if index < 0 or index >= len(self.checkpoint_indices):
            raise IndexError("checkpoint index out of range")
        self.scheduler.restored_checkpoint_index = index
        self.scheduler.submitted_steps = self.checkpoint_indices[index]

    def close(self) -> None:
        self._arena.clear()
        self.arena_capacity = 0
        self.state = ContextState.CLOSED
