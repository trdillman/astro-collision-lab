"""Pure-Python contracts for Phase 0 runtime behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .constants import (
    DEFAULT_DISPLAY_CADENCE_STEPS,
    DEFAULT_DISPLAY_CAP,
    DEFAULT_DT,
    DEFAULT_MAX_TIMER_BUDGET_MS,
    DEFAULT_MAX_TIMER_SUBMISSIONS,
    DEFAULT_SIMULATION_COUNT,
    MANIFEST_SCHEMA,
    MAX_TIMER_BUDGET_MS_HARD_LIMIT,
    MAX_TIMER_SUBMISSIONS_HARD_LIMIT,
    WARP_VERSION,
)


class Phase0Fault(RuntimeError):
    """Visible runtime fault for unmet Phase 0 host requirements."""


class RuntimeStatus(StrEnum):
    UNLOADED = "UNLOADED"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAULTED = "FAULTED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class Phase0Config:
    """Configuration bounds for the Phase 0 proof runtime."""

    simulation_count: int = DEFAULT_SIMULATION_COUNT
    display_cap: int = DEFAULT_DISPLAY_CAP
    display_cadence_steps: int = DEFAULT_DISPLAY_CADENCE_STEPS
    max_timer_submissions_per_callback: int = DEFAULT_MAX_TIMER_SUBMISSIONS
    max_timer_budget_ms: float = DEFAULT_MAX_TIMER_BUDGET_MS
    dt: float = DEFAULT_DT

    def validate(self) -> None:
        if self.simulation_count < 2:
            raise ValueError("simulation_count must be at least 2")
        if self.display_cap < 1:
            raise ValueError("display_cap must be at least 1")
        if self.display_cadence_steps < 1:
            raise ValueError("display_cadence_steps must be at least 1")
        if not (1 <= self.max_timer_submissions_per_callback <= MAX_TIMER_SUBMISSIONS_HARD_LIMIT):
            raise ValueError(
                "max_timer_submissions_per_callback must be between "
                f"1 and {MAX_TIMER_SUBMISSIONS_HARD_LIMIT}"
            )
        if not (0.0 < self.max_timer_budget_ms <= MAX_TIMER_BUDGET_MS_HARD_LIMIT):
            raise ValueError(
                f"max_timer_budget_ms must be > 0 and <= {MAX_TIMER_BUDGET_MS_HARD_LIMIT}"
            )
        if not (0.0 < self.dt <= 1.0):
            raise ValueError("dt must be > 0 and <= 1.0 for Phase 0")
        if self.display_count >= self.simulation_count:
            raise ValueError("display_count must be smaller than simulation_count")

    @property
    def display_count(self) -> int:
        return min(self.display_cap, self.simulation_count - 1)

    @property
    def display_stride(self) -> int:
        return max(1, (self.simulation_count + self.display_count - 1) // self.display_count)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["display_count"] = self.display_count
        data["display_stride"] = self.display_stride
        return data


@dataclass(frozen=True)
class ServiceResult:
    """Result from one bounded runtime service call."""

    submissions: int
    elapsed_ms: float
    step_index: int
    display_refreshed: bool
    allocation_identity: int | str | None


@dataclass(frozen=True)
class DisplaySnapshot:
    """Reduced host-staged display copy.

    ``positions`` is a ``(display_count, 3)`` float32 host buffer that the runtime
    owns and reuses: the next display refresh overwrites it in place instead of
    allocating, so a consumer that needs to retain a sample must copy it. Building
    a fresh per-point Python sequence here measures 1.02 ms per 1000 points and is
    rejected by `adr/0008-display-interop-boundary.md`.
    """

    step_index: int
    simulation_count: int
    display_count: int
    display_copy_ms: float
    positions: Any


@dataclass(frozen=True)
class EnvironmentReport:
    """Parseable environment/run manifest contract."""

    schema: str
    generated_at_utc: str
    status: str
    source_revision: str
    source_dirty: bool | None
    blender: dict[str, Any]
    python: dict[str, Any]
    warp: dict[str, Any]
    numpy: dict[str, Any]
    device: dict[str, Any]
    config: dict[str, Any]
    run: dict[str, Any]
    display: dict[str, Any]
    lifecycle: dict[str, Any]
    checks: dict[str, Any]
    faults: list[str] = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        *,
        status: RuntimeStatus,
        config: Phase0Config,
        source_revision: str = "unknown",
        source_dirty: bool | None = None,
        faults: list[str] | None = None,
    ) -> EnvironmentReport:
        now = datetime.now(UTC).isoformat()
        return cls(
            schema=MANIFEST_SCHEMA,
            generated_at_utc=now,
            status=status.value,
            source_revision=source_revision,
            source_dirty=source_dirty,
            blender={},
            python={},
            warp={"required_version": WARP_VERSION},
            numpy={"policy": "use Blender-provided NumPy; not bundled"},
            device={"required": "cuda:0"},
            config=config.to_json_dict(),
            run={},
            display={},
            lifecycle={},
            checks={},
            faults=faults or [],
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_position_after_steps(
    *,
    steps: int,
    dt: float,
    initial: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Expected output for the Phase 0 deterministic advance kernel."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    delta = float(steps) * float(dt)
    return (
        initial[0] + delta,
        initial[1] + (2.0 * delta),
        initial[2] + (0.5 * delta),
    )


def warp_version_compatible(found: str, required: str = WARP_VERSION) -> bool:
    """Is an installed Warp acceptable against the version this add-on was built for?

    A floor, not an exact match (ADR 0024). Same major, minor at or above the required
    minor. Blender ships its own Warp and it wins the import, so an exact comparison
    meant a host one patch release ahead was refused for no reason an artist could act
    on. A different major is still refused: Warp's kernel API is not stable across one,
    and accepting it would mean crashing later instead of saying so now.

    Anything that does not read as ``major.minor`` with both parts numeric is refused
    rather than guessed at, so a malformed version can never widen the gate.

    A near-copy of this lives in ``product.runtime_foundation``. The duplication is
    deliberate -- ``phase0`` stands alone and ``product`` imports nothing from it --
    and ``test_warp_version_acceptance`` asks both the same questions.
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


def assert_warp_version_compatible(version: str) -> None:
    if not warp_version_compatible(version):
        raise Phase0Fault(
            f"Astro Warp Phase 0 requires Warp {WARP_VERSION} or newer, in the same "
            f"major version; found {version!r}"
        )


def bounded_submission_count(requested: int, config: Phase0Config) -> int:
    config.validate()
    if requested <= 0:
        return 0
    return min(requested, config.max_timer_submissions_per_callback)


def should_refresh_display(step_index: int, last_display_step: int, cadence_steps: int) -> bool:
    if cadence_steps < 1:
        raise ValueError("cadence_steps must be at least 1")
    if last_display_step < 0:
        return True
    if step_index < last_display_step:
        # A backward seek --- timeline loop back to frame_start, a scrub, a checkpoint
        # restore --- is not "too soon since the last refresh", it is a different state
        # wearing an older step number. Gating it on the forward cadence leaves the
        # viewport showing the pre-seek frame until the replay climbs back past the step
        # that was last staged, which reads as the simulation having stopped.
        return True
    return (step_index - last_display_step) >= cadence_steps
