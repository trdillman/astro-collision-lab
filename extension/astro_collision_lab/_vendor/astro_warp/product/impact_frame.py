"""The frame transaction the two resumable impact runtimes share (ADR 0208, consequence C).

``GiantImpactRuntime`` and ``SpaceSimImpactRuntime`` both play one whole engine frame at a time:
one captured graph replay, one gather into the published channels, and one int read back (the
non-finite flag, reduced on the device). They used to carry two copies of that path, and a
fail-closed path that exists twice can fail open in one of them. This module holds the one copy.

What stays per engine is everything that is genuinely different: how the bodies are loaded, what
is gathered into the published channels (``_publish_state``), and the snapshot / restore /
diagnostics schema (the rock engine's bond, plastic and energy state has no counterpart in the
giant-impact engine). A subclass names its three class attributes and supplies ``_publish_state``;
the instance attributes below are set by its own ``__init__``.

Device-free at import, like the runtimes: the engines import Warp at module scope, this module
does not.
"""

from __future__ import annotations

import time
from typing import Any

from .model import Precision
from .runtime_foundation import DeviceRequirementError


class ImpactFrameRuntime:
    """The resumable-frame, display and fragment-report path of an impact runtime."""

    #: The class name the "no loaded bodies" refusal quotes.
    _ENGINE_LABEL: str
    #: The display channels this backend serves, in the order they are returned. A name the
    #: loaded scene did not allocate is omitted.
    _DISPLAY_CHANNELS: tuple[str, ...]
    #: Device-to-host readbacks one ``state()`` call costs, counted into
    #: ``explicit_snapshot_readbacks`` (the engines read different numbers of arrays).
    _LIVE_STATE_READBACKS: int

    # Set by the subclass's ``__init__``.
    device_context: Any
    config: Any
    frames_advanced: int
    last_step_ms: float
    last_solver_primitive_ms: float
    explicit_snapshot_readbacks: int
    full_state_playback_readbacks: int
    _layout: Any | None
    _solver: Any | None
    _arrays: dict[str, Any]
    _frame_started_at: float | None

    @property
    def frame_transaction_resumable(self) -> bool:
        return True

    @property
    def step_index(self) -> int:
        return self.frames_advanced

    def _require_solver(self) -> Any:
        if self._solver is None or self._layout is None:
            raise DeviceRequirementError(f"{self._ENGINE_LABEL} has no loaded bodies")
        return self._solver

    def _publish_state(self) -> None:
        """Gather the solver's state into the published display channels (per engine)."""

        raise NotImplementedError

    # -- the resumable frame ------------------------------------------------------------

    def begin_frame(self) -> None:
        self._require_solver()
        self._frame_started_at = time.perf_counter()

    def advance_frame_substep(self) -> None:
        """Run, publish, and check exactly one engine frame.

        One captured graph replay, one gather into the published channels, and one int read
        back: the non-finite flag, reduced on the device. Nothing per-particle crosses to the
        host, and a non-finite state fails closed rather than being published as a frame.
        """

        solver = self._require_solver()
        start = self._frame_started_at or time.perf_counter()
        self._frame_started_at = None
        solver.step_frame()
        self._publish_state()
        self.frames_advanced += 1
        bad = solver.nonfinite()
        self.last_solver_primitive_ms = (time.perf_counter() - start) * 1000.0
        if bad:
            raise DeviceRequirementError("non-finite particle state; step failed closed")

    def finish_frame(self) -> None:
        return None

    def step(self, steps: int = 1) -> None:
        self._require_solver()
        if steps < 1:
            raise ValueError("steps must be positive")
        start = time.perf_counter()
        for _ in range(steps):
            self.begin_frame()
            self.advance_frame_substep()
            self.finish_frame()
        self.last_step_ms = (time.perf_counter() - start) * 1000.0

    # -- display ------------------------------------------------------------------------

    def display_source(self) -> tuple[Any, int, Precision]:
        self._require_solver()
        return self._arrays["positions"], self.config.count, Precision.FLOAT32

    def display_channels(self) -> dict[str, Any]:
        """Device handles for every channel this backend serves (``_DISPLAY_CHANNELS``)."""

        self._require_solver()
        return {name: self._arrays[name] for name in self._DISPLAY_CHANNELS if name in self._arrays}

    # -- state --------------------------------------------------------------------------

    def _live_state(self) -> dict[str, Any]:
        state = self._require_solver().state()
        self.explicit_snapshot_readbacks += self._LIVE_STATE_READBACKS
        return state

    def fragment_report(self) -> dict[str, Any]:
        """What the bodies broke into, as of now (host work over one readback)."""

        np = __import__("numpy")
        from .impact import cluster_fragments

        solver = self._require_solver()
        layout = self._layout
        assert layout is not None
        state = self._live_state()
        report = cluster_fragments(
            state["positions"],
            np.asarray(layout.bodies.masses[layout.solver_slots], dtype=np.float64),
            self.config.linking_length_factor * float(solver.h),
            velocities=state["velocities"],
            gravitational_constant=self.config.gravitational_constant,
        )
        return report.to_dict()
