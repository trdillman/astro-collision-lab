"""Bounded viewport display staging for live product runtimes.

`adr/0003-display-readback-boundary.md` fixes the boundary: ordinary playback never reads
back full simulation state. `adr/0008-display-interop-boundary.md` fixes the transfer: a
device sample into a pinned host array, one persistent numpy view of it, and one upload
path in the add-on's draw handler. Phase 0 proved both, but it proved them on the Phase 0
demo runtime, which was also the only runtime the viewport handler could read. A product
context therefore solved on the device and drew nothing at all. This module is that same
staging for the product runtimes, so the proven boundary and the measured transfer serve
the real solvers instead of the demo one.

Two properties make the sample bounded staging rather than full-state readback:

- Only channels explicitly requested by a bounded consumer leave the device, and at most
  ``cap`` slots of them, whatever the body count. The viewport requests drawable channels;
  the scrub cache additionally requests the accepted raw Color By sources. Both use the
  same device selection plan and host-copy batch. Unrequested solver state never crosses
  the bus.
- The stride walks the body index order, which is the stable-id order the runtime was
  loaded in, so a display slot follows the same body for the life of the runtime instead
  of re-shuffling on every refresh (PRODUCT-009).

A body that a merge deactivated is no longer sampled at all. This module used to keep it,
on the reasoning that a compaction pass cost more than drawing a point that had stopped
moving --- but the point does not merely stop moving, it stays in the viewport forever as a
frozen ghost of a body the solver has destroyed, and every later frame draws it. ADR 0046
replaces the frozen count with a device stream compaction over the per-body ``active`` flag
every N-body kernel already honours: `compacted_plan` states the arithmetic, `_plan_kernels`
implements it, and the same mechanism serves the Emitter role, whose population grows
through that identical flag. A producer that offers no ``active`` channel keeps exactly the
strided gather it always had (`strided_plan`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .nbody import Precision, _dtypes, _registrars
from .runtime_foundation import DeviceContext, DeviceRequirementError

# Measured at 0.232 ms per refresh for 65536 points on the tested workstation
# (adr/0008-display-interop-boundary.md), which is 1.4% of a 60 Hz frame, so the default
# shows a large cloud rather than a token subset and still leaves the frame to the solver.
# ADR 0139 doubled it from 65,536 so a 100,000-body scene draws every body: the whole
# draw was 1.4 ms at 65,536 in a window, and ADR 0008's table grows linearly past it.
DEFAULT_DISPLAY_CAP = 131072
DEFAULT_DISPLAY_CADENCE_STEPS = 1

# Which producer channel each ramp reads. A ``color_source`` outside this table --- ``FLAT``
# and ``MATERIAL`` --- is a look the host resolves without per-particle data, so it stages
# no colour channel at all rather than staging a constant one body-count times.
_RAMP_FIELDS = {
    "SPEED": "velocities",
    "DENSITY": "densities",
    "DAMAGE": "damage",
    "ENERGY": "internal_energies",
}

# The four stops the live ramp interpolates between, named here rather than written into
# the kernel, because there are now two implementations of the same ramp: the device kernel
# below, which colours a sample the display stage stages itself, and `derive_ramp_colors`,
# which colours a sample that arrived from somewhere else --- a scrub-cache record, whose
# channels are fixed by its schema and never include colour. Two ramps that drifted would
# show one scene in two palettes depending on which producer served the frame, and nothing
# in the picture would say which one it was. One table, two readers, and a unit test
# holding them to each other.
VIRIDIS_STOPS: tuple[tuple[float, float, float], ...] = (
    (0.267, 0.005, 0.329),
    (0.188, 0.408, 0.557),
    (0.208, 0.718, 0.473),
    (0.993, 0.906, 0.144),
)


def ramp_scalars(color_source: str, sample: Any) -> Any:
    """The scalar field ``color_source`` ramps, read off an already-staged sample.

    ``None`` when this source needs no scalar (``FLAT``, ``MATERIAL``) or when the sample
    does not carry the channel it reads --- which is a fact about the producer, not an
    error: the gravity-only engines (direct N-body, tree) publish no densities, so a scrub
    record of theirs has nothing for ``DENSITY`` to ramp and says so by returning nothing. The
    rock-impact and giant-impact engines do publish one, and their records carry it.

    The reduction matches the device kernels exactly: ``SPEED`` is ``wp.length(velocity)``
    and ``DENSITY`` is the density itself.
    """

    if color_source in DERIVED_SCALAR_SOURCES:
        positions = getattr(sample, "positions", None)
        if positions is None or len(positions) == 0:
            return None
        points = np.asarray(positions, dtype=np.float32)
        return np.linalg.norm(points - points.mean(axis=0), axis=1).astype(np.float32, copy=False)
    field = _RAMP_FIELDS.get(color_source)
    if field is None:
        return None
    values = getattr(sample, field, None)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim == 2:
        return np.linalg.norm(array, axis=1).astype(np.float32, copy=False)
    return array.reshape(-1)


def ramp_fraction(scalars: Any, low: float, high: float) -> Any:
    """Where each scalar sits on the ramp, as float32 in ``[0, 1]``.

    The normalisation `ramp_colors` applies before it picks a colour, and the value the
    rendered points material ramps (ADR 0189): a Blender Color Ramp clamps its factor to
    ``[0, 1]``, so a raw speed of 5 renders the same colour as a speed of 1. Handing the
    material this fraction instead is what makes F12 colour by the range the viewport drew.
    """

    values = np.asarray(scalars, dtype=np.float32).reshape(-1)
    lower = float(low)
    upper = float(high)
    # A non-finite scalar (NaN speed or density, an inf from an overflowed field) must not
    # reach the ramp: NaN survives clip and `floor(NaN).astype(int32)` is undefined. It is
    # drawn at the low stop (t = 0) and excluded from the auto-range, as in the kernel.
    finite = np.isfinite(values)
    if lower >= upper:
        if finite.any():
            lower = float(values[finite].min())
            upper = float(values[finite].max())
        else:
            lower = upper = 0.0
    span = upper - lower
    if span > 0.0:
        t = np.where(finite, (np.where(finite, values, lower) - lower) / span, 0.0).astype(
            np.float32, copy=False
        )
    else:
        t = np.zeros_like(values)
    return np.clip(t, 0.0, 1.0).astype(np.float32, copy=False)


def ramp_colors(scalars: Any, low: float, high: float) -> Any:
    """`sample_colors` on the host: normalise a scalar field and ramp it to RGBA.

    Auto-ranges on equal or inverted bounds and draws the low stop for a degenerate span,
    both for the reasons the kernel does --- see `DisplayChannelSpec.ramp_low` and
    `sample_colors`. Returns a ``(N, 4)`` float32 array with alpha 1, which is the layout
    `ParticleRenderArrays` wants, so a caller hands it straight to a packet.
    """

    s = ramp_fraction(scalars, low, high) * 3.0
    # The last stop is reached at s == 3.0 exactly, which would index one past the table;
    # it is the closing end of the third segment, so it is clamped into that segment and
    # arrives as its local coordinate 1.0 instead.
    segment = np.clip(np.floor(s), 0.0, 2.0)
    local = (s - segment)[:, None]
    stops = np.asarray(VIRIDIS_STOPS, dtype=np.float32)
    index = segment.astype(np.int32)
    colors = np.empty((s.size, 4), dtype=np.float32)
    colors[:, :3] = stops[index] + (stops[index + 1] - stops[index]) * local
    colors[:, 3] = 1.0
    return colors


def derive_ramp_colors(color_source: str, sample: Any, low: float, high: float) -> Any:
    """The RGBA a look asks for, derived on the host from a sample carrying no colour.

    The live viewport does not draw the display stage's own sample during playback: under
    ADR 0088 every played frame is a scrub-cache commit, and ``_SCRUB_CHANNEL_SPEC`` fixes a
    record's channels to what the *record* has to be able to answer later, colour excluded,
    because a record outlives the look that was active when it was written. So the colour a
    played frame draws has to be derived where it is drawn, from the channels the record
    does carry. That is this function.

    ``None`` means the look cannot be served from this sample and the caller should fall
    back to flat --- and say so, which `draw._channel_fallbacks` does.
    """

    scalars = ramp_scalars(color_source, sample)
    if scalars is None:
        return None
    return ramp_colors(scalars, low, high)


# AW-011 / AW-013. A look source is one of three kinds, and the kind decides how it may be
# coloured. A *measurement* (speed, density, distance) is a number on a scale, so it goes
# through the ramp. A *label* (material, source object) is an identity, so it goes through a
# fixed palette: stretching a label over a smooth ramp recolours every survivor when one
# label disappears, which is the "glitch" AW-013 reported. A *direct* source already is a
# colour (the source object's own material colour).
CATEGORICAL_FIELDS = {"MATERIAL": "material_ids", "SOURCE_OBJECT": "source_index"}
DIRECT_COLOR_SOURCES = frozenset({"OBJECT_COLOR"})
# Distance is derived from the sample's own positions rather than staged as a channel.
DERIVED_SCALAR_SOURCES = frozenset({"DISTANCE"})

# Okabe-Ito order, which stays distinguishable under the common colour-vision deficiencies,
# plus two more so ten labels do not repeat. A label's colour is its index into this table,
# never its rank among the labels currently visible.
CATEGORICAL_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.902, 0.624, 0.000),
    (0.337, 0.706, 0.914),
    (0.000, 0.620, 0.451),
    (0.941, 0.894, 0.259),
    (0.000, 0.447, 0.698),
    (0.835, 0.369, 0.000),
    (0.800, 0.475, 0.655),
    (0.600, 0.600, 0.600),
    (0.451, 0.800, 0.200),
    (0.700, 0.200, 0.200),
)
_UNKNOWN_LABEL_COLOR = (0.35, 0.35, 0.35)


def categorical_colors(labels: Any) -> Any:
    """RGBA for integer labels: a pure function of each label, never of its neighbours.

    A negative label means "no owner" and draws a neutral grey. Returns ``(N, 4)`` float32
    with alpha 1, the layout `ramp_colors` returns.
    """

    values = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    palette = np.asarray(CATEGORICAL_PALETTE, dtype=np.float32)
    colors = np.empty((values.size, 4), dtype=np.float32)
    colors[:, :3] = palette[np.mod(values, len(palette))]
    colors[values < 0, :3] = _UNKNOWN_LABEL_COLOR
    colors[:, 3] = 1.0
    return colors


class AutoRange:
    """The colour range an auto-ranged measurement ramps over (AW-013).

    The old behaviour fitted the ramp to each frame's own minimum and maximum, so one fast
    particle arriving or leaving recoloured the whole scene. This range instead

    - uses the 1st to 99th percentile, so one stray value cannot wash out every other colour;
    - only ever grows once started, so a particle whose own value has not changed keeps its
      colour (it can only move toward the cool end as the range widens, never flicker);
    - resets when the sample's step goes backwards, i.e. when the timeline returns to the
      start, because the range then belongs to a run that no longer exists;
    - can be ``locked``, which freezes it for a render.

    It never returns an equal pair, because `ramp_colors` reads equal bounds as "auto-range
    this frame", which would put the per-frame fitting straight back.
    """

    LOW_PERCENTILE = 1.0
    HIGH_PERCENTILE = 99.0

    def __init__(self) -> None:
        self.low: float | None = None
        self.high: float | None = None
        self.locked = False
        self._step: int | None = None

    def reset(self) -> None:
        self.low = None
        self.high = None
        self._step = None

    @property
    def bounds(self) -> tuple[float, float] | None:
        if self.low is None or self.high is None:
            return None
        return self._widened(self.low, self.high)

    @staticmethod
    def _widened(low: float, high: float) -> tuple[float, float]:
        if high > low:
            return low, high
        return low, low + max(abs(low), 1.0) * 1e-6

    def _range_cannot_grow(self, values: Any) -> bool:
        """Whether this frame's percentiles provably lie inside the held range (ADR 0229).

        The range only grows, so a frame changes it only if its 1st percentile falls below
        ``low`` or its 99th rises above ``high``. Linear interpolation puts the 1st percentile
        at or above the sorted value at the floor of its virtual index, so if no more values
        than that index sit below ``low``, the percentile is at least ``low``; the 99th
        mirrors this at the ceiling. Two counts replace a partition of every value. The
        indices are taken one place on the safe side of NumPy's own, so rounding can only
        make this say "might grow" and fall through to the exact sort, never the reverse.
        """

        if self.low is None or self.high is None:
            return False
        last = values.size - 1
        below_index = int(last * (self.LOW_PERCENTILE / 100.0)) - 1
        above_index = int(last * (self.HIGH_PERCENTILE / 100.0)) + 2
        if below_index < 0 or above_index > last:
            return False
        below = int(np.count_nonzero(values < self.low))
        if below > below_index:
            return False
        above = int(np.count_nonzero(values > self.high))
        return above <= last - above_index

    def update(self, scalars: Any, step: int | None = None) -> tuple[float, float]:
        """Fold one frame's scalars in and return the range to ramp over."""

        if step is not None:
            if self._step is not None and step < self._step and not self.locked:
                self.reset()
            self._step = step
        held = self.bounds
        if held is not None and self.locked:
            return held
        values = np.asarray(scalars, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size and held is not None and self._range_cannot_grow(values):
            # The held bounds already cover this frame's percentiles, so the sort below
            # would leave them exactly as they are (ADR 0229).
            return held
        if values.size:
            # One call for both bounds: it partitions the scalars once instead of twice
            # (ADR 0222). The quantiles are float32 on purpose. NumPy interpolates a float32
            # input in float32 for a scalar quantile but in float64 for a list or float64
            # array of them, which moves the last bits; a float32 array of quantiles takes
            # the scalar's path, so the pair is bit-identical to two separate calls.
            low_value, high_value = np.percentile(
                values, np.array((self.LOW_PERCENTILE, self.HIGH_PERCENTILE), dtype=np.float32)
            )
            low = float(low_value)
            high = float(high_value)
            self.low = low if self.low is None else min(self.low, low)
            self.high = high if self.high is None else max(self.high, high)
        bounds = self.bounds
        return bounds if bounds is not None else (0.0, 1.0)


def derive_look_colors(
    color_source: str,
    sample: Any,
    low: float,
    high: float,
    auto_range: AutoRange | None = None,
    *,
    labels: Any = None,
    direct_colors: Any = None,
) -> Any:
    """The RGBA a look asks for, by the kind of source it is (AW-011, AW-013).

    ``labels`` and ``direct_colors`` carry what the sample itself cannot: a source-object
    index and an object-colour table live with the runtime's descriptor, not in a scrub
    record. ``None`` means this sample cannot serve the look and the caller falls back to
    flat and says so.
    """

    if color_source in DIRECT_COLOR_SOURCES:
        if direct_colors is None or len(direct_colors) == 0:
            return None
        return np.asarray(direct_colors, dtype=np.float32)
    field = CATEGORICAL_FIELDS.get(color_source)
    if field is not None:
        if labels is None:
            labels = getattr(sample, field, None)
        if labels is None or np.asarray(labels).size == 0:
            return None
        return categorical_colors(labels)
    scalars = ramp_scalars(color_source, sample)
    if scalars is None:
        return None
    if float(low) < float(high) or auto_range is None:
        return ramp_colors(scalars, low, high)
    step = getattr(sample, "step_index", None)
    lower, upper = auto_range.update(scalars, step=None if step is None else int(step))
    return ramp_colors(scalars, lower, upper)


# What each optional channel needs `display_channels()` to offer. Checked before the first
# launch so a lane that cannot serve a channel says which field it lacks, rather than
# raising ``KeyError`` from inside the refresh.
#
# A channel can have more than one physically valid source: ``sizes`` reads either the
# direct N-body backend's real per-body radius, or the mass-and-density derivation the
# ``sample_sizes`` kernel evaluates for a producer that publishes densities (ADR 0023 §7
# amendment --- the direct lane has no densities to derive from, but its radius already IS
# the physical size; no engine publishes densities since the compressible lane was deleted).
# Each value is therefore a tuple of *alternative* field-tuples rather than one field-tuple,
# in declaration-order preference, and a channel is served when the offered channels are a
# superset of at least one alternative. ``velocities`` has exactly one alternative, wrapped
# the same way so every caller resolves both channels through the one mechanism.
_CHANNEL_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "velocities": (("velocities",),),
    "sizes": (("radii",), ("masses", "densities")),
    "stable_ids": (("stable_ids",),),
    "damage": (("damage",),),
    "material_ids": (("material_ids",),),
    "internal_energies": (("internal_energies",),),
    "densities": (("densities",),),
}


def _channel_alternative(name: str, offered: Any) -> tuple[str, ...]:
    """The first alternative source ``name`` can be served from, given ``offered``.

    Declaration order is preference order, checked here and nowhere else: `_fields_for`,
    `requirements`, `supported_by`, and `DisplayStage._launch_channel` all resolve a
    channel's concrete source through this one function, so which fields a channel
    reads and which fields it is launched with can never disagree. When no alternative is
    fully offered, the first declared alternative is returned anyway --- it is not a claim
    that the channel is served (`supported_by` already turns it off in that case), only a
    concrete answer for a caller, such as the missing-field message in `DisplayStage.refresh`,
    that needs *a* requirement to name.
    """

    available = set(offered)
    alternatives = _CHANNEL_REQUIREMENTS[name]
    for alternative in alternatives:
        if available.issuperset(alternative):
            return alternative
    return alternatives[0]


@dataclass(frozen=True)
class DisplayChannelSpec:
    """Which optional channels a refresh stages, and how the colour ramp is normalised.

    It is a value so `ProductRuntime.ensure_display` can compare it: the pinned staging
    behind a stage is what `adr/0008-display-interop-boundary.md` measures, and rebuilding
    it on a refresh that asked for the same channels would throw that away every frame.

    The fields are named for the add-on's render properties (`draw.RenderSpec`), because
    what the viewport asks for is exactly what the device has to produce.
    """

    velocities: bool = False
    sizes: bool = False
    # Not a render property: no ramp reads it and no vertex buffer receives it. It is
    # staged because ADR 0046's compaction breaks the one thing a consumer used to be able
    # to assume --- that row *n* of the sample is body *n* of the descriptor's binding
    # order. A consumer that has to know *which* bodies it received, rather than only how
    # many, asks for this channel and reads the answer off the sample itself. The scrub
    # lane is the one such consumer today.
    stable_ids: bool = False
    # Raw D13 sources are cache carriage, not renderer inputs. They share the bounded
    # selection plan and copy batch so a scrub commit never reads full authoritative state.
    damage: bool = False
    material_ids: bool = False
    internal_energies: bool = False
    # The per-particle mass density, staged as cache carriage so a rewound frame can still be
    # coloured by it. Only the impact engines offer it (`supported_by` turns it off elsewhere).
    densities: bool = False
    color_source: str = "FLAT"
    # Equal or inverted bounds auto-range: the ramp is normalised against the sampled
    # minimum and maximum, reduced on the device in the same launch that extracts the
    # scalar. An inverted pair reads as "I did not choose bounds" rather than as a
    # reversed ramp, so a mis-dragged slider auto-ranges instead of drawing one flat
    # colour with no way to tell that from a converged field.
    ramp_low: float = 0.0
    ramp_high: float = 0.0

    @property
    def colors(self) -> bool:
        return self.color_source in _RAMP_FIELDS

    @property
    def names(self) -> tuple[str, ...]:
        """The optional channels this spec stages, in launch order."""

        enabled = [
            name
            for name in (
                "velocities",
                "sizes",
                "stable_ids",
                "damage",
                "material_ids",
                "internal_energies",
                "densities",
            )
            if getattr(self, name)
        ]
        if self.colors:
            enabled.append("colors")
        return tuple(enabled)

    def requirements(self, offered: Any) -> tuple[str, ...]:
        """Every producer field this spec reads, deduplicated and ordered.

        ``offered`` resolves which alternative a multi-source channel (``sizes``) reads
        before its fields are listed --- without it the two ``sizes`` alternatives could
        not be told apart, and a lane offering neither would be reported as missing the
        wrong fields. ``requirements()`` and `supported_by` are always called with the
        same offered channels in `DisplayStage.refresh`, so both resolve identically.
        """

        needed: list[str] = []
        for name in self.names:
            for field_name in self._fields_for(name, offered):
                if field_name not in needed:
                    needed.append(field_name)
        return tuple(needed)

    def _fields_for(self, name: str, offered: Any) -> tuple[str, ...]:
        if name == "colors":
            return (_RAMP_FIELDS[self.color_source],)
        return _channel_alternative(name, offered)

    def supported_by(self, offered: Any) -> DisplayChannelSpec:
        """This spec with every channel the producer cannot feed turned off.

        Which channels exist is a property of the lane --- a producer may carry densities, the
        direct backend carries radii instead, and the Phase 0 demo carries neither
        velocities nor masses --- while which channels are wanted is a property of the
        viewport look, and neither knows the other. Reconciling them here rather than at
        the call site keeps `_CHANNEL_REQUIREMENTS` the single statement of what a channel
        needs, alternatives included.

        Turning a channel off is not the same as hiding the mismatch: the sample says which
        channels it carries, so a consumer asked for physical sizes and handed none can say
        so. Raising instead would make a look a user is allowed to select fail the refresh.
        """

        available = set(offered)

        def served(name: str) -> bool:
            return available.issuperset(self._fields_for(name, offered))

        return replace(
            self,
            velocities=self.velocities and served("velocities"),
            sizes=self.sizes and served("sizes"),
            stable_ids=self.stable_ids and served("stable_ids"),
            damage=self.damage and served("damage"),
            material_ids=self.material_ids and served("material_ids"),
            internal_energies=self.internal_energies and served("internal_energies"),
            densities=self.densities and served("densities"),
            color_source=self.color_source if not self.colors or served("colors") else "FLAT",
        )


# What a caller means by "positions only": every optional channel off. Named once because
# three signatures default to it, and because a frozen value shared between them is one
# object rather than one per call.
POSITIONS_ONLY = DisplayChannelSpec()


@dataclass(frozen=True)
class DisplaySample:
    """One staged refresh: what was sampled, what it cost, and the host view of it.

    ``positions`` is a host buffer the stage owns and reuses: the next refresh overwrites
    it in place, so a consumer that needs to retain a sample must copy it. The optional
    channels are the same deal, and are ``None`` when the spec did not ask for them --- a
    consumer therefore tests for the channel rather than for a sentinel-filled array.
    """

    step_index: int
    # The population this sample was drawn from --- which, for a producer that offers an
    # ``active`` channel, is the *live* body count at this step and not the allocation
    # (ADR 0046). `DisplayStage.body_count` remains the allocation; the two agree for
    # every producer without an ``active`` channel, and for any step where nothing is dead.
    body_count: int
    display_count: int
    stride: int
    copy_ms: float
    positions: Any
    velocities: Any = None
    sizes: Any = None
    colors: Any = None
    # The scalar the colours were ramped from, one float32 a row, when the spec staged
    # colours (ADR 0217). It is what the host's percentile auto-range (AW-013) grows on,
    # so a played frame's range follows the device's own extraction rather than a second
    # host derivation of the same quantity. Never written to a record.
    ramp_scalars: Any = None
    # The identity of each staged row, in staged order, when the spec asked for it. Under
    # ADR 0046 this is the only thing that says which bodies a compacted sample holds: the
    # rows are the live ones in solver order, and which indices those are is a device fact
    # the host never otherwise learns.
    stable_ids: Any = None
    # Optional authoritative raw Color By sources. No selected scalar or RGBA answer is
    # carried here; cache readers derive the current display choice from these values.
    damage: Any = None
    material_ids: Any = None
    internal_energies: Any = None
    densities: Any = None
    # True when the rows are the retained identity sequence (ADR 0093): the live bodies in
    # ascending stable-id order, spread by ``floor(k * live / cap)``. Such a sample is still
    # ``strided`` above the cap --- it holds fewer rows than live bodies --- but which rows it
    # holds is the scrub contract's answer rather than the stride's, which is the difference
    # the scrub registry needs to tell apart.
    identity_ordered: bool = False

    @property
    def strided(self) -> bool:
        """True when the sample skips bodies, which is the case a stride exists for."""

        return self.display_count < self.body_count


def display_count_for(body_count: int, cap: int) -> int:
    if body_count < 1:
        raise ValueError("body_count must be positive")
    if cap < 1:
        raise ValueError("cap must be positive")
    return min(int(cap), int(body_count))


def display_stride_for(body_count: int, display_count: int) -> int:
    """Spread ``display_count`` slots across ``body_count`` bodies.

    Floor division rather than rounding: the last slot is clamped to the final body, so a
    stride that overshoots would pile the tail of the sample onto one index.
    """

    if display_count < 1:
        raise ValueError("display_count must be positive")
    return max(1, int(body_count) // int(display_count))


# ---------------------------------------------------------------------------
# ADR 0046: which body each display slot follows.
#
# Both functions below are the *host statement* of arithmetic the device kernels
# implement (`_plan_kernels`). They exist to be compared against --- a kernel is only as
# checkable as the reference it is checked against --- and to let host-side consumers
# reason about a sample without a device. `tests/unit/test_display_active_compaction.py`
# asserts the device agrees with them index for index.
# ---------------------------------------------------------------------------


def strided_plan(body_count: int, cap: int) -> tuple[np.ndarray, int, int]:
    """The pre-ADR-0046 gather: every allocated body, strided down to the cap.

    This is what a producer that offers no ``active`` channel still gets --- the bare-array
    callers, the installed HOST-004 proof --- so it is stated
    once here rather than re-derived at each site.
    """

    display_count = display_count_for(body_count, cap)
    stride = display_stride_for(body_count, display_count)
    slots = np.arange(display_count, dtype=np.int64) * stride
    # Clamped for the reason `display_stride_for` gives. Floor division makes the clamp
    # unreachable for every (body_count, cap) pair, and it is kept because the kernel
    # keeps it: the reference must not be safer than the thing it certifies.
    np.minimum(slots, body_count - 1, out=slots)
    return slots.astype(np.int32), display_count, stride


def compacted_plan(active: Any, cap: int) -> tuple[np.ndarray, int, int]:
    """The live bodies a display slot follows, given the per-body ``active`` flags.

    A *compaction*, not a prefix count. Emission alone would let a prefix work --- newly
    activated slots are appended --- but merge and absorb deactivate whichever body lost,
    wherever it sits, and both features read the same flag array. Striding across the live
    set (rather than across the allocation) is what keeps a capped sample off dead rows.

    Returns ``(indices, display_count, stride)``. An extinct population returns an empty
    plan and a count of zero: clamping to one slot would draw a body that no longer exists,
    which is precisely the defect ADR 0046 removes.
    """

    flags = np.asarray(active).reshape(-1)
    live = np.flatnonzero(flags != 0)
    total = int(live.size)
    if total == 0:
        return np.empty(0, dtype=np.int32), 0, 1
    display_count = display_count_for(total, cap)
    stride = display_stride_for(total, display_count)
    ranks = np.arange(display_count, dtype=np.int64) * stride
    return live[ranks].astype(np.int32), display_count, stride


_NO_KEYS = np.empty(0, dtype=np.uint64)


def identity_sort_keys(identity_ids: Any) -> np.ndarray:
    """The identities as the unsigned values a 64-bit FNV-1a mix means.

    The solver keeps stable ids as ``int64``, so an id with its top bit set is negative on
    the device and would sort first if read signed. Reinterpreting the same bits rather than
    converting is what keeps the order the one `runtime_stable_ids` and the scrub registry
    use.
    """

    ids = np.asarray(identity_ids).reshape(-1)
    if ids.dtype == np.dtype(np.int64):
        return ids.view(np.uint64)
    return ids.astype(np.uint64)


def identity_plan(
    active: Any, identity_ids: Any, cap: int, pinned_ids: Any = None
) -> tuple[np.ndarray, int, int]:
    """ADR 0093: the retained identity sequence of the live bodies, as display slots.

    ``identity_ids`` names every allocated body in solver order. Sorting them gives the
    identity order; the live bodies read in that order are what `retained_sample_indices`
    spreads over, and ``floor(k * live / cap)`` for ``k = 0..cap-1`` picks the retained ones.
    Within the cap that is every live body. The returned indices are solver rows, in
    ascending stable-id order, so a consumer that stages them needs no permutation of its
    own --- and a consumer that finds the staged ids *not* ascending has found a misaligned
    identity list rather than a sample.

    ``active`` of ``None`` means every body is live, which is what a producer offering no
    flag array is. Returns ``(indices, display_count, stride)`` in the same
    shape as `compacted_plan`, including the empty plan for an extinct population.

    ``pinned_ids`` names identities kept whenever they are live (ADR 0239); see
    `retained_indices`. This is the host reference the device plan is held to.
    """

    keys = identity_sort_keys(identity_ids)
    order = np.argsort(keys, kind="stable")
    if active is None:
        live_sorted = np.ones(keys.shape[0], dtype=bool)
    else:
        flags = np.asarray(active).reshape(-1)
        if flags.shape != keys.shape:
            raise ValueError("identity_ids must name every body the active flags describe")
        live_sorted = flags[order] != 0
    live = order[live_sorted]
    total = int(live.size)
    if total == 0:
        return np.empty(0, dtype=np.int32), 0, 1
    display_count = display_count_for(total, cap)
    stride = display_stride_for(total, display_count)
    pinned_sorted = np.zeros(keys.shape[0], dtype=bool)
    pinned_sorted[pinned_sorted_positions(identity_ids, pinned_ids)] = True
    pins = np.flatnonzero(pinned_sorted[live_sorted])
    return live[retained_indices(total, cap, pins)].astype(np.int32), display_count, stride


# ---------------------------------------------------------------------------
# Slice 4 scrub-cache schema (T3/T4). display.py owns the cap, the per-sample
# byte width, the maximum bounded record size, and the deterministic retained-
# identity sampling formula so ``scrub_cache.py`` consumes one statement of the
# contract rather than duplicating constants.
# ---------------------------------------------------------------------------

#: The scrub cache presents at most the same population the viewport samples.
SCRUB_PRESENTATION_CAP = DEFAULT_DISPLAY_CAP

# Stable wire identifiers for the four channels a v1 record retains. The
# availability mask assigns one bit per tag by position in the schema tuple;
# every other bit is reserved and emitted unset.
TAG_POSITION = 1
TAG_STABLE_ID = 2
TAG_VELOCITY = 3
TAG_SIZE = 4

# Only these Color By sources are backed by a stored v1 channel. SPEED is
# derived from the committed velocity when a cached sample is published, so no
# redundant Speed scalar is encoded; every reserved D13 scalar (Slice 5's
# domain) has no v1 carriage and must refuse rather than alias a stored channel.
_V1_COLOR_SOURCES = frozenset({"SPEED"})


@dataclass(frozen=True)
class ScrubChannelSpec:
    """One retained channel: its wire tag, host name, dtype, and column count."""

    tag: int
    name: str
    dtype: str  # explicit little-endian numpy dtype string, e.g. ``"<f4"``
    columns: int  # 3 for a vec3 channel, 1 for a scalar channel

    @property
    def item_bytes(self) -> int:
        return int(np.dtype(self.dtype).itemsize) * int(self.columns)


_V1_CHANNELS: tuple[ScrubChannelSpec, ...] = (
    ScrubChannelSpec(TAG_POSITION, "positions", "<f4", 3),
    ScrubChannelSpec(TAG_STABLE_ID, "stable_ids", "<u8", 1),
    ScrubChannelSpec(TAG_VELOCITY, "velocities", "<f4", 3),
    ScrubChannelSpec(TAG_SIZE, "sizes", "<f4", 1),
)

_V1_MASK_BITS = {TAG_POSITION: 0, TAG_STABLE_ID: 1, TAG_VELOCITY: 2, TAG_SIZE: 3}


class ScrubSampleSchemaV1:
    """The fixed Slice 4 scrub record: the only channels v1 stores.

    ``(3x4 position) + (1x8 stable id) + (3x4 velocity) + (1x4 size) = 36`` bytes
    per sample; the maximum bounded record adds a pinned inclusive ceiling for
    the header, field directory, alignment, and checksum. Slice 5 owns any later
    schema widening --- v1 assigns semantics only to POSITION, STABLE_ID,
    VELOCITY, and SIZE, and never defines a Damage/Material/Temperature/Energy
    or other D13 scalar.
    """

    version = 1
    presentation_cap = SCRUB_PRESENTATION_CAP
    record_header_ceiling = 4_096
    channels = _V1_CHANNELS
    per_sample_bytes = sum(spec.item_bytes for spec in _V1_CHANNELS)
    max_record_bytes = presentation_cap * per_sample_bytes + record_header_ceiling

    @classmethod
    def known_tags(cls) -> tuple[int, ...]:
        return tuple(spec.tag for spec in cls.channels)

    @classmethod
    def spec_for(cls, tag: int) -> ScrubChannelSpec | None:
        for spec in cls.channels:
            if spec.tag == tag:
                return spec
        return None

    @classmethod
    def bit_for(cls, tag: int) -> int:
        return _V1_MASK_BITS[tag]

    @classmethod
    def mask_for(cls, tags: Any) -> int:
        wanted = set(tags)
        mask = 0
        for spec in cls.channels:
            if spec.tag in wanted:
                mask |= 1 << _V1_MASK_BITS[spec.tag]
        return mask

    @classmethod
    def supports_color_source(cls, source: str) -> bool:
        return source in _V1_COLOR_SOURCES


@dataclass(frozen=True)
class ScrubCapacityForecast:
    """Exact fixed-width capacity arithmetic for a scrub cache byte budget."""

    frames: int
    bytes_consumed: int
    bytes_remaining: int
    next_record_overflow: int


def scrub_capacity_forecast(
    capacity_bytes: int, record_bytes: int = ScrubSampleSchemaV1.max_record_bytes
) -> ScrubCapacityForecast:
    """How many maximum-width records fit, and by how much the next one overflows.

    Integer floor division pins capacity: ``floor(1_073_741_824 / 4_722_688) = 227``
    complete maximum-width records leaving ``1_691_648`` bytes, and record 228 would
    exceed capacity by ``3_031_040`` bytes. Records are stored at their real width, so a
    100,000-body frame (3,604,096 bytes) fits 297 times; this is the widest case.
    """

    cap = int(capacity_bytes)
    rec = int(record_bytes)
    if rec < 1:
        raise ValueError("record_bytes must be positive")
    frames = cap // rec
    consumed = frames * rec
    remaining = cap - consumed
    next_overflow = (consumed + rec) - cap
    return ScrubCapacityForecast(frames, consumed, remaining, next_overflow)


def retained_indices(
    body_count: int, cap: int = SCRUB_PRESENTATION_CAP, pinned: Any = None
) -> np.ndarray:
    """The fixed retained source indices for a population of ``body_count`` bodies.

    Within the cap every body is retained in order. Above the cap the retained
    indices are ``floor(k * N / cap)`` for ``k = 0..cap-1`` --- an even spread over
    the stable-id order that is the same for every request at this population.

    ``pinned`` (ADR 0239) names positions that are always retained --- a Star or planet,
    which is one body, not a sample of a cloud. They take their slots first and the even
    spread runs over the remaining positions with the remaining slots,
    ``floor(k * (N - p) / (cap - p))``; with no pins that is the formula above exactly. The
    result is ascending, so the sequence stays in identity order. More pins than slots keeps
    the first ``cap`` of them.
    """

    n = int(body_count)
    c = int(cap)
    if n < 1:
        raise ValueError("body_count must be positive")
    if c < 1:
        raise ValueError("cap must be positive")
    if n <= c:
        return np.arange(n, dtype=np.int64)
    pins = _pinned_positions(pinned, n)
    p = int(pins.shape[0])
    if p >= c:
        return pins[:c]
    free = c - p
    k = np.arange(free, dtype=np.int64)
    if p == 0:
        return (k * n) // c
    rest = np.setdiff1d(np.arange(n, dtype=np.int64), pins, assume_unique=True)
    return np.union1d(pins, rest[(k * (n - p)) // free])


def _pinned_positions(pinned: Any, n: int) -> np.ndarray:
    """``pinned`` as sorted unique int64 positions inside ``0..n-1``; empty for ``None``."""

    if pinned is None:
        return np.empty(0, dtype=np.int64)
    pins = np.unique(np.asarray(pinned, dtype=np.int64).reshape(-1))
    return pins[(pins >= 0) & (pins < n)]


def pinned_sorted_positions(identity_ids: Any, pinned_ids: Any) -> np.ndarray:
    """Where each of ``pinned_ids`` sits in ``identity_ids`` sorted by identity (ADR 0239).

    Sorted unique positions; an identity the list does not name is skipped, because a pin
    asks for a body to be kept, not for one to exist.
    """

    if pinned_ids is None:
        return np.empty(0, dtype=np.int64)
    ordered = np.sort(identity_sort_keys(identity_ids), kind="stable")
    return _positions_in_sorted(ordered, identity_sort_keys(pinned_ids))


def _positions_in_sorted(ordered: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Sorted unique positions of ``want`` in the ascending keys ``ordered``; misses skipped."""

    if ordered.shape[0] == 0 or want.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    at = np.searchsorted(ordered, want)
    inside = at < ordered.shape[0]
    hit = np.zeros(want.shape[0], dtype=bool)
    hit[inside] = ordered[at[inside]] == want[inside]
    return np.unique(at[hit].astype(np.int64))


def retained_sample_indices(
    stable_ids: Any, cap: int = SCRUB_PRESENTATION_CAP, pinned_ids: Any = None
) -> np.ndarray:
    """Original-array indices of the retained bodies, in ascending stable-id order.

    ``pinned_ids`` are always retained (ADR 0239; see `retained_indices`).

    Stable IDs arrive in solver/body order (see ``runtime_stable_ids``), so the
    positional even-spread of :func:`retained_indices` alone would retain whichever
    identities happen to occupy those solver slots. The accepted contract is a fixed
    retained *identity* sequence: sort by stable id first, then take the even spread
    over that ascending order. The returned indices point back into the caller's
    original (solver-order) arrays, so the same permutation applies to every channel.
    """

    ids = np.asarray(stable_ids)
    if ids.ndim != 1:
        raise ValueError("stable_ids must be one-dimensional")
    n = int(ids.shape[0])
    if n < 1:
        # The retained identity sequence over no identities is empty, and that is a state
        # the compacted lane reaches on purpose (ADR 0046, and an emitter before its first
        # release under ADR 0047). Returning the empty selection rather than refusing keeps
        # the caller's permutation total: every channel indexed by it comes back empty too.
        # ``retained_indices`` below stays strict -- it is a positional spread over a
        # positive count, and there is nothing to spread here.
        return np.empty(0, dtype=np.int64)
    # Stable sort keeps ties in solver order so the mapping is fully deterministic.
    order = np.argsort(ids, kind="stable")
    pins = None if pinned_ids is None else pinned_sorted_positions(ids, pinned_ids)
    selection = retained_indices(n, cap=cap, pinned=pins)
    return order[selection]


def subset_indices(subset: int, retained: int) -> np.ndarray:
    """A deterministic display subset of ``subset`` slots over ``retained`` identities.

    The same floor formula is applied over the fixed retained sequence, so a subset
    is a deterministic view of fixed retained identities --- not a claim that subsets
    nest across different sizes.
    """

    m = int(subset)
    nr = int(retained)
    if m < 1:
        raise ValueError("subset must be positive")
    if nr < 1:
        raise ValueError("retained must be positive")
    if m >= nr:
        return np.arange(nr, dtype=np.int64)
    k = np.arange(m, dtype=np.int64)
    return (k * nr) // m


def normalize_display_cap(requested: int) -> tuple[int, bool]:
    """Clamp a requested display cap to the signed v1 maximum.

    A legacy or user ``display_cap`` above the cap (131,072) normalises to it and reports
    that it was clamped so the host can raise a persistent notice; the signed v1
    role surface exposes no larger cap. Values above the maximum normalise visibly
    rather than changing the retained identity sequence.
    """

    value = int(requested)
    if value < 1:
        raise ValueError("display cap must be positive")
    if value > SCRUB_PRESENTATION_CAP:
        return SCRUB_PRESENTATION_CAP, True
    return value, False


def _plan_kernels(wp: Any) -> dict[str, Any]:
    """Compile the ADR 0046 selection plan. Precision-independent: int32 throughout.

    The plan is *the* answer to "which body does display slot N follow", computed on the
    device once per refresh and then read by every gather kernel. Deriving it once rather
    than per channel is what stops a body's position and its radius from coming off
    different rows, and it is why the compaction costs one scan rather than one per channel.
    """

    warp_kernel, _ = _registrars(wp, wp.float32, wp.vec3)

    @warp_kernel
    def plan_strided(
        plan: wp.array(dtype=wp.int32),
        count: int,
        stride: int,
    ):
        """`strided_plan`, on the device: the gather a producer without ``active`` gets."""

        slot = wp.tid()
        index = slot * stride
        if index >= count:
            index = count - 1
        plan[slot] = index

    @warp_kernel
    def plan_meta(
        ranks: wp.array(dtype=wp.int32),
        size: int,
        slots: int,
        meta: wp.array(dtype=wp.int32),
    ):
        """Turn the inclusive scan's total into ``(display_count, stride, live_count)``.

        Launched at dim 1. Computing this here rather than on the host is what keeps a
        refresh at exactly one ``synchronize_device``: the host learns the live count from
        the same copy batch that brings back the staged buffers, never from a readback the
        rest of the refresh has to wait on.
        """

        total = ranks[size - 1]
        display_count = total
        if display_count > slots:
            display_count = slots
        stride = 1
        if display_count > 0:
            stride = total / display_count
            if stride < 1:
                stride = 1
        meta[0] = display_count
        meta[1] = stride
        meta[2] = total

    @warp_kernel
    def plan_compacted(
        active: wp.array(dtype=wp.int32),
        ranks: wp.array(dtype=wp.int32),
        meta: wp.array(dtype=wp.int32),
        plan: wp.array(dtype=wp.int32),
    ):
        """`compacted_plan`, on the device: scatter each live body into its display slot.

        ``ranks`` is an *inclusive* scan of ``active``, so a live body's zero-based position
        within the live set is ``ranks[i] - 1``. Every live body inspects its own rank and
        writes itself, so no ordering between threads is needed and nothing is read twice.
        """

        i = wp.tid()
        if active[i] == 0:
            return
        rank = ranks[i] - 1
        stride = meta[1]
        if rank % stride != 0:
            return
        slot = rank / stride
        if slot >= meta[0]:
            return
        plan[slot] = i

    @warp_kernel
    def plan_fill_tail(
        meta: wp.array(dtype=wp.int32),
        plan: wp.array(dtype=wp.int32),
    ):
        """Point the slots past the live count at a body that is alive.

        The staged buffers are allocated for the worst case and every gather is launched
        across all of them, so the slots beyond ``display_count`` are computed and then
        sliced off. They still have to hold a *valid, live* index: they would otherwise
        keep whatever the previous refresh left --- possibly a body merged away since ---
        and the colour ramp's auto-range reduces across the whole launch, so a stale row
        would move the bounds the visible particles are normalised against.

        Duplicating slot 0 cannot move those bounds, because slot 0's value is already in
        the reduction. Slot 0 is below ``display_count`` whenever anything is alive, so no
        thread here writes the value the others read.
        """

        slot = wp.tid()
        if slot < meta[0]:
            return
        if meta[0] == 0:
            plan[slot] = 0
        else:
            plan[slot] = plan[0]

    @warp_kernel
    def plan_gather_by_identity(
        active: wp.array(dtype=wp.int32),
        order: wp.array(dtype=wp.int32),
        by_identity: wp.array(dtype=wp.int32),
    ):
        """ADR 0093: the live flags read in ascending stable-id order.

        ``order`` is the permutation that sorts the identities, supplied by the host, which
        knows every identity in solver order already. Reading ``active`` through it is what
        lets the existing scan give each live body its rank *in identity order* --- the rank
        the retained sequence is defined over --- without a sort on the device.
        """

        j = wp.tid()
        by_identity[j] = active[order[j]]

    @warp_kernel
    def plan_identity(
        by_identity: wp.array(dtype=wp.int32),
        ranks: wp.array(dtype=wp.int32),
        order: wp.array(dtype=wp.int32),
        meta: wp.array(dtype=wp.int32),
        slots: int,
        pinned: wp.array(dtype=wp.int32),
        pinned_count: int,
        plan: wp.array(dtype=wp.int32),
    ):
        """`identity_plan`, on the device: scatter each retained body into its slot.

        The live body at identity rank ``r`` is retained when some ``k < slots`` has
        ``floor(k * live / slots) == r``. Above the cap ``live > slots``, so that floor is
        strictly increasing in ``k`` and the only candidate is ``k = ceil(r * slots / live)``;
        one multiply and one check per body, and no two bodies write the same slot. Products
        are taken in 64 bits because ``r * slots`` passes 2**31 at a few tens of thousands of
        bodies.

        ADR 0239: ``pinned`` holds the identity-order positions of bodies always kept while
        live (a Star or planet). Each body counts the live pins ranked below it (``pb``) and
        in total (``pl``) --- a handful, so a loop --- and the spread runs over its rank among
        the unpinned, ``u = r - pb``, with ``slots - pl`` slots. A body's slot is the number
        of retained bodies ranked below it: ``pb`` pins plus the spread's picks below ``u``,
        which is ``k`` for a picked body and ``ceil(u * c / n)`` for a pin.
        """

        j = wp.tid()
        if by_identity[j] == 0:
            return
        rank = ranks[j] - 1
        total = meta[2]
        if total <= slots:
            plan[rank] = order[j]
            return
        pb = int(0)  # noqa: UP018 - Warp dynamic variables, counted in the loop below
        pl = int(0)  # noqa: UP018
        me = int(0)  # noqa: UP018
        for q in range(pinned_count):
            at = pinned[q]
            if by_identity[at] != 0:
                pl += 1
                if ranks[at] - 1 < rank:
                    pb += 1
                if at == j:
                    me = 1
        if slots - pl <= 0:
            if me == 1 and pb < slots:
                plan[pb] = order[j]
            return
        u = wp.int64(rank - pb)
        n = wp.int64(total - pl)
        c = wp.int64(slots - pl)
        k = (u * c + n - wp.int64(1)) / n
        if me == 1:
            plan[wp.int32(k) + pb] = order[j]
            return
        if k >= c:
            return
        if (k * n) / c != u:
            return
        plan[wp.int32(k) + pb] = order[j]

    return {
        "strided": plan_strided,
        "meta": plan_meta,
        "compacted": plan_compacted,
        "fill_tail": plan_fill_tail,
        "gather_by_identity": plan_gather_by_identity,
        "identity": plan_identity,
    }


def _sample_kernel(wp: Any, precision: Precision):
    """Compile the planned position sample for one state precision.

    The staged array is float32 in both lanes: it exists to be uploaded to a vertex
    buffer, and the float64 lane's extra precision cannot survive that upload.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, _ = _registrars(wp, scalar, vector)

    @warp_kernel
    def sample_positions(
        positions: wp.array(dtype=vector),
        staged: wp.array(dtype=wp.vec3f),
        plan: wp.array(dtype=wp.int32),
    ):
        slot = wp.tid()
        p = positions[plan[slot]]
        staged[slot] = wp.vec3f(wp.float32(p[0]), wp.float32(p[1]), wp.float32(p[2]))

    return sample_positions


def _optional_kernels(wp: Any, precision: Precision) -> dict[str, Any]:
    """Compile the optional channel kernels for one state precision.

    One kernel per channel rather than one fused kernel taking every array: a fused kernel
    would need a null array for each channel the spec did not ask for, and Warp has no null
    array. The alternative --- a compiled variant per subset --- is twelve kernels to save
    launches that cost about five microseconds each against a refresh measured in hundreds
    of them. The launches share the device's stream, so `DisplayStage.refresh` still ends
    in exactly one ``synchronize_device``, which is the property ADR 0023 constrains.

    Every staged output is float32 whatever the state precision, for the reason
    `_sample_kernel` gives: these arrays exist to become vertex buffers.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, warp_func = _registrars(wp, scalar, vector)
    zero32 = wp.float32(0.0)
    one32 = wp.float32(1.0)
    one_third = wp.float32(1.0 / 3.0)
    quarter_pi_inverse = wp.float32(3.0 / (4.0 * math.pi))
    # Not ``inf``: Warp lowers a Python float literal into the kernel's constant table, and
    # a finite sentinel beyond every physical magnitude reduces identically while keeping
    # the bounds array readable if a probe ever prints it before the first reduction.
    unbounded = wp.float32(3.0e38)
    # Closed over at kernel-build time from the one shared table, so the device ramp and
    # `ramp_colors` interpolate between the same four colours by construction.
    stop_0 = wp.vec3f(*VIRIDIS_STOPS[0])
    stop_1 = wp.vec3f(*VIRIDIS_STOPS[1])
    stop_2 = wp.vec3f(*VIRIDIS_STOPS[2])
    stop_3 = wp.vec3f(*VIRIDIS_STOPS[3])

    @warp_func
    def body_index(plan: wp.array(dtype=wp.int32), slot: int) -> int:
        """The body a display slot follows: whatever `_plan_kernels` decided it follows.

        A one-line indirection, kept as a function so every optional channel resolves its
        source row through the same statement the position kernel does --- ADR 0046's
        invariant is that one refresh describes one set of bodies, and a channel that
        computed its own row could quietly break it.
        """

        return plan[slot]

    @warp_func
    def ramp_color(t: wp.float32) -> wp.vec3f:
        """Four-stop piecewise-linear viridis.

        A perceptually uniform, monotonically lightening, colour-vision-deficiency-safe
        ramp, so a speed or density field reads as an ordered quantity rather than as the
        banding a rainbow ramp invents. Four stops rather than a sampled table because the
        interpolation is per-particle on the device and a texture fetch for it would be a
        second resource to own for a curve this smooth.
        """

        s = wp.clamp(t, zero32, one32) * wp.float32(3.0)
        color = wp.lerp(stop_0, stop_1, s)
        if s >= wp.float32(2.0):
            color = wp.lerp(stop_2, stop_3, s - wp.float32(2.0))
        elif s >= one32:
            color = wp.lerp(stop_1, stop_2, s - one32)
        return color

    @warp_kernel
    def sample_velocities(
        velocities: wp.array(dtype=vector),
        staged: wp.array(dtype=wp.vec3f),
        plan: wp.array(dtype=wp.int32),
    ):
        slot = wp.tid()
        v = velocities[body_index(plan, slot)]
        staged[slot] = wp.vec3f(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))

    @warp_kernel
    def sample_sizes(
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        staged: wp.array(dtype=wp.float32),
        plan: wp.array(dtype=wp.int32),
    ):
        """``(3 m / 4 pi rho)**(1/3)``: the radius of a particle of mass m and density rho,
        evaluated on the device.

        Zero at non-positive density, matching that function, because a particle with no
        density has no volume to have a radius of --- and because a cube root of a negative
        number would otherwise put a NaN in a vertex buffer.
        """

        slot = wp.tid()
        index = body_index(plan, slot)
        density = wp.float32(densities[index])
        radius = zero32
        if density > zero32:
            radius = wp.pow(quarter_pi_inverse * wp.float32(masses[index]) / density, one_third)
        staged[slot] = radius

    @warp_kernel
    def sample_sizes_from_radii(
        radii: wp.array(dtype=scalar),
        staged: wp.array(dtype=wp.float32),
        plan: wp.array(dtype=wp.int32),
    ):
        """The direct backend's real per-body radius, staged as the physical size.

        A strided scalar gather, structurally identical to `sample_density` below --- the
        radius the collision/merge kernels already maintain (`nbody.py`) IS the physical
        size, so there is no derivation to guard: unlike `sample_sizes`, no floor is needed
        because a body's radius is never non-positive.
        """

        slot = wp.tid()
        staged[slot] = wp.float32(radii[body_index(plan, slot)])

    @warp_kernel
    def sample_stable_ids(
        stable_ids: wp.array(dtype=wp.int64),
        staged: wp.array(dtype=wp.int64),
        plan: wp.array(dtype=wp.int32),
    ):
        """Gather each staged row's identity through the same plan every other channel uses.

        ``int64`` end to end, matching the device arrays the N-body lanes already keep: the
        IDs are 64-bit FNV-1a mixes, so half of them do not fit a signed 64-bit range and
        are carried as the wrapped bit pattern. Nothing in this path orders or compares
        them --- it is a permutation of opaque tags --- and the host reads the buffer back
        as ``uint64`` where the unsigned value is what matters.
        """

        slot = wp.tid()
        staged[slot] = stable_ids[body_index(plan, slot)]

    @warp_kernel
    def sample_raw_scalar(
        source: wp.array(dtype=scalar),
        staged: wp.array(dtype=wp.float32),
        plan: wp.array(dtype=wp.int32),
    ):
        """Gather one authoritative scalar through the shared bounded selection plan."""

        slot = wp.tid()
        staged[slot] = wp.float32(source[body_index(plan, slot)])

    @warp_kernel
    def sample_material_ids(
        source: wp.array(dtype=wp.int64),
        staged: wp.array(dtype=wp.int64),
        plan: wp.array(dtype=wp.int32),
    ):
        """Gather persistent material identity without narrowing its signed int64 bits."""

        slot = wp.tid()
        staged[slot] = source[body_index(plan, slot)]

    @warp_kernel
    def reset_bounds(bounds: wp.array(dtype=wp.float32)):
        bounds[0] = unbounded
        bounds[1] = -unbounded

    @warp_kernel
    def sample_speed(
        velocities: wp.array(dtype=vector),
        scalars: wp.array(dtype=wp.float32),
        bounds: wp.array(dtype=wp.float32),
        plan: wp.array(dtype=wp.int32),
    ):
        slot = wp.tid()
        value = wp.float32(wp.length(velocities[body_index(plan, slot)]))
        scalars[slot] = value
        if wp.isfinite(value):
            wp.atomic_min(bounds, 0, value)
            wp.atomic_max(bounds, 1, value)

    @warp_kernel
    def sample_density(
        densities: wp.array(dtype=scalar),
        scalars: wp.array(dtype=wp.float32),
        bounds: wp.array(dtype=wp.float32),
        plan: wp.array(dtype=wp.int32),
    ):
        slot = wp.tid()
        value = wp.float32(densities[body_index(plan, slot)])
        scalars[slot] = value
        if wp.isfinite(value):
            wp.atomic_min(bounds, 0, value)
            wp.atomic_max(bounds, 1, value)

    @warp_kernel
    def sample_colors(
        scalars: wp.array(dtype=wp.float32),
        bounds: wp.array(dtype=wp.float32),
        staged: wp.array(dtype=wp.vec4f),
        low: wp.float32,
        high: wp.float32,
    ):
        """Normalise the extracted scalar and ramp it.

        The auto-range branch reads the bounds the extraction kernel reduced on the device,
        so an auto-ranged ramp costs no readback and no second synchronisation. A degenerate
        span --- one particle, or a converged field --- draws the ramp's low end rather than
        dividing by zero.
        """

        slot = wp.tid()
        lower = low
        upper = high
        if lower >= upper:
            lower = bounds[0]
            upper = bounds[1]
        span = upper - lower
        t = zero32
        # A non-finite scalar draws the low stop: NaN would otherwise survive the clamp.
        if span > zero32 and wp.isfinite(scalars[slot]):
            t = (scalars[slot] - lower) / span
        color = ramp_color(t)
        staged[slot] = wp.vec4f(color[0], color[1], color[2], one32)

    return {
        "velocities": sample_velocities,
        "sizes": sample_sizes,
        "sizes_from_radii": sample_sizes_from_radii,
        "stable_ids": sample_stable_ids,
        "raw_scalar": sample_raw_scalar,
        "material_ids": sample_material_ids,
        "reset_bounds": reset_bounds,
        "SPEED": sample_speed,
        "DENSITY": sample_density,
        # Damage and shock energy are plain per-particle scalars, ramped exactly as density
        # is (ADR 0236): the same gather-and-reduce kernel, a different source channel.
        "DAMAGE": sample_density,
        "ENERGY": sample_density,
        "colors": sample_colors,
    }


class DisplayStage:
    """Owns the device sample buffer and the pinned host staging behind it."""

    def __init__(
        self,
        device_context: DeviceContext,
        *,
        body_count: int,
        precision: Precision = Precision.FLOAT32,
        cap: int = DEFAULT_DISPLAY_CAP,
        channel_spec: DisplayChannelSpec = POSITIONS_ONLY,
    ) -> None:
        self.device_context = device_context
        self.body_count = int(body_count)
        self.cap = int(cap)
        self.precision = precision
        self.channel_spec = channel_spec
        # ``slots`` is the allocation: the largest display count this stage can ever
        # report, fixed at construction because the pinned staging behind it is what
        # `adr/0008-display-interop-boundary.md` measures. ``display_count`` is what the
        # last refresh actually sampled, which under ADR 0046 falls as bodies die and
        # rises as an emitter activates them, within that allocation.
        self.slots = display_count_for(self.body_count, self.cap)
        self.live_count = self.body_count
        self.display_count = self.slots
        self.stride = display_stride_for(self.body_count, self.slots)
        self.copies = 0
        self.last_copy_ms = 0.0
        self.last_step_index = -1
        self.latest: DisplaySample | None = None

        device_context.initialize()
        wp = device_context.wp
        device = device_context.device
        self._wp = wp
        self._kernel = _sample_kernel(wp, precision)
        self._plan_kernels = _plan_kernels(wp)
        # Zeros, not uninitialised: every slot must hold a valid body index from the first
        # launch, because the gather kernels read the plan unconditionally.
        self._plan: Any | None = wp.zeros(self.slots, dtype=wp.int32, device=device)
        # (display_count, stride) for the refresh in flight, computed on the device by
        # `plan_meta` and brought back in the same copy batch as the staged buffers so a
        # refresh still costs exactly one ``synchronize_device``.
        self._meta = wp.zeros(3, dtype=wp.int32, device=device)
        self._meta_host = wp.zeros(3, dtype=wp.int32, device="cpu", pinned=True)
        self._meta_view = self._meta_host.numpy()
        # Allocated on the first refresh that offers an ``active`` channel, sized to that
        # array: a producer that never offers one never pays for the scan buffer.
        self._ranks: Any | None = None
        # ADR 0093's identity selection, allocated on the first refresh that asks for it.
        # The order is uploaded once per identity list and reused until the list changes:
        # the identities of a running population are fixed, so re-sorting them every frame
        # would put an O(N log N) host sort on the draw path for an answer that never moves.
        self.identity_ordered = False
        self._order: Any | None = None
        self._order_keys: np.ndarray | None = None
        self._by_identity: Any | None = None
        self._all_live: Any | None = None
        # ADR 0239: the pinned positions in identity order, uploaded once per (list, pins), and
        # the sorted identity keys they are looked up in, kept from the one sort per list.
        self._pins: Any | None = None
        self._pin_count = 0
        self._pin_keys: np.ndarray | None = None
        self._sorted_keys: np.ndarray = _NO_KEYS
        self._staged: Any | None = wp.zeros(self.slots, dtype=wp.vec3f, device=device)
        # Pinned so the device-to-host hop is a DMA into memory this stage already owns,
        # and viewed once so the upload reads the pinned allocation itself.
        self._host: Any | None = wp.zeros(
            self.slots,
            dtype=wp.vec3f,
            device="cpu",
            pinned=True,
        )
        self._view = self._host.numpy()

        self._channel_kernels: dict[str, Any] = {}
        self._channel_staged: dict[str, Any] = {}
        self._channel_host: dict[str, Any] = {}
        self._channel_views: dict[str, Any] = {}
        self._bounds: Any | None = None
        if channel_spec.names:
            self._channel_kernels = _optional_kernels(wp, precision)
            for name, dtype in (
                ("velocities", wp.vec3f),
                ("sizes", wp.float32),
                ("stable_ids", wp.int64),
                ("damage", wp.float32),
                ("material_ids", wp.int64),
                ("internal_energies", wp.float32),
                ("densities", wp.float32),
                ("colors", wp.vec4f),
            ):
                if name not in channel_spec.names:
                    continue
                self._channel_staged[name] = wp.zeros(self.slots, dtype=dtype, device=device)
                self._channel_host[name] = wp.zeros(
                    self.slots, dtype=dtype, device="cpu", pinned=True
                )
                self._channel_views[name] = self._channel_host[name].numpy()
            if channel_spec.colors:
                # The ramp reads the extracted scalar in the next launch, so an auto-ranged
                # colour costs no second synchronisation. It is also copied to the host
                # (ADR 0217): the percentile auto-range a played frame grows (AW-013) is
                # the host's, and reading one float a body here is what lets the host stop
                # deriving the scalar and the colour itself from the staged velocities.
                self._channel_staged["ramp_scalars"] = wp.zeros(
                    self.slots, dtype=wp.float32, device=device
                )
                self._channel_host["ramp_scalars"] = wp.zeros(
                    self.slots, dtype=wp.float32, device="cpu", pinned=True
                )
                self._channel_views["ramp_scalars"] = self._channel_host["ramp_scalars"].numpy()
                self._bounds = wp.zeros(2, dtype=wp.float32, device=device)
        # The ramp bounds a refresh asks for, when they are not the spec's (ADR 0217). The
        # spec decides which staging exists and so is compared by `ensure_display`; the
        # bounds change from frame to frame as an auto-range grows and must not rebuild it.
        self._ramp_bounds: tuple[float, float] | None = None

    @property
    def closed(self) -> bool:
        return self._staged is None

    def refresh(
        self,
        source: Any,
        *,
        step_index: int,
        identity_ids: Any = None,
        ramp_bounds: tuple[float, float] | None = None,
        pinned_ids: Any = None,
    ) -> DisplaySample:
        """Stage one refresh from ``display_channels()``, or from a bare position array.

        The bare form is what the position-only callers and the installed HOST-004 proof
        pass, and it stays supported because a stage with no optional channels reads
        nothing else --- widening its call sites would be churn for an argument that would
        always hold one key.

        ``identity_ids``, when given, names every allocated body in solver order and asks for
        the retained identity sequence rather than the solver-order stride (ADR 0093). It is
        what the scrub cache needs above the presentation cap, where the rows the stride
        would drop are the ones the contract keeps.

        ``pinned_ids`` names identities the identity selection always keeps while live
        (ADR 0239): a Star among half a million cloud particles is not a sample of the cloud.
        Ignored without ``identity_ids``.
        """

        if self._staged is None or self._host is None:
            raise DeviceRequirementError("display stage is closed")
        channels = source if isinstance(source, dict) else {"positions": source}
        missing = [
            name for name in self.channel_spec.requirements(channels) if name not in channels
        ]
        if missing:
            raise DeviceRequirementError(
                f"runtime offers no {', '.join(missing)} for the requested display channels"
            )
        started = time.perf_counter()
        wp = self._wp
        device = self.device_context.device
        compacted = self._plan_refresh(channels, device, identity_ids, pinned_ids)
        self._ramp_bounds = ramp_bounds
        # Every gather runs across the whole allocation, not across the live count: the
        # live count is still on the device at this point, and waiting for it here is
        # exactly the second synchronisation ADR 0046 is written to avoid. The slots past
        # it are computed and then sliced off below.
        wp.launch(
            self._kernel,
            dim=self.slots,
            inputs=[channels["positions"], self._staged, self._plan],
            device=device,
        )
        for name in self.channel_spec.names:
            self._launch_channel(name, channels, device)
        wp.copy(self._host, self._staged)
        for name in self.channel_spec.names:
            wp.copy(self._channel_host[name], self._channel_staged[name])
        if "ramp_scalars" in self._channel_host:
            wp.copy(self._channel_host["ramp_scalars"], self._channel_staged["ramp_scalars"])
        if compacted:
            wp.copy(self._meta_host, self._meta)
        # One synchronisation for the whole refresh, whatever the channel count: every
        # launch and every copy above was issued on this device's stream in order.
        wp.synchronize_device(device)
        if compacted:
            self.display_count = int(self._meta_view[0])
            self.stride = int(self._meta_view[1])
            self.live_count = int(self._meta_view[2])
        self.last_copy_ms = (time.perf_counter() - started) * 1000.0
        self.copies += 1
        self.last_step_index = int(step_index)
        end = self.display_count
        self.latest = DisplaySample(
            step_index=int(step_index),
            body_count=self.live_count,
            display_count=self.display_count,
            stride=self.stride,
            copy_ms=self.last_copy_ms,
            positions=self._view[:end],
            velocities=self._sliced("velocities", end),
            sizes=self._sliced("sizes", end),
            colors=self._sliced("colors", end),
            stable_ids=self._sliced("stable_ids", end),
            damage=self._sliced("damage", end),
            material_ids=self._sliced("material_ids", end),
            internal_energies=self._sliced("internal_energies", end),
            densities=self._sliced("densities", end),
            ramp_scalars=self._sliced("ramp_scalars", end),
            identity_ordered=self.identity_ordered,
        )
        return self.latest

    def _sliced(self, name: str, end: int) -> Any:
        view = self._channel_views.get(name)
        return None if view is None else view[:end]

    def _plan_refresh(
        self,
        channels: dict[str, Any],
        device: Any,
        identity_ids: Any = None,
        pinned_ids: Any = None,
    ) -> bool:
        """Fill the selection plan for this refresh. True when it was compacted.

        A producer that offers ``active`` gets ADR 0046's stream compaction: an inclusive
        scan of the flags gives each live body its rank within the live set, one dim-1
        kernel turns the total into ``(display_count, stride)``, and each live body
        scatters itself into the slot its rank names. A producer that does not --- the
        bare-array callers, the installed HOST-004 proof --- gets the strided
        plan it always had, whose count is known on the host and needs no readback at all.

        A caller that passes ``identity_ids`` gets ADR 0093's identity selection instead,
        with or without ``active``; see `_plan_identity`.
        """

        wp = self._wp
        active = channels.get("active")
        self.identity_ordered = identity_ids is not None
        if identity_ids is not None:
            self._plan_identity(active, identity_ids, device, pinned_ids)
            return True
        if active is None:
            self.live_count = self.body_count
            self.display_count = self.slots
            self.stride = display_stride_for(self.body_count, self.slots)
            wp.launch(
                self._plan_kernels["strided"],
                dim=self.slots,
                inputs=[self._plan, self.body_count, self.stride],
                device=device,
            )
            return False
        size = int(active.shape[0])
        if self._ranks is None or int(self._ranks.shape[0]) != size:
            self._ranks = wp.zeros(size, dtype=wp.int32, device=device)
        # An inclusive scan, so a live body at index i has rank ``ranks[i] - 1`` within the
        # live set and ``ranks[size - 1]`` is the live total. Warp's own scan rather than a
        # hand-rolled one: it is the primitive this compaction exists on top of.
        wp.utils.array_scan(active, self._ranks, inclusive=True)
        wp.launch(
            self._plan_kernels["meta"],
            dim=1,
            inputs=[self._ranks, size, self.slots, self._meta],
            device=device,
        )
        wp.launch(
            self._plan_kernels["compacted"],
            dim=size,
            inputs=[active, self._ranks, self._meta, self._plan],
            device=device,
        )
        wp.launch(
            self._plan_kernels["fill_tail"],
            dim=self.slots,
            inputs=[self._meta, self._plan],
            device=device,
        )
        return True

    def _plan_identity(
        self, active: Any, identity_ids: Any, device: Any, pinned_ids: Any = None
    ) -> None:
        """ADR 0093: stage the retained identity sequence, in O(N) and with no device sort.

        The same scan-and-scatter as the compaction, run over the flags read in identity
        order: gather ``active`` through the sorting permutation, scan it inclusively so each
        live body has its rank in identity order, let `plan_meta` turn the total into the
        counts, and let every live body the even spread names write itself into its slot.
        A producer with no ``active`` channel is treated as all-live, which is what it is.
        Still no synchronisation here: the counts travel back with the staged buffers.
        """

        wp = self._wp
        size = self.body_count if active is None else int(active.shape[0])
        order = self._identity_order(identity_ids, size, device)
        pins = self._pinned(pinned_ids, device)
        if active is None:
            if self._all_live is None or int(self._all_live.shape[0]) != size:
                self._all_live = wp.full(size, 1, dtype=wp.int32, device=device)
            active = self._all_live
        if self._by_identity is None or int(self._by_identity.shape[0]) != size:
            self._by_identity = wp.zeros(size, dtype=wp.int32, device=device)
        if self._ranks is None or int(self._ranks.shape[0]) != size:
            self._ranks = wp.zeros(size, dtype=wp.int32, device=device)
        wp.launch(
            self._plan_kernels["gather_by_identity"],
            dim=size,
            inputs=[active, order, self._by_identity],
            device=device,
        )
        wp.utils.array_scan(self._by_identity, self._ranks, inclusive=True)
        wp.launch(
            self._plan_kernels["meta"],
            dim=1,
            inputs=[self._ranks, size, self.slots, self._meta],
            device=device,
        )
        wp.launch(
            self._plan_kernels["identity"],
            dim=size,
            inputs=[
                self._by_identity,
                self._ranks,
                order,
                self._meta,
                self.slots,
                pins,
                self._pin_count,
                self._plan,
            ],
            device=device,
        )
        wp.launch(
            self._plan_kernels["fill_tail"],
            dim=self.slots,
            inputs=[self._meta, self._plan],
            device=device,
        )

    def _identity_order(self, identity_ids: Any, size: int, device: Any) -> Any:
        """The device permutation that sorts ``identity_ids``, uploaded once per list.

        Refused when the list does not name exactly the allocation: an order over the wrong
        number of bodies would read ``active`` out of bounds or leave bodies unrankable, and
        either way the staged rows would not be the identities the caller believes.
        """

        keys = identity_sort_keys(identity_ids)
        if int(keys.shape[0]) != size:
            raise DeviceRequirementError(
                f"identity selection names {int(keys.shape[0])} bodies for an allocation of {size}"
            )
        if (
            self._order is not None
            and self._order_keys is not None
            and np.array_equal(self._order_keys, keys)
        ):
            return self._order
        order = np.argsort(keys, kind="stable").astype(np.int32)
        self._order = self._wp.array(order, dtype=self._wp.int32, device=device)
        self._order_keys = keys.copy()
        self._sorted_keys = keys[order]
        self._pins = None
        return self._order

    def _pinned(self, pinned_ids: Any, device: Any) -> Any:
        """The device array of pinned identity-order positions (ADR 0239), uploaded on change.

        Never empty, because a launch wants an array; ``_pin_count`` says how much of it is
        real. Positions are into the sorted identity list `_identity_order` uploaded, whose
        sorted keys it keeps, so a played frame pays a lookup of the few pinned ids and never
        a sort of the whole population. A new identity list clears the cache.
        """

        want = identity_sort_keys(pinned_ids) if pinned_ids is not None else _NO_KEYS
        if (
            self._pins is not None
            and self._pin_keys is not None
            and np.array_equal(self._pin_keys, want)
        ):
            return self._pins
        pins = _positions_in_sorted(self._sorted_keys, want).astype(np.int32)
        self._pin_count = int(pins.shape[0])
        padded = pins if pins.shape[0] else np.zeros(1, dtype=np.int32)
        self._pins = self._wp.array(padded, dtype=self._wp.int32, device=device)
        self._pin_keys = want.copy()
        return self._pins

    def _launch_channel(self, name: str, channels: dict[str, Any], device: Any) -> None:
        """Issue one optional channel's launches. Never synchronises; see `refresh`."""

        wp = self._wp
        kernels = self._channel_kernels
        count = self.slots
        if name in ("velocities", "stable_ids", "material_ids"):
            inputs = [channels[name], self._channel_staged[name]]
            kernel = kernels[name]
        elif name in ("damage", "internal_energies", "densities"):
            inputs = [channels[name], self._channel_staged[name]]
            kernel = kernels["raw_scalar"]
        elif name == "sizes":
            # Sizes has two alternative sources (`_CHANNEL_REQUIREMENTS`): resolving through
            # `_channel_alternative` here, the same function `supported_by` already resolved
            # this spec's ``sizes`` flag through, is what keeps "which fields does this
            # channel need" and "which fields does this launch read" from disagreeing.
            if _channel_alternative(name, channels) == ("radii",):
                inputs = [channels["radii"], self._channel_staged[name]]
                kernel = kernels["sizes_from_radii"]
            else:
                inputs = [channels["masses"], channels["densities"], self._channel_staged[name]]
                kernel = kernels["sizes"]
        else:
            scalars = self._channel_staged["ramp_scalars"]
            ramp_low, ramp_high = self.channel_spec.ramp_low, self.channel_spec.ramp_high
            if self._ramp_bounds is not None:
                ramp_low, ramp_high = self._ramp_bounds
            wp.launch(kernels["reset_bounds"], dim=1, inputs=[self._bounds], device=device)
            wp.launch(
                kernels[self.channel_spec.color_source],
                dim=count,
                inputs=[
                    channels[_RAMP_FIELDS[self.channel_spec.color_source]],
                    scalars,
                    self._bounds,
                    self._plan,
                ],
                device=device,
            )
            wp.launch(
                kernels["colors"],
                dim=count,
                inputs=[
                    scalars,
                    self._bounds,
                    self._channel_staged[name],
                    float(ramp_low),
                    float(ramp_high),
                ],
                device=device,
            )
            return
        wp.launch(
            kernel,
            dim=count,
            inputs=[*inputs, self._plan],
            device=device,
        )

    def report(self) -> dict[str, Any]:
        """What the stage did, for diagnostics and for installed proofs."""

        return {
            "cap": self.cap,
            "body_count": self.body_count,
            "slots": self.slots,
            "live_count": self.live_count,
            "display_count": self.display_count,
            "stride": self.stride,
            "copies": self.copies,
            "last_copy_ms": self.last_copy_ms,
            "last_step_index": self.last_step_index,
            "strided": self.display_count < self.live_count,
            "identity_ordered": self.identity_ordered,
            "bounded_by_cap": self.display_count <= self.cap,
            "channels": ("positions", *self.channel_spec.names),
            "color_source": self.channel_spec.color_source,
            # The bounds the ramp normalised against are device-resident, so the report says
            # whether they were chosen or reduced rather than what they were: reading them
            # back to print them would be exactly the readback this module exists to avoid.
            "ramp_auto_ranged": self.channel_spec.colors
            and self.channel_spec.ramp_low >= self.channel_spec.ramp_high,
        }

    def close(self) -> None:
        self._staged = None
        self._host = None
        self._view = None
        self._plan = None
        self._ranks = None
        self._order = None
        self._order_keys = None
        self._by_identity = None
        self._all_live = None
        self._pins = None
        self._pin_keys = None
        self._sorted_keys = _NO_KEYS
        self._channel_staged.clear()
        self._channel_host.clear()
        self._channel_views.clear()
        self._bounds = None
        self.latest = None
