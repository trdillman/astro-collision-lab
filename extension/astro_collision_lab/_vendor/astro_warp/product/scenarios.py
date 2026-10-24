"""Scenario templates: the starting scene a setup tool cannot infer on its own.

A setup tool binds whatever the user had selected. It cannot know what makes a two-body
orbit an orbit rather than two objects falling together, so a first run either starts from
someone else's numbers or from none. A scenario carries those numbers: the objects to
create, the setup tool to run them through, and the property values that make the result a
scene worth pressing play on.

DEC-005 asks that a template instantiate a local graph copy. Nothing here is a graph. A
scenario is property values only, and a context compiles its own graph from its own
properties the way every hand-authored context does, so editing this file cannot reach a
context somebody already instantiated and editing that context cannot reach this file.

Every number below is one an installed proof already ran, cited per scenario, rather than
one chosen to look plausible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .orbit import circular_velocity

ScenarioValue = float | int | str | bool


@dataclass(frozen=True)
class ScenarioObject:
    """One template object: what to create, and what its binding is worth.

    ``setup_selected`` is the selection the setup tool sees. An object outside it is bound
    afterwards as an ordinary analytical body, which is how a scenario mixes one sampled
    cloud with one companion star under a tool that would otherwise sample both.
    """

    name: str
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    display_radius: float = 1.0
    setup_selected: bool = True
    binding: Mapping[str, ScenarioValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", tuple(float(value) for value in self.location))
        object.__setattr__(self, "binding", MappingProxyType(dict(self.binding)))


@dataclass(frozen=True)
class Scenario:
    """A named starting scene: objects, the setup tool, and context property values."""

    key: str
    label: str
    description: str
    setup: str
    context: Mapping[str, ScenarioValue] = field(default_factory=dict)
    objects: tuple[ScenarioObject, ...] = ()
    primary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def setup_objects(self) -> tuple[ScenarioObject, ...]:
        """The objects the setup tool binds, in selection order."""

        return tuple(spec for spec in self.objects if spec.setup_selected)

    def companion_objects(self) -> tuple[ScenarioObject, ...]:
        """The objects bound after the setup tool has run."""

        return tuple(spec for spec in self.objects if not spec.setup_selected)

    def validate(self) -> None:
        """Reject a template that cannot instantiate, at import rather than at click."""

        names = [spec.name for spec in self.objects]
        if not names:
            raise ValueError(f"{self.key}: a scenario needs at least one object")
        if any(not name for name in names):
            raise ValueError(f"{self.key}: every scenario object needs a name")
        if len(set(names)) != len(names):
            raise ValueError(f"{self.key}: scenario object names must be unique")
        if not self.setup_objects():
            raise ValueError(f"{self.key}: the setup tool needs at least one selected object")
        if self.primary and self.primary not in [spec.name for spec in self.setup_objects()]:
            raise ValueError(f"{self.key}: primary {self.primary!r} is not a setup object")
        for spec in self.objects:
            if len(spec.location) != 3:
                raise ValueError(f"{self.key}: {spec.name} needs a three-component location")
            count = spec.binding.get("particle_count")
            if count is not None and int(count) < 1:
                raise ValueError(f"{self.key}: {spec.name} needs a positive particle count")


# Frames 1 through 251 is 250 intervals, so a scenario that wants one full period across
# its frame range divides the period by 250 rather than by the frame count.
_FRAME_START = 1
_ORBIT_FRAME_COUNT = 250

# Two-body geometry, from the Phase 2 installed orbit proof: G = 1, a thousand-to-one mass
# ratio, and ten length units of separation. The proof authored the circular state as
# barycentric velocities; the template authors the same orbit as orbital elements instead,
# which is the path Orbit Setup and the predicted-trajectory overlay are built on.
_TWO_BODY_G = 1.0
_TWO_BODY_PRIMARY_MASS = 1000.0
_TWO_BODY_ORBITER_MASS = 1.0
_TWO_BODY_SEMI_MAJOR_AXIS = 10.0
_TWO_BODY_PERIOD = math.tau * math.sqrt(
    _TWO_BODY_SEMI_MAJOR_AXIS**3 / (_TWO_BODY_G * (_TWO_BODY_PRIMARY_MASS + _TWO_BODY_ORBITER_MASS))
)

TWO_BODY_ORBIT = Scenario(
    key="TWO_BODY_ORBIT",
    label="Two-Body Orbit",
    description="A planet on a circular orbit around a primary, one period across the frame range",
    setup="ORBIT",
    primary="Astro Primary Star",
    context={
        "backend": "direct",
        "units_length": "sim-length",
        "units_mass": "sim-mass",
        "units_time": "sim-time",
        "gravitational_constant": _TWO_BODY_G,
        # Two separated bodies never approach the singularity, so softening would only
        # bias an orbit whose whole point is that it closes.
        "softening": 0.0,
        "frame_start": _FRAME_START,
        "frame_end": _FRAME_START + _ORBIT_FRAME_COUNT,
        "seconds_per_frame": _TWO_BODY_PERIOD / _ORBIT_FRAME_COUNT,
        # A thousand leapfrog steps per period, which is the resolution the Phase 2 proof
        # ran the same orbit at.
        "substeps": 4,
    },
    objects=(
        ScenarioObject(
            name="Astro Primary Star",
            location=(0.0, 0.0, 0.0),
            display_radius=0.4,
            binding={"mass": _TWO_BODY_PRIMARY_MASS, "radius": 0.4},
        ),
        ScenarioObject(
            # At the periapsis of a circular orbit, which is where true anomaly zero puts
            # it, so the object sits where the solver starts it.
            name="Astro Orbiting Planet",
            location=(_TWO_BODY_SEMI_MAJOR_AXIS, 0.0, 0.0),
            display_radius=0.18,
            binding={
                "mass": _TWO_BODY_ORBITER_MASS,
                "radius": 0.18,
                "semi_major_axis": _TWO_BODY_SEMI_MAJOR_AXIS,
                "eccentricity": 0.0,
                "true_anomaly": 0.0,
            },
        ),
    ),
)

# The cluster is the scene ADR 0017 measured Auto against: 65536 sampled bodies in one
# binding plus a companion primary. The masses, the radii and the separation are that
# measured scene's. The frame time, the softening and the initial velocities are derived
# from them here, because a template a user presses play on has to evolve while they watch.
_CLUSTER_G = 1.0
_CLUSTER_COUNT = 65536
_CLUSTER_CLOUD_MASS = 1.0e3
_CLUSTER_CLOUD_RADIUS = 40.0
_CLUSTER_PRIMARY_MASS = 5.0e3
_CLUSTER_PRIMARY_RADIUS = 8.0
_CLUSTER_SEPARATION = 60.0
_CLUSTER_TOTAL_MASS = _CLUSTER_CLOUD_MASS + _CLUSTER_PRIMARY_MASS

# One orbit of the cloud about the primary across the frame range, which is the rule the
# two-body template already follows. ADR 0017 ran this scene at a thousandth of a second per
# frame so that a single step drifted the cloud by a part in ten thousand while it measured
# backend cost. At that rate the whole 250-frame range spans 8% of the cluster's dynamical
# time `sqrt(R^3 / G M)`, so the step counter advances and the picture holds still: a still
# image being redrawn is not a simulation playing. The measurement keeps its own scene --
# `validation/live_gui_phase3_probe.py` authors the frame time it measures at -- and the
# template a user loads gets the timescale of what it actually contains.
_CLUSTER_PERIOD = math.tau * math.sqrt(_CLUSTER_SEPARATION**3 / (_CLUSTER_G * _CLUSTER_TOTAL_MASS))

# Circular about the barycentre, so the pair orbits rather than the cloud falling straight
# down the primary's well. That is not a step size the template could fix: a fixed global
# step cannot integrate a particle that reaches the primary's softening length, and a cold
# cloud aimed at a mass five times its own sends thousands of them there.
_CLUSTER_RELATIVE_SPEED = circular_velocity(_CLUSTER_G * _CLUSTER_TOTAL_MASS, _CLUSTER_SEPARATION)
_CLUSTER_CLOUD_SPEED = _CLUSTER_RELATIVE_SPEED * _CLUSTER_PRIMARY_MASS / _CLUSTER_TOTAL_MASS
_CLUSTER_PRIMARY_SPEED = _CLUSTER_RELATIVE_SPEED * _CLUSTER_CLOUD_MASS / _CLUSTER_TOTAL_MASS

# The mean interparticle spacing, which is the length `sampling.py` already hands each
# sampled particle as its radius. Softening below it resolves pairs the sampling never
# represented and hands the fixed step accelerations it cannot integrate; the 0.05 this
# scene carried is twenty times below it.
_CLUSTER_SOFTENING = _CLUSTER_CLOUD_RADIUS / _CLUSTER_COUNT ** (1.0 / 3.0)

CLUSTER = Scenario(
    key="CLUSTER",
    label="Cluster",
    description="A sampled cluster above the tree crossover, with Auto choosing the backend",
    setup="DISTRIBUTION",
    context={
        "backend": "auto",
        "units_length": "sim-length",
        "units_mass": "sim-mass",
        "units_time": "sim-time",
        "gravitational_constant": _CLUSTER_G,
        "softening": _CLUSTER_SOFTENING,
        "frame_start": _FRAME_START,
        "frame_end": _FRAME_START + _ORBIT_FRAME_COUNT,
        "seconds_per_frame": _CLUSTER_PERIOD / _ORBIT_FRAME_COUNT,
        # One step per frame. ADR 0017 measured the tree step on this scene at 16.8 ms, which
        # fits the one service slice a tick allows at the shipped 24 fps rate (20.833 ms); a
        # second substep would not, so the frame would need a second tick and the scene would
        # play below the rate the artist authored. The shortest orbit the scene contains -- a
        # cloud-edge particle 20 units from the primary -- still crosses in 53 leapfrog steps
        # at this frame time.
        "substeps": 1,
    },
    objects=(
        ScenarioObject(
            name="Astro Cluster Cloud",
            location=(0.0, 0.0, 0.0),
            display_radius=_CLUSTER_CLOUD_RADIUS,
            binding={
                "particle_count": _CLUSTER_COUNT,
                "mass": _CLUSTER_CLOUD_MASS,
                "radius": _CLUSTER_CLOUD_RADIUS,
                "sampling_seed": 0,
                # Every sampled particle carries the cloud's bulk velocity, which is what
                # puts the cloud on the orbit rather than each particle on its own.
                "velocity_y": _CLUSTER_CLOUD_SPEED,
            },
        ),
        ScenarioObject(
            # Outside the setup selection: Distribution Setup would sample it into a second
            # cloud, and this one is a single body the cluster orbits.
            name="Astro Cluster Primary",
            location=(_CLUSTER_SEPARATION, 0.0, 0.0),
            display_radius=_CLUSTER_PRIMARY_RADIUS,
            setup_selected=False,
            binding={
                "mass": _CLUSTER_PRIMARY_MASS,
                "radius": _CLUSTER_PRIMARY_RADIUS,
                # Opposite the cloud and lighter in the same ratio, so the barycentre holds
                # still and the orbit stays in frame instead of drifting out of it.
                "velocity_y": -_CLUSTER_PRIMARY_SPEED,
            },
        ),
    ),
)


def _registry(*scenarios: Scenario) -> Mapping[str, Scenario]:
    for scenario in scenarios:
        scenario.validate()
    return MappingProxyType({scenario.key: scenario for scenario in scenarios})


SCENARIOS: Mapping[str, Scenario] = _registry(TWO_BODY_ORBIT, CLUSTER)
