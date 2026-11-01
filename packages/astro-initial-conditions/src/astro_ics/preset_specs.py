"""Numerical authoring recipes, independent of Blender and the simulation runtime.

All values are normalized simulation units. These recipes establish initial
conditions; they do not constrain the outcome of the subsequent simulation.
"""

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class BodySpec:
    name: str
    kind: str
    count: int
    mass: float
    radius: float
    location: tuple[float, float, float]
    velocity: tuple[float, float, float]
    material: str
    spin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: tuple[float, float, float, float] = (0.35, 0.65, 1.0, 1.0)
    galaxy_id: str = ""
    scale_length: float = 0.0
    thickness: float = 0.1
    seed: int = 0
    rotation_fraction: float = 1.0


@dataclass(frozen=True)
class SceneSpec:
    name: str
    mode: str
    seconds_per_frame: float
    bodies: tuple[BodySpec, ...]


def moon_collision() -> SceneSpec:
    """Offset collision with zero initial center of mass and bulk momentum.

    Counts follow volume, 1 : 0.46**3, to give the fixed-h IISPH core common
    particle spacing. Both bodies remain fully simulated: exactly 100,000 total.
    """
    separation = 1.60
    transverse = 0.80
    longitudinal = sqrt(separation * separation - transverse * transverse)
    relative_speed = 1.20
    target_fraction = 1.0 / 1.1
    impactor_fraction = 0.1 / 1.1
    target = BodySpec(
        name="Proto-Earth",
        kind="SPHERE",
        count=91_130,
        mass=1.0,
        radius=1.0,
        location=(-longitudinal * impactor_fraction, -transverse * impactor_fraction, 0.0),
        velocity=(relative_speed * impactor_fraction, 0.0, 0.0),
        material="ROCK",
        color=(0.18, 0.48, 0.95, 1.0),
    )
    impactor = BodySpec(
        name="Theia",
        kind="SPHERE",
        count=8_870,
        mass=0.1,
        radius=0.46,
        location=(longitudinal * target_fraction, transverse * target_fraction, 0.0),
        velocity=(-relative_speed * target_fraction, 0.0, 0.0),
        material="ROCK",
        color=(1.0, 0.38, 0.10, 1.0),
    )
    return SceneSpec("Moon Collision", "IMPACT", 0.005, (target, impactor))


def galaxy_collision() -> SceneSpec:
    """Native two-Galaxy preset, including persistent component associations.

    Matches the compound root defaults: one unit mass, 20% disk / 80% halo,
    50K particles per Galaxy. These are candidates for live-solver isolation
    qualification; this recipe does not certify equilibrium or performance.
    """
    bodies = []
    for label, seed, x, y, vx, color in (
        ("A", 0, -8.0, -0.2, 0.12, (0.16, 0.55, 1.0, 1.0)),
        ("B", 1, 8.0, 0.2, -0.12, (1.0, 0.40, 0.14, 1.0)),
    ):
        for kind, count, mass, radius in (
            ("DISK", 10_000, 0.2, 5.0),
            ("HALO", 40_000, 0.8, 10.0),
        ):
            bodies.append(BodySpec(
                name=f"Galaxy {label} {kind.title()}",
                kind=kind, count=count, mass=mass, radius=radius,
                location=(x, y, 0.0), velocity=(vx, 0.0, 0.0),
                material="ROCK", color=color if kind == "DISK" else
                    ((.08, .12, .2, .12) if label == "A" else (.2, .08, .04, .12)),
                galaxy_id=f"galaxy-{label.lower()}",
                scale_length=1.0, thickness=0.4, seed=seed,
                rotation_fraction=1.0 if kind == "DISK" else 0.0,
            ))
    return SceneSpec("Galaxy Collision", "GALAXY", 0.02, tuple(bodies))
