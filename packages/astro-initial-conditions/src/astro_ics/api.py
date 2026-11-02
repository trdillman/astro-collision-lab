"""Friendly, typed front door to the galaxy and impact initial-condition samplers.

``galaxy_models`` and ``preset_specs`` are kept byte-identical to the Astro Collision Lab
add-on (see ``tools/sync_packages.py``). This module adds plain dataclasses and helpers so
other projects can use them without Blender or the add-on's solver.

All values are normalized simulation units (G defaults to 1).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import galaxy_models, preset_specs

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
KINDS = ("SPHERE", "HALO", "BULGE", "DISK")
Vector = tuple[float, float, float]


@dataclass(frozen=True)
class Component:
    """One particle component: a Plummer sphere/halo/bulge or an exponential disk.

    Components that share a non-empty ``galaxy_id`` are one galaxy: each feels the
    others' gravity when its velocities are set up. ``rotation`` is a 3x3 row-major
    matrix that turns the component's local frame into world space.
    """

    name: str
    kind: str = "SPHERE"
    particle_count: int = 10_000
    mass: float = 1.0
    radius: float = 1.0
    position: Vector = (0.0, 0.0, 0.0)
    velocity: Vector = (0.0, 0.0, 0.0)
    spin: Vector = (0.0, 0.0, 0.0)
    rotation: tuple[Vector, Vector, Vector] = IDENTITY
    thickness: float = 0.1
    galaxy_id: str = ""
    scale_length: float = 0.0
    toomre_q: float = 1.5
    rotation_fraction: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if isinstance(self.particle_count, bool) or int(self.particle_count) != self.particle_count \
                or self.particle_count < 1:
            raise ValueError("particle_count must be a positive integer")
        for label in ("mass", "radius"):
            value = getattr(self, label)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{label} must be finite and positive")
        if not (math.isfinite(self.scale_length) and self.scale_length >= 0):
            raise ValueError("scale_length must be finite and >= 0 (0 picks a default)")
        if self.kind == "DISK" and not (math.isfinite(self.thickness) and self.thickness > 0):
            raise ValueError("a DISK needs a finite, positive thickness")
        if not -1.0 <= self.rotation_fraction <= 1.0:
            raise ValueError("rotation_fraction must be in [-1, 1]")
        matrix = np.asarray(self.rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be a 3x3 orthonormal matrix")


@dataclass(frozen=True)
class Config:
    """Sampler settings. ``mode`` only matters for ``preview`` ("GALAXY" or "IMPACT")."""

    seed: int = 0
    gravitational_constant: float = 1.0
    softening: float = 0.03
    mode: str = "GALAXY"
    particle_count: int = 0


@dataclass(frozen=True)
class Sample:
    """Concatenated particle arrays for every component, in input order."""

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    radii: np.ndarray
    component: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.masses)

    def diagnostics(self, config: Config | None = None) -> dict:
        cfg = config or Config()
        return diagnostics(self.positions, self.velocities, self.masses,
                           cfg.gravitational_constant, cfg.softening)


def plummer_sphere(name: str = "Plummer", **kwargs) -> Component:
    """A Plummer sphere (``kind="SPHERE"``). Velocities come from its distribution function."""
    return Component(name=name, kind=kwargs.pop("kind", "SPHERE"), **kwargs)


def exponential_disk(name: str = "Disk", **kwargs) -> Component:
    """An exponential disk with sech^2 vertical profile, rotating by default."""
    kwargs.setdefault("rotation_fraction", 1.0)
    return Component(name=name, kind="DISK", **kwargs)


def disk_galaxy(galaxy_id: str, *, particle_count: int = 50_000, mass: float = 1.0,
                disk_fraction: float = 0.2, disk_radius: float = 5.0, halo_radius: float = 10.0,
                scale_length: float = 1.0, thickness: float = 0.4, position: Vector = (0.0, 0.0, 0.0),
                velocity: Vector = (0.0, 0.0, 0.0), rotation=IDENTITY,
                seed: int = 0) -> tuple[Component, Component]:
    """A disk plus dark halo sharing ``galaxy_id`` (the add-on's default galaxy recipe)."""
    if not 0.0 < disk_fraction < 1.0:
        raise ValueError("disk_fraction must be between 0 and 1")
    disk_count = max(1, round(particle_count * disk_fraction))
    common = {"galaxy_id": galaxy_id, "position": position, "velocity": velocity,
              "rotation": rotation, "scale_length": scale_length, "seed": seed}
    return (
        exponential_disk(f"{galaxy_id} disk", particle_count=disk_count, mass=mass * disk_fraction,
                         radius=disk_radius, thickness=thickness, **common),
        plummer_sphere(f"{galaxy_id} halo", kind="HALO", particle_count=max(1, particle_count - disk_count),
                       mass=mass * (1 - disk_fraction), radius=halo_radius, **common),
    )


def sample(components: Sequence[Component], config: Config | None = None) -> Sample:
    """Draw positions and equilibrium velocities for every component (deterministic per seed)."""
    components = tuple(components)
    if not components:
        raise ValueError("at least one component is required")
    rows = galaxy_models.sample(components, config or Config(), np)
    return Sample(
        positions=np.concatenate([r[0] for r in rows]),
        velocities=np.concatenate([r[1] for r in rows]),
        masses=np.concatenate([r[2] for r in rows]),
        radii=np.concatenate([r[3] for r in rows]),
        component=np.concatenate([np.full(len(r[2]), i, dtype=np.int32) for i, r in enumerate(rows)]),
    )


def preview(components: Sequence[Component], config: Config | None = None,
            limit: int | None = None) -> list[np.ndarray]:
    """Fast positions-only preview per component, at most ``limit`` points each."""
    return galaxy_models.preview(tuple(components), config or Config(), limit=limit)


def diagnostics(positions, velocities, masses, gravitational_constant: float = 1.0,
                softening: float = 0.03) -> dict:
    """Lagrangian radii, energy, angular momentum and bound fraction of a particle set."""
    return galaxy_models.diagnostics(np.asarray(positions, float), np.asarray(velocities, float),
                                     np.asarray(masses, float), gravitational_constant, softening)


def assess(baseline: dict, samples: Sequence[dict], periods: float) -> dict:
    """Pass/fail stability check: >=5 periods, <=5% radius, <=1% energy and spin drift."""
    return galaxy_models.assess(baseline, list(samples), periods)


def euler_xyz_matrix(angles: Vector) -> tuple[Vector, Vector, Vector]:
    """Rotation matrix for XYZ Euler angles in radians (Blender's default order)."""
    x, y, z = angles
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return tuple(tuple(float(v) for v in row) for row in rz @ ry @ rx)


def from_preset(body: preset_specs.BodySpec) -> Component:
    """Convert one preset body (``preset_specs.BodySpec``) into a ``Component``."""
    return Component(
        name=body.name, kind=body.kind, particle_count=body.count, mass=body.mass,
        radius=body.radius, position=body.location, velocity=body.velocity, spin=body.spin,
        rotation=euler_xyz_matrix(body.rotation), thickness=body.thickness,
        galaxy_id=body.galaxy_id, scale_length=body.scale_length, seed=body.seed,
        rotation_fraction=body.rotation_fraction,
    )


def preset(name: str) -> tuple[tuple[Component, ...], Config]:
    """Ready-made scenes: ``"galaxy_collision"`` or ``"moon_collision"``."""
    builders = {"galaxy_collision": preset_specs.galaxy_collision,
                "moon_collision": preset_specs.moon_collision}
    if name not in builders:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(builders)}")
    scene = builders[name]()
    components = tuple(from_preset(b) for b in scene.bodies)
    return components, Config(mode=scene.mode, particle_count=sum(c.particle_count for c in components))
