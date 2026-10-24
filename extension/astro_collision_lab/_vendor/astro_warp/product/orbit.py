"""Orbit setup helpers for authoring and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OrbitalElements:
    semi_major_axis: float
    eccentricity: float = 0.0
    true_anomaly: float = 0.0
    inclination: float = 0.0
    longitude_of_ascending_node: float = 0.0
    argument_of_periapsis: float = 0.0


@dataclass(frozen=True)
class OrbitState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]


def state_from_orbital_elements(elements: OrbitalElements, mu: float) -> OrbitState:
    """Convert Keplerian elements to an inertial Cartesian state."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if elements.semi_major_axis <= 0.0:
        raise ValueError("semi_major_axis must be positive")
    if not 0.0 <= elements.eccentricity < 1.0:
        raise ValueError("elliptic eccentricity must be in [0, 1)")

    a = elements.semi_major_axis
    e = elements.eccentricity
    nu = elements.true_anomaly
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    sqrt_mu_p = math.sqrt(mu / p)

    perifocal_position = (r * math.cos(nu), r * math.sin(nu), 0.0)
    perifocal_velocity = (-sqrt_mu_p * math.sin(nu), sqrt_mu_p * (e + math.cos(nu)), 0.0)
    return OrbitState(
        position=_rotate_perifocal(
            perifocal_position,
            elements.longitude_of_ascending_node,
            elements.inclination,
            elements.argument_of_periapsis,
        ),
        velocity=_rotate_perifocal(
            perifocal_velocity,
            elements.longitude_of_ascending_node,
            elements.inclination,
            elements.argument_of_periapsis,
        ),
    )


def circular_velocity(mu: float, radius: float) -> float:
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return math.sqrt(mu / radius)


def _rotate_perifocal(
    vector: tuple[float, float, float],
    raan: float,
    inclination: float,
    arg_periapsis: float,
) -> tuple[float, float, float]:
    cos_o = math.cos(raan)
    sin_o = math.sin(raan)
    cos_i = math.cos(inclination)
    sin_i = math.sin(inclination)
    cos_w = math.cos(arg_periapsis)
    sin_w = math.sin(arg_periapsis)

    r11 = cos_o * cos_w - sin_o * sin_w * cos_i
    r12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    r13 = sin_o * sin_i
    r21 = sin_o * cos_w + cos_o * sin_w * cos_i
    r22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    r23 = -cos_o * sin_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i
    r33 = cos_i

    x, y, z = vector
    return (
        r11 * x + r12 * y + r13 * z,
        r21 * x + r22 * y + r23 * z,
        r31 * x + r32 * y + r33 * z,
    )
