"""Astro initial conditions: stable galaxy and impact starting setups for N-body codes.

Quick start::

    import astro_ics as ics
    parts = ics.disk_galaxy("milky", particle_count=20_000)
    s = ics.sample(parts)            # positions, velocities, masses, radii
    print(len(s), s.diagnostics()["bound_fraction"])
"""
from . import galaxy_models, preset_specs
from .api import (
                  IDENTITY,
                  KINDS,
                  Component,
                  Config,
                  Sample,
                  assess,
                  diagnostics,
                  disk_galaxy,
                  euler_xyz_matrix,
                  exponential_disk,
                  from_preset,
                  plummer_sphere,
                  preset,
                  preview,
                  sample,
)

__version__ = "0.1.0"
__all__ = ["IDENTITY", "KINDS", "Component", "Config", "Sample", "assess", "diagnostics",
           "disk_galaxy", "euler_xyz_matrix", "exponential_disk", "from_preset", "galaxy_models",
           "plummer_sphere", "preset", "preset_specs", "preview", "sample"]
