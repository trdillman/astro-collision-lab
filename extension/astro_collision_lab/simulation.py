"""Independent, live-only particle simulation facade for Astro Collision Lab.

Initialization samples authored primitives on the CPU and uploads once. The live step and
display methods never request snapshots, fragment reports, host particle arrays or a history
cache. Vendor runtimes retain their small nonfinite/CFL/overflow scalar checks. IMPACT also
reduces its live validity guards to one 20-byte status read per completed step. IMPACT is an
IISPH/bond VFX model; COMPRESSIBLE evolves specific energy with synthetic normalized material
EOS. Neither uses a moon guide or guarantees lunar formation. No performance is asserted.
"""
from __future__ import annotations

import math
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class BodySpec:
    name: str
    kind: str = "SPHERE"
    particle_count: int = 50000
    mass: float = 1.0
    radius: float = 1.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[tuple[float, float, float], ...] = IDENTITY
    thickness: float = 0.1
    material: str = "ROCK"
    galaxy_id: str = ""
    scale_length: float = 0.0
    toomre_q: float = 1.5
    rotation_fraction: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class SimulationConfig:
    mode: str = "GALAXY"
    particle_count: int = 100000
    gravitational_constant: float = 1.0
    dt: float = 0.005
    softening: float = 0.03
    theta: float = 0.6
    substeps: int = 1
    seed: int = 0
    # Explicit normalized, synthetic material controls, not a library of physical planets.
    bulk_modulus: float = 10.0
    shear_modulus: float = 0.5
    yield_stress: float = 0.02
    melt_energy: float = 5.0
    specific_internal_energy: float = 0.0
    courant_number: float = 0.3
    precision: str = "float32"

    @property
    def density_label(self) -> str:
        return "Mass density [code mass / code length^3]"

    @property
    def energy_label(self) -> str:
        return {
            "GALAXY": "Not evolved in collisionless gravity",
            "IMPACT": "Shock/fracture heat proxy [code length^2 / code time^2]",
            "COMPRESSIBLE": "Specific internal energy [code length^2 / code time^2]",
        }.get(self.mode.upper(), "Unknown model")

    def validate(self) -> None:
        if self.mode.upper() not in ("GALAXY", "IMPACT", "COMPRESSIBLE"):
            raise ValueError("mode must be GALAXY, IMPACT or COMPRESSIBLE")
        if isinstance(self.particle_count, bool) or not isinstance(self.particle_count, int) or self.particle_count < 1:
            raise ValueError("particle_count must be a positive integer")
        for key in ("gravitational_constant", "dt", "theta", "bulk_modulus"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("softening", "shear_modulus", "yield_stress", "melt_energy", "specific_internal_energy"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.softening == 0:
            raise ValueError("positive softening is required for the live tree solver")
        if self.substeps < 1 or not isinstance(self.substeps, int):
            raise ValueError("substeps must be a positive integer")
        if not 0 < self.courant_number <= 1:
            raise ValueError("courant_number must be in (0,1]")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        if self.mode.upper() == "IMPACT" and self.precision != "float32":
            raise ValueError("the IISPH impact core uses float32")
        if self.theta > 0.8:
            raise ValueError("theta must be at most 0.8 for the tree gravity core")
        if self.mode.upper() == "IMPACT" and self.substeps != 1:
            raise ValueError("IISPH chooses adaptive inner steps; substeps must be 1")


@dataclass(frozen=True)
class PreparedScene:
    bodies: Any
    specifications: tuple[BodySpec, ...]
    materials: Any = None
    shells: tuple = ()
    impact_layout: Any = None
    spacing_ratio: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _validate_bodies(specs: Sequence[BodySpec], config: SimulationConfig, np: Any) -> tuple[BodySpec, ...]:
    specs = tuple(specs)
    if not specs:
        raise ValueError("at least one authored primitive is required")
    if sum(s.particle_count for s in specs) != config.particle_count:
        raise ValueError("body particle counts must sum to the exact requested particle_count")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("body names must be unique")
    for s in specs:
        if isinstance(s.particle_count, bool) or not isinstance(s.particle_count, int) or s.particle_count < 1:
            raise ValueError(f"{s.name}: count must be a positive integer")
        if s.kind.upper() not in ("SPHERE", "DISK", "HALO", "BULGE"):
            raise ValueError(f"{s.name}: kind must be SPHERE, DISK or HALO")
        if config.mode.upper() != "GALAXY" and s.kind.upper() != "SPHERE":
            raise ValueError("impact models accept SPHERE primitives only")
        if not all(math.isfinite(v) and v > 0 for v in (s.mass, s.radius)):
            raise ValueError(f"{s.name}: positive finite mass and radius are required")
        if not math.isfinite(s.scale_length) or s.scale_length < 0:
            raise ValueError(f"{s.name}: scale length must be finite and nonnegative")
        if not math.isfinite(s.toomre_q) or s.toomre_q < 1:
            raise ValueError(f"{s.name}: Toomre Q must be finite and at least one")
        if not math.isfinite(s.rotation_fraction) or abs(s.rotation_fraction)>1:
            raise ValueError(f"{s.name}: rotation fraction must be in [-1,1]")
        if not isinstance(s.seed,int) or s.seed<0:
            raise ValueError(f"{s.name}: seed must be a nonnegative integer")
        if s.kind.upper() == "DISK" and (not math.isfinite(s.thickness) or s.thickness <= 0):
            raise ValueError(f"{s.name}: disk thickness must be positive")
        for key in ("position", "velocity", "spin"):
            a = np.asarray(getattr(s, key), dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"{s.name}: {key} must be a finite vector")
        matrix = np.asarray(s.rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or np.linalg.det(matrix) <= 0:
            raise ValueError(f"{s.name}: rotation must be a proper orthonormal 3x3 matrix; scale belongs in radius/thickness")
    if config.mode.upper() == "GALAXY":
        for index, s in enumerate(specs):
            for other in specs[index + 1:]:
                if not s.galaxy_id or s.galaxy_id != other.galaxy_id:
                    continue
                if math.dist(s.position, other.position) > 1e-5 * max(s.radius, other.radius):
                    raise ValueError("Galaxy components must share their parent centre; move the Galaxy root")
                if not np.allclose(s.velocity, other.velocity, rtol=0, atol=1e-8):
                    raise ValueError("Galaxy components must share a common bulk velocity; edit the Galaxy root")
                if not np.allclose(s.rotation, other.rotation, rtol=0, atol=1e-5):
                    raise ValueError("Galaxy components must share their parent orientation; rotate the Galaxy root")
    if config.mode.upper() == "IMPACT":
        # The fixed-h IISPH core has 63 usable neighbor slots. Independent source
        # resolutions can overflow/truncate pressure neighborhoods before contact.
        # Allow only small integer-count rounding, then check actual GPU occupancy
        # after load. Never silently redistribute the user's authored counts.
        spacing = [(4*math.pi*s.radius**3/(3*s.particle_count))**(1/3) for s in specs]
        ratio = max(spacing)/min(spacing)
        if ratio > 1.01:
            raise ValueError(
                f"IMPACT requires common spacing across sources (ratio {ratio:.6g} > 1.01). "
                "Allocate counts in proportion to radius cubed, preserving the total count, "
                "or adjust source radii. Authored counts have not been changed."
            )
    return specs


def validate_scene(config: SimulationConfig, bodies: Sequence[BodySpec]) -> tuple[BodySpec, ...]:
    """Preflight authored settings without sampling particles or starting CUDA."""
    config.validate()
    import numpy as np

    return _validate_bodies(bodies, config, np)


def _normalized_material(spec: BodySpec, index: int, config: SimulationConfig):
    from ._vendor.astro_warp.product.materials import (
        MaterialDescriptor, SolidEosSpec, SolidEos, StrengthModel,
    )
    key = spec.material.upper().replace("SYNTHETIC-", "").replace("_", "-")
    if key not in ("ROCK", "IRON", "FLUID", "TILLOTSON", "VAPORIZING-ROCK", "SYNTHETIC-TILLOTSON"):
        raise ValueError(f"{spec.name}: unknown material {spec.material!r}")
    rho = spec.mass / (4 * math.pi * spec.radius**3 / 3)
    if key in ("TILLOTSON", "VAPORIZING-ROCK", "SYNTHETIC-TILLOTSON"):
        eos = SolidEosSpec(SolidEos.TILLOTSON, {
            "reference_density": rho, "a": 0.5, "b": 1.5,
            "bulk_modulus_a": config.bulk_modulus, "bulk_modulus_b": config.bulk_modulus,
            "reference_energy": 100.0, "incipient_vaporization_energy": 10.0,
            "complete_vaporization_energy": 20.0, "alpha": 5.0, "beta": 5.0,
        })
    else:
        eos = SolidEosSpec(SolidEos.MURNAGHAN, {
            "reference_density": rho, "bulk_modulus": config.bulk_modulus,
            "exponent": 4.0,
        })
    fluid = key == "FLUID"
    return MaterialDescriptor(material_id=index, name=f"{spec.name}: normalized synthetic {key}",
        eos=eos, shear_modulus=0.0 if fluid else config.shear_modulus,
        strength=StrengthModel.NONE if fluid else StrengthModel.VON_MISES,
        parameters={} if fluid else {"yield_stress": config.yield_stress, "melt_energy": config.melt_energy})


def _sample_galaxy(specs: tuple[BodySpec, ...], config: SimulationConfig, np: Any):
    from .galaxy_models import sample
    return sample(specs, config, np)


def prepare_scene(bodies: Sequence[BodySpec], config: SimulationConfig) -> PreparedScene:
    """Pure initialization work. Exactly the authored count is simulated; no idle mirrors."""
    import numpy as np
    from ._vendor.astro_warp.product.sampling import BodySet
    specs = validate_scene(config, bodies)
    n = config.particle_count
    stable = np.arange(n, dtype=np.int64)
    active = np.ones(n, dtype=np.int32)
    if config.mode.upper() == "GALAXY":
        rows = _sample_galaxy(specs, config, np)
        body_set = BodySet(positions=np.concatenate([r[0] for r in rows]),
            velocities=np.concatenate([r[1] for r in rows]), masses=np.concatenate([r[2] for r in rows]),
            radii=np.concatenate([r[3] for r in rows]), stable_ids=stable, active=active)
        return PreparedScene(body_set, specs, metadata={"active_count": n, "idle_count": 0,
            "initial_support": "Plummer DF / axisymmetric disk and Jeans candidate; isolated qualification required"})
    from ._vendor.astro_warp.product.materials import MaterialRegistry
    from ._vendor.astro_warp.product.planetary_sampling import ShellSpec, sample_layered_body
    from ._vendor.astro_warp.product.giant_impact_runtime import GiantImpactBodies
    from ._vendor.astro_warp.product.spacesim_impact_runtime import SpaceSimImpactBodies, assign_fault_cells, fault_cell_count
    materials = MaterialRegistry(tuple(_normalized_material(s, i, config) for i, s in enumerate(specs)))
    rows = []
    shells = []
    spacing = []
    fault_ids = []
    body_ids = []
    for i, spec in enumerate(specs):
        rho = materials.get(i).reference_density
        sampled = sample_layered_body((ShellSpec(spec.radius, i, rho),), spec.particle_count,
            seed=[config.seed, i], center=spec.position, velocity=spec.velocity)
        offsets = sampled.positions - spec.position
        positions = offsets @ np.asarray(spec.rotation).T + spec.position
        velocities = sampled.velocities + np.cross(np.asarray(spec.spin), positions-spec.position)
        # A rigid turn must not alter the authored centre-of-mass drift.
        correction = (velocities * sampled.masses[:, None]).sum(0) / sampled.masses.sum() - spec.velocity
        velocities -= correction
        rows.append((positions, velocities, sampled.masses, sampled.radii, sampled.material_ids))
        spacing.append(float(sampled.radii[0]))
        shells.append((spec.name, ((1.0, i, rho),)))
        body_ids.append(np.full(spec.particle_count, i, dtype=np.int32))
        fault_ids.append(assign_fault_cells(positions, fault_cell_count(spec.particle_count, 400),
                                          np.random.default_rng([config.seed, i, 1])))
    body_set = BodySet(positions=np.concatenate([r[0] for r in rows]),
        velocities=np.concatenate([r[1] for r in rows]), masses=np.concatenate([r[2] for r in rows]),
        radii=np.concatenate([r[3] for r in rows]) * ((3/(4*math.pi))**(1/3) if config.mode.upper()=="IMPACT" else 1.0), material_ids=np.concatenate([r[4] for r in rows]),
        stable_ids=stable, active=active, body_flags=np.ones(n,dtype=np.int32))
    speed = float(np.linalg.norm(body_set.velocities, axis=1).max())
    speed = max(speed, math.sqrt(2*config.gravitational_constant*sum(s.mass for s in specs)/min(s.radius for s in specs)))
    # The common grid must reach the coarsest sampling; every source particle remains active.
    layout = GiantImpactBodies(body_set, np.arange(n, dtype=np.int64), np.arange(n,dtype=np.int32),
        np.concatenate(body_ids), np.asarray([s.position for s in specs]), max(spacing), speed,
        tuple(s.name for s in specs))
    return PreparedScene(body_set, specs, materials, tuple(shells),
        SpaceSimImpactBodies(layout, np.concatenate(fault_ids)), max(spacing)/min(spacing),
        {"active_count": n, "idle_count": 0, "material_model": "normalized synthetic constants; density derived from authored mass/volume"})


class AstroCollisionSimulation:
    """Headless API used by the independent add-on's authoring and GPU viewport layers."""
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.config.validate()
        self.runtime = None
        self.device_context = None
        self.count = 0
        self.physical_time = 0.0
        self.stats: dict[str, Any] = {}
        self._channels: dict[str, Any] = {}
        self._authoring_metadata: dict[str, Any] = {}
        self._frames = 0
        self._radius_writer = None
        self._health_guard = None
        self._step_failure: str | None = None

    def initialize(self, bodies: Sequence[BodySpec]):
        try:
            return self._initialize(bodies)
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise

    def _initialize(self, bodies: Sequence[BodySpec]):
        if self.runtime is not None:
            self.close()
        started = time.perf_counter()
        prepared = prepare_scene(bodies, self.config)
        from ._vendor.astro_warp.product.runtime_foundation import DeviceContext
        from ._vendor.astro_warp.product.model import Precision
        context = DeviceContext()
        self.device_context = context
        mode = self.config.mode.upper()
        cfg = self.config
        if mode == "GALAXY":
            from ._vendor.astro_warp.product.tree import BarnesHutConfig, BarnesHutRuntime
            runtime = BarnesHutRuntime(context, BarnesHutConfig(capacity=cfg.particle_count,
                count=cfg.particle_count, gravitational_constant=cfg.gravitational_constant,
                dt=cfg.dt, substeps=cfg.substeps, softening=cfg.softening, theta=cfg.theta,
                precision=Precision(cfg.precision)))
            self.runtime = runtime
            runtime.load_bodies(prepared.bodies)
        elif mode == "IMPACT":
            from ._vendor.astro_warp.product.spacesim_impact_runtime import SpaceSimImpactConfig, SpaceSimImpactRuntime
            runtime = SpaceSimImpactRuntime(context, SpaceSimImpactConfig(count=cfg.particle_count,
                gravitational_constant=cfg.gravitational_constant, seconds_per_frame=cfg.dt,
                materials=prepared.materials, shells=prepared.shells,
                tree_theta=cfg.theta, gravity_softening=cfg.softening))
            self.runtime = runtime
            runtime.load_layout(prepared.impact_layout)
        else:
            from ._vendor.astro_warp.product.sph_runtime import SphProductConfig
            from ._vendor.astro_warp.product.planetary_sph_runtime import PlanetarySphRuntime
            runtime = PlanetarySphRuntime(context, SphProductConfig(count=cfg.particle_count,
                gravitational_constant=cfg.gravitational_constant, seconds_per_frame=cfg.dt,
                substeps=cfg.substeps, softening=cfg.softening, theta=cfg.theta,
                gravity_backend="tree", precision=Precision(cfg.precision), materials=prepared.materials,
                shells=prepared.shells, internal_energy=cfg.specific_internal_energy,
                courant_number=cfg.courant_number))
            self.runtime = runtime
            runtime.load_bodies(prepared.bodies, body_counts=tuple(s.particle_count for s in prepared.specifications))
        self.device_context = context
        self.runtime = runtime
        self.count = cfg.particle_count
        self.physical_time = 0.0
        self._refresh_channels()
        self._frames = 0
        import numpy as np
        stable_ids = np.asarray(prepared.bodies.stable_ids, dtype=np.int64).copy()
        source_index = np.repeat(np.arange(len(prepared.specifications), dtype=np.int32),
                                 [s.particle_count for s in prepared.specifications])
        stable_ids.setflags(write=False)
        source_index.setflags(write=False)
        self._authoring_metadata = {"stable_ids": stable_ids, "source_index": source_index,
            "source_names": tuple(s.name for s in prepared.specifications),
            "source_mass": tuple(s.mass for s in prepared.specifications),
            "source_radius": tuple(s.radius for s in prepared.specifications),
            "initial_count": self.count}
        self.stats = {"mode": mode, "active_count": self.count, "idle_count": 0,
            "physical_dt": cfg.dt, "softening_used": cfg.softening, "theta_used": cfg.theta, "initialization_ms": (time.perf_counter()-started)*1000,
            "spacing_ratio": prepared.spacing_ratio, "density_label": cfg.density_label,
            "energy_label": cfg.energy_label, "metadata": dict(prepared.metadata),
            "particle_readback_policy": "none in live step/display; small adaptive/error counters remain"}
        if mode == 'IMPACT':
            solver=runtime._require_solver()
            overflow = int(solver.neighbor_overflow)
            if overflow:
                raise ValueError(
                    f"IMPACT initial neighbor capacity exceeded ({overflow} dropped candidates). "
                    "Use common source spacing and initially separated spheres; simulation refused."
                )
            self.stats['initial_neighbor_overflow'] = overflow
            self.stats['softening_used']=math.sqrt(float(solver.eps2))
            self.stats['theta_used']=float(solver.settings.tree_theta) if hasattr(solver,'settings') else cfg.theta
            self.stats['gravity_backend']=str(solver.gravity_model_resolved)
            if int(solver.n)!=cfg.particle_count:
                raise RuntimeError('solver did not retain the exact active particle count')
        elif mode == 'GALAXY':
            self.stats['gravity_backend']='TREE'
        else:
            self.stats['gravity_backend']=runtime.hydro.gravity.selected_backend
        return self

    def step(self, dt: float | None = None) -> Mapping[str, Any]:
        if self.runtime is None:
            raise RuntimeError("initialize the simulation before stepping")
        self._raise_if_step_failed()
        if dt is not None and not math.isclose(dt, self.config.dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("dt changed: reinitialize instead of replaying a stale captured graph")
        started = time.perf_counter()
        frame = self._frames + 1
        try:
            self.runtime.step()
            if self.config.mode.upper() == "IMPACT":
                self._check_impact_health()
            physical_time = (
                self.runtime.hydro.simulation_time
                if self.config.mode.upper() == "COMPRESSIBLE"
                else frame * self.config.dt
            )
            solver_ms = float(self.runtime.last_step_ms)
            self._refresh_channels()
        except BaseException as error:
            # Vendor channels may already have changed in place. Neither a new
            # display request nor another step may expose that failed state.
            self._step_failure = f"{type(error).__name__}: {error}"
            self._channels = {}
            raise
        self._frames = frame
        self.physical_time = physical_time
        self.stats.update(
            {
                "last_step_ms": (time.perf_counter() - started) * 1000,
                "solver_reported_ms": solver_ms,
                "frame": frame,
                "physical_time": self.physical_time,
            }
        )
        return self._channels

    def _check_impact_health(self):
        started = time.perf_counter()
        if self._health_guard is None:
            from ._gpu_health import DeviceImpactHealth

            self._health_guard = DeviceImpactHealth(self.device_context)
        counters = self._health_guard.sample(self.runtime._require_solver())
        self.stats.update(
            {
                "live_guardrails": counters,
                "live_guardrail_ms": (time.perf_counter() - started) * 1000,
                "live_guardrail_readback_bytes": self._health_guard.READBACK_BYTES,
                "live_guardrail_readback_count": 1,
            }
        )
        failed = [f"{name}={value}" for name, value in counters.items() if value != 0]
        if failed:
            raise RuntimeError("IMPACT solver guardrail failed: " + ", ".join(failed))

    def _raise_if_step_failed(self):
        if self._step_failure is not None:
            raise RuntimeError(
                "simulation stopped after a failed step; reset or reinitialize: "
                + self._step_failure
            )

    def _refresh_channels(self):
        channels = self.runtime.display_channels()
        if self.config.mode.upper() == "COMPRESSIBLE":
            if self._radius_writer is None:
                from ._gpu_radii import DeviceRadii
                self._radius_writer = DeviceRadii(self.device_context, self.count)
            channels = {**channels, "radii": self._radius_writer.update(channels)}
        self._channels = {**channels, **({"sizes": channels["radii"]} if "radii" in channels else {})}

    def display_channels(self) -> Mapping[str, Any]:
        if self.runtime is None:
            raise RuntimeError("initialize the simulation before requesting device channels")
        self._raise_if_step_failed()
        return self._channels

    def authoring_metadata(self) -> Mapping[str, Any]:
        """Static init mapping: sorted int64 stable_ids to int32 source_index and names."""
        if self.runtime is None:
            raise RuntimeError("initialize before requesting authoring metadata")
        return self._authoring_metadata

    def close(self):
        try:
            if self.runtime is not None:
                self.runtime.close()
        finally:
            if self.device_context is not None:
                self.device_context.close()
            self.runtime = None
            self.device_context = None
            self._channels = {}
            self._radius_writer = None
            self._health_guard = None
            self._step_failure = None
            self._authoring_metadata = {}
            self.count = 0


# Compatibility spelling for a frontend that calls the return value GPU channels.
AstroCollisionSimulation.channels = AstroCollisionSimulation.display_channels
