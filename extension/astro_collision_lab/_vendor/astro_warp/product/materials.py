"""Material capability model and the condensed-matter equation-of-state registry.

This is the material half of `docs/source-audit/SPEC/07_IMPACT_SOLID_MATERIAL_PHYSICS.md`
and carries the registry side of SOLID-001. `hydro.py` owns the gas closures the
compressible SPH core needs; this module owns the closures a *solid* needs and the
descriptor that says which constitutive modules a material switches on. The split is the
requirement, not a filing convenience: SOLID-001 asks for stress, plasticity, damage and
porosity to be separate modules, and a material that declares no strength must not pull the
solid pipeline into the graph at all.

Two closures are registered here, both from the miluphCUDA donor
(`schaefer2016miluph`, `schaefer2020miluphcuda`; SRC-MILUPH in the source manifest):

- Murnaghan, the cold solid test EOS. Barotropic, so ``c**2 = dP/drho`` exactly and the
  sound speed is gated against the pressure's own derivative the way SPH-003 gates the
  barotropic gas closures.
- Tillotson, the impact EOS. Not barotropic: pressure depends on the specific internal
  energy as well as the density, so the sound speed is the adiabatic derivative
  ``c**2 = dP/drho|_E + (P/rho**2) dP/dE|_rho`` and both partials are analytic here. They
  are gated against complex-step differentiation, which is exact to round-off for an
  analytic function, so the gate is a round-off bound rather than a chosen tolerance.

No material *values* ship with this module. The equations are sourced; the basalt/iron/ice
parameter tables in the donor's `material-config/` are not embedded in this repository's
read-only evidence, so publishing numbers attributed to them would be an invented citation.
`MaterialDescriptor` therefore carries the schema - unit, range, rationale and source per
parameter, which is the impact-physics acceptance criterion - and a scenario supplies the
values. A sourced material library is separate work and is recorded as such in
`adr/0022-solid-constitutive-validation-thresholds.md`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Device-side EOS selector. `hydro.py` owns 0-2 for the gas closures; the solid closures
# continue that numbering rather than starting a second space, so one integer identifies a
# closure everywhere it is reported. `tests/unit/test_solid_material_contracts.py` asserts
# the two ranges stay disjoint - a silent collision would make a proof name the wrong EOS.
EOS_ID_MURNAGHAN = 3
EOS_ID_TILLOTSON = 4

# The constitutive pipeline order, stated once. A material emits a subset of this in this
# order, and a registry's union is this filtered by what its materials emit - deriving both
# from one tuple is what stops the two from disagreeing. They did: the union was a separate
# five-element literal that predated ADR 0020 adding `condensed_eos`, so `manifest()`
# under-reported the stage every descriptor was already emitting. The names are literals
# rather than the `MODULE_*` constants in `solids.py` because that module imports Warp and
# this one is host-only; `tests/unit/test_solids_graph.py` holds the two in agreement.
SOLID_PIPELINE_ORDER = (
    "astro.solids.stress_rate",
    "astro.solids.plasticity",
    "astro.solids.damage",
    "astro.solids.porosity",
    "astro.solids.condensed_eos",
    "astro.solids.stress_force",
)


class MaterialCapability(StrEnum):
    """What a material can do, in the vocabulary of SPEC 07's capability model.

    The solver graph is compiled from these. A capability that no registered module can
    serve fails configuration rather than becoming a dormant branch inside a kernel.
    """

    HYDRODYNAMIC = "hydrodynamic"
    ELASTIC = "elastic"
    PLASTIC = "plastic"
    BRITTLE_DAMAGE = "brittle_damage"
    POROUS = "porous"
    THERMAL = "thermal"
    TABULATED_EOS = "tabulated_eos"
    GRANULAR = "granular"
    VOID_OR_TRACER = "void_or_tracer"


class SolidEos(StrEnum):
    MURNAGHAN = "murnaghan"
    TILLOTSON = "tillotson"


class StrengthModel(StrEnum):
    NONE = "none"
    VON_MISES = "von_mises"
    DRUCKER_PRAGER = "drucker_prager"


class DamageModel(StrEnum):
    NONE = "none"
    GRADY_KIPP = "grady_kipp"


class PorosityModel(StrEnum):
    NONE = "none"
    P_ALPHA = "p_alpha"


@dataclass(frozen=True)
class ParameterSpec:
    """Unit, admissible range, rationale and source for one material parameter.

    SPEC 07's first impact-physics acceptance criterion is that every material parameter
    has all four. They live in a table rather than in prose so `MaterialDescriptor.validate`
    can enforce the range instead of trusting the caller to have read the documentation.
    """

    unit: str
    minimum: float
    maximum: float
    rationale: str
    source: str
    #: An optional parameter may be absent from a descriptor (ADR 0204). It is checked against
    #: its range when present, and a descriptor without it means "this model is not used":
    #: every material authored, saved or hand-built before the parameter existed stays valid.
    optional: bool = False

    def check(self, name: str, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"{name}={value!r} outside the admissible range "
                f"[{self.minimum}, {self.maximum}] {self.unit}".rstrip()
            )


_MILUPH = "SRC-MILUPH schaefer2016miluph, schaefer2020miluphcuda"
_SPACESIM_THERMAL = (
    "docs/reference-analysis/SPACESIM_IMPACT_CONTRACT.md section 2.2 (ComputeStress.comp); "
    "preset values: adr/0204"
)
_INF = math.inf

MURNAGHAN_PARAMETERS: dict[str, ParameterSpec] = {
    "reference_density": ParameterSpec(
        "kg/m^3", 0.0, _INF, "density at which the cold pressure vanishes", _MILUPH
    ),
    "bulk_modulus": ParameterSpec(
        "Pa", 0.0, _INF, "isothermal bulk modulus at the reference density", _MILUPH
    ),
    "exponent": ParameterSpec(
        "1", 1.0, 20.0, "pressure derivative of the bulk modulus, n = dK/dP", _MILUPH
    ),
}

TILLOTSON_PARAMETERS: dict[str, ParameterSpec] = {
    "reference_density": ParameterSpec(
        "kg/m^3", 0.0, _INF, "uncompressed density of the condensed phase", _MILUPH
    ),
    "a": ParameterSpec("1", 0.0, 1.0, "cold-term Gruneisen coefficient", _MILUPH),
    "b": ParameterSpec("1", 0.0, _INF, "thermal-term coefficient", _MILUPH),
    "bulk_modulus_a": ParameterSpec("Pa", 0.0, _INF, "linear compression modulus A", _MILUPH),
    "bulk_modulus_b": ParameterSpec("Pa", -_INF, _INF, "quadratic compression modulus B", _MILUPH),
    "reference_energy": ParameterSpec(
        "J/kg", 0.0, _INF, "energy scale E0 of the thermal denominator", _MILUPH
    ),
    "incipient_vaporization_energy": ParameterSpec(
        "J/kg", 0.0, _INF, "E_iv, lower edge of the mixed-phase interpolation", _MILUPH
    ),
    "complete_vaporization_energy": ParameterSpec(
        "J/kg", 0.0, _INF, "E_cv, upper edge of the mixed-phase interpolation", _MILUPH
    ),
    "alpha": ParameterSpec("1", 0.0, _INF, "expanded-phase density decay exponent", _MILUPH),
    "beta": ParameterSpec("1", 0.0, _INF, "expanded-phase cold-term decay exponent", _MILUPH),
}

EOS_PARAMETERS: dict[SolidEos, dict[str, ParameterSpec]] = {
    SolidEos.MURNAGHAN: MURNAGHAN_PARAMETERS,
    SolidEos.TILLOTSON: TILLOTSON_PARAMETERS,
}

EOS_IDS: dict[SolidEos, int] = {
    SolidEos.MURNAGHAN: EOS_ID_MURNAGHAN,
    SolidEos.TILLOTSON: EOS_ID_TILLOTSON,
}


def _murnaghan_pressure(
    density: float, energy: float, p: dict[str, float], exp: Callable[[Any], Any]
) -> Any:
    del energy, exp
    eta = density / p["reference_density"]
    return (p["bulk_modulus"] / p["exponent"]) * (eta ** p["exponent"] - 1.0)


def _tillotson_pressure(
    density: float, energy: float, p: dict[str, float], exp: Callable[[Any], Any]
) -> Any:
    """Tillotson pressure, written so that a complex argument evaluates the same expression.

    ``exp`` is injected rather than taken from `math` because the derivative oracle in the
    validation suite differentiates this function by the complex step: the analytic partials
    below are the ones the kernel uses, and an oracle that re-derived them by hand would be
    checking one transcription against another. Substituting `cmath.exp` makes the oracle a
    property of this expression instead.

    Branch selection compares real parts, so the branch a complex step lands in is the branch
    of the point it perturbs. The step never crosses an edge: it is imaginary and ~1e-20.
    """

    rho0 = p["reference_density"]
    eta = density / rho0
    mu = eta - 1.0
    e0 = p["reference_energy"]
    a = p["a"]
    b = p["b"]
    big_a = p["bulk_modulus_a"]
    big_b = p["bulk_modulus_b"]
    e_iv = p["incipient_vaporization_energy"]
    e_cv = p["complete_vaporization_energy"]

    omega = 1.0 + energy / (e0 * eta * eta)
    compressed = (a + b / omega) * density * energy + big_a * mu + big_b * mu * mu

    real_density = density.real if isinstance(density, complex) else density
    real_energy = energy.real if isinstance(energy, complex) else energy
    if real_density >= rho0 or real_energy <= e_iv:
        return compressed

    inverse = rho0 / density - 1.0
    expanded = (
        a * density * energy
        + (b * density * energy / omega + big_a * mu * exp(-p["beta"] * inverse))
        * exp(-p["alpha"] * inverse * inverse)
    )
    if real_energy >= e_cv:
        return expanded

    # Mixed phase: linear in the specific energy between the two branches, which is the
    # donor's interpolation and the reason E_iv < E_cv is a validated constraint.
    weight = (energy - e_iv) / (e_cv - e_iv)
    return weight * expanded + (1.0 - weight) * compressed


_PRESSURE_FUNCTIONS: dict[SolidEos, Callable[..., Any]] = {
    SolidEos.MURNAGHAN: _murnaghan_pressure,
    SolidEos.TILLOTSON: _tillotson_pressure,
}


@dataclass(frozen=True)
class SolidEosSpec:
    """One condensed-matter closure and the parameters that instantiate it."""

    name: SolidEos
    parameters: dict[str, float] = field(default_factory=dict)

    @property
    def eos_id(self) -> int:
        return EOS_IDS[self.name]

    @property
    def evolves_internal_energy(self) -> bool:
        """Murnaghan is barotropic; Tillotson is not, so only Tillotson needs the energy."""

        return self.name == SolidEos.TILLOTSON

    def validate(self) -> None:
        schema = EOS_PARAMETERS[self.name]
        missing = sorted(set(schema) - set(self.parameters))
        if missing:
            raise ValueError(f"{self.name} is missing parameters {missing}")
        unknown = sorted(set(self.parameters) - set(schema))
        if unknown:
            raise ValueError(f"{self.name} does not take parameters {unknown}")
        for key, spec in schema.items():
            spec.check(key, float(self.parameters[key]))
        if self.parameters["reference_density"] <= 0.0:
            raise ValueError("reference_density must be positive")
        if self.name == SolidEos.MURNAGHAN and self.parameters["bulk_modulus"] <= 0.0:
            raise ValueError(
                "bulk_modulus must be positive: it sets the sound speed and the Courant step"
            )
        if self.name == SolidEos.TILLOTSON:
            if self.parameters["reference_energy"] <= 0.0:
                raise ValueError("reference_energy must be positive")
            if (
                self.parameters["incipient_vaporization_energy"]
                >= self.parameters["complete_vaporization_energy"]
            ):
                raise ValueError(
                    "incipient_vaporization_energy must be below "
                    "complete_vaporization_energy: the mixed phase interpolates between them"
                )

    def pressure(self, density: float, internal_energy: float = 0.0) -> float:
        if density <= 0.0:
            raise ValueError("density must be positive")
        function = _PRESSURE_FUNCTIONS[self.name]
        return float(function(density, internal_energy, self.parameters, math.exp))

    def pressure_expression(self, density: Any, internal_energy: Any, exp: Callable[[Any], Any]):
        """The pressure with the exponential injected, for the complex-step oracle."""

        return _PRESSURE_FUNCTIONS[self.name](density, internal_energy, self.parameters, exp)

    def pressure_density_derivative(self, density: float, internal_energy: float = 0.0) -> float:
        """``dP/drho`` at fixed specific internal energy."""

        p = self.parameters
        if self.name == SolidEos.MURNAGHAN:
            eta = density / p["reference_density"]
            return (p["bulk_modulus"] / p["reference_density"]) * eta ** (p["exponent"] - 1.0)
        return self._tillotson_partials(density, internal_energy)[0]

    def pressure_energy_derivative(self, density: float, internal_energy: float = 0.0) -> float:
        """``dP/dE`` at fixed density. Zero for a barotropic closure by definition."""

        if self.name == SolidEos.MURNAGHAN:
            return 0.0
        return self._tillotson_partials(density, internal_energy)[1]

    def sound_speed(self, density: float, internal_energy: float = 0.0) -> float:
        """The adiabatic sound speed.

        For a barotropic closure this collapses to ``sqrt(dP/drho)``; the general form adds
        the ``(P/rho**2) dP/dE`` term, which is the part a code that reuses the barotropic
        expression for Tillotson silently drops, and with it the correct Courant step.
        """

        if density <= 0.0:
            return 0.0
        pressure = self.pressure(density, internal_energy)
        squared = self.pressure_density_derivative(density, internal_energy) + (
            pressure / (density * density)
        ) * self.pressure_energy_derivative(density, internal_energy)
        return math.sqrt(squared) if squared > 0.0 else 0.0

    def _tillotson_partials(self, density: float, energy: float) -> tuple[float, float]:
        p = self.parameters
        rho0 = p["reference_density"]
        eta = density / rho0
        mu = eta - 1.0
        e0 = p["reference_energy"]
        a = p["a"]
        b = p["b"]
        big_a = p["bulk_modulus_a"]
        big_b = p["bulk_modulus_b"]
        e_iv = p["incipient_vaporization_energy"]
        e_cv = p["complete_vaporization_energy"]

        omega = 1.0 + energy / (e0 * eta * eta)
        # d(omega)/d(rho) = -2 E / (E0 eta^2 rho); d(omega)/dE = 1 / (E0 eta^2).
        d_omega_d_rho = -2.0 * energy / (e0 * eta * eta * density)
        d_omega_d_e = 1.0 / (e0 * eta * eta)

        cold_d_rho = big_a / rho0 + 2.0 * big_b * mu / rho0
        compressed_d_rho = (
            (a + b / omega) * energy
            - (b / (omega * omega)) * d_omega_d_rho * density * energy
            + cold_d_rho
        )
        compressed_d_e = (a + b / omega) * density - (b / (omega * omega)) * d_omega_d_e * (
            density * energy
        )

        if density >= rho0 or energy <= e_iv:
            return compressed_d_rho, compressed_d_e

        inverse = rho0 / density - 1.0
        d_inverse_d_rho = -rho0 / (density * density)
        decay = math.exp(-p["alpha"] * inverse * inverse)
        cold_decay = math.exp(-p["beta"] * inverse)
        thermal = b * density * energy / omega
        cold = big_a * mu * cold_decay
        d_thermal_d_rho = b * energy / omega - (b * density * energy / (omega * omega)) * (
            d_omega_d_rho
        )
        d_thermal_d_e = b * density / omega - (b * density * energy / (omega * omega)) * d_omega_d_e
        d_cold_d_rho = (big_a / rho0) * cold_decay + big_a * mu * cold_decay * (
            -p["beta"] * d_inverse_d_rho
        )
        d_decay_d_rho = decay * (-2.0 * p["alpha"] * inverse * d_inverse_d_rho)

        expanded_d_rho = (
            a * energy + (d_thermal_d_rho + d_cold_d_rho) * decay + (thermal + cold) * d_decay_d_rho
        )
        expanded_d_e = a * density + d_thermal_d_e * decay

        if energy >= e_cv:
            return expanded_d_rho, expanded_d_e

        weight = (energy - e_iv) / (e_cv - e_iv)
        d_weight_d_e = 1.0 / (e_cv - e_iv)
        compressed = (a + b / omega) * density * energy + big_a * mu + big_b * mu * mu
        expanded = a * density * energy + (thermal + cold) * decay
        return (
            weight * expanded_d_rho + (1.0 - weight) * compressed_d_rho,
            weight * expanded_d_e
            + (1.0 - weight) * compressed_d_e
            + d_weight_d_e * (expanded - compressed),
        )


@dataclass(frozen=True)
class MaterialDescriptor:
    """One material: its closure, its moduli, and the constitutive modules it switches on.

    ``capabilities`` is derived rather than declared. A descriptor that named a capability
    its models cannot serve would compile a graph stage that does nothing, which is the
    failure SPEC 07 asks configuration to reject rather than hide inside a kernel branch.
    """

    material_id: int
    name: str
    eos: SolidEosSpec
    shear_modulus: float = 0.0
    strength: StrengthModel = StrengthModel.NONE
    damage: DamageModel = DamageModel.NONE
    porosity: PorosityModel = PorosityModel.NONE
    parameters: dict[str, float] = field(default_factory=dict)

    @property
    def reference_density(self) -> float:
        return float(self.eos.parameters["reference_density"])

    @property
    def capabilities(self) -> frozenset[MaterialCapability]:
        found = {MaterialCapability.HYDRODYNAMIC}
        if self.shear_modulus > 0.0:
            found.add(MaterialCapability.ELASTIC)
        if self.strength is not StrengthModel.NONE:
            found.add(MaterialCapability.PLASTIC)
        if self.strength is StrengthModel.DRUCKER_PRAGER:
            found.add(MaterialCapability.GRANULAR)
        if self.damage is not DamageModel.NONE:
            found.add(MaterialCapability.BRITTLE_DAMAGE)
        if self.porosity is not PorosityModel.NONE:
            found.add(MaterialCapability.POROUS)
        if self.eos.evolves_internal_energy:
            found.add(MaterialCapability.THERMAL)
        return frozenset(found)

    @property
    def graph_modules(self) -> tuple[str, ...]:
        """The graph modules this material requires, in constitutive pipeline order.

        This is the SOLID-001 separation made executable: a material with no shear modulus
        returns nothing, so a hydrodynamic scenario never compiles the solid stages, and a
        material with damage but no strength is rejected by `validate` rather than quietly
        evolving a damage variable that scales a stress nobody computes.
        """

        modules: list[str] = []
        if MaterialCapability.ELASTIC in self.capabilities:
            modules.append("astro.solids.stress_rate")
        if MaterialCapability.PLASTIC in self.capabilities:
            modules.append("astro.solids.plasticity")
        if MaterialCapability.BRITTLE_DAMAGE in self.capabilities:
            modules.append("astro.solids.damage")
        if MaterialCapability.POROUS in self.capabilities:
            modules.append("astro.solids.porosity")
        if modules:
            # The condensed-matter EOS is the solid pressure stage; it feeds the stress
            # divergence alongside the deviatoric stress, so it precedes stress_force to match
            # SolidRuntime.evaluate_equation_of_state order.
            modules.append("astro.solids.condensed_eos")
            modules.append("astro.solids.stress_force")
        return tuple(modules)

    def validate(self) -> None:
        if self.material_id < 0:
            raise ValueError("material_id cannot be negative")
        if not self.name.strip():
            raise ValueError("material name is required")
        if not math.isfinite(self.shear_modulus) or self.shear_modulus < 0.0:
            raise ValueError("shear_modulus must be finite and non-negative")
        self.eos.validate()
        elastic = self.shear_modulus > 0.0
        if self.strength is not StrengthModel.NONE and not elastic:
            raise ValueError(
                f"{self.name}: {self.strength} needs a positive shear_modulus - plasticity "
                "corrects a deviatoric stress, and without elasticity there is none"
            )
        if self.damage is not DamageModel.NONE and not elastic:
            raise ValueError(
                f"{self.name}: {self.damage} needs a positive shear_modulus - damage scales "
                "the tensile and deviatoric capacity of an elastic material"
            )
        for key, spec in STRENGTH_PARAMETERS[self.strength].items():
            if key not in self.parameters:
                if spec.optional:
                    continue
                raise ValueError(f"{self.name}: {self.strength} requires {key}")
            spec.check(key, float(self.parameters[key]))
        for key, spec in DAMAGE_PARAMETERS[self.damage].items():
            if key not in self.parameters:
                if spec.optional:
                    continue
                raise ValueError(f"{self.name}: {self.damage} requires {key}")
            spec.check(key, float(self.parameters[key]))
        for key, spec in POROSITY_PARAMETERS[self.porosity].items():
            if key not in self.parameters:
                if spec.optional:
                    continue
                raise ValueError(f"{self.name}: {self.porosity} requires {key}")
            spec.check(key, float(self.parameters[key]))
        required = (
            set(STRENGTH_PARAMETERS[self.strength])
            | set(DAMAGE_PARAMETERS[self.damage])
            | set(POROSITY_PARAMETERS[self.porosity])
        )
        unknown = sorted(set(self.parameters) - required)
        if unknown:
            raise ValueError(f"{self.name}: no registered model reads {unknown}")


MATERIAL_PRESET_FIELD_UNITS: dict[str, str] = {
    "shear_modulus": "Pa",
}
"""The declared units of ``MaterialDescriptor``'s own dimensional fields (ADR 0041).

The parameter tables below already carry a unit per nested ``ParameterSpec``, so this is
the half of the metadata the descriptor's *own* fields need: the one-pass SI-to-code
converter in `unit_presets` reads it to convert each top-level dimensional field and
refuses rather than guessing when a dimensional field has no entry. ``shear_modulus`` is
today the only such field --- identity, name and the model choices are counts or
enumerations, not quantities --- and the conversion tests fail if that ever drifts.
"""


STRENGTH_PARAMETERS: dict[StrengthModel, dict[str, ParameterSpec]] = {
    StrengthModel.NONE: {},
    StrengthModel.VON_MISES: {
        "yield_stress": ParameterSpec(
            "Pa", 0.0, _INF, "deviatoric stress at which the material yields", _MILUPH
        ),
        # Read by the rock-impact backend only (ADR 0204): the compressible solids pipeline has
        # its own thermal path (the Tillotson energy) and does not read it. 0, or absent, means
        # the yield stress does not soften with heat.
        "melt_energy": ParameterSpec(
            "J/kg",
            0.0,
            _INF,
            "specific energy of shock heat at which the yield stress has softened to zero: "
            "SpaceSim's Y0(T) = Y_limit max(1 - T/T_melt, 0) written for the heat above the "
            "ambient state, specific heat x (melting point - ambient)",
            _SPACESIM_THERMAL,
            optional=True,
        ),
    },
    StrengthModel.DRUCKER_PRAGER: {
        "cohesion": ParameterSpec(
            "Pa", 0.0, _INF, "shear strength extrapolated to zero pressure", _MILUPH
        ),
        "friction_coefficient": ParameterSpec(
            "1", 0.0, _INF, "slope of the yield surface in pressure", _MILUPH
        ),
        "yield_stress": ParameterSpec(
            "Pa", 0.0, _INF, "ceiling the pressure-dependent surface saturates at", _MILUPH
        ),
    },
}

DAMAGE_PARAMETERS: dict[DamageModel, dict[str, ParameterSpec]] = {
    DamageModel.NONE: {},
    DamageModel.GRADY_KIPP: {
        "weibull_k": ParameterSpec(
            "m^-3", 0.0, _INF, "Weibull flaw number density coefficient", _MILUPH
        ),
        "weibull_m": ParameterSpec("1", 0.0, _INF, "Weibull modulus", _MILUPH),
        "crack_growth_speed": ParameterSpec(
            "m/s", 0.0, _INF, "crack tip speed, a fraction of the shear wave speed", _MILUPH
        ),
        "young_modulus": ParameterSpec(
            "Pa", 0.0, _INF, "modulus converting the activation strain to a stress", _MILUPH
        ),
    },
}

POROSITY_PARAMETERS: dict[PorosityModel, dict[str, ParameterSpec]] = {
    PorosityModel.NONE: {},
    PorosityModel.P_ALPHA: {
        "initial_distention": ParameterSpec(
            "1", 1.0, _INF, "alpha_0 = rho_solid / rho_porous at zero pressure", _MILUPH
        ),
        "elastic_pressure": ParameterSpec(
            "Pa", 0.0, _INF, "p_e, below which compaction is reversible", _MILUPH
        ),
        "solid_pressure": ParameterSpec(
            "Pa", 0.0, _INF, "p_s, at which the distention reaches one", _MILUPH
        ),
        "crush_exponent": ParameterSpec(
            "1", 1.0, 8.0, "exponent n of the quasi-static crush curve", _MILUPH
        ),
    },
}


class MaterialRegistry:
    """The materials one context may reference, keyed by the id the device carries."""

    def __init__(self, materials: tuple[MaterialDescriptor, ...]) -> None:
        for material in materials:
            material.validate()
        ids = [material.material_id for material in materials]
        if len(ids) != len(set(ids)):
            raise ValueError("material ids must be unique within a registry")
        names = [material.name for material in materials]
        if len(names) != len(set(names)):
            raise ValueError("material names must be unique within a registry")
        self._materials = {material.material_id: material for material in materials}

    def __len__(self) -> int:
        return len(self._materials)

    def get(self, material_id: int) -> MaterialDescriptor:
        try:
            return self._materials[material_id]
        except KeyError as exc:
            raise KeyError(f"unknown material_id {material_id}") from exc

    @property
    def materials(self) -> tuple[MaterialDescriptor, ...]:
        return tuple(self._materials[key] for key in sorted(self._materials))

    @property
    def capabilities(self) -> frozenset[MaterialCapability]:
        found: set[MaterialCapability] = set()
        for material in self._materials.values():
            found.update(material.capabilities)
        return frozenset(found)

    @property
    def graph_modules(self) -> tuple[str, ...]:
        """Union of the required modules, in pipeline order and without duplicates."""

        order: list[str] = []
        for material in self.materials:
            for module in material.graph_modules:
                if module not in order:
                    order.append(module)
        return tuple(module for module in SOLID_PIPELINE_ORDER if module in order)

    def manifest(self) -> dict[str, Any]:
        """Everything needed to reconstruct the material configuration of a run."""

        return {
            "materials": [
                {
                    "material_id": material.material_id,
                    "name": material.name,
                    "eos": str(material.eos.name),
                    "eos_id": material.eos.eos_id,
                    "eos_parameters": dict(sorted(material.eos.parameters.items())),
                    "shear_modulus": material.shear_modulus,
                    "strength": str(material.strength),
                    "damage": str(material.damage),
                    "porosity": str(material.porosity),
                    "parameters": dict(sorted(material.parameters.items())),
                    "capabilities": sorted(str(item) for item in material.capabilities),
                    "graph_modules": list(material.graph_modules),
                }
                for material in self.materials
            ],
            "graph_modules": list(self.graph_modules),
        }


def registry_from_manifest(manifest: dict[str, Any]) -> MaterialRegistry:
    """Rebuild a registry from what `MaterialRegistry.manifest` wrote.

    A compiled plan carries the materials as node parameters so a changed yield surface
    moves the plan hash, which means the runtime has to read them back out of JSON rather
    than off a live object. Only the authored fields are read: `eos_id`, `capabilities`
    and `graph_modules` are derived, and reading them back would let a hand-edited manifest
    claim a capability its models do not serve - exactly the disagreement
    `MaterialDescriptor.capabilities` exists to make impossible.
    """

    return MaterialRegistry(
        tuple(
            MaterialDescriptor(
                material_id=int(entry["material_id"]),
                name=str(entry["name"]),
                eos=SolidEosSpec(
                    name=SolidEos(str(entry["eos"])),
                    parameters={
                        key: float(value)
                        for key, value in dict(entry["eos_parameters"]).items()
                    },
                ),
                shear_modulus=float(entry.get("shear_modulus", 0.0)),
                strength=StrengthModel(str(entry.get("strength", StrengthModel.NONE))),
                damage=DamageModel(str(entry.get("damage", DamageModel.NONE))),
                porosity=PorosityModel(str(entry.get("porosity", PorosityModel.NONE))),
                parameters={
                    key: float(value)
                    for key, value in dict(entry.get("parameters", {})).items()
                },
            )
            for entry in manifest["materials"]
        )
    )


ALL_PARAMETERS: dict[str, ParameterSpec] = {
    name: spec
    for table in (
        *EOS_PARAMETERS.values(),
        *STRENGTH_PARAMETERS.values(),
        *DAMAGE_PARAMETERS.values(),
        *POROSITY_PARAMETERS.values(),
    )
    for name, spec in table.items()
}


def parameter_default(name: str) -> float:
    """The value an authoring surface starts a parameter at.

    The bottom of the admissible range, because that is the only value the tables themselves
    know is legal for every parameter. It is deliberately not a plausible physical value: a
    field that arrives pre-filled with something believable is a field nobody edits, and the
    numbers this build can offer are not measurements. Presets exist to supply real starting
    points, and they set every key they need.
    """

    minimum = ALL_PARAMETERS[name].minimum
    return float(minimum) if math.isfinite(minimum) else 0.0


@dataclass(frozen=True)
class MaterialPreset:
    """A named starting point for authoring one material.

    Every number here is *synthetic*: dimensionally right, inside the admissible ranges and
    internally consistent, but not transcribed from a published measurement. The read-only
    evidence set carries no material tables, and a preset that named a real rock while
    holding invented numbers would be worse than no library at all - it would look sourced.
    So the names say synthetic, `note` says it again where the user reads it, and every
    parameter stays editable so a user holding sourced values enters them over the top.
    """

    key: str
    label: str
    note: str
    eos: SolidEosSpec
    shear_modulus: float = 0.0
    strength: StrengthModel = StrengthModel.NONE
    damage: DamageModel = DamageModel.NONE
    porosity: PorosityModel = PorosityModel.NONE
    parameters: dict[str, float] = field(default_factory=dict)

    def descriptor(self, material_id: int, name: str = "") -> MaterialDescriptor:
        """This preset as a validated material carrying ``material_id``.

        Validated here rather than at compile time so a preset that drifts out of its own
        admissible range fails in the unit tests instead of in a user's first impact.
        """

        descriptor = MaterialDescriptor(
            material_id=int(material_id),
            name=name or self.key,
            eos=SolidEosSpec(name=self.eos.name, parameters=dict(self.eos.parameters)),
            shear_modulus=self.shear_modulus,
            strength=self.strength,
            damage=self.damage,
            porosity=self.porosity,
            parameters=dict(self.parameters),
        )
        descriptor.validate()
        return descriptor


MATERIAL_PRESETS: tuple[MaterialPreset, ...] = (
    MaterialPreset(
        key="synthetic-fluid",
        label="Synthetic Fluid",
        note="no shear modulus: hydrodynamic only, and the solid stages never compile",
        eos=SolidEosSpec(
            name=SolidEos.MURNAGHAN,
            parameters={"reference_density": 1000.0, "bulk_modulus": 2.2e9, "exponent": 7.0},
        ),
    ),
    MaterialPreset(
        key="synthetic-ice",
        label="Synthetic Ice",
        note="weak volatile shell; the impactor of the layered fixture in the impact suite",
        eos=SolidEosSpec(
            name=SolidEos.MURNAGHAN,
            parameters={"reference_density": 917.0, "bulk_modulus": 9.5e9, "exponent": 5.0},
        ),
        shear_modulus=3.6e9,
        strength=StrengthModel.VON_MISES,
        # ADR 0204: ~1.8 kJ/(kg K) x (273 K - 200 K ambient).
        parameters={"yield_stress": 1.0e7, "melt_energy": 1.3e5},
    ),
    MaterialPreset(
        key="synthetic-rock",
        label="Synthetic Rock",
        note="silicate mantle; the target of the layered fixture in the impact suite",
        eos=SolidEosSpec(
            name=SolidEos.MURNAGHAN,
            parameters={"reference_density": 2700.0, "bulk_modulus": 2.67e10, "exponent": 4.0},
        ),
        shear_modulus=2.27e10,
        strength=StrengthModel.VON_MISES,
        # ADR 0204: ~1.0 kJ/(kg K) x (1400 K solidus - 200 K ambient).
        parameters={"yield_stress": 3.5e9, "melt_energy": 1.2e6},
    ),
    MaterialPreset(
        key="synthetic-iron",
        label="Synthetic Iron",
        note="dense core: the inner shell a differentiated body is differentiated by",
        eos=SolidEosSpec(
            name=SolidEos.MURNAGHAN,
            parameters={"reference_density": 7800.0, "bulk_modulus": 1.28e11, "exponent": 5.5},
        ),
        shear_modulus=7.9e10,
        strength=StrengthModel.VON_MISES,
        # ADR 0204: ~0.6 kJ/(kg K) x (1811 K - 200 K ambient).
        parameters={"yield_stress": 1.0e9, "melt_energy": 1.0e6},
    ),
    MaterialPreset(
        key="synthetic-vaporizing-rock",
        label="Synthetic Vaporizing Rock",
        note="Tillotson: the only preset whose EOS tracks vaporization, so energy evolves",
        eos=SolidEosSpec(
            name=SolidEos.TILLOTSON,
            parameters={
                "reference_density": 2700.0,
                "a": 0.5,
                "b": 1.5,
                "bulk_modulus_a": 2.67e10,
                "bulk_modulus_b": 2.67e10,
                "reference_energy": 5.0e8,
                "incipient_vaporization_energy": 5.0e6,
                "complete_vaporization_energy": 2.0e7,
                "alpha": 5.0,
                "beta": 5.0,
            },
        ),
        shear_modulus=2.27e10,
        strength=StrengthModel.VON_MISES,
        parameters={"yield_stress": 3.5e9, "melt_energy": 1.2e6},
    ),
)

PRESETS_BY_KEY: dict[str, MaterialPreset] = {preset.key: preset for preset in MATERIAL_PRESETS}
