"""Sole owner of preset scale constants and SI/code-unit conversion (ADR 0041).

The solver core runs in nondimensional code units with no editable ``G``, so one module
has to own the mapping between the physical quantities a user authors (km/s, Kelvin,
pascals) and the numbers the integrator consumes. This is that module. UI adapters,
scenario builders, and the material converters all route through it; none of them carries
a duplicate conversion table or an inline ``G`` formula.

The stored constants are *defining* constants, not display values. ``G_SI`` is CODATA
2022; the mass parameters are the IAU 2015 Resolution B3 nominal solar and terrestrial
conversion constants, exact by definition rather than measured masses. Mass scales are
the nominal ``GM`` divided by ``G_SI`` and inherit ``G``'s measured uncertainty. The
parsec is derived as ``(648000 / pi) au`` in ``decimal.Decimal`` with at least 50
significant digits of pi and converted to binary64 once; recomputing it from
already-rounded ``math.pi`` double-rounds and lands one ulp above the accepted Galactic
``Gcode``.

``Gcode`` follows two routes, each pinned by the ADR. Preset scales, mass scales, and
``Gcode`` are all derived inside one ``DERIVATION_PRECISION``-digit ``Decimal`` context
from the defining constants and converted to binary64 once per value; recomputing
``Gcode`` from those already-rounded doubles instead would double-round and land one ulp
above the accepted Galactic constant. Custom ``Gcode`` uses the binary64 formula
``G_SI * M0 * T0^2 / L0^3`` directly, because the ADR's Slice 2 migration stores
``mass_scale_kg = 1.0 / G_SI`` and requires that the result stay exactly ``1.0``; the
Decimal route would return ``1.0000000000000002`` and change legacy dynamics.

``UnitScale`` is frozen and its ``gravitational_constant`` is always the derived value;
there is no editable gravity field anywhere. Construct scales with
:func:`resolve_unit_scale`, which validates Custom scales and rejects finite inputs
whose derived ``Gcode`` or any conversion factor would underflow to zero or overflow to
non-finite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Any

from .materials import (
    DAMAGE_PARAMETERS,
    EOS_PARAMETERS,
    MATERIAL_PRESET_FIELD_UNITS,
    POROSITY_PARAMETERS,
    STRENGTH_PARAMETERS,
    MaterialDescriptor,
    SolidEosSpec,
)
from .model import Units

G_SI = 6.67430e-11
"""CODATA 2022 Newtonian constant of gravitation, m3 kg-1 s-2."""

NOMINAL_TERRESTRIAL_GM = 3.986004418e14
"""IAU 2015 Resolution B3 nominal terrestrial mass parameter, m3 s-2 (exact)."""

NOMINAL_SOLAR_GM = 1.3271244e20
"""IAU 2015 Resolution B3 nominal solar mass parameter, m3 s-2 (exact)."""

ASTRONOMICAL_UNIT_M = 149597870700.0
"""IAU 2012 Resolution B2 exact astronomical unit, metres."""

JULIAN_YEAR_S = 31557600.0
"""Julian year of 365.25 days of 86400 s, seconds."""

GALACTIC_SOLAR_MASS_COUNT = 1e9
"""Galactic mass scale in nominal solar mass parameters."""

DERIVATION_PRECISION = 60
"""Significant decimal digits for every defining derivation; the ADR requires at least 50."""

_DECIMAL_PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923"
    "0781640628620899862803482534211706798"
)
"""Over 100 significant digits of pi; verified against an independently computed value."""

_DECIMAL_G_SI = Decimal("6.67430e-11")
_DECIMAL_TERRESTRIAL_GM = Decimal("3.986004418e14")
_DECIMAL_SOLAR_GM = Decimal("1.3271244e20")
_DECIMAL_ASTRONOMICAL_UNIT = Decimal(149597870700)
_DECIMAL_KILOMETRE = Decimal(1000)
_DECIMAL_HOUR = Decimal(3600)
_DECIMAL_JULIAN_YEAR = Decimal(31557600)
_DECIMAL_MEGAYEAR = Decimal(10) ** 6
_DECIMAL_KILOPARSEC = Decimal(1000)
_DECIMAL_GALACTIC_SOLAR_COUNT = Decimal(10) ** 9


def _decimal_parsec() -> Decimal:
    """``(648000 / pi) au`` in Decimal at derivation precision -- the single parsec formula."""

    with localcontext() as context:
        context.prec = DERIVATION_PRECISION
        return Decimal(648000) / _DECIMAL_PI * _DECIMAL_ASTRONOMICAL_UNIT


def _parsec_metres() -> float:
    """The parsec in metres, converted to binary64 exactly once."""

    return float(_decimal_parsec())


PARSEC_M = _parsec_metres()
"""One parsec in metres, derived rather than pasted as a rounded display literal."""


@dataclass(frozen=True)
class _PresetScales:
    """Base scales, derived ``Gcode``, and descriptor labels for one accepted preset."""

    length_scale_m: float
    mass_scale_kg: float
    time_scale_s: float
    gravitational_constant: float
    length_label: str
    mass_label: str
    time_label: str


def _decimal_gcode(
    gravitational_constant: Decimal,
    mass_scale: Decimal,
    time_scale: Decimal,
    length_scale: Decimal,
) -> Decimal:
    """``G_SI * M0 * T0^2 / L0^3`` from Decimal defining operands, not rounded doubles."""

    return (
        gravitational_constant
        * mass_scale
        * time_scale
        * time_scale
        / (length_scale * length_scale * length_scale)
    )


def _preset(
    length: Decimal,
    mass: Decimal,
    time: Decimal,
    length_label: str,
    mass_label: str,
    time_label: str,
) -> _PresetScales:
    """One table row: Decimal defining scales converted to binary64 once each."""

    return _PresetScales(
        length_scale_m=float(length),
        mass_scale_kg=float(mass),
        time_scale_s=float(time),
        gravitational_constant=float(_decimal_gcode(_DECIMAL_G_SI, mass, time, length)),
        length_label=length_label,
        mass_label=mass_label,
        time_label=time_label,
    )


def _build_presets() -> dict[str, _PresetScales]:
    """Derive the accepted preset table from the defining constants.

    Everything is computed inside one ``DERIVATION_PRECISION``-digit context and converted
    to binary64 once per value. Recomputing ``Gcode`` from the already-rounded doubles
    instead would double-round and land one ulp above the accepted Galactic value.
    """

    with localcontext() as context:
        context.prec = DERIVATION_PRECISION
        parsec = _decimal_parsec()
        terrestrial_mass = _DECIMAL_TERRESTRIAL_GM / _DECIMAL_G_SI
        solar_mass = _DECIMAL_SOLAR_GM / _DECIMAL_G_SI
        galactic_mass = _DECIMAL_GALACTIC_SOLAR_COUNT * _DECIMAL_SOLAR_GM / _DECIMAL_G_SI
        megayear = _DECIMAL_JULIAN_YEAR * _DECIMAL_MEGAYEAR
        return {
            "PLANETARY": _preset(
                _DECIMAL_KILOMETRE, terrestrial_mass, _DECIMAL_HOUR, "km", "Mearth", "hour"
            ),
            "STELLAR": _preset(
                _DECIMAL_ASTRONOMICAL_UNIT,
                solar_mass,
                _DECIMAL_JULIAN_YEAR,
                "au",
                "Msun",
                "Julian year",
            ),
            "GALACTIC": _preset(
                _DECIMAL_KILOPARSEC * parsec, galactic_mass, megayear, "kpc", "1e9 Msun", "Myr"
            ),
        }


_PRESETS = _build_presets()

_CUSTOM_LABELS = ("custom-L", "custom-M", "custom-T")
CUSTOM_UNIT_LABELS = _CUSTOM_LABELS
"""Exact descriptor labels identifying a Custom scale in persisted runtime units."""


def _custom_gcode(mass_scale_kg: float, time_scale_s: float, length_scale_m: float) -> float:
    """``G_SI * M0 * T0^2 / L0^3`` for Custom scales in binary64 arithmetic.

    Binary64 on purpose: the Slice 2 migration stores ``mass_scale_kg = 1.0 / G_SI`` and
    requires ``Gcode`` to stay exactly ``1.0``, which only this route reproduces. The
    cubed denominator is checked before dividing because a finite positive ``L0`` below
    roughly ``8e-108`` cubes to exact zero and would otherwise raise ``ZeroDivisionError``
    instead of the contracted validation error.
    """

    length_cubed = length_scale_m * length_scale_m * length_scale_m
    if length_cubed == 0.0:
        return 0.0
    return G_SI * mass_scale_kg * (time_scale_s * time_scale_s) / length_cubed


def _si_to_code_factors(
    length_scale_m: float, mass_scale_kg: float, time_scale_s: float
) -> dict[str, float]:
    """The single SI-to-code factor table for base scales ``L0``, ``M0``, ``T0``.

    One table, keyed by the closed unit-tag vocabulary. Every conversion path -- the
    dispatcher, the convenience methods, and Custom validation -- reads these factors, so
    no formula exists twice in the product.
    """

    length_cubed = length_scale_m * length_scale_m * length_scale_m
    time_squared = time_scale_s * time_scale_s
    return {
        "m": 1.0 / length_scale_m,
        "kg": 1.0 / mass_scale_kg,
        "s": 1.0 / time_scale_s,
        "m/s": time_scale_s / length_scale_m,
        "kg/m^3": length_cubed / mass_scale_kg,
        "Pa": length_scale_m * time_squared / mass_scale_kg,
        "J/kg": time_squared / (length_scale_m * length_scale_m),
        "m^-3": length_cubed,
        "rad/s": time_scale_s,
        "K": 1.0,
        "1": 1.0,
    }


@dataclass(frozen=True)
class UnitScale:
    """One validated SI/code-unit scale pair; build it with :func:`resolve_unit_scale`.

    ``gravitational_constant`` is the derived code-unit ``G`` and is never editable: ADR
    0037 forbids an authored gravity, and ADR 0041 derives it from the base scales. The
    descriptor labels are the exact strings the runtime ``Units`` carries.
    """

    preset: str
    length_label: str
    mass_label: str
    time_label: str
    length_scale_m: float
    mass_scale_kg: float
    time_scale_s: float
    gravitational_constant: float

    def _factor(self, unit_tag: str) -> float:
        factors = _si_to_code_factors(self.length_scale_m, self.mass_scale_kg, self.time_scale_s)
        try:
            return factors[unit_tag]
        except KeyError:
            raise ValueError(f"unsupported unit tag '{unit_tag}'") from None

    # -- base conversions -----------------------------------------------------

    def length_to_code(self, metres: float) -> float:
        """Metres to code length: ``m / L0``."""

        return self.value_to_code(metres, "m")

    def mass_to_code(self, kilograms: float) -> float:
        """Kilograms to code mass: ``kg / M0``."""

        return self.value_to_code(kilograms, "kg")

    def time_to_code(self, seconds: float) -> float:
        """Seconds to code time: ``s / T0``."""

        return self.value_to_code(seconds, "s")

    def speed_to_code(self, metres_per_second: float) -> float:
        """Metres per second to code speed: ``(m/s) * T0 / L0``."""

        return self.value_to_code(metres_per_second, "m/s")

    def density_to_code(self, kilograms_per_cubic_metre: float) -> float:
        """kg/m^3 to code density: ``(kg/m^3) * L0^3 / M0``."""

        return self.value_to_code(kilograms_per_cubic_metre, "kg/m^3")

    def pressure_to_code(self, pascals: float) -> float:
        """Pascals to code pressure or modulus: ``Pa * L0 * T0^2 / M0``."""

        return self.value_to_code(pascals, "Pa")

    def specific_energy_to_code(self, joules_per_kilogram: float) -> float:
        """J/kg to code specific energy: ``(J/kg) * T0^2 / L0^2``."""

        return self.value_to_code(joules_per_kilogram, "J/kg")

    def inverse_volume_to_code(self, per_cubic_metre: float) -> float:
        """1/m^3 to code inverse volume: ``(1/m^3) * L0^3``."""

        return self.value_to_code(per_cubic_metre, "m^-3")

    def descriptor_units(self, *, softening: float = 0.0) -> Units:
        """The runtime ``Units`` for this scale; the only ``Units`` the role bridge builds."""

        return Units(
            length=self.length_label,
            mass=self.mass_label,
            time=self.time_label,
            gravitational_constant=self.gravitational_constant,
            softening=softening,
            length_scale_m=self.length_scale_m,
            mass_scale_kg=self.mass_scale_kg,
            time_scale_s=self.time_scale_s,
        )

    # -- central dispatch -------------------------------------------------------

    def value_to_code(self, value: float, unit_tag: str) -> float:
        """One SI value to code units for a closed unit tag.

        ``rad/s`` multiplies by ``T0`` only, because radians are dimensionless; ``K`` and
        ``1`` are unchanged. Any other tag raises rather than passing through as if it
        were dimensionless.
        """

        return value * self._factor(unit_tag)

    def value_to_si(self, value: float, unit_tag: str) -> float:
        """One code-unit value back to SI for a closed unit tag."""

        return value / self._factor(unit_tag)

    # -- material descriptors ---------------------------------------------------

    def material_descriptor_to_code(self, descriptor: MaterialDescriptor) -> MaterialDescriptor:
        """One complete SI material descriptor, converted to code units exactly once.

        ADR 0041's one-pass contract for the material half of the authoring surface: the
        descriptor a user authors is SI, the solver consumes code units, and exactly one
        conversion may stand between them. Every dimensional number is dispatched through
        :meth:`value_to_code` under the unit declared for it --- ``MATERIAL_PRESET_FIELD_UNITS``
        for the descriptor's own fields, the parameter tables for every nested
        ``ParameterSpec`` --- so an untagged field or an unknown unit raises here rather
        than passing through as if it were dimensionless. The caller's descriptor is never
        mutated: the converted descriptor and both of its parameter dicts are new objects,
        which is what keeps the source ``MATERIAL_PRESETS`` byte for byte intact however
        many times a compile converts them.

        The complete SI input is validated in SI units before any conversion runs, and the
        converted descriptor is validated before it is returned, for the same reason
        `MaterialPreset.descriptor` validates: a descriptor selecting a model without one
        of its parameters, or carrying a key no registered model reads, must fail with
        the model's own message rather than a ``KeyError`` out of the rebuild (or worse,
        the key silently dropped); and a scale extreme enough to round a converted
        modulus or density to ``inf`` is refused by the range check rather than handed to
        the solver.
        """

        descriptor.validate()

        # PEP 563 keeps annotations as strings, so a float field is "float"; a future
        # `float | None` field must be added to MATERIAL_PRESET_FIELD_UNITS in the same
        # commit --- the metadata tests fail if that drifts.
        dimensional = {
            field.name for field in fields(MaterialDescriptor) if str(field.type) == "float"
        }
        untagged = sorted(dimensional - set(MATERIAL_PRESET_FIELD_UNITS))
        if untagged:
            raise ValueError(
                f"{descriptor.name}: no declared unit for dimensional field(s) {untagged}"
            )

        eos_parameters = {
            key: self.value_to_code(value, EOS_PARAMETERS[descriptor.eos.name][key].unit)
            for key, value in descriptor.eos.parameters.items()
        }
        own_fields = {
            name: self.value_to_code(float(getattr(descriptor, name)), unit)
            for name, unit in MATERIAL_PRESET_FIELD_UNITS.items()
        }
        # An optional parameter a descriptor does not carry stays absent (ADR 0204): converting
        # it would invent a value, and "absent" is the descriptor's way of saying "unused".
        parameters = {
            key: self.value_to_code(float(descriptor.parameters[key]), spec.unit)
            for key, spec in STRENGTH_PARAMETERS[descriptor.strength].items()
            if key in descriptor.parameters or not spec.optional
        }
        parameters.update(
            {
                key: self.value_to_code(float(descriptor.parameters[key]), spec.unit)
                for key, spec in DAMAGE_PARAMETERS[descriptor.damage].items()
                if key in descriptor.parameters or not spec.optional
            }
        )
        parameters.update(
            {
                key: self.value_to_code(float(descriptor.parameters[key]), spec.unit)
                for key, spec in POROSITY_PARAMETERS[descriptor.porosity].items()
                if key in descriptor.parameters or not spec.optional
            }
        )
        converted = MaterialDescriptor(
            material_id=descriptor.material_id,
            name=descriptor.name,
            eos=SolidEosSpec(name=descriptor.eos.name, parameters=eos_parameters),
            strength=descriptor.strength,
            damage=descriptor.damage,
            porosity=descriptor.porosity,
            parameters=parameters,
            **own_fields,
        )
        converted.validate()
        return converted


def _positive_scale(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}")
    return float(value)


def resolve_unit_scale(
    preset: str,
    *,
    length_scale_m: float = 1.0,
    mass_scale_kg: float = 1.0,
    time_scale_s: float = 1.0,
) -> UnitScale:
    """Resolve one preset or Custom System choice into a validated :class:`UnitScale`.

    Presets use their defining constants and ignore the Custom fields. Custom requires
    three finite, strictly positive scales, derives ``Gcode``, and additionally rejects
    finite scales whose ``Gcode`` or any closed-tag conversion factor underflows to zero
    or overflows to non-finite, so an extreme scale fails loudly instead of silently
    producing ``Gcode = 0`` or ``inf``. Unknown presets are refused.
    """

    if preset == "CUSTOM":
        length = _positive_scale("length_scale_m", length_scale_m)
        mass = _positive_scale("mass_scale_kg", mass_scale_kg)
        time = _positive_scale("time_scale_s", time_scale_s)
        labels = _CUSTOM_LABELS
        # A finite positive L0 below roughly 8e-108 cubes to exact zero, and below roughly
        # 1.5e-162 even its square does. Both would otherwise reach a division by that
        # zero in the derived Gcode or a conversion factor, so refuse here first; this
        # check precedes any arithmetic that could divide.
        if length * length * length == 0.0:
            raise ValueError(
                "custom scale combination produces a non-finite or zero gravitational_constant"
            )
        gcode = _custom_gcode(mass, time, length)
        for tag, factor in _si_to_code_factors(length, mass, time).items():
            if not math.isfinite(factor) or factor <= 0.0:
                raise ValueError(
                    f"custom scale combination produces a non-finite or zero "
                    f"{tag} conversion factor"
                )
    elif preset in _PRESETS:
        base = _PRESETS[preset]
        length = base.length_scale_m
        mass = base.mass_scale_kg
        time = base.time_scale_s
        labels = (base.length_label, base.mass_label, base.time_label)
        gcode = base.gravitational_constant
    else:
        raise ValueError(f"unknown unit preset '{preset}'")

    if not math.isfinite(gcode) or gcode <= 0.0:
        raise ValueError(
            "custom scale combination produces a non-finite or zero gravitational_constant"
        )
    return UnitScale(
        preset=preset,
        length_label=labels[0],
        mass_label=labels[1],
        time_label=labels[2],
        length_scale_m=length,
        mass_scale_kg=mass,
        time_scale_s=time,
        gravitational_constant=gcode,
    )


# ---------------------------------------------------------------------------
# Stored-scene unit contract (ADR 0041 "Slice 2 saved-scene migration and break policy")
# ---------------------------------------------------------------------------

UNIT_CONTRACT_VERSION = 1
"""The only contract version this build understands, stored as raw ID property
``astro_warp_unit_contract_version`` by three writers: proven-safe manual adoption, the
successful explicit migration, and fresh Quick Effects System creation
(:func:`mark_fresh_quick_effect_system`, reserved for that writer and unused until Quick
Effects lands)."""

# The raw ID property names the contract owns on a System object. Kept here, next to the
# classifier that reads them, so no Blender-side caller has to spell them twice.
UNIT_CONTRACT_MARKER_KEY = "astro_warp_unit_contract_version"
UNIT_CONTRACT_PROVENANCE_KEY = "astro_warp_unit_migration_provenance"

MIGRATION_PROVENANCE = "slice2-generic-g1"
"""Provenance tag the explicit migration writes alongside marker 1."""

ADOPTED_EXISTING_PROVENANCE = "adopted-existing"
"""Provenance tag for recording a contract that was already current, unchanged.

Distinct from :data:`MIGRATION_PROVENANCE` on purpose: that tag means "these numbers were
remapped to preserve their dynamics", this one means "these numbers were already a valid
current contract and nothing about them changed". Conflating the two would make the
provenance unable to answer the only question it exists for -- whether a System's stored
numbers were ever rewritten."""

LEGACY_GENERIC_SIGNATURE = ("sim-length", "sim-mass", "sim-time", 1.0)
"""The exact descriptor signature Slice 2's role bridge compiled for every preset choice.

The bridge mapped the UI choice ``GALACTIC`` to these labels with ``Gcode=1.0``
(``product_registry._ROLE_UNIT_PRESETS``), which is also the convention the shipped
scenarios and the galaxy-merger reference scene author in, so this one signature is the
complete definition of a legacy System."""

MIGRATION_CUSTOM_SCALES = {"length_scale_m": 1.0, "time_scale_s": 1.0, "mass_scale_kg": 1.0 / G_SI}
"""The migration's Custom scales: ``Gcode`` stays exactly ``1.0`` through the binary64
``_custom_gcode`` route, so every code-space mass, position, velocity, radius, softening,
and seconds-per-frame is numerically unchanged."""


def _matches_migrated_scale(stored: Any, migrated: float) -> bool:
    """True if ``stored`` is ``migrated``, allowing for Blender's 32-bit FloatProperty
    round-trip: the scale is written as a binary64 constant but read back truncated to
    single precision, so an exact ``==`` never matches for large values like
    ``1.0 / G_SI``."""

    if not isinstance(stored, (int, float)):
        return False
    return math.isclose(stored, migrated, rel_tol=1e-6)


class UnitContractError(ValueError):
    """A stored System carries a unit contract this build refuses to guess about.

    Subclasses :class:`ValueError` so the existing operator error lanes catch it without a
    new except-clause shape, while still being a name the migration guard can raise and
    report distinctly."""


class UnitContractState(StrEnum):
    """Which unit contract a stored System carries.

    ``str``-valued so a stored descriptor's ``Units`` labels and the enum render the same
    way in messages and comparisons (``state.value == "CURRENT"``)."""

    CURRENT = "CURRENT"
    LEGACY_G1 = "LEGACY_G1"


@dataclass(frozen=True)
class StoredUnitSignature:
    """The stored descriptor's unit block, or ``None`` when nothing is compiled yet."""

    length: str
    mass: str
    time: str
    gravitational_constant: float


def _current_signature() -> dict[tuple[str, str, str], float]:
    """Preset label triples to their exact ``Gcode``.

    Built from the same ``_PRESETS`` table :func:`resolve_unit_scale` reads, so a preset
    added there reaches the classifier without a second copy of its constants.
    """

    return {
        (base.length_label, base.mass_label, base.time_label): base.gravitational_constant
        for base in _PRESETS.values()
    }


def _custom_labels() -> tuple[str, str, str]:
    return _CUSTOM_LABELS


def classify_unit_contract(
    *, raw_marker: int | None, stored_units: StoredUnitSignature | None
) -> UnitContractState:
    """Classify a stored System's unit contract; every ambiguous case refuses.

    Total over ADR 0041's table:

    - missing marker plus the exact generic signature: LEGACY_G1
    - marker 1 plus no stored descriptor: CURRENT (a fresh, not-yet-compiled System)
    - marker 1 plus an exact current preset or Custom signature: CURRENT
    - unknown marker, marker 1 plus generic/partial/nonfinite/contradictory signature,
      missing marker plus any other signature: :class:`UnitContractError`

    Pure: reads its two inputs and writes nothing, so the load scan, the lifecycle guard,
    and the migration operator all share one classification that cannot drift.
    """

    if raw_marker is not None and type(raw_marker) is not int:
        raise UnitContractError(
            f"unit-contract marker must be an integer, got {type(raw_marker).__name__} "
            f"{raw_marker!r}; refusing to guess the unit contract"
        )

    if stored_units is not None:
        labels = (stored_units.length, stored_units.mass, stored_units.time)
        gcode = stored_units.gravitational_constant
        complete = all(labels) and all(isinstance(label, str) for label in labels)
        finite_positive = type(gcode) is float and math.isfinite(gcode) and gcode > 0.0
        if not finite_positive:
            raise UnitContractError(
                f"stored unit signature has a non-finite or non-positive "
                f"gravitational_constant {gcode!r}; refusing to guess the unit contract"
            )
        if not complete:
            raise UnitContractError(
                f"stored unit signature {stored_units!r} is partial; refusing to guess "
                f"the unit contract"
            )
    else:
        labels = None

    if raw_marker is not None and raw_marker != UNIT_CONTRACT_VERSION:
        raise UnitContractError(
            f"unknown unit-contract marker {raw_marker!r}; this build understands only "
            f"version {UNIT_CONTRACT_VERSION}"
        )

    if raw_marker == UNIT_CONTRACT_VERSION:
        if labels is None:
            return UnitContractState.CURRENT
        if labels == LEGACY_GENERIC_SIGNATURE[:3]:
            raise UnitContractError(
                f"marker {UNIT_CONTRACT_VERSION} contradicts the stored generic "
                f"signature {stored_units!r}; refusing to guess the unit contract"
            )
        preset_gcode = _current_signature().get(labels)
        if preset_gcode is not None:
            # FloatProperty truncation: a preset Gcode stored as a binary64 constant
            # reads back truncated to float32 (GALACTIC -> 0.004498502239584923), so
            # compare within the round-to-nearest bound of half a float32 ulp --
            # 2.0**-24 provably covers every float32 round-trip while still refusing
            # a 1e-7 relative mismatch (the wrong-Gcode contract test). The same trap
            # _matches_migrated_scale and the Slice 2 golden proof already pinned.
            if not math.isclose(
                stored_units.gravitational_constant, preset_gcode, rel_tol=2.0**-24
            ):
                raise UnitContractError(
                    f"marker {UNIT_CONTRACT_VERSION} contradicts the stored signature "
                    f"{stored_units!r}; refusing to guess the unit contract"
                )
            return UnitContractState.CURRENT
        if labels == _custom_labels():
            # Custom labels carry any finite positive Gcode; the role bridge separately
            # recomputes and compares it from the stored scales at compile time.
            return UnitContractState.CURRENT
        raise UnitContractError(
            f"marker {UNIT_CONTRACT_VERSION} contradicts the stored signature "
            f"{stored_units!r}; refusing to guess the unit contract"
        )

    # No marker: only the exact stored generic signature is legacy. A marker-free
    # System with no descriptor has no evidence that distinguishes Slice 2 from a
    # partially authored/corrupt object, so the accepted break policy refuses it.
    if (
        labels == LEGACY_GENERIC_SIGNATURE[:3]
        and stored_units.gravitational_constant == LEGACY_GENERIC_SIGNATURE[3]
    ):
        return UnitContractState.LEGACY_G1
    raise UnitContractError(
        f"no unit-contract marker with stored signature {stored_units!r}; refusing to "
        f"guess the unit contract"
    )


def signature_records_a_current_contract(
    stored_units: StoredUnitSignature | None,
) -> bool:
    """Whether ``stored_units`` would classify as CURRENT if it carried marker 1.

    The question an unmarked System cannot answer for itself. ``classify_unit_contract``
    refuses a marker-free non-legacy signature because "no marker" and "these numbers"
    together do not say whether the numbers were authored under this contract or are a
    partial/corrupt object -- it must not guess. Asking whether the signature is
    *self-consistent* under marker 1 is a different, decidable question: preset labels
    must pair with that preset's exact ``Gcode``, Custom labels need a finite positive
    one. A signature that fails that test is refused as firmly as before.

    This is the predicate the explicit adopt path is allowed to act on, and only that
    path: it records a fact the classifier can already verify rather than resolving the
    ambiguity the classifier refuses. Pure, and writes nothing.
    """

    if stored_units is None:
        return False
    try:
        state = classify_unit_contract(
            raw_marker=UNIT_CONTRACT_VERSION, stored_units=stored_units
        )
    except UnitContractError:
        return False
    return state is UnitContractState.CURRENT


def system_records_a_current_contract(system: Any, *, scene: Any | None = None) -> bool:
    """Whether ``system``'s stored signature is already a valid current contract.

    The System-level form of :func:`signature_records_a_current_contract`, so the adopt
    path does not have to reach for the private stored-signature reader. Reads two inputs
    and writes nothing.
    """

    return signature_records_a_current_contract(
        _stored_signature_from_system(system, scene=scene)
    )


def _stored_signature_from_system(
    system: Any, *, scene: Any | None = None
) -> StoredUnitSignature | None:
    """The System's stored descriptor signature, or ``None`` when nothing is compiled.

    Reads the context row this System controls (``astro_warp_context_id`` then the
    controller-name fallback the role bridge itself uses) and parses its stored descriptor
    JSON. A row whose JSON cannot be parsed is *evidence of a signature we cannot read*,
    which classifies as an error rather than as "nothing stored": ADR 0041's "unknown or
    partial evidence counts" rule.
    """

    if scene is None:
        return None
    controller = system.name
    stored_context_id = system.get("astro_warp_context_id", "") or ""
    items = getattr(scene, "astro_warp_contexts", ()) or ()
    if stored_context_id:
        matches = [item for item in items if item.context_id == stored_context_id]
        if len(matches) != 1:
            raise UnitContractError(
                f"System '{controller}' stores context id {stored_context_id!r}, but it "
                f"resolves to {len(matches)} rows; refusing to guess the unit contract"
            )
        candidate = matches[0]
        row_controller = getattr(candidate, "controller_object_name", "")
        if row_controller != controller:
            raise UnitContractError(
                f"System '{controller}' stores context id {stored_context_id!r}, but its "
                f"row belongs to {row_controller!r}; refusing to guess the unit contract"
            )
    else:
        matches = [
            item
            for item in items
            if getattr(item, "controller_object_name", "") == controller
        ]
        if len(matches) > 1:
            raise UnitContractError(
                f"System '{controller}' resolves to {len(matches)} controller rows; "
                "refusing to guess the unit contract"
            )
        if not matches:
            return None
        candidate = matches[0]
    payload = getattr(candidate, "descriptor_json", "") or ""
    if not payload:
        raise UnitContractError(
            f"System '{controller}' has a context row with no stored descriptor; "
            "refusing to guess the unit contract"
        )
    try:
        import json

        units = json.loads(payload)["units"]
        labels = (units["length"], units["mass"], units["time"])
        gcode = units["gravitational_constant"]
        if not all(type(label) is str for label in labels) or type(gcode) is not float:
            raise UnitContractError(
                f"System '{controller}' has a stored descriptor whose unit block has "
                "invalid value types; refusing to guess the unit contract"
            )
        return StoredUnitSignature(
            length=labels[0],
            mass=labels[1],
            time=labels[2],
            gravitational_constant=gcode,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UnitContractError(
            f"System '{controller}' has a stored descriptor whose unit block cannot be "
            f"read ({exc}); refusing to guess the unit contract"
        ) from exc


def classify_system_unit_contract(
    system: Any, *, scene: Any | None = None
) -> UnitContractState:
    """Classify one real Blender System using its owning Scene's context rows.

    Blender Objects do not expose a fictional ``astro_warp_scene`` attribute. Callers
    already own the operation context, so they pass ``context.scene`` explicitly; pure
    tests may pass an equivalent scene-shaped object. Marker values are forwarded
    without coercion so bool/string/float corruption cannot masquerade as version 1.
    """

    raw_marker = system.get(UNIT_CONTRACT_MARKER_KEY, None)
    stored = _stored_signature_from_system(system, scene=scene)
    return classify_unit_contract(raw_marker=raw_marker, stored_units=stored)


def require_current_unit_contract(
    system: Any, operation: str, *, scene: Any | None = None
) -> UnitContractState:
    """The one shared lifecycle gate: classify ``system`` or raise before any mutation.

    Compile, Initialize, Play, and Bake (and later Quick Effects reuse) call this before
    touching state, so no entry point duplicates classification and none can mutate a
    legacy or corrupt System. ``system`` may be ``None`` for a scene with no role System
    (the pre-role context-item path), which passes as CURRENT because there is no stored
    unit contract to gate.
    """

    if system is None:
        return UnitContractState.CURRENT
    stored: StoredUnitSignature | None = None
    try:
        stored = _stored_signature_from_system(system, scene=scene)
        state = classify_unit_contract(
            raw_marker=system.get(UNIT_CONTRACT_MARKER_KEY, None),
            stored_units=stored,
        )
    except UnitContractError as exc:
        # A successful explicit migration intentionally leaves the old compiled row
        # untouched and tells the user to Compile Context next. That one operation must
        # be able to replace the known-stale generic row; Initialize/Play/Bake still
        # refuse it, so the old descriptor can never run under the new marker.
        settings = getattr(getattr(system, "astro_warp_role", None), "system_settings", None)
        migrated = (
            operation == "Compile Context"
            and type(system.get(UNIT_CONTRACT_MARKER_KEY, None)) is int
            and system.get(UNIT_CONTRACT_MARKER_KEY, None) == UNIT_CONTRACT_VERSION
            and system.get(UNIT_CONTRACT_PROVENANCE_KEY, None) == MIGRATION_PROVENANCE
            and settings is not None
            and getattr(settings, "unit_preset", None) == "CUSTOM"
            and _matches_migrated_scale(
                getattr(settings, "length_scale_m", None),
                MIGRATION_CUSTOM_SCALES["length_scale_m"],
            )
            and _matches_migrated_scale(
                getattr(settings, "mass_scale_kg", None),
                MIGRATION_CUSTOM_SCALES["mass_scale_kg"],
            )
            and _matches_migrated_scale(
                getattr(settings, "time_scale_s", None),
                MIGRATION_CUSTOM_SCALES["time_scale_s"],
            )
            and stored == StoredUnitSignature(*LEGACY_GENERIC_SIGNATURE)
        )
        if migrated:
            return UnitContractState.CURRENT
        raise UnitContractError(
            f"'{operation}' refused: System '{system.name}' has an unusable unit "
            f"contract ({exc}); use {MIGRATE_LABEL} after backing up"
        ) from exc
    if state is not UnitContractState.CURRENT:
        raise UnitContractError(
            f"'{operation}' refused: System '{system.name}' still carries the legacy "
            f"generic G=1 unit contract; run {MIGRATE_LABEL} "
            f"(F3-search '{MIGRATE_IDNAME}') to map it to Custom scales without "
            f"changing its dynamics"
        )
    return state


def require_current_row_unit_contract(
    row: Any, controller: Any, operation: str
) -> UnitContractState:
    """Gate the role-free lane on the active context row's own stored unit fields.

    A role System's gate reads the raw marker on the System plus the stored descriptor
    the row holds for it. When no role System is active -- the role was moved away, or a
    pre-role Slice 2 file where no object ever carried a role -- that same contract is
    still readable as the active row's unit fields plus the raw marker on the row's
    controller object, so this lane classifies through the same pure classifier instead
    of passing unmarked rows through. The refusal names the context row, not a role
    System, because the row is what this lane owns.

    The refusal set is the contract's actual target (ADR 0041 acceptance scope), not the
    classifier's full table: an *unmarked* row refuses only on the exact stored Slice 2
    generic-G1 signature, because every other unmarked shape is fresh authoring this
    surface still emits (create_context's SI defaults, the guided flow's scaled labels)
    and a refusal there would be a dead end the migration operator cannot recover from.
    A *marked* row has declared a contract, so it classifies fully: unknown or
    contradicted markers refuse. The classifier's own legacy-triple determination decides
    the unmarked branch, so this lane carries no second copy of the table.

    ``controller`` may be ``None`` (a row with no controller carries no marker); reads
    two inputs and writes nothing, exactly like the System gate.
    """

    name = getattr(row, "name", "") or getattr(row, "context_id", "") or "context"
    labels = (
        getattr(row, "units_length", "") or "",
        getattr(row, "units_mass", "") or "",
        getattr(row, "units_time", "") or "",
    )
    raw_gcode = getattr(row, "gravitational_constant", None)
    try:
        gcode = None if raw_gcode is None else float(raw_gcode)
    except (TypeError, ValueError) as exc:
        raise UnitContractError(
            f"context row '{name}' has a non-numeric gravitational_constant "
            f"{raw_gcode!r}; refusing to guess the unit contract"
        ) from exc
    stored = (
        None
        if gcode is None and not any(labels)
        else StoredUnitSignature(*labels, gravitational_constant=gcode)
    )
    raw_marker = None if controller is None else controller.get(UNIT_CONTRACT_MARKER_KEY, None)
    if raw_marker is None:
        # Unmarked row: refuse only the exact generic-G1 signature -- the shape Slice 2
        # saved and the migration operator exists to carry. Every other unmarked shape
        # is fresh authoring, so it passes; the classifier itself decides which side of
        # that line the signature falls on.
        try:
            state = classify_unit_contract(raw_marker=None, stored_units=stored)
        except UnitContractError:
            return UnitContractState.CURRENT
        if state is UnitContractState.LEGACY_G1:
            raise UnitContractError(
                f"'{operation}' refused: context row '{name}' still carries the legacy "
                f"generic G=1 unit contract; run {MIGRATE_LABEL} "
                f"(F3-search '{MIGRATE_IDNAME}') to map it to Custom scales without "
                f"changing its dynamics"
            )
        return state
    try:
        state = classify_unit_contract(raw_marker=raw_marker, stored_units=stored)
    except UnitContractError as exc:
        raise UnitContractError(
            f"'{operation}' refused: context row '{name}' has an unusable unit "
            f"contract ({exc}); use {MIGRATE_LABEL} after backing up"
        ) from exc
    if state is not UnitContractState.CURRENT:
        raise UnitContractError(
            f"'{operation}' refused: context row '{name}' still carries the legacy "
            f"generic G=1 unit contract; run {MIGRATE_LABEL} "
            f"(F3-search '{MIGRATE_IDNAME}') to map it to Custom scales without "
            f"changing its dynamics"
        )
    return state


def mark_fresh_quick_effect_system(system: Any) -> None:
    """Write marker 1 for a System freshly created inside a Quick Effects transaction.

    Reserved for Task 7's writer; defined here so the marker key and its three legal
    writers stay in one module rather than spreading the key name across the tree.
    """

    system[UNIT_CONTRACT_MARKER_KEY] = UNIT_CONTRACT_VERSION


def migration_unit_scale() -> UnitScale:
    """The Custom scale the explicit migration maps a legacy System to."""

    return resolve_unit_scale(preset="CUSTOM", **MIGRATION_CUSTOM_SCALES)


MIGRATE_LABEL = "Migrate Legacy Generic G=1 Units"
MIGRATE_IDNAME = "astro_warp_phase0.migrate_legacy_generic_units"
