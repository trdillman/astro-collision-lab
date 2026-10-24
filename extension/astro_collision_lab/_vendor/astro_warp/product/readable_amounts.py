"""Plain readouts for a Body's mass and radius when Blender's field cannot show them (ADR 0235).

Blender draws a float field with a fixed number of decimals (at most six), so an Earth in a
Stellar System -- 0.000003 Sun masses -- reads "0.00000" and looks like no mass at all. The
field stays the place to type; this module writes the line drawn under it: the value to three
significant figures in the System's own unit, and the same amount against a familiar object
("about 1.00 Earth masses", "about 6,370 km"). Pure arithmetic, no ``bpy``, so it is tested
without Blender.
"""

from __future__ import annotations

import math

#: Familiar masses, kg (IAU 2015 nominal values for Sun and Earth; Moon and Jupiter by GM / G).
_MASS_REFERENCES = (
    ("Moon masses", 7.346e22),
    ("Earth masses", 5.9722e24),
    ("Jupiter masses", 1.8982e27),
    ("Sun masses", 1.9885e30),
)

#: What a preset's descriptor mass label reads as in a sentence.
_MASS_UNIT_WORDS = {
    "Mearth": "Earth masses",
    "Msun": "Sun masses",
    "1e9 Msun": "billion Sun masses",
}

#: The familiar-object name each preset unit is a multiple of. A comparison against that same
#: object only restates the number ("5 billion Sun masses, about 5 × 10⁹ Sun masses"), so it is
#: left out.
_MASS_UNIT_BASES = {
    "Mearth": "Earth masses",
    "Msun": "Sun masses",
    "1e9 Msun": "Sun masses",
}

#: What a preset's descriptor length label reads as in a sentence.
_LENGTH_UNIT_WORDS = {"km": "km", "au": "au", "kpc": "kpc"}

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

#: A field value inside this band already reads correctly in Blender's fixed decimals.
_READABLE_LOW = 0.01
_READABLE_HIGH = 1e6

_AU_M = 1.495978707e11
_LIGHT_YEAR_M = 9.4607304725808e15

#: Past this many au a distance reads better in light-years than as a long number of au.
_LIGHT_YEAR_FROM_AU = 1e5


def significant(value: float, digits: int = 3) -> str:
    """``value`` to ``digits`` significant figures, in a form a non-programmer reads.

    Ordinary numbers stay ordinary ("6,370", "0.0420"); only very small or very large ones use
    a power of ten, written with a real superscript ("3.00 × 10⁻⁶"), never "3e-06".
    """

    if value == 0.0 or not math.isfinite(value):
        return f"{value:g}"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = round(value / 10.0**exponent, digits - 1)
    if abs(mantissa) >= 10.0:  # 9.996 rounds up into the next decade
        mantissa /= 10.0
        exponent += 1
    if -2 <= exponent < 6:
        decimals = max(digits - 1 - exponent, 0)
        return f"{mantissa * 10.0**exponent:,.{decimals}f}"
    power = str(exponent).translate(_SUPERSCRIPT)
    return f"{mantissa:.{digits - 1}f} × 10{power}"


def needs_readout(value: float) -> bool:
    """True when Blender's fixed decimals would misstate ``value`` (it rounds to 0 or bloats)."""

    magnitude = abs(float(value))
    return magnitude != 0.0 and math.isfinite(magnitude) and not (
        _READABLE_LOW <= magnitude < _READABLE_HIGH
    )


def mass_readout(value: float, mass_scale_kg: float, mass_label: str = "") -> str | None:
    """The line under a Body's Mass field, or ``None`` when the field already reads right."""

    if not needs_readout(value) or not (mass_scale_kg > 0.0 and math.isfinite(mass_scale_kg)):
        return None
    unit = _MASS_UNIT_WORDS.get(mass_label, "System mass units")
    kilograms = float(value) * mass_scale_kg
    name, reference = min(
        _MASS_REFERENCES, key=lambda pair: abs(math.log10(kilograms / pair[1]))
    )
    text = f"Mass {significant(float(value))} {unit}"
    if name != _MASS_UNIT_BASES.get(mass_label, unit):
        text += f", about {significant(kilograms / reference)} {name}"
    return text


def radius_readout(value: float, length_scale_m: float, length_label: str = "") -> str | None:
    """The line under a Body's Radius field, or ``None`` when the field already reads right."""

    if not needs_readout(value) or not (length_scale_m > 0.0 and math.isfinite(length_scale_m)):
        return None
    unit = _LENGTH_UNIT_WORDS.get(length_label, "System length units")
    metres = float(value) * length_scale_m
    familiar = (
        f"{significant(metres)} m" if metres < 1e3
        else f"{significant(metres / 1e3)} km" if metres < 0.01 * _AU_M
        else f"{significant(metres / _AU_M)} au" if metres <= _LIGHT_YEAR_FROM_AU * _AU_M
        else f"{significant(metres / _LIGHT_YEAR_M)} light-years"
    )
    text = f"Radius {significant(float(value))} {unit}"
    if not familiar.endswith(f" {unit}"):
        text += f", about {familiar}"
    return text
