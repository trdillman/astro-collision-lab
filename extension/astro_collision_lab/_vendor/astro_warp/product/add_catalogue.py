"""The Add-menu catalogue: one row per object an artist can add (AW-007, ADR 0164).

Pure Python, no Blender. ``CATALOGUE`` is the single source for the Add > Astro Warp
submenus, their tooltips and every number a freshly added object starts with; the Blender half
(``add_objects``) draws and applies, and this module decides. :func:`plan_object` turns a preset
plus the 3D cursor, an optional central Body and the redo-panel multipliers into an
:class:`ObjectPlan` in the *System's own units*, so a test can check what will be written
without a window.

Every size and mass here is SI, the way a reference book states it, and is converted through
``resolve_unit_scale`` for the preset's own unit preset, so a row cannot carry a number in the
wrong System's units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .unit_presets import G_SI, resolve_unit_scale

# -- constants -----------------------------------------------------------------------------

MSUN_KG = 1.3271244e20 / G_SI
"""Nominal solar mass in kg (IAU 2015 B3 GM over CODATA G)."""
MEARTH_KG = 5.972168e24
AU_M = 149597870700.0
PARSEC_M = 648000.0 / math.pi * AU_M
KPC_M = PARSEC_M * 1000.0
KM_M = 1000.0
C_M_S = 299792458.0

MIN_BODY_MASS = 1e-12
"""The smallest Mass the Body and Cloud Total Mass fields store, in System mass units."""

REARTH_M = 6.371e6
HANDLE_EARTH_UNITS = 0.1
"""ADR 0233: the sphere an Earth-sized Body is added as, in its System's length units.

At true scale a planet is tens of thousands of times smaller than its orbit (an Earth is 0.04 mm
across where 1 AU is a metre), so the sphere the artist sees and grabs is a handle, about 25
pixels across from Blender's default view (0.02 was tried first and a window showed it hidden
behind the 3D cursor). A Sun's handle (about 0.48) still leaves a planet added 1 AU out well
clear of it. The handle is never smaller
than the real Body, otherwise this times the cube root of its real size over Earth's, so a Sun
still looks bigger than a planet and an asteroid smaller. Only the sphere: the Body's Radius stays
the real one, and gravity and collisions use that.
"""
HANDLE_MIN_FRACTION = 0.25
"""No handle is smaller than this share of an Earth's (about a comet's), so a 30 km black hole
can still be seen and grabbed. An orbiting Body's cap below still wins."""
HANDLE_ORBIT_FRACTION = 0.25
"""An orbiting Body's handle never reaches past this share of the way to what it orbits."""


def handle_radius(real_radius: float, real_radius_m: float, distance: float = 0.0) -> float:
    """ADR 0233: the radius of the sphere a Body is added as, in System length units.

    ``real_radius`` is the Body's Radius in those units and ``real_radius_m`` the same in metres;
    ``distance`` is how far it is from what it orbits (zero when it orbits nothing).
    """

    handle = HANDLE_EARTH_UNITS * math.cbrt(max(real_radius_m, 0.0) / REARTH_M)
    handle = max(handle, HANDLE_MIN_FRACTION * HANDLE_EARTH_UNITS)
    if distance > 0.0:
        handle = min(handle, HANDLE_ORBIT_FRACTION * distance)
    return max(real_radius, handle)

CATEGORIES: tuple[str, ...] = (
    "Planets",
    "Moons",
    "Minor bodies",
    "Stars",
    "Black holes",
    "Disks & Groups",
    "Stellar systems",
    "Interstellar matter",
)
"""The eight submenus, in menu order."""


def menu_idname(category: str) -> str:
    """The Blender menu id of a category submenu (``Disks & Groups`` -> ``..._disks_groups``)."""

    slug = "".join(ch if ch.isalnum() else "_" for ch in category.lower())
    slug = "_".join(part for part in slug.split("_") if part)
    return f"VIEW3D_MT_astro_warp_add_{slug}"


HEADROOM_CEILING: dict[str, int] = {
    "body": 1,
    "ball": 5000,
    "disk": 2000,
    "galaxy": 1500,
}
"""ADR 0076's measured headroom by what the preset is: the Cluster's 5,000 bodies (46 percent
of the 20.833 ms budget), the Protoplanetary Disk's 2,000 (43 percent) and the Galaxy
Collision's 1,500 per galaxy (48 percent at 3,000 total)."""

ORBIT_NONE = "none"
ORBIT_OPTIONAL = "optional"
ORBIT_REQUIRED = "required"

SUPPORTED_UNIT_PRESETS: tuple[str, ...] = ("STELLAR", "GALACTIC")

# What a System this catalogue *makes* gets, because the System defaults (one galactic
# megayear per frame, a softening of 1.0) are wrong for a stellar scene. An existing System
# is never touched. First values, not measured ones (ADR 0164 Consequences).
NEW_SYSTEM_PACING: dict[str, dict[str, float]] = {
    "STELLAR": {"seconds_per_frame": 1.0 / 365.25, "softening": 1e-4, "substeps": 8},
    "GALACTIC": {"seconds_per_frame": 1.0, "softening": 0.01, "substeps": 1},
}


class AddRefused(ValueError):
    """The add cannot be done; the message is for the artist."""


@dataclass(frozen=True)
class AddPreset:
    preset_id: str
    category: str
    label: str
    tooltip: str
    unit_preset: str
    role: str  # "BODY" | "CLOUD"
    headroom: str  # key of HEADROOM_CEILING
    size_m: float  # Body radius, or a cloud's outer radius (ring: see ring_radii)
    mass_kg: float  # Body mass, or a cloud's total mass
    count: int = 1
    shape: str = ""
    velocity_model: str = ""
    orbit: str = ORBIT_NONE
    speed_factor: float = 1.0
    default_distance_m: float = 0.0
    pinned: bool = False
    support_fraction: float = 1.0
    inner_m: float = 0.0
    thickness_m: float = 0.0
    centre_concentration: float = 0.0
    ring_radii: tuple[float, float] = field(default=(0.0, 0.0))
    """For a ring sized in the central Body's radii: (inner, outer). Zero means absolute."""
    softening_fraction: float = 0.0
    """A galactic cloud's softening for a System this add makes, as a fraction of its radius."""
    centred: bool = False
    """With ``orbit == ORBIT_REQUIRED``: the item is centred on the central Body and carried with
    it, not set orbiting it (ADR 0202). The Planetary nebula's shell is round its star."""
    expansion_speed_m_s: float = 0.0
    """For ``velocity_model == "EXPANSION"``: the outward speed in SI. Zero is the cloud's own
    escape speed, the model's default."""
    expansion_proportional: bool = False
    """For ``velocity_model == "EXPANSION"``: the speed grows with distance (a Hubble flow)."""
    shearing: bool = False
    """The item is a local shearing box (ADR 0202): it makes a System of its own with the
    Shearing Box option on, sized from ``box_m`` and turning at the orbital rate of a patch
    ``shear_radius_m`` from a body of ``shear_mass_kg``, and it is placed at the world origin the
    box is centred on, whatever the 3D cursor says."""
    box_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """For a shearing item: the box's width (radial, X), length (azimuthal, Y) and height (Z)."""
    shear_radius_m: float = 0.0
    shear_mass_kg: float = 0.0
    """For a shearing item: the patch sits ``shear_radius_m`` from a body of ``shear_mass_kg``,
    which sets the box's angular speed ``sqrt(G M / r^3)``. That body is not added: the box
    replaces its pull with the tidal field."""
    substeps: int = 0
    """Substeps per frame for a System this add makes; zero keeps the scale's default. The
    shearing box advances its tidal and Coriolis terms exactly, so only the particles' mutual
    pull needs resolving, and at the sheet's pace it needs far fewer than the stellar default."""


def _p(**kwargs) -> AddPreset:
    return AddPreset(**kwargs)


_GRAVITY_ONLY = (
    " Gravity only: no pressure, no heating, no drag and no glow, so it holds together or "
    "collapses by its own weight and nothing else."
)

CATALOGUE: tuple[AddPreset, ...] = (
    _p(
        preset_id="rocky_planet", category="Planets", label="Rocky planet",
        tooltip="One Earth-sized rocky planet at the 3D cursor. It orbits the active Body, or "
        "the heaviest Body in the System, if there is one; otherwise it starts at rest.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=6.371e6, mass_kg=MEARTH_KG, orbit=ORBIT_OPTIONAL, default_distance_m=AU_M,
    ),
    _p(
        preset_id="gas_giant", category="Planets", label="Gas giant",
        tooltip="One Jupiter-sized gas giant at the 3D cursor. It orbits the active Body, or "
        "the heaviest Body in the System, if there is one; otherwise it starts at rest.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=6.9911e7, mass_kg=1.8982e27, orbit=ORBIT_OPTIONAL,
        default_distance_m=5.2 * AU_M,
    ),
    _p(
        preset_id="ice_giant", category="Planets", label="Ice giant",
        tooltip="One Neptune-sized ice giant at the 3D cursor. It orbits the active Body, or "
        "the heaviest Body in the System, if there is one; otherwise it starts at rest.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=2.4622e7, mass_kg=1.02409e26, orbit=ORBIT_OPTIONAL,
        default_distance_m=30.0 * AU_M,
    ),
    _p(
        preset_id="moon", category="Moons", label="Moon",
        tooltip="One Moon-sized moon at the 3D cursor. Select the planet first and it orbits "
        "that planet; with nothing selected it orbits the heaviest Body in the System.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=1.7374e6, mass_kg=7.342e22, orbit=ORBIT_OPTIONAL, default_distance_m=3.844e8,
    ),
    _p(
        preset_id="asteroid", category="Minor bodies", label="Asteroid",
        tooltip="One Vesta-sized asteroid at the 3D cursor. It orbits the active Body, or the "
        "heaviest Body in the System, if there is one; otherwise it starts at rest.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=2.627e5, mass_kg=2.59e20, orbit=ORBIT_OPTIONAL, default_distance_m=2.7 * AU_M,
    ),
    _p(
        preset_id="comet", category="Minor bodies", label="Comet",
        tooltip="One comet nucleus at the 3D cursor, started at the far end of a stretched "
        "orbit round the active Body or the heaviest Body in the System. The Body Mass field "
        "cannot go below about 2e18 kg, so this is a large Centaur-class nucleus, not a small "
        "one. There is no tail.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=1.0e5, mass_kg=5.0e18, orbit=ORBIT_OPTIONAL, speed_factor=0.5,
        default_distance_m=5.0 * AU_M,
    ),
    _p(
        preset_id="star", category="Stars", label="Star",
        tooltip="One Sun-like star at the 3D cursor. It stays where it is unless the System "
        "moves it.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=6.957e8, mass_kg=MSUN_KG,
    ),
    _p(
        preset_id="black_hole", category="Black holes", label="Black hole",
        tooltip="A ten-solar-mass black hole at the 3D cursor. It is a Pinned Body: it holds "
        "still and pulls with its full mass. It does not swallow anything by itself; what "
        "happens to a particle that reaches it follows the System's Collision Mode, and "
        "Absorb (Direct solver only) is what makes it eat.",
        unit_preset="STELLAR", role="BODY", headroom="body",
        size_m=2.0 * G_SI * 10.0 * MSUN_KG / C_M_S**2, mass_kg=10.0 * MSUN_KG, pinned=True,
    ),
    _p(
        preset_id="planetary_ring", category="Disks & Groups", label="Planetary ring",
        tooltip="A ring of particles round a planet, about Saturn's rings in proportion. "
        "Select the planet first; the ring is centred on it and orbits it.",
        unit_preset="STELLAR", role="CLOUD", headroom="disk",
        size_m=0.0, mass_kg=1e-11 * MSUN_KG, count=2000, shape="DISK_ANNULUS",
        velocity_model="KEPLERIAN", orbit=ORBIT_REQUIRED, ring_radii=(1.2, 2.3),
        thickness_m=0.0,
    ),
    _p(
        preset_id="accretion_disk", category="Disks & Groups", label="Accretion disk",
        tooltip="A disk of particles orbiting a star or black hole, from 0.01 to 0.1 au and "
        "denser towards the middle. Select the star or black hole first. Orbiting particles "
        "only: no heating, no glow.",
        unit_preset="STELLAR", role="CLOUD", headroom="disk",
        size_m=0.1 * AU_M, mass_kg=1e-8 * MSUN_KG, count=2000, shape="DISK_ANNULUS",
        velocity_model="KEPLERIAN", orbit=ORBIT_REQUIRED, inner_m=0.01 * AU_M,
        thickness_m=0.005 * AU_M, centre_concentration=1.0,
    ),
    _p(
        preset_id="orbiting_group", category="Disks & Groups", label="Orbiting group",
        tooltip="A small ball of particles at the 3D cursor, moving round the active Body at "
        "the speed of a circular orbit. Select the Body first.",
        unit_preset="STELLAR", role="CLOUD", headroom="ball",
        size_m=0.02 * AU_M, mass_kg=1e-9 * MSUN_KG, count=1000, shape="BALL",
        velocity_model="VIRIALIZE", orbit=ORBIT_REQUIRED, default_distance_m=AU_M,
    ),
    _p(
        preset_id="free_group", category="Disks & Groups", label="Free group",
        tooltip="A ball of particles at the 3D cursor, at rest. Set its Bulk Velocity to make "
        "it drift.",
        unit_preset="STELLAR", role="CLOUD", headroom="ball",
        size_m=0.1 * AU_M, mass_kg=1e-9 * MSUN_KG, count=2000, shape="BALL",
        velocity_model="UNIFORM",
    ),
    _p(
        preset_id="scatter", category="Disks & Groups", label="Scatter",
        tooltip="A ball of particles at the 3D cursor with more motion than its own gravity "
        "can hold, so it flies apart.",
        unit_preset="STELLAR", role="CLOUD", headroom="ball",
        size_m=0.1 * AU_M, mass_kg=1e-9 * MSUN_KG, count=2000, shape="BALL",
        velocity_model="VIRIALIZE", support_fraction=2.0,
    ),
    _p(
        preset_id="shearing_sheet", category="Disks & Groups", label="Shearing sheet",
        tooltip="A small box cut out of a rotating disk: 0.01 au square, 1 au from a Sun-like "
        "star, with 2,000 particles. It runs in a frame that turns with the box, so the inner "
        "edge of the sheet moves faster than the outer and the sheet shears into streaks, and "
        "its sides wrap round (a shearing box). It makes a System of its own, placed at the "
        "world origin whatever the 3D cursor says, and it needs the Direct solver. Its "
        "particles start on circular orbits with no random motion, so gravity gathers them "
        "into clumps and wakes. No star is added: the box stands in for its pull.",
        unit_preset="STELLAR", role="CLOUD", headroom="disk",
        size_m=0.01 * AU_M, mass_kg=3e-9 * MSUN_KG, count=2000, shape="OBJECT_VOLUME",
        velocity_model="UNIFORM", shearing=True,
        box_m=(0.01 * AU_M, 0.01 * AU_M, 0.0005 * AU_M),
        shear_radius_m=AU_M, shear_mass_kg=MSUN_KG, softening_fraction=0.01, substeps=2,
    ),
    _p(
        preset_id="disk_galaxy", category="Stellar systems", label="Disk galaxy",
        tooltip="A flat, rotating galaxy disk of 1,500 particles at the 3D cursor, 10 kpc "
        "across, held up by its own gravity. The dark matter halo is the separate Cold dark "
        "matter halo item.",
        unit_preset="GALACTIC", role="CLOUD", headroom="galaxy",
        size_m=10.0 * KPC_M, mass_kg=5e10 * MSUN_KG, count=1500, shape="DISK_ANNULUS",
        velocity_model="ROTATION_CURVE", thickness_m=1.0 * KPC_M, centre_concentration=1.0,
        softening_fraction=0.02,
    ),
    _p(
        preset_id="globular_cluster", category="Stellar systems", label="Globular cluster",
        tooltip="A ball of 5,000 stars at the 3D cursor, 20 parsecs across, in balance "
        "against its own gravity.",
        unit_preset="GALACTIC", role="CLOUD", headroom="ball",
        size_m=20.0 * PARSEC_M, mass_kg=5e5 * MSUN_KG, count=5000, shape="BALL",
        velocity_model="VIRIALIZE", softening_fraction=0.02,
    ),
    _p(
        preset_id="elliptical_galaxy", category="Stellar systems", label="Elliptical galaxy",
        tooltip="An approximate elliptical galaxy: a ball of 1,500 particles, 5 kpc across, "
        "in balance against its own gravity. It has no realistic brightness profile.",
        unit_preset="GALACTIC", role="CLOUD", headroom="galaxy",
        size_m=5.0 * KPC_M, mass_kg=1e11 * MSUN_KG, count=1500, shape="BALL",
        velocity_model="VIRIALIZE", softening_fraction=0.02,
    ),
    _p(
        preset_id="dark_matter_halo", category="Stellar systems",
        label="Cold dark matter halo",
        tooltip="A ball of 1,500 particles of dark matter, 30 kpc across, to put round a "
        "galaxy. It pulls like dark matter but is drawn like any other particle cloud; if it "
        "hides the galaxy, give it its own colour (Color Source: Source Object).",
        unit_preset="GALACTIC", role="CLOUD", headroom="galaxy",
        size_m=30.0 * KPC_M, mass_kg=1e12 * MSUN_KG, count=1500, shape="BALL",
        velocity_model="VIRIALIZE", softening_fraction=0.02,
    ),
    _p(
        preset_id="gas_cloud", category="Interstellar matter", label="Gas cloud",
        tooltip="A cloud of gas particles at the 3D cursor, 20 parsecs across." + _GRAVITY_ONLY,
        unit_preset="GALACTIC", role="CLOUD", headroom="ball",
        size_m=20.0 * PARSEC_M, mass_kg=1e5 * MSUN_KG, count=3000, shape="BALL",
        velocity_model="VIRIALIZE", softening_fraction=0.02,
    ),
    _p(
        preset_id="dust_cloud", category="Interstellar matter", label="Dust cloud",
        tooltip="A cloud of dust particles at the 3D cursor, 5 parsecs across." + _GRAVITY_ONLY,
        unit_preset="GALACTIC", role="CLOUD", headroom="ball",
        size_m=5.0 * PARSEC_M, mass_kg=1e3 * MSUN_KG, count=3000, shape="BALL",
        velocity_model="VIRIALIZE", softening_fraction=0.02,
    ),
    _p(
        preset_id="planetary_nebula", category="Interstellar matter", label="Planetary nebula",
        tooltip="A thin shell of gas puffed off a dying star, drifting outward. Select the star "
        "first; the shell is centred on it and carried with it. It starts 20 au across and "
        "leaves at 30 km/s. A real nebula grows thousands of times wider; this one is sized "
        "to play in a star's own System. Drawn as particles that only feel gravity: no "
        "pressure, no heating and no glow.",
        unit_preset="STELLAR", role="CLOUD", headroom="ball",
        size_m=10.0 * AU_M, mass_kg=0.2 * MSUN_KG, count=3000, shape="SHELL",
        velocity_model="EXPANSION", orbit=ORBIT_REQUIRED, centred=True,
        inner_m=9.0 * AU_M, expansion_speed_m_s=30.0 * KM_M,
    ),
)

BY_ID: dict[str, AddPreset] = {preset.preset_id: preset for preset in CATALOGUE}

NOT_BUILT: tuple[str, ...] = ()
"""Recorded in ADR 0164 as later; deliberately given no catalogue row and no menu entry.
The Planetary nebula and the Shearing sheet left this list with ADR 0202: the first on a Shell
shape and an Expansion flow, the second on a real shearing-box frame in the Direct solver.
Custom presets left it with ADR 0203: they are the ``Custom`` submenu in ``custom_presets`` --
files an artist saved, not catalogue rows -- so ``CATEGORIES`` stays eight."""


def presets_in(category: str) -> tuple[AddPreset, ...]:
    return tuple(preset for preset in CATALOGUE if preset.category == category)


def preset(preset_id: str) -> AddPreset:
    try:
        return BY_ID[preset_id]
    except KeyError:
        raise AddRefused(f"'{preset_id}' is not an object this add-on can add") from None


def default_count(preset_id: str) -> int:
    return preset(preset_id).count


# -- the plan -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Central:
    """The Body something orbits, in the target System's units."""

    name: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    mass: float
    radius: float


@dataclass(frozen=True)
class ObjectPlan:
    preset_id: str
    name: str
    unit_preset: str
    role: str
    location: tuple[float, float, float]
    proxy: str  # "SPHERE" | "ANNULUS" | "BOX"
    proxy_radius: float
    proxy_inner: float
    proxy_thickness: float
    body: dict | None
    cloud: dict | None
    central_name: str | None
    new_system_pacing: dict
    central_relation: str = "orbiting"
    """How the item relates to its central Body, for the INFO line: "orbiting" or "centred on"."""
    proxy_dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """For ``proxy == "BOX"``: the box's width (X), length (Y) and height (Z)."""


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def circular_velocity(g: float, central_mass: float, own_mass: float, offset) -> tuple:
    """The velocity of a circular orbit about the central Body, perpendicular to ``offset`` in
    the XY plane, counter-clockwise seen from +Z. Zero when the offset is zero."""

    distance = _norm(offset)
    if distance <= 0.0:
        return (0.0, 0.0, 0.0)
    speed = math.sqrt(g * (central_mass + own_mass) / distance)
    horizontal = math.hypot(offset[0], offset[1])
    if horizontal <= 1e-12 * distance:
        return (speed, 0.0, 0.0)
    return (-offset[1] / horizontal * speed, offset[0] / horizontal * speed, 0.0)


def plan_object(
    preset_id: str,
    *,
    cursor,
    central: Central | None = None,
    count: int = 0,
    size_scale: float = 1.0,
    mass_scale: float = 1.0,
    counter_clockwise: bool = True,
) -> ObjectPlan:
    """Everything the operator writes for one added object, in the preset's System units."""

    spec = preset(preset_id)
    if not (size_scale > 0.0 and math.isfinite(size_scale)):
        raise AddRefused("Size must be above zero")
    if not (mass_scale > 0.0 and math.isfinite(mass_scale)):
        raise AddRefused("Mass must be above zero")
    if spec.orbit == ORBIT_REQUIRED and central is None:
        if spec.centred:
            raise AddRefused(f"{spec.label}: select the star it should be centred on first")
        raise AddRefused(
            f"{spec.label}: select the planet, star or black hole it should orbit first"
        )
    units = resolve_unit_scale(spec.unit_preset)
    length = units.length_scale_m
    mass_unit = units.mass_scale_kg
    g = units.gravitational_constant
    mass = max(spec.mass_kg / mass_unit * mass_scale, MIN_BODY_MASS)
    size = spec.size_m / length * size_scale
    location = tuple(float(v) for v in cursor)
    base_velocity = (0.0, 0.0, 0.0)
    orbiting = central is not None and spec.orbit != ORBIT_NONE
    if orbiting:
        base_velocity = central.velocity
        if spec.ring_radii[1] > 0.0 or spec.shape == "DISK_ANNULUS" or spec.centred:
            # a ring or disk turns about it, not near it; a centred shell is round it
            location = central.position
        else:
            offset = _sub(location, central.position)
            if _norm(offset) <= max(central.radius, 1e-12):
                offset = (spec.default_distance_m / length, 0.0, 0.0)
                location = _add(central.position, offset)
    pacing = dict(NEW_SYSTEM_PACING[spec.unit_preset])
    if spec.substeps:
        pacing["substeps"] = spec.substeps
    if spec.softening_fraction > 0.0:
        pacing["softening"] = spec.softening_fraction * size
    if spec.role == "BODY":
        velocity = base_velocity
        if orbiting:
            offset = _sub(location, central.position)
            orbital = circular_velocity(g, central.mass, mass, offset)
            orbital = tuple(v * spec.speed_factor for v in orbital)
            if not counter_clockwise:
                orbital = tuple(-v for v in orbital)
            velocity = _add(base_velocity, orbital)
        distance = _norm(_sub(location, central.position)) if orbiting else 0.0
        handle = handle_radius(size, spec.size_m * size_scale, distance)
        return ObjectPlan(
            preset_id=spec.preset_id, name=spec.label, unit_preset=spec.unit_preset,
            role="BODY", location=location, proxy="SPHERE", proxy_radius=handle,
            proxy_inner=0.0, proxy_thickness=0.0,
            body={"mass": mass, "radius": size, "initial_velocity": velocity,
                  "pinned": spec.pinned},
            cloud=None, central_name=central.name if orbiting else None,
            new_system_pacing=pacing,
        )
    particles = count if count > 0 else spec.count
    cloud = {
        "shape": spec.shape,
        "count": particles,
        "total_mass": mass,
        "velocity_model": spec.velocity_model,
        "support_fraction": spec.support_fraction,
        "annulus_inner_radius": 0.0,
        "annulus_surface_density_exponent": spec.centre_concentration,
        "bulk_velocity": (0.0, 0.0, 0.0),
        "keplerian_counter_clockwise": counter_clockwise,
        "keplerian_central": None,
    }
    if spec.shape == "DISK_ANNULUS":
        if spec.ring_radii[1] > 0.0:
            if central.radius <= 0.0:
                raise AddRefused(
                    f"{spec.label}: '{central.name}' has no Radius to size the ring from; "
                    "give it one first"
                )
            inner = spec.ring_radii[0] * central.radius * size_scale
            outer = spec.ring_radii[1] * central.radius * size_scale
            thickness = 0.02 * outer
        else:
            inner = spec.inner_m / length * size_scale
            outer = size
            thickness = spec.thickness_m / length * size_scale
        cloud["annulus_inner_radius"] = inner
        if spec.velocity_model == "KEPLERIAN":
            cloud["keplerian_central"] = central.name
        return ObjectPlan(
            preset_id=spec.preset_id, name=spec.label, unit_preset=spec.unit_preset,
            role="CLOUD", location=location, proxy="ANNULUS", proxy_radius=outer,
            proxy_inner=inner, proxy_thickness=thickness, body=None, cloud=cloud,
            central_name=central.name if orbiting else None, new_system_pacing=pacing,
        )
    if spec.shearing:
        # A local shearing box (ADR 0202): the object is the box, centred on the world origin the
        # frame is centred on, and the System it joins is made for it with the option on and the
        # angular speed of a patch `shear_radius_m` from a body of `shear_mass_kg`.
        width, span, height = (value / length * size_scale for value in spec.box_m)
        omega_si = math.sqrt(G_SI * spec.shear_mass_kg / spec.shear_radius_m**3)
        pacing["shearing_box"] = {
            "omega": omega_si * units.time_scale_s,
            "box_x": width,
            "box_y": span,
        }
        return ObjectPlan(
            preset_id=spec.preset_id, name=spec.label, unit_preset=spec.unit_preset,
            role="CLOUD", location=(0.0, 0.0, 0.0), proxy="BOX", proxy_radius=width,
            proxy_inner=0.0, proxy_thickness=height, body=None, cloud=cloud,
            central_name=None, new_system_pacing=pacing,
            proxy_dimensions=(width, span, height),
        )
    if spec.shape == "SHELL":
        # The hole in the middle, as the Inner Radius a ring has (ADR 0202); the redo-panel Size
        # scales it with the shell so a bigger shell stays as thin.
        cloud["annulus_inner_radius"] = spec.inner_m / length * size_scale
    if spec.velocity_model == "EXPANSION":
        cloud["expansion_speed"] = units.speed_to_code(spec.expansion_speed_m_s)
        cloud["expansion_proportional"] = spec.expansion_proportional
    if spec.centred and orbiting:
        # Round the central Body and carried with it: its own motion and nothing else.
        cloud["bulk_velocity"] = base_velocity
    elif orbiting:
        offset = _sub(location, central.position)
        orbital = circular_velocity(g, central.mass, mass, offset)
        if not counter_clockwise:
            orbital = tuple(-v for v in orbital)
        cloud["bulk_velocity"] = _add(base_velocity, orbital)
    return ObjectPlan(
        preset_id=spec.preset_id, name=spec.label, unit_preset=spec.unit_preset,
        role="CLOUD", location=location, proxy="SPHERE", proxy_radius=size,
        proxy_inner=0.0, proxy_thickness=0.0, body=None, cloud=cloud,
        central_name=central.name if orbiting else None, new_system_pacing=pacing,
        central_relation="centred on" if spec.centred else "orbiting",
    )
