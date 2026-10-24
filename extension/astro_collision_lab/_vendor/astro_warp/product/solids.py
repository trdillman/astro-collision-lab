"""Solid constitutive pipeline: strain rate, stress rate, plasticity, damage, porosity, force.

This is stages S1-S6 of `docs/source-audit/SPEC/07_IMPACT_SOLID_MATERIAL_PHYSICS.md` and
carries the module half of SOLID-001. `materials.py` owns the capability model and the
condensed-matter closures; this module owns the state each capability evolves and the Warp
kernels that evolve it. The five stages are separate launches with separate host oracles
because SOLID-001 asks for separate modules: a material with strength but no damage must
compile the stress rate and the plasticity return and nothing else, and the graph modules
`MaterialDescriptor.graph_modules` derives are exactly the launches recorded here.

The boundary COLL-002 asks for is structural rather than defensive. What this module takes
from `nbody.py` is `Precision` and the lane dtype helper - the shared numeric plumbing every
backend uses - and nothing else; the continuum force below resolves contact through the SPH
pressure and stress gradients of overlapping bodies. A resolved SPH body therefore has no
path to the point-particle collision module at all.
`tests/unit/test_solid_material_contracts.py` gates that by import graph, pinning the
`nbody` surface to exactly those two names, which is a stronger statement than a runtime
check because it cannot be satisfied by a branch that happens never to be taken.

Every gate the validation suite applies to these kernels is a round-off bound or a scaling
law with a measured constant, never a chosen tolerance:

- The strain rate uses the inverse-consistency correction, so a linear velocity field is
  reproduced exactly and the gate is round-off rather than a resolution study.
- The Jaumann rate is objective in the continuum but is integrated by explicit Euler, which
  under rigid rotation drifts at first order in the step. The gate is that first-order law
  with a measured constant, not a magnitude.
- The von Mises return, the Drucker-Prager surface, the P-alpha crush curve and the Weibull
  flaw thresholds are closed-form algebra, so the device has to reproduce the host oracle to
  round-off.
- The damage growth ODE has a closed-form solution at fixed activation, so the integrator is
  gated against the exact solution rather than against a finer step of itself.
- The pair force is antisymmetric by construction, so total momentum is conserved to
  round-off - the same argument the hydrodynamic force gate rests on.

`adr/0022-solid-constitutive-validation-thresholds.md` records each derivation.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from typing import Any

from .materials import (
    EOS_ID_MURNAGHAN,
    EOS_ID_TILLOTSON,
    DamageModel,
    MaterialDescriptor,
    MaterialRegistry,
    PorosityModel,
    StrengthModel,
)
from .nbody import Precision, _dtypes
from .runtime_foundation import DeviceRequirementError
from .sph import KERNEL_REGISTRY, NeighborRuntime, SphKernel

# Device selectors. Model choice is a runtime integer for the same reason the smoothing
# kernel is one in `hydro.py`: it is uniform across a launch, so it costs a predicted branch,
# and specialising the Warp module on it would multiply the compile count by the size of the
# capability lattice for no measurable gain.
STRENGTH_ID_NONE = 0
STRENGTH_ID_VON_MISES = 1
STRENGTH_ID_DRUCKER_PRAGER = 2

DAMAGE_ID_NONE = 0
DAMAGE_ID_GRADY_KIPP = 1

POROSITY_ID_NONE = 0
POROSITY_ID_P_ALPHA = 1

_STRENGTH_IDS: dict[StrengthModel, int] = {
    StrengthModel.NONE: STRENGTH_ID_NONE,
    StrengthModel.VON_MISES: STRENGTH_ID_VON_MISES,
    StrengthModel.DRUCKER_PRAGER: STRENGTH_ID_DRUCKER_PRAGER,
}
_DAMAGE_IDS: dict[DamageModel, int] = {
    DamageModel.NONE: DAMAGE_ID_NONE,
    DamageModel.GRADY_KIPP: DAMAGE_ID_GRADY_KIPP,
}
_POROSITY_IDS: dict[PorosityModel, int] = {
    PorosityModel.NONE: POROSITY_ID_NONE,
    PorosityModel.P_ALPHA: POROSITY_ID_P_ALPHA,
}

# Flat device layout of the material parameter tables. One row per material, fixed slots, so
# a kernel reads a parameter by a compile-time index rather than by a name lookup. The slot
# names are the schema keys in `materials.py`; `_material_tables` is the only writer and the
# contract test asserts every schema key has a slot.
EOS_SLOT_REFERENCE_DENSITY = 0
EOS_SLOT_BULK_MODULUS = 1
EOS_SLOT_EXPONENT = 2
EOS_SLOT_A = 3
EOS_SLOT_B = 4
EOS_SLOT_BULK_MODULUS_A = 5
EOS_SLOT_BULK_MODULUS_B = 6
EOS_SLOT_REFERENCE_ENERGY = 7
EOS_SLOT_ENERGY_IV = 8
EOS_SLOT_ENERGY_CV = 9
EOS_SLOT_ALPHA = 10
EOS_SLOT_BETA = 11
EOS_SLOTS = 12

STRENGTH_SLOT_YIELD_STRESS = 0
STRENGTH_SLOT_COHESION = 1
STRENGTH_SLOT_FRICTION = 2
STRENGTH_SLOTS = 3

DAMAGE_SLOT_WEIBULL_K = 0
DAMAGE_SLOT_WEIBULL_M = 1
DAMAGE_SLOT_CRACK_SPEED = 2
DAMAGE_SLOT_YOUNG_MODULUS = 3
DAMAGE_SLOTS = 4

POROSITY_SLOT_INITIAL_DISTENTION = 0
POROSITY_SLOT_ELASTIC_PRESSURE = 1
POROSITY_SLOT_SOLID_PRESSURE = 2
POROSITY_SLOT_CRUSH_EXPONENT = 3
POROSITY_SLOTS = 4

_EOS_SLOT_NAMES: dict[str, int] = {
    "reference_density": EOS_SLOT_REFERENCE_DENSITY,
    "bulk_modulus": EOS_SLOT_BULK_MODULUS,
    "exponent": EOS_SLOT_EXPONENT,
    "a": EOS_SLOT_A,
    "b": EOS_SLOT_B,
    "bulk_modulus_a": EOS_SLOT_BULK_MODULUS_A,
    "bulk_modulus_b": EOS_SLOT_BULK_MODULUS_B,
    "reference_energy": EOS_SLOT_REFERENCE_ENERGY,
    "incipient_vaporization_energy": EOS_SLOT_ENERGY_IV,
    "complete_vaporization_energy": EOS_SLOT_ENERGY_CV,
    "alpha": EOS_SLOT_ALPHA,
    "beta": EOS_SLOT_BETA,
}
_STRENGTH_SLOT_NAMES: dict[str, int] = {
    "yield_stress": STRENGTH_SLOT_YIELD_STRESS,
    "cohesion": STRENGTH_SLOT_COHESION,
    "friction_coefficient": STRENGTH_SLOT_FRICTION,
}
_DAMAGE_SLOT_NAMES: dict[str, int] = {
    "weibull_k": DAMAGE_SLOT_WEIBULL_K,
    "weibull_m": DAMAGE_SLOT_WEIBULL_M,
    "crack_growth_speed": DAMAGE_SLOT_CRACK_SPEED,
    "young_modulus": DAMAGE_SLOT_YOUNG_MODULUS,
}
_POROSITY_SLOT_NAMES: dict[str, int] = {
    "initial_distention": POROSITY_SLOT_INITIAL_DISTENTION,
    "elastic_pressure": POROSITY_SLOT_ELASTIC_PRESSURE,
    "solid_pressure": POROSITY_SLOT_SOLID_PRESSURE,
    "crush_exponent": POROSITY_SLOT_CRUSH_EXPONENT,
}

MODULE_STRESS_RATE = "astro.solids.stress_rate"
MODULE_PLASTICITY = "astro.solids.plasticity"
MODULE_DAMAGE = "astro.solids.damage"
MODULE_POROSITY = "astro.solids.porosity"
MODULE_CONDENSED_EOS = "astro.solids.condensed_eos"
MODULE_STRESS_FORCE = "astro.solids.stress_force"

# Slot 1 of the two-element timestep array `hydro.py` publishes: [0] is the Courant candidate
# the next evaluation would take, [1] the step the current one is integrating. The rate stages
# want the step actually being taken, so they read slot 1 - and they read it on the device,
# because the coupled path records them into a graph that is replayed with a different step
# every time. Standalone, `SolidRuntime` owns a two-slot array of its own with the same layout.
TIMESTEP_SLOT_COMMITTED = 1

# Every field a coupled solid stage reads that the hydrodynamic stage already owns.
# `bind_continuum` swaps exactly these and nothing else: the stress, damage, distention,
# velocity gradient, flaw tables and `accelerations` stay this runtime's own, because no
# other stage produces them and S6 assigns rather than accumulates into the last.
CONTINUUM_ARRAYS = (
    "velocities",
    "masses",
    "densities",
    "omegas",
    "internal_energies",
    "pressures",
    "sound_speeds",
    "reach",
    "timesteps",
)

# The kernel selector `hydro.py` publishes. Duplicating the integers would let the two
# modules disagree silently, so they are imported at call time instead; see `_kernel_id`.

_ONE_THIRD = 1.0 / 3.0


# ---------------------------------------------------------------------------------------
# Host oracles. Every one of these is the float64 statement of what a kernel below computes.
# The validation suite compares the device against these, and these against closed-form
# solutions, so a transcription error has to survive two independent statements to escape.
# ---------------------------------------------------------------------------------------


def symmetric_from_components(diagonal: tuple[float, float, float],
                              offdiagonal: tuple[float, float, float]
                              ) -> list[list[float]]:
    """Expand the six stored components into the full symmetric tensor.

    Storage is six components because that is what SPEC 07 specifies and because the three
    redundant entries are the ones an asymmetric update would corrupt without being caught.
    """

    xx, yy, zz = diagonal
    xy, xz, yz = offdiagonal
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


def components_from_symmetric(tensor: list[list[float]]) -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
    return (
        (tensor[0][0], tensor[1][1], tensor[2][2]),
        (tensor[0][1], tensor[0][2], tensor[1][2]),
    )


def second_invariant(tensor: list[list[float]]) -> float:
    """``J2 = 0.5 * S:S`` for a deviatoric tensor."""

    total = 0.0
    for row in range(3):
        for column in range(3):
            total += tensor[row][column] * tensor[row][column]
    return 0.5 * total


def max_principal_deviatoric(tensor: list[list[float]]) -> float:
    """Largest eigenvalue of a traceless symmetric tensor, in closed form.

    A traceless symmetric 3x3 has characteristic polynomial ``l**3 - J2 l - J3 = 0``, whose
    roots are ``2 sqrt(J2/3) cos(theta - 2 pi k / 3)`` with
    ``theta = arccos(1.5 sqrt(3) J3 / J2**1.5) / 3``. The ``k = 0`` root is the largest, so
    the tensile activation criterion needs no iterative eigensolver and no tolerance.
    """

    j2 = second_invariant(tensor)
    if j2 <= 0.0:
        return 0.0
    determinant = (
        tensor[0][0] * (tensor[1][1] * tensor[2][2] - tensor[1][2] * tensor[2][1])
        - tensor[0][1] * (tensor[1][0] * tensor[2][2] - tensor[1][2] * tensor[2][0])
        + tensor[0][2] * (tensor[1][0] * tensor[2][1] - tensor[1][1] * tensor[2][0])
    )
    argument = 1.5 * math.sqrt(3.0) * determinant / (j2 * math.sqrt(j2))
    argument = min(1.0, max(-1.0, argument))
    return 2.0 * math.sqrt(j2 * _ONE_THIRD) * math.cos(math.acos(argument) * _ONE_THIRD)


def jaumann_stress_rate(
    stress: list[list[float]], gradient: list[list[float]], shear_modulus: float
) -> list[list[float]]:
    """``dS/dt = 2 mu dev(strain_rate) + Omega S - S Omega``.

    ``gradient[a][b]`` is ``d v_a / d x_b``. The rotation term is the commutator with the
    spin tensor, which is what makes the rate objective: a stress carried by a rigidly
    rotating body rotates with it instead of accumulating an artificial elastic response.
    """

    strain = [[0.0] * 3 for _ in range(3)]
    spin = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            strain[row][column] = 0.5 * (gradient[row][column] + gradient[column][row])
            spin[row][column] = 0.5 * (gradient[row][column] - gradient[column][row])
    trace = strain[0][0] + strain[1][1] + strain[2][2]
    rate = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            deviatoric = strain[row][column] - (trace * _ONE_THIRD if row == column else 0.0)
            rotation = 0.0
            for index in range(3):
                rotation += spin[row][index] * stress[index][column]
                rotation -= stress[row][index] * spin[index][column]
            rate[row][column] = 2.0 * shear_modulus * deviatoric + rotation
    return rate


def yield_strength(
    strength: StrengthModel, pressure: float, parameters: dict[str, float]
) -> float:
    """The yield surface at this pressure.

    von Mises is pressure independent. Drucker-Prager is the pressure-dependent surface a
    granular or fractured material needs, clamped below at zero - a material in tension has
    no cohesive shear strength once the friction term has eaten the cohesion - and above at
    the intact yield stress, which is the ceiling the surface saturates at.
    """

    if strength is StrengthModel.NONE:
        return math.inf
    if strength is StrengthModel.VON_MISES:
        return float(parameters["yield_stress"])
    surface = parameters["cohesion"] + parameters["friction_coefficient"] * pressure
    return min(max(surface, 0.0), float(parameters["yield_stress"]))


def radial_return_factor(stress: list[list[float]], limit: float) -> float:
    """The von Mises scale factor ``min(1, Y / sqrt(3 J2))``.

    Scaling the deviator leaves its direction and the pressure untouched, which is why the
    return is radial: plastic flow is incompressible, so the correction cannot touch the
    trace.
    """

    if limit >= math.inf:
        return 1.0
    j2 = second_invariant(stress)
    if j2 <= 0.0:
        return 1.0
    equivalent = math.sqrt(3.0 * j2)
    if equivalent <= limit:
        return 1.0
    return limit / equivalent


def weibull_threshold(flaw_index: int, flaw_density: float, weibull_modulus: float) -> float:
    """Activation strain of the ``flaw_index``-th flaw of a Weibull population.

    Inverting ``n(eps) = k V eps**m`` at the integers gives ``eps_i = ((i+1)/(k V))**(1/m)``.
    Assigning thresholds this way rather than by sampling makes the population exactly
    Weibull at every particle count, so the resolution dependence SPEC 07 asks to be
    documented is the flaw count per particle and nothing else.
    """

    if flaw_density <= 0.0 or weibull_modulus <= 0.0:
        return math.inf
    return ((flaw_index + 1) / flaw_density) ** (1.0 / weibull_modulus)


def activated_flaw_count(
    strain: float,
    flaw_density: float,
    weibull_modulus: float,
    flaw_offset: int,
    flaw_stride: int,
    flaw_count: int,
) -> int:
    """How many of this particle's flaws have activated at ``strain``.

    Flaws are dealt round-robin, so particle ``r`` of a material with ``n`` particles holds
    global flaws ``r, r+n, r+2n, ...``. Counting the activated ones is then a closed-form
    inequality rather than a scan over a stored threshold list, which is why a particle costs
    two integers instead of a fixed-size flaw array.
    """

    if flaw_count <= 0 or flaw_stride <= 0 or strain <= 0.0:
        return 0
    if flaw_density <= 0.0 or weibull_modulus <= 0.0:
        return 0
    # The same form S4 evaluates (ADR 0173): `k V eps**m` as `(eps / eps_1)**m`.
    reached = (strain / weibull_threshold(0, flaw_density, weibull_modulus)) ** weibull_modulus
    highest = math.floor((reached - flaw_offset - 1.0) / flaw_stride)
    return int(min(max(highest + 1.0, 0.0), float(flaw_count)))


def damage_growth(
    damage: float,
    activated: int,
    flaw_count: int,
    crack_growth_speed: float,
    particle_radius: float,
    timestep: float,
) -> float:
    """One explicit step of ``d(D**(1/3))/dt = activated**(1/3) c_g / R``.

    The cube-root form is the Grady-Kipp crack volume: ``activated`` cracks each growing a
    sphere of radius ``c_g t`` fill a fraction ``D`` of the particle volume. It is linear in
    ``D**(1/3)`` at fixed activation, so the closed-form solution the validation suite gates
    against is exact rather than a finer-step reference. The local ceiling is the fraction of
    this particle's flaws that have activated: a single activated flaw cannot fully damage a
    particle that carries many.

    Damage is irreversible - a Grady-Kipp crack does not heal when its particle unloads - so
    the result is the running max against the incoming ``damage``, the host mirror of the
    device kernel's own running-max latch.
    """

    if activated <= 0 or particle_radius <= 0.0 or flaw_count <= 0:
        return damage
    root = damage ** _ONE_THIRD + (activated**_ONE_THIRD) * crack_growth_speed * timestep / (
        particle_radius
    )
    return max(damage, min(root**3, min(1.0, activated / flaw_count)))


def crush_curve_distention(
    pressure: float,
    initial_distention: float,
    elastic_pressure: float,
    solid_pressure: float,
    crush_exponent: float,
) -> float:
    """Quasi-static P-alpha crush curve.

    ``alpha`` falls from ``alpha_0`` at the elastic limit to one at the solid pressure and is
    monotone non-increasing in pressure by construction, which is the property that makes
    compaction irreversible when the update takes the running minimum.
    """

    if pressure <= elastic_pressure:
        return initial_distention
    if pressure >= solid_pressure:
        return 1.0
    span = (solid_pressure - pressure) / (solid_pressure - elastic_pressure)
    return 1.0 + (initial_distention - 1.0) * span**crush_exponent


def porous_pressure(
    solid_pressure_value: float, distention: float
) -> float:
    """``P_porous = P_solid(alpha rho, E) / alpha``.

    The solid closure is evaluated at the matrix density ``alpha rho`` because that is the
    density of the material between the pores; dividing by ``alpha`` converts the matrix
    stress back to the stress carried across the porous cross-section.
    """

    return solid_pressure_value / distention


def particle_radius(mass: float, density: float) -> float:
    """Radius of the sphere of this particle's own volume, ``(3 m / 4 pi rho)**(1/3)``."""

    if density <= 0.0:
        return 0.0
    return (3.0 * mass / (4.0 * math.pi * density)) ** _ONE_THIRD


def damaged_stress(
    pressure: float, stress: list[list[float]], damage: float
) -> list[list[float]]:
    """The stress a damaged particle actually carries.

    Damage removes tensile and deviatoric capacity and leaves compressive capacity intact:
    a fully fragmented rock still resists compression, which is why only a negative pressure
    is scaled. Without the asymmetry a damaged body would pass through itself.
    """

    scale = 1.0 - damage
    carried = -pressure * scale if pressure < 0.0 else -pressure
    total = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            total[row][column] = scale * stress[row][column]
            if row == column:
                total[row][column] += carried
    return total


# ---------------------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------------------


def _matrix_type(wp: Any, precision: Precision) -> Any:
    return wp.mat33d if precision == Precision.FLOAT64 else wp.mat33f


def _kernel_id(kernel: SphKernel) -> int:
    from .hydro import sph_kernel_ids

    return sph_kernel_ids()[kernel]


def _solid_registrars(wp: Any, scalar: Any, vector: Any, matrix: Any) -> tuple[Any, Any]:
    """Resolve stringized annotations against the solid type namespace.

    `nbody._registrars` resolves against ``wp``, ``scalar`` and ``vector`` only, and a
    constitutive kernel has to name a tensor type in its signature. Rather than widen the
    shared helper - which every existing kernel module depends on and which the hydrodynamic
    session is currently editing - the same three-line resolver is restated here with
    ``matrix`` added. The mechanism is identical: `from __future__ import annotations` makes
    every annotation a string, and Warp needs the evaluated type.
    """

    namespace = {"wp": wp, "scalar": scalar, "vector": vector, "matrix": matrix}

    def resolve(func: Any) -> Any:
        func.__annotations__ = inspect.get_annotations(func, eval_str=True, locals=namespace)
        return func

    def register_kernel(func: Any = None, **options: Any) -> Any:
        if func is None:
            return lambda inner: wp.kernel(resolve(inner), **options)
        return wp.kernel(resolve(func), **options)

    def register_func(func: Any) -> Any:
        return wp.func(resolve(func))

    return register_kernel, register_func


def _solid_kernels(wp: Any, precision: Precision = Precision.FLOAT32):
    """Compile the constitutive kernels for one state precision.

    The smoothing-kernel shape derivative is restated here rather than imported from
    `hydro.py`: a Warp function is bound to the module that registered it, and the
    hydrodynamic factory does not publish its internal functions. That duplication is a
    known, ungated drift risk: `shape_derivative` below is an inner `@warp_func` with no
    host entry point, so nothing currently compares it against `sph.KernelSpec.gradient`,
    the host statement both device copies answer to. The solid suite's gates are all
    round-off or scaling statements that a wrong-but-smooth shape would still satisfy. The
    proof records this in `limitations`; closing it means a device probe section, not a unit
    test.
    """

    scalar, vector, _ = _dtypes(wp, precision)
    matrix = _matrix_type(wp, precision)
    warp_kernel, warp_func = _solid_registrars(wp, scalar, vector, matrix)
    zero = scalar(0.0)
    half = scalar(0.5)
    one = scalar(1.0)
    two = scalar(2.0)
    three = scalar(3.0)
    third = scalar(_ONE_THIRD)
    # Warp types a float literal written inside a kernel body as float32 and only then casts
    # it to the lane's scalar, so a constant that is not an exact binary fraction arrives with
    # float32 error - about 3e-8 relative - even in the float64 lane. Constants built out here
    # are already lane-typed Python objects and reach the kernel exactly. Every non-dyadic
    # constant this module needs is therefore named here and referenced by name below.
    four_thirds = scalar(4.0 / 3.0)
    negative_seven_thirds = scalar(-7.0 / 3.0)
    four_pi = scalar(4.0 * math.pi)
    # cbrt(eps) is the central-difference spacing that balances truncation against round-off.
    difference_spacing = scalar(2.220446049250313e-16**_ONE_THIRD)
    # sqrt(float32 eps): the standard reciprocal-condition-number floor. Compared against a
    # determinant normalized by the moment matrix's own scale (see `compute_velocity_gradient`),
    # not against the raw determinant, so the test tracks conditioning rather than magnitude.
    moment_condition_floor = scalar(3.4526698e-4)

    from .hydro import KERNEL_ID_CUBIC_SPLINE, KERNEL_ID_WENDLAND_C2

    @warp_func
    def shape_derivative(kernel_id: int, q: scalar):
        t = one - half * q
        u = two - q
        shape = negative_seven_thirds * q * t * t * t * t * t * (two + scalar(5.0) * q)
        if kernel_id == KERNEL_ID_CUBIC_SPLINE:
            if q < one:  # noqa: SIM108
                shape = scalar(-3.0) * q + scalar(2.25) * q * q
            else:
                shape = scalar(-0.75) * u * u
        if kernel_id == KERNEL_ID_WENDLAND_C2:
            shape = scalar(-5.0) * q * t * t * t
        return shape

    @warp_func
    def kernel_gradient(
        kernel_id: int, normalization: scalar, support_ratio: scalar, r: scalar, h: scalar
    ):
        """``dW/dr``, zero outside the support and at the origin."""

        if h <= zero or r <= zero:
            return zero
        q = r / h
        if q >= support_ratio:
            return zero
        h2 = h * h
        return normalization / (h2 * h2) * shape_derivative(kernel_id, q)

    @warp_func
    def stress_tensor(diagonal: vector, offdiagonal: vector):
        return matrix(
            diagonal[0],
            offdiagonal[0],
            offdiagonal[1],
            offdiagonal[0],
            diagonal[1],
            offdiagonal[2],
            offdiagonal[1],
            offdiagonal[2],
            diagonal[2],
        )

    @warp_func
    def invariant_j2(stress: matrix):
        total = zero
        for row in range(3):
            for column in range(3):
                total = total + stress[row, column] * stress[row, column]
        return half * total

    @warp_func
    def principal_maximum(stress: matrix):
        """Largest eigenvalue of the traceless symmetric ``stress``; see the host oracle."""

        j2 = invariant_j2(stress)
        if j2 <= zero:
            return zero
        determinant = wp.determinant(stress)
        argument = scalar(1.5) * wp.sqrt(three) * determinant / (j2 * wp.sqrt(j2))
        argument = wp.min(one, wp.max(-one, argument))
        return two * wp.sqrt(j2 * third) * wp.cos(wp.acos(argument) * third)

    @warp_kernel
    def seed_reach(
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        support_ratio: scalar,
    ):
        i = wp.tid()
        reach[i] = support_ratio * smoothing_lengths[i]

    @warp_kernel
    def scatter_reach(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        support_ratio: scalar,
    ):
        """Raise every neighbour's query radius to this particle's own support.

        Same two-pass construction the hydrodynamic stage uses, and for the same reason: a
        pair interacts out to ``support_ratio * max(h_i, h_j)``, and the small-support
        particle cannot discover the large-support one by searching, because the partner
        sits beyond the radius it would search with. The large one announces itself instead.
        ``wp.atomic_max`` keeps the result exactly order independent, so the pair set does
        not depend on how the launch interleaves.
        """

        i = wp.tid()
        p = positions[i]
        span = support_ratio * smoothing_lengths[i]
        query = wp.hash_grid_query(grid, p, span)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j != i and wp.length(p - positions[j]) <= span:
                wp.atomic_max(reach, j, span)

    @warp_kernel
    def compute_velocity_gradient(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        number_density: int,
        correct: int,
        gradients: wp.array(dtype=matrix),
        correction_failures: wp.array(dtype=wp.int32),
    ):
        """S1: the SPH velocity gradient, optionally inverse-consistency corrected.

        ``number_density`` weights every pair by the particle's own volume ``m_i/rho_i``
        rather than the neighbour's ``m_j/rho_j`` (ADR 0167). That is the gradient the
        number-density divergence in `compute_stress_forces` does its work against, so the
        elastic energy S2 stores is the work S6 takes out of the motion, pair by pair, even
        where neighbours differ in mass. The correction below divides the weight back out, so
        the corrected gradient differs only through the neighbours' volumes.

        The raw estimator ``L = sum_j (m_j/rho_j) (v_j - v_i) grad_i W`` reproduces a linear
        velocity field only up to the discretisation error of the neighbourhood. Multiplying
        by the inverse of ``B = sum_j (m_j/rho_j) (x_j - x_i) grad_i W`` makes it exact for a
        linear field at any particle arrangement, which turns the strain-rate gate into a
        round-off bound and removes the free-surface bias from the rotation test.

        The correction is a versioned solver choice, not a silent default: a neighbourhood
        too degenerate to invert falls back to the raw estimator and increments a counter, so
        a run cannot quietly change formulation particle by particle without saying so.
        """

        i = wp.tid()
        h_i = smoothing_lengths[i]
        if h_i <= zero:
            return
        p = positions[i]
        v_i = velocities[i]
        radius = reach[i]
        own = zero
        if densities[i] > zero:
            own = masses[i] / densities[i]
        raw = matrix(zero)
        moment = matrix(zero)
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j == i:
                continue
            h_j = smoothing_lengths[j]
            if h_j <= zero:
                continue
            rho_j = densities[j]
            if rho_j <= zero:
                continue
            offset = p - positions[j]
            r = wp.length(offset)
            if r <= zero:
                continue
            # Symmetric reach: the pair is admitted from both sides or from neither, which is
            # the condition that keeps the corrected gradient a property of the pair set
            # rather than of the traversal order.
            span = support_ratio * wp.max(h_i, h_j)
            if r >= wp.min(span, wp.min(radius, reach[j])):
                continue
            derivative = kernel_gradient(kernel_id, normalization, support_ratio, r, h_i)
            if derivative == zero:
                continue
            grad = offset * (derivative / r)
            weight = masses[j] / rho_j
            if number_density != 0:
                weight = own
            raw = raw + wp.outer(velocities[j] - v_i, grad) * weight
            moment = moment + wp.outer(positions[j] - p, grad) * weight
        if correct != 0:
            # Singular moment matrices are real: a particle with fewer than three
            # non-coplanar neighbours has no linearly complete neighbourhood to correct
            # against. Falling back to the raw estimator is the only honest answer, and the
            # counter is what stops it being a silent one.
            #
            # A determinant that merely clears zero can still belong to a matrix too
            # ill-conditioned to invert usefully: a neighbourhood ten smoothing lengths
            # across and one barely-independent direction produce determinants many orders
            # apart, so an absolute floor would call one singular and wave the other through
            # for the same underlying degeneracy. Normalizing by the matrix's own scale -
            # its squared Frobenius norm, which `invariant_j2` already computes generically -
            # makes the test track conditioning instead of raw magnitude, the same reasoning
            # `principal_maximum` above uses to clamp its determinant-derived argument.
            frobenius_squared = two * invariant_j2(moment)
            scale_cubed = frobenius_squared * wp.sqrt(frobenius_squared)
            if wp.abs(wp.determinant(moment)) > moment_condition_floor * scale_cubed:
                gradients[i] = raw * wp.inverse(moment)
                return
            wp.atomic_add(correction_failures, 0, 1)
        gradients[i] = raw

    @warp_kernel
    def advance_deviatoric_stress(
        gradients: wp.array(dtype=matrix),
        material_index: wp.array(dtype=wp.int32),
        shear_moduli: wp.array(dtype=scalar),
        stress_diagonal: wp.array(dtype=vector),
        stress_offdiagonal: wp.array(dtype=vector),
        timesteps: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        damage: wp.array(dtype=scalar),
        heated_energies: wp.array(dtype=scalar),
    ):
        """S2: one explicit step of the Jaumann objective deviatoric stress rate.

        The step arrives as a device array rather than a launch parameter because the
        coupled path records this into a replayed graph, where a host float would be frozen
        at capture time and every later step would integrate the step the first one took.

        The elastic store is `V (1 - D) S:S / 4 mu`. The stress force pays for the `S:dS`
        part (ADR 0167); the store also changes with the volume at a fixed `S`, by
        `(1 - D) S:S / 4 mu` per unit of `dV = V tr(L) dt`, and nothing else pays for that.
        It is drawn from the internal energy here (ADR 0171), so the books close.
        """

        i = wp.tid()
        timestep = timesteps[TIMESTEP_SLOT_COMMITTED]
        material = material_index[i]
        if material < 0:
            return
        mu = shear_moduli[material]
        if mu <= zero:
            return
        stress = stress_tensor(stress_diagonal[i], stress_offdiagonal[i])
        gradient = gradients[i]
        strain = (gradient + wp.transpose(gradient)) * half
        spin = (gradient - wp.transpose(gradient)) * half
        trace = strain[0, 0] + strain[1, 1] + strain[2, 2]
        deviatoric = strain - wp.identity(n=3, dtype=scalar) * (trace * third)
        rate = deviatoric * (two * mu) + spin * stress - stress * spin
        rho = densities[i]
        if rho > zero:
            # `S:S / 4 mu` is `J2 / 2 mu`.
            heated_energies[i] = heated_energies[i] - (one - damage[i]) * invariant_j2(
                stress
            ) * trace * timestep / (two * mu * rho)
        updated = stress + rate * timestep
        # Written back through the six stored components, which is also the projection that
        # keeps the stored tensor exactly symmetric: the rate is symmetric analytically, and
        # storing six components means round-off cannot make it otherwise.
        stress_diagonal[i] = vector(updated[0, 0], updated[1, 1], updated[2, 2])
        stress_offdiagonal[i] = vector(updated[0, 1], updated[0, 2], updated[1, 2])

    @warp_kernel
    def apply_plasticity(
        material_index: wp.array(dtype=wp.int32),
        strength_ids: wp.array(dtype=wp.int32),
        strength_parameters: wp.array2d(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        shear_moduli: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        damage: wp.array(dtype=scalar),
        stress_diagonal: wp.array(dtype=vector),
        stress_offdiagonal: wp.array(dtype=vector),
        heated_energies: wp.array(dtype=scalar),
        yielded: wp.array(dtype=wp.int32),
    ):
        """S3: the radial return onto the yield surface.

        The elastic energy the return takes off the stress, `(1 - D) (S:S - S':S') / 4 mu rho`
        per unit mass, is plastic work and becomes heat (ADR 0171). Dropped, a hypervelocity
        contact lost over a third of its energy to nowhere.
        """

        i = wp.tid()
        material = material_index[i]
        if material < 0:
            return
        model = strength_ids[material]
        if model == STRENGTH_ID_NONE:
            return
        limit = strength_parameters[material, STRENGTH_SLOT_YIELD_STRESS]
        if model == STRENGTH_ID_DRUCKER_PRAGER:
            surface = (
                strength_parameters[material, STRENGTH_SLOT_COHESION]
                + strength_parameters[material, STRENGTH_SLOT_FRICTION] * pressures[i]
            )
            limit = wp.min(wp.max(surface, zero), limit)
        stress = stress_tensor(stress_diagonal[i], stress_offdiagonal[i])
        j2 = invariant_j2(stress)
        if j2 <= zero:
            return
        equivalent = wp.sqrt(three * j2)
        if equivalent <= limit:
            return
        factor = limit / equivalent
        stress_diagonal[i] = stress_diagonal[i] * factor
        stress_offdiagonal[i] = stress_offdiagonal[i] * factor
        mu = shear_moduli[material]
        rho = densities[i]
        if mu > zero and rho > zero:
            # `S:S = 2 J2`, and the return scales `J2` by `factor**2`.
            heated_energies[i] = heated_energies[i] + (one - damage[i]) * j2 * (
                one - factor * factor
            ) / (two * mu * rho)
        wp.atomic_add(yielded, 0, 1)

    @warp_kernel
    def advance_damage(
        material_index: wp.array(dtype=wp.int32),
        damage_ids: wp.array(dtype=wp.int32),
        damage_parameters: wp.array2d(dtype=scalar),
        flaw_strain: wp.array(dtype=scalar),
        flaw_stride: wp.array(dtype=wp.int32),
        flaw_offset: wp.array(dtype=wp.int32),
        flaw_count: wp.array(dtype=wp.int32),
        pressures: wp.array(dtype=scalar),
        stress_diagonal: wp.array(dtype=vector),
        stress_offdiagonal: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        damage: wp.array(dtype=scalar),
        activated_out: wp.array(dtype=wp.int32),
        timesteps: wp.array(dtype=scalar),
        shear_moduli: wp.array(dtype=scalar),
        heated_energies: wp.array(dtype=scalar),
    ):
        """S4: Weibull flaw activation and Grady-Kipp crack growth.

        The step is a device array for the same reason S2's is: this stage is recorded into
        the coupled graph, and a captured host float never changes again.

        A damaged particle carries `(1 - D) S`, so it stores `(1 - D) S:S / 4 mu` per unit
        volume. The share a growing crack takes, `dD S:S / 4 mu rho` per unit mass, is
        fracture work and becomes heat (ADR 0171).
        """

        i = wp.tid()
        timestep = timesteps[TIMESTEP_SLOT_COMMITTED]
        material = material_index[i]
        if material < 0:
            activated_out[i] = 0
            return
        if damage_ids[material] == DAMAGE_ID_NONE:
            activated_out[i] = 0
            return
        count = flaw_count[i]
        stride = flaw_stride[material]
        if count <= 0 or stride <= 0:
            activated_out[i] = 0
            return
        modulus = damage_parameters[material, DAMAGE_SLOT_YOUNG_MODULUS]
        if modulus <= zero:
            activated_out[i] = 0
            return
        stress = stress_tensor(stress_diagonal[i], stress_offdiagonal[i])
        # Tension is a positive principal stress, and the isotropic part shifts every
        # eigenvalue by the same amount, so the largest principal stress of the full tensor
        # is the largest principal deviator less the pressure.
        principal = principal_maximum(stress) - pressures[i]
        strain = principal / modulus
        if strain <= zero:
            # Unloaded: the count is a running peak (see below), so keep what is banked.
            return
        # `k V eps**m` evaluated as `(eps / eps_1)**m`, where `eps_1 = (k V)**(-1/m)` is the
        # strain the material's first flaw activates at. For a body the size of an asteroid
        # `k V` is past float32's range and `eps**m` below it; the ratio is neither (ADR 0173).
        first = flaw_strain[material]
        weibull_m = damage_parameters[material, DAMAGE_SLOT_WEIBULL_M]
        if first <= zero or weibull_m <= zero:
            activated_out[i] = 0
            return
        reached = wp.pow(strain / first, weibull_m)
        highest = wp.floor((reached - scalar(flaw_offset[i]) - one) / scalar(stride))
        activated = int(wp.min(wp.max(highest + one, zero), scalar(count)))
        # A flaw that has already nucleated keeps its crack growing even through a step whose
        # instantaneous strain implies fewer newly-reached flaws, so the count driving growth
        # can never fall below the largest this particle has already banked. `activated_out`
        # already carries the prior launch's stored count, so the max against it is that
        # running peak - the mirror of the damage running-max below.
        activated = wp.max(activated, activated_out[i])
        activated_out[i] = activated
        if activated <= 0:
            return
        radius = densities[i]
        if radius <= zero:
            return
        radius = wp.pow(three * masses[i] / (four_pi * densities[i]), third)
        if radius <= zero:
            return
        speed = damage_parameters[material, DAMAGE_SLOT_CRACK_SPEED]
        root = wp.pow(damage[i], third) + wp.pow(scalar(activated), third) * speed * timestep / (
            radius
        )
        ceiling = wp.min(one, scalar(activated) / scalar(count))
        # Grady-Kipp damage is cumulative crack volume: irreversible, so unloading must never
        # reduce it. The running max against the prior stored value is the mirror of
        # `advance_porosity`'s running min for irreversible pore crush below.
        previous = damage[i]
        grown = wp.max(previous, wp.min(root * root * root, ceiling))
        damage[i] = grown
        mu = shear_moduli[material]
        if grown > previous and mu > zero:
            heated_energies[i] = heated_energies[i] + (grown - previous) * invariant_j2(
                stress
            ) / (two * mu * densities[i])

    @warp_kernel
    def advance_porosity(
        material_index: wp.array(dtype=wp.int32),
        porosity_ids: wp.array(dtype=wp.int32),
        porosity_parameters: wp.array2d(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        distention: wp.array(dtype=scalar),
    ):
        """S5: irreversible P-alpha compaction.

        The running minimum is the irreversibility: unloading follows the elastic branch at
        the distention already reached, so a crushed pore never reopens.
        """

        i = wp.tid()
        material = material_index[i]
        if material < 0:
            return
        if porosity_ids[material] == POROSITY_ID_NONE:
            return
        initial = porosity_parameters[material, POROSITY_SLOT_INITIAL_DISTENTION]
        elastic = porosity_parameters[material, POROSITY_SLOT_ELASTIC_PRESSURE]
        solid = porosity_parameters[material, POROSITY_SLOT_SOLID_PRESSURE]
        exponent = porosity_parameters[material, POROSITY_SLOT_CRUSH_EXPONENT]
        pressure = pressures[i]
        curve = initial
        if pressure >= solid:
            curve = one
        elif pressure > elastic:
            span = (solid - pressure) / (solid - elastic)
            curve = one + (initial - one) * wp.pow(span, exponent)
        distention[i] = wp.min(distention[i], curve)

    @warp_func
    def solid_pressure(eos_id: int, parameters: wp.array2d(dtype=scalar), material: int,
                       density: scalar, energy: scalar):
        """Murnaghan and Tillotson, in the same expressions as `materials.py`."""

        rho0 = parameters[material, EOS_SLOT_REFERENCE_DENSITY]
        eta = density / rho0
        if eos_id == EOS_ID_MURNAGHAN:
            exponent = parameters[material, EOS_SLOT_EXPONENT]
            bulk = parameters[material, EOS_SLOT_BULK_MODULUS]
            return (bulk / exponent) * (wp.pow(eta, exponent) - one)
        mu = eta - one
        a = parameters[material, EOS_SLOT_A]
        b = parameters[material, EOS_SLOT_B]
        big_a = parameters[material, EOS_SLOT_BULK_MODULUS_A]
        big_b = parameters[material, EOS_SLOT_BULK_MODULUS_B]
        e0 = parameters[material, EOS_SLOT_REFERENCE_ENERGY]
        e_iv = parameters[material, EOS_SLOT_ENERGY_IV]
        e_cv = parameters[material, EOS_SLOT_ENERGY_CV]
        omega = one + energy / (e0 * eta * eta)
        compressed = (a + b / omega) * density * energy + big_a * mu + big_b * mu * mu
        if density >= rho0 or energy <= e_iv:
            return compressed
        inverse = rho0 / density - one
        decay = wp.exp(-parameters[material, EOS_SLOT_ALPHA] * inverse * inverse)
        cold_decay = wp.exp(-parameters[material, EOS_SLOT_BETA] * inverse)
        expanded = a * density * energy + (
            b * density * energy / omega + big_a * mu * cold_decay
        ) * decay
        if energy >= e_cv:
            return expanded
        weight = (energy - e_iv) / (e_cv - e_iv)
        return weight * expanded + (one - weight) * compressed

    @warp_kernel
    def apply_solid_equation_of_state(
        material_index: wp.array(dtype=wp.int32),
        eos_ids: wp.array(dtype=wp.int32),
        eos_parameters: wp.array2d(dtype=scalar),
        porosity_ids: wp.array(dtype=wp.int32),
        densities: wp.array(dtype=scalar),
        internal_energies: wp.array(dtype=scalar),
        distention: wp.array(dtype=scalar),
        shear_moduli: wp.array(dtype=scalar),
        damage: wp.array(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        sound_speeds: wp.array(dtype=scalar),
    ):
        """The condensed-matter closure, evaluated at the matrix density when porous.

        The sound speed is the adiabatic one, differentiated numerically about the evaluated
        point rather than by a second transcription of the analytic partials. A central
        difference at ``cbrt(eps)`` relative spacing is accurate to about ``eps**(2/3)``,
        which is far below the accuracy any Courant condition needs and cannot disagree with
        the pressure the force actually uses. The float64 lane's analytic partials in
        `materials.py` are what the validation suite gates this against.
        """

        i = wp.tid()
        material = material_index[i]
        if material < 0:
            return
        density = densities[i]
        if density <= zero:
            pressures[i] = zero
            sound_speeds[i] = zero
            return
        alpha = one
        if porosity_ids[material] != POROSITY_ID_NONE:
            alpha = distention[i]
        matrix_density = density * alpha
        eos_id = eos_ids[material]
        energy = internal_energies[i]
        pressure = solid_pressure(eos_id, eos_parameters, material, matrix_density, energy)
        pressures[i] = pressure / alpha

        step = difference_spacing * matrix_density
        upper = solid_pressure(eos_id, eos_parameters, material, matrix_density + step, energy)
        lower = solid_pressure(eos_id, eos_parameters, material, matrix_density - step, energy)
        squared = (upper - lower) / (two * step)
        if energy > zero:
            energy_step = difference_spacing * energy
            hot = solid_pressure(
                eos_id, eos_parameters, material, matrix_density, energy + energy_step
            )
            cold = solid_pressure(
                eos_id, eos_parameters, material, matrix_density, energy - energy_step
            )
            squared = squared + (pressure / (matrix_density * matrix_density)) * (
                (hot - cold) / (two * energy_step)
            )
        # The elastic wave is faster than the bulk wave: a solid transmits shear, so the
        # Courant condition has to see ``sqrt(c**2 + 4 mu / 3 rho)`` or a strong body will be
        # stepped as though it were a fluid. A cracked particle carries `(1 - D) S` (S6), so
        # the wave it transmits is `(1 - D) mu`'s; charging it the whole `mu` over a density
        # thinned towards zero gave a shattered fragment a runaway Courant speed (ADR 0174).
        mu = (one - damage[i]) * shear_moduli[material]
        if mu > zero:
            squared = squared + four_thirds * mu / matrix_density
        if squared <= zero:
            sound_speeds[i] = zero
        else:
            sound_speeds[i] = wp.sqrt(squared)

    @warp_kernel
    def compute_stress_forces(
        grid: wp.uint64,
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        densities: wp.array(dtype=scalar),
        omegas: wp.array(dtype=scalar),
        smoothing_lengths: wp.array(dtype=scalar),
        reach: wp.array(dtype=scalar),
        pressures: wp.array(dtype=scalar),
        stress_diagonal: wp.array(dtype=vector),
        stress_offdiagonal: wp.array(dtype=vector),
        damage: wp.array(dtype=scalar),
        kernel_id: int,
        normalization: scalar,
        support_ratio: scalar,
        include_pressure: int,
        number_density: int,
        accelerations: wp.array(dtype=vector),
    ):
        """S6: the symmetric SPH divergence of the full Cauchy stress.

        ``a_i = sum_j m_j (sigma_i / (Omega_i rho_i**2) + sigma_j / (Omega_j rho_j**2)) .
        grad_i W`` with each tensor paired against the gradient taken at its own smoothing
        length. The pair contribution to ``j`` is the same two matrix-vector products with
        the gradients negated, so ``m_i a_i + m_j a_j`` cancels term by term and total
        momentum is conserved to round-off - the property the gate measures, and the reason
        the per-particle term is formed once rather than averaged across the pair.

        This is also where COLL-002 is satisfied physically: two resolved bodies in contact
        exchange momentum here, through the pressure and deviatoric gradients of overlapping
        support, with no collision event, no impulse, and no pair list of any other kind.

        ``include_pressure`` selects which half of the Cauchy stress this launch carries.
        Standalone it is one and the divergence is the whole tensor, which is the form every
        `adr/0022` gate is stated against. Spliced into the hydrodynamic step it is zero,
        because there the isotropic part is already carried by the pressure gradient the
        hydro force kernel forms - the condensed EOS has overwritten `pressures` for solid
        particles by then, so that gradient is the solid one. Two symmetrised pair forms over
        the same pressure field would double it. Each term stays antisymmetric pair by pair
        either way, so the momentum statement holds for both settings.

        ``number_density`` takes the form the fluid's pressure gradient takes under a number
        density (ADR 0166): ``m_i a_i = sum_j (m_i**2 term_i grad_i + m_j**2 term_j grad_j)``,
        still antisymmetric pair by pair. Spliced in, this is the deviatoric half of the
        stress whose isotropic half that gradient carries, so the two halves must share a
        form; and its work is balanced by S1 taken with the own-volume weight (ADR 0167).
        """

        # Grid order: consecutive threads take particles that share cells, so a warp walks
        # the same neighbours together. Each thread writes only entry `i`.
        slot = wp.tid()
        i = wp.hash_grid_point_id(grid, slot)
        if i < 0:
            return
        h_i = smoothing_lengths[i]
        if h_i <= zero:
            return
        rho_i = densities[i]
        if rho_i <= zero:
            return
        p = positions[i]
        radius = reach[i]
        sigma_i = stress_tensor(stress_diagonal[i], stress_offdiagonal[i])
        scale_i = one - damage[i]
        carried_i = zero
        if include_pressure != 0:
            pressure_i = pressures[i]
            carried_i = -pressure_i
            if pressure_i < zero:
                carried_i = -pressure_i * scale_i
        sigma_i = sigma_i * scale_i + wp.identity(n=3, dtype=scalar) * carried_i
        term_i = sigma_i * (one / (omegas[i] * rho_i * rho_i))

        acceleration = vector()
        query = wp.hash_grid_query(grid, p, radius)
        j = int(0)  # noqa: UP018
        while wp.hash_grid_query_next(query, j):
            if j == i:
                continue
            h_j = smoothing_lengths[j]
            if h_j <= zero:
                continue
            rho_j = densities[j]
            if rho_j <= zero:
                continue
            offset = p - positions[j]
            r = wp.length(offset)
            if r <= zero:
                continue
            span = support_ratio * wp.max(h_i, h_j)
            if r >= wp.min(span, wp.min(radius, reach[j])):
                continue
            derivative_i = kernel_gradient(kernel_id, normalization, support_ratio, r, h_i)
            derivative_j = kernel_gradient(kernel_id, normalization, support_ratio, r, h_j)
            grad_i = offset * (derivative_i / r)
            grad_j = offset * (derivative_j / r)
            sigma_j = stress_tensor(stress_diagonal[j], stress_offdiagonal[j])
            scale_j = one - damage[j]
            carried_j = zero
            if include_pressure != 0:
                pressure_j = pressures[j]
                carried_j = -pressure_j
                if pressure_j < zero:
                    carried_j = -pressure_j * scale_j
            sigma_j = sigma_j * scale_j + wp.identity(n=3, dtype=scalar) * carried_j
            term_j = sigma_j * (one / (omegas[j] * rho_j * rho_j))
            if number_density == 0:
                acceleration = acceleration + (term_i * grad_i + term_j * grad_j) * masses[j]
            else:
                m_i = masses[i]
                m_j = masses[j]
                acceleration = acceleration + (term_i * grad_i) * m_i + (term_j * grad_j) * (
                    m_j * (m_j / m_i)
                )
        accelerations[i] = acceleration

    @warp_kernel
    def accumulate_solid_forces(
        solid_accelerations: wp.array(dtype=vector),
        accelerations: wp.array(dtype=vector),
    ):
        """Add the deviatoric divergence into a caller's total acceleration.

        Separate from S6 for the reason `selfgravity.apply_gravity` is separate from the
        traversal: the hydrodynamic force kernel *assigns* the acceleration, so anything that
        contributes has to follow it and accumulate rather than write. Keeping the write in
        S6 also leaves the standalone stage - the one every `adr/0022` gate measures -
        untouched by the coupling.
        """

        i = wp.tid()
        accelerations[i] = accelerations[i] + solid_accelerations[i]

    return _SolidKernels(
        seed_reach=seed_reach,
        scatter_reach=scatter_reach,
        compute_velocity_gradient=compute_velocity_gradient,
        advance_deviatoric_stress=advance_deviatoric_stress,
        apply_plasticity=apply_plasticity,
        advance_damage=advance_damage,
        advance_porosity=advance_porosity,
        apply_solid_equation_of_state=apply_solid_equation_of_state,
        compute_stress_forces=compute_stress_forces,
        accumulate_solid_forces=accumulate_solid_forces,
    )


@dataclass(frozen=True)
class _SolidKernels:
    seed_reach: Any
    scatter_reach: Any
    compute_velocity_gradient: Any
    advance_deviatoric_stress: Any
    apply_plasticity: Any
    advance_damage: Any
    advance_porosity: Any
    apply_solid_equation_of_state: Any
    compute_stress_forces: Any
    accumulate_solid_forces: Any


# ---------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolidConfig:
    """Configuration of the solid stages for one context."""

    capacity: int
    count: int
    registry: MaterialRegistry
    precision: Precision = Precision.FLOAT32
    correct_velocity_gradient: bool = True
    flaws_per_particle: int = 10
    # The number-density pair (ADR 0167): S6's divergence in the form the fluid's pressure
    # gradient takes under `HydroConfig.number_density` (ADR 0166), and S1 weighted by the
    # particle's own volume, which is the gradient that force does its work against. Off is
    # the mass-sum pair every `adr/0022` gate is stated against. The two are one number where
    # neighbours share a mass; where they do not - an iron core under a rock mantle - pairing
    # one form's force with the other's gradient makes elastic energy from nothing.
    number_density: bool = False

    def validate(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 < self.count <= self.capacity:
            raise ValueError("count must be positive and within capacity")
        if len(self.registry) == 0:
            raise ValueError("a solid configuration needs at least one material")
        if self.flaws_per_particle < 1:
            raise ValueError(
                "flaws_per_particle must be at least one: a Weibull population with no flaws "
                "cannot activate, and the resolution dependence of the damage model is "
                "exactly this count"
            )

    @property
    def graph_modules(self) -> tuple[str, ...]:
        return self.registry.graph_modules


@dataclass(frozen=True)
class SolidDiagnostics:
    active_count: int
    material_count: int
    damage_max: float
    distention_min: float
    yielded_particles: int
    correction_failures: int
    stage_ms: float
    graph_modules: tuple[str, ...]


def _material_tables(registry: MaterialRegistry) -> dict[str, list[Any]]:
    """Flatten the registry into the row-per-material device layout.

    Materials are addressed by their index in `MaterialRegistry.materials`, not by their
    `material_id`: the ids are a scenario's naming and may be sparse, and a sparse device
    table would size the parameter arrays by the largest id a scene happened to use.
    """

    materials: tuple[MaterialDescriptor, ...] = registry.materials
    tables: dict[str, list[Any]] = {
        "eos_ids": [],
        "eos_parameters": [],
        "strength_ids": [],
        "strength_parameters": [],
        "damage_ids": [],
        "damage_parameters": [],
        "porosity_ids": [],
        "porosity_parameters": [],
        "shear_moduli": [],
    }
    for material in materials:
        tables["eos_ids"].append(material.eos.eos_id)
        eos_row = [0.0] * EOS_SLOTS
        for key, value in material.eos.parameters.items():
            eos_row[_EOS_SLOT_NAMES[key]] = float(value)
        tables["eos_parameters"].append(eos_row)

        tables["shear_moduli"].append(float(material.shear_modulus))
        tables["strength_ids"].append(_STRENGTH_IDS[material.strength])
        tables["damage_ids"].append(_DAMAGE_IDS[material.damage])
        tables["porosity_ids"].append(_POROSITY_IDS[material.porosity])

        strength_row = [0.0] * STRENGTH_SLOTS
        damage_row = [0.0] * DAMAGE_SLOTS
        porosity_row = [0.0] * POROSITY_SLOTS
        for key, value in material.parameters.items():
            if key in _STRENGTH_SLOT_NAMES:
                strength_row[_STRENGTH_SLOT_NAMES[key]] = float(value)
            if key in _DAMAGE_SLOT_NAMES:
                damage_row[_DAMAGE_SLOT_NAMES[key]] = float(value)
            if key in _POROSITY_SLOT_NAMES:
                porosity_row[_POROSITY_SLOT_NAMES[key]] = float(value)
        # An unyielding von Mises material would be one whose yield stress slot stayed zero,
        # which is a valid configuration; Drucker-Prager saturating at zero is not, and is
        # already rejected by the descriptor's schema.
        tables["strength_parameters"].append(strength_row)
        tables["damage_parameters"].append(damage_row)
        tables["porosity_parameters"].append(porosity_row)
    return tables


def _material_identity_arrays(
    np: Any,
    material_ids: Any,
    *,
    material_rows: dict[int, int],
    count: int,
    capacity: int,
) -> tuple[Any, Any]:
    """Build separate transient table rows and persistent material identities.

    Constitutive kernels index compact parameter tables, while Color By Material must publish
    the authored registry ID. They can coincide for a small registry and still mean different
    things, so both arrays are retained explicitly on the device.
    """

    supplied = np.asarray(material_ids, dtype=np.int64)
    if supplied.shape != (count,):
        raise ValueError(f"material_ids must have shape ({count},)")
    rows = np.full(capacity, -1, dtype=np.int32)
    persistent = np.full(capacity, -1, dtype=np.int64)
    persistent[:count] = supplied
    for index, material_id in enumerate(supplied.tolist()):
        # -1 is the coupled fluid sentinel and remains absent in both representations.
        if material_id < 0:
            continue
        if material_id not in material_rows:
            raise KeyError(f"unknown material_id {material_id}")
        rows[index] = material_rows[material_id]
    return rows, persistent


def _damage_source_is_active(material_ids: Any, registry: MaterialRegistry) -> bool:
    """Whether any material actually used by this state runs a damage model."""

    return any(
        material_id >= 0 and registry.get(int(material_id)).damage is not DamageModel.NONE
        for material_id in material_ids.tolist()
    )


class SolidRuntime:
    """Own the solid state and launch the constitutive stages over a neighbour service.

    The stages are separate methods rather than one `step` because that is what SOLID-001
    asks to be validatable: the validation ladder gates the stress rate, the yield surface,
    the crush curve and the damage growth independently, and a single fused advance would
    make each gate a statement about the composition instead of about the model.

    Nothing here captures a CUDA graph. The solid pipeline is not on a playback path yet - it
    is driven by the validation suite and by the impact scenarios of Phase 6 - and the graph
    belongs with the step that will drive it, next to the hydrodynamic capture that already
    covers the neighbour rebuild.
    """

    def __init__(self, neighbors: NeighborRuntime, config: SolidConfig) -> None:
        config.validate()
        # `compute_stress_forces` walks in grid order: thread `slot` takes the grid's
        # `slot`-th point. That is a permutation of this stage's particles only if the grid
        # holds exactly them, so a count that differs would silently skip some particles
        # and compute others twice.
        if config.count != neighbors.config.count:
            raise ValueError(
                f"solid count {config.count} must match the neighbour count "
                f"{neighbors.config.count}"
            )
        self.neighbors = neighbors
        self.config = config
        self.last_stage_ms = 0.0
        self._wp: Any | None = None
        self._kernels: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._material_rows: dict[int, int] = {
            material.material_id: index
            for index, material in enumerate(config.registry.materials)
        }
        self._recording = False
        self._continuum_bound = False
        # Where S3 and S4 book the elastic energy they release (ADR 0171); `None` means
        # this runtime's own `internal_energies`, looked up at launch time.
        self._heated_energies: Any = None
        self._damage_source_active = False

    def close(self) -> None:
        """Drop the device arrays this runtime owns (the hydro owner calls this on close)."""

        self._arrays.clear()
        self._heated_energies = None
        self._continuum_bound = False

    # Everything the constitutive model integrates, as opposed to derives: a scrub restore that
    # reloads positions and velocities alone hands back intact, stress-free material (SR-12).
    # `gradients`, `pressures`, `sound_speeds` and the rest are re-evaluated by the first
    # `prime`, so they are not state.
    STATE_ARRAYS = (
        "stress_diagonal",
        "stress_offdiagonal",
        "damage",
        "activated",
        "distention",
    )

    def capture_state(self) -> dict[str, Any]:
        """Host copies of the integrated solid state, trimmed to the live particles."""

        if not self._arrays:
            raise DeviceRequirementError("SolidRuntime has no loaded state to capture")
        count = self.config.count
        return {name: self._arrays[name].numpy()[:count].copy() for name in self.STATE_ARRAYS}

    def restore_state(self, state: dict[str, Any]) -> None:
        """Write a `capture_state` result back in place, keeping every array's identity.

        In place because the arrays are bound into recorded stages; a fresh array would leave
        a captured graph or a coupled stage reading the old one.
        """

        staged = self.stage_state(state)
        count = self.config.count
        for name in self.STATE_ARRAYS:
            target = self._arrays[name]
            buffer = target.numpy()
            buffer[:count] = staged[name]
            target.assign(buffer)

    # Model-valid ranges (SR-68). Damage is a fraction and the stress/heat terms use
    # (1 - damage); a flaw count cannot be negative. Everything is finite. No STATE_ARRAYS key
    # is optional: every snapshot that carries `solid_state` was written by `capture_state`,
    # which writes all of them, so a missing key is corruption, not a legacy snapshot (a
    # snapshot with no `solid_state` at all is the legacy case and is handled by the caller).
    _STATE_RANGES: dict[str, tuple[float | None, float | None]] = {
        "damage": (0.0, 1.0),
        "activated": (0.0, None),
    }

    def stage_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate a `capture_state` result and return it as host arrays; writes nothing.

        Raises `ValueError` on a missing key, a wrong shape, a non-finite value or a value
        outside the model's range, so a rejected restore leaves the live state untouched.
        """

        if not self._arrays:
            raise DeviceRequirementError("SolidRuntime has no loaded state to restore into")
        np = __import__("numpy")
        count = self.config.count
        staged: dict[str, Any] = {}
        for name in self.STATE_ARRAYS:
            if name not in state:
                raise ValueError(f"solid state is missing required field {name!r}")
            buffer = self._arrays[name].numpy()
            supplied = np.asarray(state[name], dtype=buffer.dtype)
            if supplied.shape != (count, *buffer.shape[1:]):
                raise ValueError(
                    f"solid state {name!r} has shape {supplied.shape}, "
                    f"expected {(count, *buffer.shape[1:])}"
                )
            if not bool(np.isfinite(supplied).all()):
                raise ValueError(f"solid state {name!r} has non-finite values")
            low, high = self._STATE_RANGES.get(name, (None, None))
            if supplied.size and (
                (low is not None and bool((supplied < low).any()))
                or (high is not None and bool((supplied > high).any()))
            ):
                raise ValueError(f"solid state {name!r} is outside [{low}, {high}]")
            staged[name] = supplied
        return staged

    # -- state ---------------------------------------------------------------------------

    def load_state(
        self,
        positions: Any,
        masses: Any,
        densities: Any,
        internal_energies: Any,
        material_ids: Any,
        velocities: Any = None,
        omegas: Any = None,
    ) -> None:
        """Upload the solid state and allocate the constitutive arrays.

        ``densities`` and ``omegas`` are supplied rather than summed here: the density
        estimate and the ``grad-h`` term belong to the SPH core, and re-deriving them in the
        solid module would give an impact run two density fields that could disagree.
        """

        self.neighbors.load_points(positions)
        wp = self.neighbors.device_context.wp
        self._wp = wp
        np = __import__("numpy")
        config = self.config
        scalar, vector, state_dtype = _dtypes(wp, config.precision)
        matrix = _matrix_type(wp, config.precision)
        self._kernels = _solid_kernels(wp, config.precision)
        device = self.neighbors.device_context.device
        count = config.count
        capacity = config.capacity

        def padded_scalars(values: Any, label: str, positive: bool) -> Any:
            supplied = np.asarray(values, dtype=state_dtype)
            if supplied.shape != (count,):
                raise ValueError(f"{label} must have shape ({count},), got {supplied.shape}")
            if positive and not bool(np.all(supplied > 0.0)):
                raise ValueError(f"{label} must be strictly positive")
            padded = np.zeros(capacity, dtype=state_dtype)
            padded[:count] = supplied
            return wp.array(padded, dtype=scalar, device=device)

        def padded_vectors(values: Any, label: str) -> Any:
            padded = np.zeros((capacity, 3), dtype=state_dtype)
            if values is not None:
                supplied = np.asarray(values, dtype=state_dtype)
                if supplied.shape != (count, 3):
                    raise ValueError(f"{label} must have shape ({count}, 3)")
                padded[:count] = supplied
            return wp.array(padded, dtype=vector, device=device)

        rows, persistent_material_ids = _material_identity_arrays(
            np,
            material_ids,
            material_rows=self._material_rows,
            count=count,
            capacity=capacity,
        )
        self._damage_source_active = _damage_source_is_active(
            persistent_material_ids[:count], config.registry
        )

        ones = np.zeros(capacity, dtype=state_dtype)
        ones[:count] = 1.0

        tables = _material_tables(config.registry)
        materials = len(config.registry)

        def table_scalars(name: str) -> Any:
            return wp.array(np.asarray(tables[name], dtype=state_dtype), dtype=scalar,
                            device=device)

        def table_rows(name: str) -> Any:
            return wp.array(np.asarray(tables[name], dtype=state_dtype), dtype=scalar,
                            ndim=2, device=device)

        def table_ints(name: str) -> Any:
            return wp.array(np.asarray(tables[name], dtype=np.int32), dtype=wp.int32,
                            device=device)

        flaw_offset = np.zeros(capacity, dtype=np.int32)
        flaw_count = np.zeros(capacity, dtype=np.int32)
        flaw_stride = np.zeros(materials, dtype=np.int32)
        flaw_strain = np.zeros(materials, dtype=state_dtype)
        volumes = np.zeros(materials, dtype=np.float64)
        host_masses = np.asarray(masses, dtype=np.float64)
        host_densities = np.asarray(densities, dtype=np.float64)
        for row in range(materials):
            selected = np.nonzero(rows[:count] == row)[0]
            flaw_stride[row] = selected.size
            # Rank within the material, in particle order: deterministic, and reproducible
            # across runs and across particle counts, which is what a seeded permutation
            # would have to promise anyway.
            flaw_offset[selected] = np.arange(selected.size, dtype=np.int32)
            flaw_count[selected] = config.flaws_per_particle
            if selected.size:
                volumes[row] = float(
                    np.sum(host_masses[selected] / host_densities[selected])
                )
        for index, material in enumerate(config.registry.materials):
            weibull_k = float(material.parameters.get("weibull_k", 0.0))
            weibull_m = float(material.parameters.get("weibull_m", 0.0))
            # In float64 on the host, then cast: `k V` itself does not fit float32 (ADR 0173).
            # Zero means "no flaws", which S4 reads as nothing ever activating.
            first = weibull_threshold(0, weibull_k * volumes[index], weibull_m)
            flaw_strain[index] = first if math.isfinite(first) else 0.0

        self._arrays = {
            "material_index": wp.array(rows, dtype=wp.int32, device=device),
            "material_ids": wp.array(persistent_material_ids, dtype=wp.int64, device=device),
            "masses": padded_scalars(masses, "masses", True),
            "densities": padded_scalars(densities, "densities", True),
            "internal_energies": padded_scalars(internal_energies, "internal_energies", False),
            "velocities": padded_vectors(velocities, "velocities"),
            "omegas": (
                wp.array(ones, dtype=scalar, device=device)
                if omegas is None
                else padded_scalars(omegas, "omegas", True)
            ),
            "distention": wp.array(
                self._initial_distention(np, rows, capacity, count, state_dtype),
                dtype=scalar,
                device=device,
            ),
            "stress_diagonal": wp.zeros(capacity, dtype=vector, device=device),
            "stress_offdiagonal": wp.zeros(capacity, dtype=vector, device=device),
            "gradients": wp.zeros(capacity, dtype=matrix, device=device),
            "pressures": wp.zeros(capacity, dtype=scalar, device=device),
            "sound_speeds": wp.zeros(capacity, dtype=scalar, device=device),
            "damage": wp.zeros(capacity, dtype=scalar, device=device),
            "activated": wp.zeros(capacity, dtype=wp.int32, device=device),
            "accelerations": wp.zeros(capacity, dtype=vector, device=device),
            "reach": wp.zeros(capacity, dtype=scalar, device=device),
            "yielded": wp.zeros(1, dtype=wp.int32, device=device),
            "correction_failures": wp.zeros(1, dtype=wp.int32, device=device),
            # Same two-slot layout `hydro.py` publishes; `bind_continuum` swaps this for the
            # hydrodynamic stage's own array so the coupled rate stages read the step that
            # stage committed rather than a second, independently maintained one.
            "timesteps": wp.zeros(2, dtype=scalar, device=device),
            "flaw_offset": wp.array(flaw_offset, dtype=wp.int32, device=device),
            "flaw_count": wp.array(flaw_count, dtype=wp.int32, device=device),
            "flaw_stride": wp.array(flaw_stride, dtype=wp.int32, device=device),
            "flaw_strain": wp.array(flaw_strain, dtype=scalar, device=device),
            "eos_ids": table_ints("eos_ids"),
            "eos_parameters": table_rows("eos_parameters"),
            "strength_ids": table_ints("strength_ids"),
            "strength_parameters": table_rows("strength_parameters"),
            "damage_ids": table_ints("damage_ids"),
            "damage_parameters": table_rows("damage_parameters"),
            "porosity_ids": table_ints("porosity_ids"),
            "porosity_parameters": table_rows("porosity_parameters"),
            "shear_moduli": table_scalars("shear_moduli"),
        }

    def _initial_distention(
        self, np: Any, rows: Any, capacity: int, count: int, state_dtype: str
    ) -> Any:
        values = np.ones(capacity, dtype=state_dtype)
        for index, material in enumerate(self.config.registry.materials):
            if material.porosity is PorosityModel.NONE:
                continue
            values[:count][rows[:count] == index] = float(
                material.parameters["initial_distention"]
            )
        return values

    # -- stages --------------------------------------------------------------------------

    def _launch(self, kernel: Any, inputs: list[Any], dim: int | None = None) -> None:
        wp = self._require_wp()
        if self._recording:
            # A recorded launch is written into the caller's sequence, not run: synchronising
            # is illegal inside a capture, and timing a launch that has not happened yet
            # would report the recording rather than the work.
            wp.launch(
                kernel,
                dim=self.config.count if dim is None else dim,
                inputs=inputs,
                device=self.neighbors.device_context.device,
            )
            return
        start = time.perf_counter()
        wp.launch(
            kernel,
            dim=self.config.count if dim is None else dim,
            inputs=inputs,
            device=self.neighbors.device_context.device,
        )
        wp.synchronize_device(self.neighbors.device_context.device)
        self.last_stage_ms = (time.perf_counter() - start) * 1000.0

    def _commit_timestep(self, timestep: float | None) -> None:
        """Publish a host step into the slot the rate kernels read.

        Only the standalone path reaches the write. Coupled, the slot is the hydrodynamic
        stage's own and is written on the device by its `commit_timestep` launch, so the step
        the rate stages integrate is the step the integrator took - by construction, not by
        two copies agreeing. Passing a step anyway is the error, not passing none: it would
        read as authority over a value this runtime does not own.
        """

        if self._continuum_bound:
            if timestep is not None:
                raise DeviceRequirementError(
                    "the timestep of a coupled SolidRuntime is the hydrodynamic stage's; "
                    "advance_stress/advance_damage take theirs from the device"
                )
            return
        if timestep is None:
            raise DeviceRequirementError(
                "an unbound SolidRuntime has no step on the device; pass one"
            )
        wp = self._require_wp()
        np = __import__("numpy")
        scalar, _, state_dtype = _dtypes(wp, self.config.precision)
        slots = np.zeros(2, dtype=state_dtype)
        slots[TIMESTEP_SLOT_COMMITTED] = timestep
        self._arrays["timesteps"] = wp.array(
            slots, dtype=scalar, device=self.neighbors.device_context.device
        )

    def _support(self) -> tuple[int, float, float]:
        spec = KERNEL_REGISTRY[self.neighbors.config.kernel]
        return _kernel_id(spec.name), spec.normalization, spec.support_ratio

    def _prepare_pairs(self) -> None:
        """Re-solve the neighbourhood and publish symmetric query radii.

        Both the gradient and the force stage need the pair set to be symmetric under
        adaptive ``h``, so both start here. Nothing is read back to the host: the reach is
        built on the device by the same seed-then-scatter pass the hydrodynamic stage uses.

        Coupled, that pass is the hydrodynamic stage's and has already been recorded over the
        same neighbour service with the same kernel, and `bind_continuum` took its `reach`.
        Repeating it here would submit host work inside a graph capture, where `solve()` is
        illegal, and would build a second grid from the same positions.
        """

        if self._recording:
            return
        kernels = self._require_kernels()
        _, _, support_ratio = self._support()
        self.neighbors.solve()
        self._launch(kernels.seed_reach, [
            self.neighbors.smoothing_length_array,
            self._arrays["reach"],
            support_ratio,
        ])
        self._launch(kernels.scatter_reach, [
            self.neighbors.grid_id,
            self.neighbors.positions,
            self.neighbors.smoothing_length_array,
            self._arrays["reach"],
            support_ratio,
        ])

    def compute_velocity_gradient(self) -> None:
        """S1."""

        kernels = self._require_kernels()
        kernel_id, normalization, support_ratio = self._support()
        arrays = self._arrays
        self._prepare_pairs()
        arrays["correction_failures"].zero_()
        self._launch(
            kernels.compute_velocity_gradient,
            [
                self.neighbors.grid_id,
                self.neighbors.positions,
                arrays["velocities"],
                arrays["masses"],
                arrays["densities"],
                self.neighbors.smoothing_length_array,
                arrays["reach"],
                kernel_id,
                normalization,
                support_ratio,
                1 if self.config.number_density else 0,
                1 if self.config.correct_velocity_gradient else 0,
                arrays["gradients"],
                arrays["correction_failures"],
            ],
        )

    def advance_stress(self, timestep: float | None = None) -> None:
        """S2. ``None`` means the step is already on the device, which is the coupled case."""

        kernels = self._require_kernels()
        self._commit_timestep(timestep)
        arrays = self._arrays
        self._launch(
            kernels.advance_deviatoric_stress,
            [
                arrays["gradients"],
                arrays["material_index"],
                arrays["shear_moduli"],
                arrays["stress_diagonal"],
                arrays["stress_offdiagonal"],
                arrays["timesteps"],
                arrays["densities"],
                arrays["damage"],
                self._heat_sink(),
            ],
        )

    def apply_plasticity(self) -> None:
        """S3."""

        kernels = self._require_kernels()
        arrays = self._arrays
        arrays["yielded"].zero_()
        self._launch(
            kernels.apply_plasticity,
            [
                arrays["material_index"],
                arrays["strength_ids"],
                arrays["strength_parameters"],
                arrays["pressures"],
                arrays["shear_moduli"],
                arrays["densities"],
                arrays["damage"],
                arrays["stress_diagonal"],
                arrays["stress_offdiagonal"],
                self._heat_sink(),
                arrays["yielded"],
            ],
        )

    def _heat_sink(self) -> Any:
        """The energy S3 and S4 credit with the elastic energy they take out of the stress.

        Coupled, `internal_energies` is the hydrodynamic stage's predicted state, which the
        next step overwrites; the heat has to land in the committed energy instead.
        """

        if self._heated_energies is not None:
            return self._heated_energies
        return self._arrays["internal_energies"]

    def advance_damage(self, timestep: float | None = None) -> None:
        """S4. ``None`` means the step is already on the device, which is the coupled case."""

        kernels = self._require_kernels()
        self._commit_timestep(timestep)
        arrays = self._arrays
        self._launch(
            kernels.advance_damage,
            [
                arrays["material_index"],
                arrays["damage_ids"],
                arrays["damage_parameters"],
                arrays["flaw_strain"],
                arrays["flaw_stride"],
                arrays["flaw_offset"],
                arrays["flaw_count"],
                arrays["pressures"],
                arrays["stress_diagonal"],
                arrays["stress_offdiagonal"],
                arrays["masses"],
                arrays["densities"],
                arrays["damage"],
                arrays["activated"],
                arrays["timesteps"],
                arrays["shear_moduli"],
                self._heat_sink(),
            ],
        )

    def advance_porosity(self) -> None:
        """S5."""

        kernels = self._require_kernels()
        arrays = self._arrays
        self._launch(
            kernels.advance_porosity,
            [
                arrays["material_index"],
                arrays["porosity_ids"],
                arrays["porosity_parameters"],
                arrays["pressures"],
                arrays["distention"],
            ],
        )

    def evaluate_equation_of_state(self) -> None:
        kernels = self._require_kernels()
        arrays = self._arrays
        self._launch(
            kernels.apply_solid_equation_of_state,
            [
                arrays["material_index"],
                arrays["eos_ids"],
                arrays["eos_parameters"],
                arrays["porosity_ids"],
                arrays["densities"],
                arrays["internal_energies"],
                arrays["distention"],
                arrays["shear_moduli"],
                arrays["damage"],
                arrays["pressures"],
                arrays["sound_speeds"],
            ],
        )

    def compute_stress_forces(self, include_pressure: bool = True) -> None:
        """S6."""

        kernels = self._require_kernels()
        kernel_id, normalization, support_ratio = self._support()
        arrays = self._arrays
        self._prepare_pairs()
        self._launch(
            kernels.compute_stress_forces,
            [
                self.neighbors.grid_id,
                self.neighbors.positions,
                arrays["masses"],
                arrays["densities"],
                arrays["omegas"],
                self.neighbors.smoothing_length_array,
                arrays["reach"],
                arrays["pressures"],
                arrays["stress_diagonal"],
                arrays["stress_offdiagonal"],
                arrays["damage"],
                kernel_id,
                normalization,
                support_ratio,
                1 if include_pressure else 0,
                1 if self.config.number_density else 0,
                arrays["accelerations"],
            ],
        )

    # -- continuum coupling --------------------------------------------------------------

    def bind_continuum(self, arrays: dict[str, Any]) -> None:
        """Adopt the hydrodynamic stage's arrays for every field both stages read.

        Nothing is copied, the same way `SelfGravityRuntime.bind` takes the neighbour
        stage's positions: a coupled impact must not carry two densities, two pressures or
        two timesteps that could disagree, and the density and the ``grad-h`` term are the
        SPH core's to produce - `load_state` already refuses to derive them.

        What stays this runtime's own is the state no other stage produces - the stress, the
        damage, the distention, the velocity gradient and the flaw tables - plus
        `accelerations`, which S6 assigns and `record_apply` then accumulates into the
        caller's total.

        ``heated_energies`` is where S3 and S4 book the elastic energy they release (ADR
        0171). It is the caller's committed specific energy, not the predicted state bound
        as `internal_energies`: the predictor rewrites that every step, so a credit there
        would be lost.
        """

        def checked(name: str, owned: Any) -> Any:
            if name not in arrays:
                raise ValueError(f"bind_continuum requires the {name!r} array")
            supplied = arrays[name]
            if owned is None:
                raise DeviceRequirementError("SolidRuntime has no loaded state")
            if supplied.dtype is not owned.dtype:
                raise ValueError(f"{name} dtype does not match the configured precision")
            if len(supplied) < len(owned):
                raise ValueError(f"{name} is shorter than the allocated capacity")
            return supplied

        for name in CONTINUUM_ARRAYS:
            self._arrays[name] = checked(name, self._arrays.get(name))
        self._heated_energies = checked(
            "heated_energies", self._arrays.get("internal_energies")
        )
        self._continuum_bound = True

    def _record(self, stages: tuple[Any, ...]) -> None:
        """Write `stages` into the caller's sequence instead of submitting them.

        The flag rather than a parallel set of recording methods: every stage would
        otherwise exist twice, and the pair that drifted would be the one no gate covers.
        """

        if not self._continuum_bound:
            raise DeviceRequirementError(
                "record_* requires bind_continuum; an unbound SolidRuntime owns its own "
                "densities and timestep and must be driven stage by stage"
            )
        self._recording = True
        try:
            for stage in stages:
                stage()
        finally:
            self._recording = False

    def record_equation_of_state(self) -> None:
        """Write the condensed closure in, over the gas closure the caller just applied.

        Recorded immediately after the hydrodynamic equation of state rather than instead of
        it: the kernel returns early where `material_index` is negative, so a hydrodynamic-only
        particle keeps the gas pressure and sound speed, and a solid particle has both
        overwritten before anything reads them. The elastic wave enters `sound_speeds` here,
        which is what lets the caller's existing Courant reduction bound the step without a
        second constant.
        """

        self._record((self.evaluate_equation_of_state,))

    def velocity_gradient_outputs(self) -> tuple[int, Any, Any]:
        """S1's switch and outputs, for a caller that computes S1 inside its own traversal.

        Coupled, S1 is not a launch of its own. Its query - radius `reach[i]` about
        `positions[i]`, on the neighbour service both stages share, over the arrays
        `bind_continuum` adopted - is the one the hydrodynamic force kernel already makes, so
        `HydroRuntime` computes it in that walk (ADR 0127) and `record_stage` starts at S2.
        This zeroes the failure counter the way `compute_velocity_gradient` does before its
        launch, and returns ``(correct, gradients, correction_failures)`` in that kernel's
        argument order. The standalone path is unchanged and still launches S1 itself.
        """

        if not self._continuum_bound:
            raise DeviceRequirementError(
                "velocity_gradient_outputs requires bind_continuum; an unbound SolidRuntime "
                "computes S1 with compute_velocity_gradient"
            )
        arrays = self._arrays
        arrays["correction_failures"].zero_()
        return (
            1 if self.config.correct_velocity_gradient else 0,
            arrays["gradients"],
            arrays["correction_failures"],
        )

    def tension_damage(self) -> Any:
        """The damage the caller's pressure gradient cuts tension by (ADR 0172).

        Coupled, the hydrodynamic force kernel carries the pressure, so the rule S6 applies
        standalone - a damaged particle carries `(1 - D)` of a negative pressure - has to be
        applied there, from this array.
        """

        return self._arrays["damage"]

    def record_stage(self) -> None:
        """Write S2..S6 into the caller's sequence, contributing the deviatoric force only.

        S1 is not here: the caller computes it in its own force traversal and has already
        written `gradients` by the time this runs; see `velocity_gradient_outputs`.

        `include_pressure=False` because the caller's pressure gradient already carries the
        isotropic part - over the pressure this runtime itself wrote in
        `record_equation_of_state` - and adding it twice would double the whole compressive
        response. Standalone the same kernel carries the entire Cauchy stress, which is the
        form every `adr/0022` gate is stated against, so the coupling leaves those numbers
        untouched.
        """

        self._record(
            (
                self.advance_stress,
                self.apply_plasticity,
                self.advance_damage,
                self.advance_porosity,
                lambda: self.compute_stress_forces(include_pressure=False),
            )
        )

    def record_apply(self, accelerations: Any) -> None:
        """Add the deviatoric divergence into the caller's total acceleration.

        Apart from `record_stage` for the reason `SelfGravityRuntime.record_apply` is apart
        from its traversal: the hydrodynamic force kernel assigns the acceleration, so this
        has to follow it rather than precede it.
        """

        wp = self._require_wp()
        kernels = self._require_kernels()
        wp.launch(
            kernels.accumulate_solid_forces,
            dim=self.config.count,
            inputs=[self._arrays["accelerations"], accelerations],
            device=self.neighbors.device_context.device,
        )

    # -- readback ------------------------------------------------------------------------

    def color_source_channels(self) -> dict[str, Any]:
        """Authoritative raw Solid Color By handles, with no host readback.

        Material is always the persistent registry ID, never ``material_index``. Damage is
        absent unless at least one configured material actually activates a damage model;
        allocating the scratch array alone is not evidence that Damage exists physically.
        """

        if not self._arrays:
            raise DeviceRequirementError("SolidRuntime has no loaded state")
        channels = {"material_ids": self._arrays["material_ids"]}
        if self._damage_source_active:
            channels["damage"] = self._arrays["damage"]
        return channels

    def read(self, name: str) -> Any:
        """Explicit host readback of one array, for validation only.

        Named `read` rather than exposed as properties so an accidental call in a playback
        path is visible in review: the solid stages never read back on their own.
        """

        if name not in self._arrays:
            raise KeyError(f"unknown solid array {name!r}")
        return self._arrays[name].numpy()[: self.config.count]

    def set_scalar_field(self, name: str, values: Any) -> None:
        """Upload one scalar field, for the validation ladder's prescribed states.

        The ladder gates each constitutive stage against a state it dictates - a chosen
        pressure for the crush curve, a chosen stress for the yield surface - so that a
        failure names one model rather than the composition that produced the state. This is
        the only writer of that kind, and it is not on any solver path.
        """

        if name not in self._arrays:
            raise KeyError(f"unknown solid array {name!r}")
        wp = self._require_wp()
        np = __import__("numpy")
        scalar, _, state_dtype = _dtypes(wp, self.config.precision)
        padded = np.zeros(self.config.capacity, dtype=state_dtype)
        padded[: self.config.count] = np.asarray(values, dtype=state_dtype)
        self._arrays[name] = wp.array(
            padded, dtype=scalar, device=self.neighbors.device_context.device
        )

    def set_stress(self, diagonal: Any, offdiagonal: Any) -> None:
        """Upload a deviatoric stress field, for the validation ladder's prescribed states."""

        wp = self._require_wp()
        np = __import__("numpy")
        _, vector, state_dtype = _dtypes(wp, self.config.precision)
        device = self.neighbors.device_context.device
        for name, values in (
            ("stress_diagonal", diagonal),
            ("stress_offdiagonal", offdiagonal),
        ):
            padded = np.zeros((self.config.capacity, 3), dtype=state_dtype)
            padded[: self.config.count] = np.asarray(values, dtype=state_dtype)
            self._arrays[name] = wp.array(padded, dtype=vector, device=device)

    def set_velocity_gradient(self, gradients: Any) -> None:
        """Upload a prescribed velocity gradient, bypassing S1.

        The rigid-rotation gate is a statement about the stress rate, not about the SPH
        gradient estimator; feeding the analytic gradient keeps the two gates independent so
        a failure names one module.
        """

        wp = self._require_wp()
        np = __import__("numpy")
        _, _, state_dtype = _dtypes(wp, self.config.precision)
        matrix = _matrix_type(wp, self.config.precision)
        padded = np.zeros((self.config.capacity, 3, 3), dtype=state_dtype)
        padded[: self.config.count] = np.asarray(gradients, dtype=state_dtype)
        self._arrays["gradients"] = wp.array(
            padded, dtype=matrix, device=self.neighbors.device_context.device
        )

    def diagnostics(self) -> SolidDiagnostics:
        arrays = self._arrays
        if not arrays:
            raise DeviceRequirementError("SolidRuntime has no loaded state")
        count = self.config.count
        damage = arrays["damage"].numpy()[:count]
        distention = arrays["distention"].numpy()[:count]
        return SolidDiagnostics(
            active_count=count,
            material_count=len(self.config.registry),
            damage_max=float(damage.max()) if count else 0.0,
            distention_min=float(distention.min()) if count else 1.0,
            yielded_particles=int(arrays["yielded"].numpy()[0]),
            correction_failures=int(arrays["correction_failures"].numpy()[0]),
            stage_ms=self.last_stage_ms,
            graph_modules=self.config.graph_modules,
        )

    def _require_wp(self) -> Any:
        if self._wp is None:
            raise DeviceRequirementError("SolidRuntime has no loaded state")
        return self._wp

    def _require_kernels(self) -> Any:
        if self._kernels is None:
            raise DeviceRequirementError("SolidRuntime has no loaded state")
        return self._kernels


def strength_ids() -> dict[StrengthModel, int]:
    return dict(_STRENGTH_IDS)


def damage_ids() -> dict[DamageModel, int]:
    return dict(_DAMAGE_IDS)


def porosity_ids() -> dict[PorosityModel, int]:
    return dict(_POROSITY_IDS)


def parameter_slot_names() -> dict[str, dict[str, int]]:
    """Every schema key's device slot, so a contract test can assert none is unmapped."""

    return {
        "eos": dict(_EOS_SLOT_NAMES),
        "strength": dict(_STRENGTH_SLOT_NAMES),
        "damage": dict(_DAMAGE_SLOT_NAMES),
        "porosity": dict(_POROSITY_SLOT_NAMES),
    }


def solid_eos_ids() -> tuple[int, int]:
    """The device ids of the condensed-matter closures, for the disjointness contract."""

    return EOS_ID_MURNAGHAN, EOS_ID_TILLOTSON
