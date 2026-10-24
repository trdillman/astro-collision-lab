# ruff: noqa: UP018 - Warp refuses to accumulate into a bare literal inside a loop, so every
# kernel accumulator is declared as float(0.0) / int(0) (as in `giant_impact`).
"""Real-time rock impact: SpaceSim's pressure and fracture, with bond strength, on a two-rate clock.

The engine behind ADR 0193 (which takes up ADR 0185). What it keeps from SpaceSim's GPU path
(`docs/reference-analysis/SPACESIM_IMPACT_CONTRACT.md`):

- **Pressure** is implicit incompressible SPH (IISPH): a fixed, even number of relaxed Jacobi
  sweeps with omega 0.5, pressure clamped at zero so nothing pulls, and Monaghan artificial
  viscosity whose signal speed is ``0.2 h / dt`` (``AdvectVelocities``/``PredictDensities``/
  ``DijPj``/``PressureIteration``/``ComputeForce.comp``).
- **Reference bonds**: every particle's in-body neighbours inside the kernel support are
  captured once into a row of at most 63 (``InitializeStress.comp``'s ``initial_neighbors``).
- **Strength is clamped at yield, with no history** (``ComputeStress.comp``: ``S = s min(Y0/|s|,
  1)``): a bond whose strain passes the yield strain ``Y0 / E`` does not carry more, and it does
  not break there either - its reference length moves so the strain sits at the yield strain
  (ADR 0204), which is the permanent set a rock keeps after it has flowed.
- **Thermal softening** (``Y0(T) = Y_limit max(1 - T / T_melt, 0)``): the yield stress falls
  with the particle's thermal energy, so shock-heated rock goes fluid. That energy is the
  viscous dissipation and the elastic energy a bond loses when its reference length moves or it
  is cut (ADR 0204); it is kept, never cooled, and a ``heat`` that glows and cools for the
  Energy colour mode runs beside it.
- **Fracture** is geometric: a bond is gone for good once its length passes its break length
  (``ApplyStress.comp`` prunes beyond ``2 h``), or the body's critical strain on its current
  reference length, and bonds that cross a body's fault cells break at a small activation
  strain, SpaceSim's pre-fractured rubble.
- **Gravity** is direct tiled all-pairs, as in `giant_impact`.

What it changes, both measured on the shipped Asteroid Impact scene (ADR 0193 section 2):

- **Strength is bond-based, not corotated SPH stress.** SpaceSim's ``s = 2 mu eps`` over the
  reference kernel gradient, run on its own, grows without bound from rest under self-gravity
  at every step size tried (58 ms and 29 ms alike): it leans on the pressure, viscosity and
  viscosity-solve chain SpaceSim runs every substep. Each bond here is a peridynamic spring
  (micromodulus ``18 K / (pi delta^4)``, ``K = 5/3 mu``, horizon = the kernel support) damped
  along its own line: central, equal and opposite, blind to rigid motion, and with no
  zero-energy mode to ring up. A body at rest holds to 2e-4 of a spacing.
- **The clock has two rates.** An explicit spring is stable only while a step is shorter than
  about a wave's crossing of a particle spacing - a few hundred milliseconds for a 50 km rock,
  against a frame of tens of seconds - while the pressure solve, the neighbour search and
  gravity are what cost. So each frame runs **outer** steps sized by how far the fastest
  particle may travel (SpaceSim's ``collision_step_factor`` 0.2 of a smoothing length), each
  rebuilding the neighbour list and solving pressure once, and inside each, **inner** steps
  sized by the bond network's own stability limit, which touch only the bonds. Pressure and
  viscosity land as one velocity impulse per outer step (held as a force across the inner
  steps they turn into anti-damping); gravity uses symmetric kicks at cadence boundaries. With one
  inner step per outer step this is SpaceSim's own substep.
- **The pressure leaves a body's elastic compression to its bonds** (``pressure_tolerance``):
  a pressure solve that drives an intact, self-gravitating body back to its rest density every
  outer step fights the bonds and pumps energy in; above the tolerance - at contact, in debris
  piling up - it acts as SpaceSim's does.
- **The yield is per bond, and symmetric.** SpaceSim clamps a corotated stress tensor; a bond has
  one scalar strain, so the clamp is on it: past ``|strain| > eps_y`` the reference length moves
  to ``r / (1 +- eps_y)``, in tension and in compression alike, with the yield strain
  ``eps_y = Y0 / E`` (``E = 5/2 mu``) of the softer, hotter end. Both ends of a bond compute
  the same update from the same inputs, so the pair stays equal and opposite.
- **Break strains.** Past the critical strain on the bond's *current* reference length, or past
  ``2 h``, a bond is cut; across a fault cell the activation strain is the limit (a fault is a
  weakness). A material whose yield strain is above its critical strain is brittle (it breaks
  before it yields); below it, ductile (it flows until ``2 h`` pulls it apart).

Smaller deviations: the kernel gradient is the true derivative (the shader writes 1 where the
derivative has 3; contract item 7); a broken bond is removed by stable compaction, so a row is
always its reference row minus its broken bonds, in order; a checkpoint restores the rows bit
for bit from one bit mask of broken bonds, the reference lengths that have moved and each
particle's thermal energy; there is no viscosity solve; plastic work is shared equally by the two
ends of a bond as energy, not by mass.

A frame is one captured CUDA graph, replayed, exactly like `giant_impact`. Its outer steps are
sized on the device, one at a time (SpaceSim's auto_substep, ADR 0220): each step measures the
fastest particle and holds it to ``collision_step_factor`` smoothing lengths, never longer than
the planned step, and a CUDA conditional loop runs steps until the gravity interval is covered,
up to its share of ``max_outer_steps``. Each step is a function of the state at its start, so a
restored state replays it. Everything is float32 on the device; nothing crosses to the host
while playing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .giant_impact import _cell, _CellGrid, _span
from .model import DEFAULT_TREE_THETA, DeviceDescriptor, tree_accuracy_tolerance
from .spacesim_precision import finite_float32

#: Neighbour and bond row capacity. A row ends at the first -1, so 63 entries fit.
CAPACITY = 64
_CAP = wp.constant(CAPACITY)

#: Gravity tile, as in `giant_impact`.
TILE = 128
_TILE = wp.constant(TILE)

_INV_PI = wp.constant(0.3183098861837907)

#: The pressure solve's density error is summed in units of 2^-40 of rest density, each
#: particle's share capped at 4 rest densities, so two million particles cannot overflow the
#: 64-bit total (ADR 0219).
_ERROR_SCALE = wp.constant(1099511627776.0)
_ERROR_CLAMP = wp.constant(4.0)

#: Young's modulus over shear modulus at the bond model's fixed Poisson ratio of 1/4.
YOUNG_OVER_SHEAR = 2.5

#: Strain, on a bond's current reference length, past which it is cut: SpaceSim's per-body
#: ``critical_strain`` (``validation/fixtures/spacesim_impact/authored_collision.json``).
DEFAULT_CRITICAL_STRAIN = 0.1

#: The yield strain of a solid that has no yield limit: it never yields. A finite float32 so the
#: kernels need no special case.
NO_YIELD_STRAIN = 1.0e30

#: A bond's reference length never shrinks below this fraction of the particle spacing, however
#: far a particle is pushed through its partner.
MIN_BOND_LENGTH_SPACINGS = 0.1

#: The bond network's exact stability limit on an interior lattice site, as a fraction of the
#: longitudinal wave's crossing time of one spacing (measured on the shipped lattice: 0.49 s for
#: synthetic iron at a 3.83 km spacing). The compile-time step estimate uses it.
BOND_LIMIT_CROSSINGS = 0.70

#: How self-gravity can be summed (ADR 0218).
GRAVITY_MODELS = ("AUTO", "TREE", "DIRECT")

#: ``AUTO`` walks the Barnes-Hut tree from this many particles up and sums all pairs below it.
#: It lies between the measured 9,875-particle loss (tree pass 1.52 ms against 0.67 ms) and the
#: 49,847-particle win (2.86 ms against 4.41 ms) of the frame ledger; the ledger calibrates it.
#: Below it the tree also gives away momentum conservation (its forces are not pairwise equal
#: and opposite) for no speed in return.
TREE_GRAVITY_MIN_PARTICLES = 32768


def resolve_gravity_model(model: str, n: int, gravitational_constant: float, theta: float) -> str:
    """The summation a solver of ``n`` particles runs: ``"TREE"`` or ``"DIRECT"`` (ADR 0218).

    ``AUTO`` is the tree from :data:`TREE_GRAVITY_MIN_PARTICLES` up when it can run (G > 0) and
    its error is certified (``theta`` in (0, 0.8], ``model.tree_accuracy_tolerance``), else
    direct. An explicit ``TREE`` that cannot run is refused rather than quietly swapped.
    """

    if model not in GRAVITY_MODELS:
        raise ValueError(f"gravity_model must be one of {', '.join(GRAVITY_MODELS)}, not {model!r}")
    if model == "DIRECT":
        return "DIRECT"
    runnable = gravitational_constant > 0.0
    certified = tree_accuracy_tolerance(theta) is not None
    if model == "TREE":
        if not runnable:
            raise ValueError("gravity_model TREE needs a positive gravitational constant")
        if not certified:
            raise ValueError(
                "tree_theta must be in (0, 0.8], the range the tree's error is certified over"
            )
        return "TREE"
    return "TREE" if n >= TREE_GRAVITY_MIN_PARTICLES and runnable and certified else "DIRECT"


def _tree_config(count: int, gravitational_constant: float, theta: float, softening: float):
    """The Barnes-Hut configuration a solver of ``count`` particles walks, validated.

    Imported here rather than at module scope: the tree module is the N-body lane's and this
    engine must stay importable without it on a host that only runs the kernels' host twins.
    Raises ``ValueError`` where the tree cannot hold ``count`` particles (its node budget).
    """

    from .tree import BarnesHutConfig

    config = BarnesHutConfig(
        capacity=count,
        count=count,
        gravitational_constant=gravitational_constant,
        # Unused by a gravity pass; the tree's integrator never runs here.
        dt=1.0,
        theta=theta,
        softening=softening,
    )
    config.validate()
    return config


def _place_gravity_model(
    requested: str,
    resolved: str,
    is_cuda: bool,
    count: int,
    gravitational_constant: float,
    theta: float,
    softening: float,
) -> tuple[str, Any]:
    """The summation a solver runs on its device, and the tree configuration if it is the tree.

    ``AUTO`` only takes the tree where it can run: it is the N-body lane's CUDA kernel set, and
    its node budget caps the particle count. An explicit ``TREE`` that cannot is refused, never
    swapped (ADR 0218).
    """

    if resolved != "TREE":
        return resolved, None
    try:
        if not is_cuda:
            raise ValueError("gravity_model TREE needs a CUDA device")
        return "TREE", _tree_config(count, gravitational_constant, theta, softening)
    except ValueError:
        if requested == "TREE":
            raise
        return "DIRECT", None


# ----------------------------------------------------------------------------- kernel
# SpaceSim's cubic spline (``SphUtils.glsl``): h1 = 0.75 h, support 2 h1 = 1.5 h.


@wp.func
def _kernel(dr: wp.vec3, h: float):
    """``(grad W, W)`` at separation ``dr`` (``x_i - x_j``) for smoothing length ``h``."""
    h1 = 0.75 * h
    inv = 1.0 / h1
    dist = wp.length(dr)
    q = dist * inv
    p0 = wp.max(1.0 - q, 0.0)
    p1 = wp.max(2.0 - q, 0.0)
    inv3 = inv * inv * inv
    w = _INV_PI * inv3 * (0.25 * p1 * p1 * p1 - p0 * p0 * p0)
    g = wp.vec3(0.0, 0.0, 0.0)
    if dist > 0.0:
        # dW/dq = (1/pi) h1^-3 (-0.75 p1^2 + 3 p0^2), times dq/dr = 1/h1 along dr / |dr|.
        dwdq = _INV_PI * inv3 * (-0.75 * p1 * p1 + 3.0 * p0 * p0)
        g = dr * (dwdq * inv / dist)
    return g, w


def kernel_gradient(dr: np.ndarray, h: float) -> np.ndarray:
    """Host twin of `_kernel`'s gradient (float64), for separations ``dr`` (M x 3)."""
    dr = np.asarray(dr, dtype=np.float64)
    inv = 1.0 / (0.75 * h)
    dist = np.linalg.norm(dr, axis=1)
    q = dist * inv
    p0 = np.maximum(1.0 - q, 0.0)
    p1 = np.maximum(2.0 - q, 0.0)
    dwdq = (1.0 / math.pi) * inv**3 * (-0.75 * p1 * p1 + 3.0 * p0 * p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(dist > 0.0, dwdq * inv / dist, 0.0)
    return dr * scale[:, None]


def kernel_value(dr: np.ndarray, h: float) -> np.ndarray:
    """Host twin of `_kernel`'s value (float64)."""
    dist = np.linalg.norm(np.asarray(dr, dtype=np.float64), axis=1)
    inv = 1.0 / (0.75 * h)
    q = dist * inv
    p0 = np.maximum(1.0 - q, 0.0)
    p1 = np.maximum(2.0 - q, 0.0)
    return (1.0 / math.pi) * inv**3 * (0.25 * p1**3 - p0**3)


# ----------------------------------------------------------------------------- neighbours


@wp.kernel
def _build_neighbors(
    start: wp.array(dtype=int),
    end: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    inv: float,
    x: wp.array(dtype=wp.vec3),
    h: float,
    nbr: wp.array2d(dtype=int),
    overflow: wp.array(dtype=int),
):
    # Every particle inside the exact support, itself included (the density sum takes the self
    # term), in the grid's deterministic visiting order. Rows are slot-major ([k, i]) so a warp
    # reads one slot for 32 particles in one transaction.
    i = wp.tid()
    xi = x[i]
    reach = 1.5 * h
    reach2 = reach * reach
    count = int(0)
    x0, x1 = _span(xi[0], reach, inv)
    y0, y1 = _span(xi[1], reach, inv)
    z0, z1 = _span(xi[2], reach, inv)
    for cz in range(z0, z1 + 1):
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                cell = _cell(cx, cy, cz)
                for k in range(start[cell], end[cell]):
                    j = ids[k]
                    d = xi - x[j]
                    if wp.dot(d, d) < reach2:
                        if count < _CAP - 1:
                            nbr[count, i] = j
                            count += 1
                        else:
                            wp.atomic_add(overflow, 0, 1)
    nbr[count, i] = -1


# ----------------------------------------------------------------------------- pressure


@wp.kernel
def _density_dii(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    h: float,
    nbr: wp.array2d(dtype=int),
    a_other: wp.array(dtype=wp.vec3),
    clock: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    dii: wp.array(dtype=wp.vec3),
    v_adv: wp.array(dtype=wp.vec3),
):
    # AdvectVelocities.comp: the density, the diagonal displacement term, and the velocity the
    # non-pressure forces alone would give. ``clock[0]`` is the outer step (ADR 0220).
    i = wp.tid()
    dt = clock[0]
    xi = x[i]
    r = float(0.0)
    gsum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(_CAP):
        j = nbr[k, i]
        if j < 0:
            break
        g, w = _kernel(xi - x[j], h)
        r += m[j] * w
        gsum += m[j] * g
    rho[i] = r
    c = dt / r
    dii[i] = gsum * (-(c * c))
    v_adv[i] = v[i] + a_other[i] * dt


@wp.kernel
def _predict(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    h: float,
    nbr: wp.array2d(dtype=int),
    rho: wp.array(dtype=float),
    dii: wp.array(dtype=wp.vec3),
    v_adv: wp.array(dtype=wp.vec3),
    clock: wp.array(dtype=float),
    rho_adv: wp.array(dtype=float),
    aii: wp.array(dtype=float),
    p: wp.array(dtype=float),
):
    # PredictDensities.comp.
    i = wp.tid()
    dt = clock[0]
    xi = x[i]
    ri = rho[i]
    di = dii[i]
    vi = v_adv[i]
    wi = dt * dt * m[i] / (ri * ri)
    drho = float(0.0)
    a = float(0.0)
    for k in range(_CAP):
        j = nbr[k, i]
        if j < 0:
            break
        if j != i:
            g, w = _kernel(xi - x[j], h)
            drho += m[j] * wp.dot(vi - v_adv[j], g)
            a += m[j] * wp.dot(di - g * wi, g)
    rho_adv[i] = wp.max(ri + drho * dt, 0.01 * ri)
    if a == 0.0:
        a = -1.0
    aii[i] = a
    p[i] = 0.0


@wp.kernel
def _dij_pj(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    h: float,
    nbr: wp.array2d(dtype=int),
    rho: wp.array(dtype=float),
    p: wp.array(dtype=float),
    clock: wp.array(dtype=float),
    done: wp.array(dtype=int),
    out: wp.array(dtype=wp.vec3),
):
    # DijPj.comp. A converged solve skips its remaining sweeps (ADR 0219).
    i = wp.tid()
    if done[0] != 0:
        return
    dt = clock[0]
    xi = x[i]
    s = wp.vec3(0.0, 0.0, 0.0)
    for k in range(_CAP):
        j = nbr[k, i]
        if j < 0:
            break
        if j != i:
            g, w = _kernel(xi - x[j], h)
            rj = rho[j]
            s += g * (m[j] * p[j] / (rj * rj))
    out[i] = s * (-(dt * dt))


@wp.kernel
def _pressure_iteration(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    h: float,
    nbr: wp.array2d(dtype=int),
    rho: wp.array(dtype=float),
    rho0: wp.array(dtype=float),
    rho_adv: wp.array(dtype=float),
    dii: wp.array(dtype=wp.vec3),
    aii: wp.array(dtype=float),
    dij_pj: wp.array(dtype=wp.vec3),
    p_in: wp.array(dtype=float),
    clock: wp.array(dtype=float),
    omega: float,
    tolerance: float,
    done: wp.array(dtype=int),
    measure: int,
    p_out: wp.array(dtype=float),
    residual: wp.array(dtype=wp.int64),
    pushing: wp.array(dtype=int),
    sweeps: wp.array(dtype=int),
):
    # PressureIteration.comp: one relaxed Jacobi sweep, pressure never negative. ``tolerance``
    # (1 + the compression the pressure leaves to the bonds) scales the rest density. A
    # converged solve skips its remaining sweeps; a ``measure`` sweep also sums the density
    # error the pressure it read leaves, over the particles it pushes on (ADR 0219).
    i = wp.tid()
    if done[0] != 0:
        return
    if i == 0:
        sweeps[0] = sweeps[0] + 1
    dt = clock[0]
    xi = x[i]
    ri = rho[i]
    pi = p_in[i]
    dpi = dij_pj[i]
    vi = dt * dt * m[i] / (ri * ri)
    s = float(0.0)
    for k in range(_CAP):
        j = nbr[k, i]
        if j < 0:
            break
        if j != i:
            g, w = _kernel(xi - x[j], h)
            delta = dpi - dii[j] * p_in[j] - dij_pj[j] + g * (vi * pi)
            s += m[j] * wp.dot(delta, g)
    p2 = wp.max((rho0[i] * tolerance - rho_adv[i] - s) / aii[i], 0.0)
    if measure != 0 and (pi > 0.0 or p2 > 0.0):
        # |a_ii (p2 - p_i)| is how far the density at ``p_in`` misses the target. Summed in
        # fixed point, so the total, and the decision taken from it, is the same whatever
        # order the threads add in.
        error = wp.min(wp.abs(aii[i] * (p2 - pi)) / rho0[i], _ERROR_CLAMP)
        wp.atomic_add(residual, 0, wp.int64(wp.ceil(error * _ERROR_SCALE)))
        wp.atomic_add(pushing, 0, 1)
    p_out[i] = (1.0 - omega) * pi + omega * p2


@wp.kernel
def _pressure_open(
    done: wp.array(dtype=int),
    residual: wp.array(dtype=wp.int64),
    pushing: wp.array(dtype=int),
):
    done[0] = 0
    residual[0] = wp.int64(0)
    pushing[0] = 0


@wp.kernel
def _pressure_check(
    max_error: float,
    residual: wp.array(dtype=wp.int64),
    pushing: wp.array(dtype=int),
    done: wp.array(dtype=int),
):
    # Converged when the mean density error over the pushed-on particles is at most
    # ``max_error`` of rest density (none pushed on: nothing left to solve).
    bound = wp.float64(max_error) * wp.float64(_ERROR_SCALE) * wp.float64(pushing[0])
    if wp.float64(residual[0]) <= bound:
        done[0] = 1
    residual[0] = wp.int64(0)
    pushing[0] = 0


@wp.kernel
def _hydro_impulse(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    h: float,
    nbr: wp.array2d(dtype=int),
    rho: wp.array(dtype=float),
    p: wp.array(dtype=float),
    clock: wp.array(dtype=float),
    alpha_av: float,
    a_hydro: wp.array(dtype=wp.vec3),
    heat: wp.array(dtype=float),
    energy: wp.array(dtype=float),
):
    # ComputeForce.comp: the pressure force plus Monaghan viscosity with the dt-scaled signal
    # speed, delivered by the first inner step as one impulse. The viscosity's dissipation,
    # 0.5 sum m_j Pi_ij v_ij . grad W, is heat twice over (ADR 0204): the ``heat`` that glows
    # and cools (the Energy colour mode), and the persistent ``energy`` the yield softens with.
    # ``clock`` holds the step's dt and the heat decay over it (ADR 0220).
    i = wp.tid()
    dt = clock[0]
    heat_decay = clock[3]
    xi = x[i]
    vi = v[i]
    ri = rho[i]
    pi = p[i] / (ri * ri)
    a = wp.vec3(0.0, 0.0, 0.0)
    du = float(0.0)
    cs = (0.2 / dt) * h
    for k in range(_CAP):
        j = nbr[k, i]
        if j < 0:
            break
        if j != i:
            d = xi - x[j]
            g, w = _kernel(d, h)
            rj = rho[j]
            a -= g * (m[j] * (pi + p[j] / (rj * rj)))
            dv = vi - v[j]
            vr = wp.dot(dv, d)
            if vr < 0.0:
                mu = h * vr / (wp.dot(d, d) + 0.01 * h * h)
                visc = (-alpha_av * cs * mu) / (0.5 * (ri + rj))
                a -= g * (m[j] * visc)
                du += 0.5 * m[j] * visc * wp.dot(dv, g)
    a_hydro[i] = a
    gain = wp.max(du, 0.0) * dt
    heat[i] = heat[i] * heat_decay + gain
    energy[i] = energy[i] + gain


# ----------------------------------------------------------------------------- gravity


@wp.kernel
def _pack(
    x: wp.array(dtype=wp.vec3), gm: wp.array(dtype=float), n: int, pm: wp.array(dtype=wp.vec4)
):
    i = wp.tid()
    if i < n:
        q = x[i]
        pm[i] = wp.vec4(q[0], q[1], q[2], gm[i])
    else:
        pm[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _gravity(
    pm: wp.array(dtype=wp.vec4), n_tiles: int, eps2: float, n: int, acc: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    pi = pm[i]
    a = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_tiles):
        t = wp.tile_load(pm, shape=_TILE, offset=k * _TILE)
        for j in range(_TILE):
            q = t[j]
            d = wp.vec3(q[0] - pi[0], q[1] - pi[1], q[2] - pi[2])
            r2 = wp.dot(d, d) + eps2
            inv = 1.0 / wp.sqrt(r2)
            a += (q[3] * inv * inv * inv) * d
    if i < n:
        acc[i] = a


# ----------------------------------------------------------------------------- strength


@wp.func
def _yielded_length(r: float, length: float, yield_strain: float, min_length: float):
    # The reference length after the yield clamp (SpaceSim's S = s min(Y0 / |s|, 1), per bond):
    # unchanged while |r - L| / L stays within the yield strain, otherwise the length that
    # leaves exactly that strain, so the bond carries no more than the yield and keeps the
    # elongation (or the shortening) for good. A zero yield strain (fully softened) follows
    # the separation; a yield strain of 1 or more cannot shorten a bond.
    out = float(length)
    if r > length * (1.0 + yield_strain):
        out = r / (1.0 + yield_strain)
    elif r < length * (1.0 - yield_strain):
        out = wp.max(r / (1.0 - yield_strain), min_length)
    return out


@wp.func
def _elastic_energy(stiffness: float, r: float, length: float):
    # What a bond stores at separation ``r`` over reference length ``length``: its force is
    # k (r - L) / L, so the energy is k (r - L)^2 / (2 L).
    return 0.5 * stiffness * (r - length) * (r - length) / length


@wp.kernel
def _soften(
    energy: wp.array(dtype=float),
    y_cold: wp.array(dtype=float),
    inv_melt: wp.array(dtype=float),
    y_eff: wp.array(dtype=float),
):
    # Thermal softening, SpaceSim's Y0(T) = Y_limit max(1 - T / T_melt, 0), with the particle's
    # persistent thermal energy standing for the temperature above the ambient state and
    # ``inv_melt`` = 1 / (the energy that melts the material). A material with no melt energy
    # has ``inv_melt`` 0 and never softens.
    i = wp.tid()
    y_eff[i] = y_cold[i] * wp.max(1.0 - energy[i] * inv_melt[i], 0.0)


@wp.kernel
def _zero_float(slot: int, out: wp.array(dtype=float)):
    out[slot] = 0.0


@wp.kernel
def _max_speed(
    v: wp.array(dtype=wp.vec3), v_ref: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)
):
    # The fastest particle relative to the scene's mean velocity: what sizes the next outer
    # step (ADR 0220). A maximum is the same whatever order the threads land in, so the value is
    # a function of the velocities alone.
    i = wp.tid()
    wp.atomic_max(out, 0, wp.length(v[i] - v_ref[0]))


@wp.func
def _rail(vn: wp.vec3, v_ref: wp.vec3, vmax: float, clamped: wp.array(dtype=int)):
    # The speed rail, in the frame the steps are sized in (ADR 0224): a particle's speed
    # against the scene's mean velocity is held at ``vmax``, so a boost of the whole scene
    # changes nothing. With a mean of zero it is the absolute rail bit for bit.
    rel = vn - v_ref
    s = wp.length(rel)
    if s > vmax:
        wp.atomic_add(clamped, 0, 1)
        return v_ref + rel * (vmax / s)
    return vn


@wp.kernel
def _max_accel(a_hydro: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    # The largest pressure acceleration the last pressure solve left, kept beside the fastest
    # speed in ``out[1]``: the other half of what sizes the next outer step (ADR 0220). It is one
    # number of state, carried by a checkpoint, so a restore sizes its first step as the
    # uninterrupted run did.
    i = wp.tid()
    wp.atomic_max(out, 1, wp.length(a_hydro[i]))


@wp.kernel
def _bond_step(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    a_hydro: wp.array(dtype=wp.vec3),
    clock: wp.array(dtype=float),
    first: int,
    bonds: wp.array2d(dtype=int),
    bdata: wp.array2d(dtype=wp.vec4),
    count0: wp.array(dtype=int),
    y_eff: wp.array(dtype=float),
    cap2: float,
    min_length: float,
    heat: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    v_ref: wp.array(dtype=wp.vec3),
    v_next: wp.array(dtype=wp.vec3),
    damage: wp.array(dtype=float),
    x_next: wp.array(dtype=wp.vec3),
    clamped: wp.array(dtype=int),
):
    # Each bond is a spring along its own line, k (r - L) / L, damped along the line by its
    # relative speed; ``bdata[k, i]`` is ((1 + critical strain)^2, L, k, gamma), one load. L is
    # state: past the yield strain it moves (``_yielded_length``), so the strain never
    # exceeds the yield and the elongation stays, and the elastic energy that moving it removes
    # is dissipated as heat, half to each end (ADR 0204): added to the particle's glowing
    # ``heat`` and to its persistent ``energy``, each particle written by its own thread only.
    # A bond that is cut gives up what it still held in the same way, so the energy of the
    # springs, the motion and the heat adds up. A bond is cut when it is longer than
    # min(2 h, (1 + critical strain) L) - on the length it had when this step began. A pair's
    # force is exactly equal and opposite (one k, gamma and L per bond, updated identically at
    # both ends from the same r and the same yield strain, the smaller of the two ends'),
    # central, and blind to rigid motion. The force is summed before any bond is removed, and
    # the tests read only ``x``, so both ends of a bond see one length and sever it in the same
    # pass. Velocities read ``v`` and write ``v_next``, positions read ``x`` and write
    # ``x_next``: no thread reads a value another writes in this pass. The step's clock gives
    # the inner dt and the speed rail; the ``first`` inner step also delivers the pressure
    # impulse over the outer dt (ADR 0220).
    i = wp.tid()
    kick = float(0.0)
    if first != 0:
        kick = clock[0]
    dt = clock[1]
    vmax = clock[2]
    xi = x[i]
    vi = v[i]
    yi = y_eff[i]
    f = wp.vec3(0.0, 0.0, 0.0)
    count = int(0)
    broke = int(0)
    released = float(0.0)
    for k in range(_CAP):
        j = bonds[k, i]
        if j < 0:
            break
        count += 1
        b = bdata[k, i]
        length = b[1]
        d = x[j] - xi
        r2 = wp.dot(d, d)
        r = wp.sqrt(r2)
        moved = _yielded_length(r, length, wp.min(yi, y_eff[j]), min_length)
        if r2 > wp.min(cap2, b[0] * length * length):
            # Marked in place for the compaction below; -1 stays the only terminator. What the
            # bond held goes to heat, half to this end (the other end counts the same half).
            broke += 1
            bonds[k, i] = -2 - j
            released += 0.5 * _elastic_energy(b[2], r, length)
        elif moved != length:
            bdata[k, i] = wp.vec4(b[0], moved, b[2], b[3])
            # The elastic energy that moving the reference length took out, at the same r.
            released += 0.5 * wp.max(
                _elastic_energy(b[2], r, length) - _elastic_energy(b[2], r, moved), 0.0
            )
        if r > 0.0:
            e = d / r
            f += e * (b[2] * (r - moved) / moved + b[3] * wp.dot(v[j] - vi, e))
    if released > 0.0:
        warmed = released / m[i]
        heat[i] = heat[i] + warmed
        energy[i] = energy[i] + warmed
    if broke > 0:
        # Stable compaction: a row stays its reference row minus its broken bonds, in order.
        kept = int(0)
        for k in range(count):
            j = bonds[k, i]
            if j >= 0:
                bonds[kept, i] = j
                bdata[kept, i] = bdata[k, i]
                kept += 1
        # Blank every slot the compaction freed, not only the first: a row's slots past its
        # terminator are then always -1, so no reader can mistake a stale partner for a bond.
        for k in range(kept, count):
            bonds[k, i] = -1
        count = kept
    c0 = count0[i]
    if c0 > 0:
        damage[i] = 1.0 - float(count) / float(c0)
    vn = _rail(vi + a_hydro[i] * kick + f / m[i] * dt, v_ref[0], vmax, clamped)
    v_next[i] = vn
    x_next[i] = xi + vn * dt


@wp.kernel
def _gravity_kick(
    v: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    dt: float,
    vmax: float,
    v_ref: wp.array(dtype=wp.vec3),
    clamped: wp.array(dtype=int),
):
    # The same speed rail the bond step applies: a closing half-kick lands after the last bond
    # step, so without it a frame could end above the cap the rest of the integrator holds.
    i = wp.tid()
    v[i] = _rail(v[i] + a[i] * dt, v_ref[0], vmax, clamped)


# ----------------------------------------------------------------------------- step clock


@wp.kernel
def _interval_open(
    span: wp.float64,
    budget: int,
    remaining: wp.array(dtype=wp.float64),
    steps_left: wp.array(dtype=int),
    more: wp.array(dtype=wp.int32),
):
    # A gravity interval of ``span`` time units, to be covered in at most ``budget`` outer steps.
    remaining[0] = span
    steps_left[0] = budget
    more[0] = wp.int32(1)


@wp.kernel
def _plan_step(
    speed: wp.array(dtype=float),
    reach: wp.float64,
    dt_floor: wp.float64,
    inner: int,
    cap_length: wp.float64,
    inv_half_life: wp.float64,
    remaining: wp.array(dtype=wp.float64),
    steps_left: wp.array(dtype=int),
    clock: wp.array(dtype=float),
    tally: wp.array(dtype=int),
    more: wp.array(dtype=wp.int32),
):
    # SpaceSim's auto_substep, one outer step at a time (ADR 0220). The step the fastest
    # particle allows (``reach`` = collision_step_factor h), never longer than the plan's, sets
    # how many steps the rest of the interval needs; the rest is split evenly over them, so the
    # interval ends exactly on its boundary. The split keeps what is left within ``steps_left``
    # plan steps, so no step is ever longer than the plan's (the bond step stays inside its
    # limit). Out of budget, a step outruns the fastest particle's reach: ``tally[4]`` counts
    # those steps and the speed rail (counted by ``clamped``) binds where it must, never silently.
    # The reach bounds the speed the step will reach, not only the speed it starts at: the
    # pressure impulse lands on the first bond step and is held for the whole step, so the
    # fastest particle moves about (s + a dt) dt. With the last solve's largest pressure
    # acceleration ``a`` the step is the root of (s + a dt) dt = reach, written in the form that
    # stays exact as a or s goes to zero.
    left = remaining[0]
    dt_step = dt_floor
    s = wp.float64(speed[0])
    a = wp.float64(speed[1])
    if not wp.isfinite(s):
        s = wp.float64(0.0)
    if not wp.isfinite(a):
        a = wp.float64(0.0)
    if s > wp.float64(0.0) or a > wp.float64(0.0):
        root = s + wp.sqrt(s * s + wp.float64(4.0) * a * reach)
        dt_step = wp.min(wp.float64(2.0) * reach / root, dt_floor)
    n = int(wp.ceil(left / dt_step - wp.float64(1.0e-9)))
    if n > steps_left[0]:
        n = steps_left[0]
        tally[4] = tally[4] + 1
    n = wp.max(n, 1)
    dt = left / wp.float64(n)
    clock[0] = float(dt)
    clock[1] = float(dt / wp.float64(inner))
    clock[2] = float(cap_length / dt)
    clock[3] = float(wp.pow(wp.float64(0.5), dt * inv_half_life))
    if n == 1:
        remaining[0] = wp.float64(0.0)
        more[0] = wp.int32(0)
    else:
        remaining[0] = left - dt
        more[0] = wp.int32(1)
    steps_left[0] = steps_left[0] - 1
    tally[0] = tally[0] + 1


@wp.kernel
def _tally_open(tally: wp.array(dtype=int)):
    tally[0] = 0


@wp.kernel
def _tally_close(planned: int, fixed: int, tally: wp.array(dtype=int), hist: wp.array(dtype=int)):
    # ``tally`` = (this frame's steps, the most any frame took, frames above the plan, all
    # steps, steps cut short by the budget); ``hist[k]`` counts the frames that took k steps. A
    # fixed plan counts its own.
    steps = tally[0]
    if fixed > 0:
        steps = fixed
        tally[0] = fixed
    if steps < hist.shape[0]:
        hist[steps] = hist[steps] + 1
    tally[1] = wp.max(tally[1], steps)
    if steps > planned:
        tally[2] = tally[2] + 1
    tally[3] = tally[3] + steps


@wp.kernel
def _add(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def _bond_acceleration(
    x: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    bonds: wp.array2d(dtype=int),
    bdata: wp.array2d(dtype=wp.vec4),
    y_eff: wp.array(dtype=float),
    out: wp.array(dtype=wp.vec3),
):
    # The springs' acceleration alone (no damping), for the pressure solve's advected velocity.
    # The strain is clamped at the yield strain, the strain the bond step leaves it at.
    i = wp.tid()
    xi = x[i]
    yi = y_eff[i]
    f = wp.vec3(0.0, 0.0, 0.0)
    for k in range(_CAP):
        j = bonds[k, i]
        if j < 0:
            break
        b = bdata[k, i]
        d = x[j] - xi
        r = wp.length(d)
        if r > 0.0:
            limit = wp.min(yi, y_eff[j])
            f += (d / r) * (b[2] * wp.clamp((r - b[1]) / b[1], -limit, limit))
    out[i] = f / m[i]


@wp.kernel
def _damp(v: wp.array(dtype=wp.vec3), f: float):
    i = wp.tid()
    v[i] = v[i] * f


@wp.kernel
def _flag_nonfinite(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), flag: wp.array(dtype=int)
):
    i = wp.tid()
    p = x[i]
    q = v[i]
    if not wp.isfinite(p[0] + p[1] + p[2] + q[0] + q[1] + q[2]):
        flag[0] = 1


@wp.kernel
def _raise_flag(src: wp.array(dtype=int), flag: wp.array(dtype=int)):
    """Raise ``flag`` if ``src`` is raised: folds the tree's sticky overflow flag into ``bad``."""
    if src[0] != 0:
        flag[0] = 1


# ----------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class SpaceSimImpactSettings:
    """Solver knobs. Lengths are multiples of the particle spacing, so a setting means the same
    thing at every particle count."""

    frame_dt: float = 1.0
    #: Smoothing length in particle spacings; the kernel's support (and the bond horizon) is
    #: 1.5 of it.
    smoothing: float = 1.5
    #: SpaceSim's iisph.max_iterations: the GPU loop runs at most this many sweeps, rounded up
    #: to even.
    pressure_iterations: int = 6
    #: SpaceSim's iisph.min_iterations, rounded up to even and never below 2: from here, after
    #: every even sweep short of the maximum, the solve stops once it has converged (ADR 0219).
    #: At or above ``pressure_iterations`` every step runs the full count, as SpaceSim's GPU
    #: dispatch does.
    pressure_min_iterations: int = 2
    #: Converged: the mean density error the pressure leaves, over the particles it pushes on,
    #: is at most this fraction of rest density. SPlisHSPlasH's default maximum density error
    #: (0.01 %); 0 stops early only when nothing is left to solve.
    pressure_max_error: float = 1.0e-4
    #: Hard-coded on SpaceSim's GPU path whatever the setting says.
    omega: float = 0.5
    #: Compression the pressure solve leaves to the bonds, as a fraction of rest density.
    pressure_tolerance: float = 0.01
    alpha_av: float = 0.3
    #: SpaceSim's collision_step_factor: the fastest particle moves at most this many smoothing
    #: lengths per outer step.
    collision_step_factor: float = 0.2
    #: Speed the planned outer step is sized for, as a multiple of the characteristic speed.
    ejecta_margin: float = 1.5
    #: Speeds against the scene's mean velocity are clamped at this many smoothing lengths per
    #: outer step (a safety rail; how often it binds is counted). It is the frame the steps are
    #: sized in, so a scene moving as a whole is railed as if at rest (ADR 0224).
    speed_cap_lengths: float = 1.0
    #: Gravitational softening in particle spacings.
    softening: float = 1.0
    #: How self-gravity is summed (ADR 0218): ``DIRECT`` is the exact tiled all-pairs sum,
    #: ``TREE`` walks the Barnes-Hut tree the N-body solver ships at ``tree_theta``, and ``AUTO``
    #: (the default) takes the tree from ``TREE_GRAVITY_MIN_PARTICLES`` particles up, where the
    #: O(N^2) pass stops being worth its exactness, and the exact sum below.
    gravity_model: str = "AUTO"
    #: The tree's opening angle (the plan layer's shipped value); it must lie in the certified
    #: range (0, 0.8] for the tree to run.
    tree_theta: float = DEFAULT_TREE_THETA
    #: Gravity intervals per frame, spread over outer steps; each has two central half-kicks.
    #: Adjacent intervals share their endpoint force evaluation (ADR 0199).
    gravity_passes_per_frame: int = 1
    #: Each bond's damping as a fraction of its own critical damping.
    bond_damping: float = 0.05
    #: The inner step as a fraction of the bond network's exact static stability limit. 0.4:
    #: the impact's outcome stops changing at or below it (measured: 0.42 converged with 0.29
    #: and 0.20; 0.62 dispersed the whole target).
    bond_dt_safety: float = 0.4
    #: Shock heat halves over this many frames.
    heat_half_life_frames: float = 4.0
    settle_frames: int = 0
    settle_damping: float = 0.9
    #: Fixed step counts override the sizing rules (0 sizes them).
    outer_steps: int = 0
    inner_steps: int = 0
    #: Adaptive outer stepping (SpaceSim's auto_substep, ADR 0220): the most outer steps a frame
    #: may take. Above the planned count, each outer step is sized on the device from the
    #: fastest particle at its own start, never longer than the planned step. 0 (the default),
    #: or a value at or below the planned count, keeps every frame at the planned count; a fixed
    #: ``outer_steps`` does too.
    max_outer_steps: int = 0


@dataclass(frozen=True)
class SpaceSimMaterial:
    """One material in code units: shear modulus, yield limit, reference density, and the two
    numbers that say how it fails.

    A bond yields at the material's yield strain ``Y / E``, ``E = 5/2 mu``: past it the bond
    carries no more and its reference length moves (permanent set). It is cut once it is
    stretched past ``critical_strain`` of its current length (default
    :data:`DEFAULT_CRITICAL_STRAIN`, SpaceSim's), or past ``2 h``; a material whose yield strain
    is above its critical strain is brittle. ``melt_energy`` is the specific shock heat at which
    the yield stress has softened to zero (``0``: it never softens). A material with no yield
    limit never yields; one with no shear modulus is a fluid: its particles form no bonds and are
    held by the pressure solve and gravity alone."""

    shear_modulus: float
    yield_limit: float
    reference_density: float
    critical_strain: float | None = None
    melt_energy: float = 0.0

    @property
    def is_solid(self) -> bool:
        return self.shear_modulus > 0.0

    def validate(self) -> None:
        density = float(self.reference_density)
        if not (math.isfinite(density) and density > 0.0):
            raise ValueError("reference_density must be positive and finite")
        for name in ("shear_modulus", "yield_limit", "melt_energy"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and not negative")
        if self.is_solid and self.critical_strain is None and not self.yield_limit > 0.0:
            raise ValueError("a solid needs a yield limit or a critical strain to break at")
        if self.critical_strain is not None:
            value = float(self.critical_strain)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError("critical_strain must be positive and finite")

    @property
    def yield_strain(self) -> float:
        """``Y / E`` at the cold state; :data:`NO_YIELD_STRAIN` for a solid with no yield
        limit (it never yields) and 0 for a fluid (it has no bonds to yield)."""
        if not self.is_solid:
            return 0.0
        if not self.yield_limit > 0.0:
            return NO_YIELD_STRAIN
        return self.yield_limit / (YOUNG_OVER_SHEAR * self.shear_modulus)

    @property
    def break_strain(self) -> float:
        """The strain, on a bond's current reference length, it is cut at."""
        if self.critical_strain is not None:
            return float(self.critical_strain)
        if not self.is_solid:
            return 0.0
        return DEFAULT_CRITICAL_STRAIN


def wave_speed(material: SpaceSimMaterial) -> float:
    """The bond network's longitudinal wave speed in this material: springs at the peridynamic
    micromodulus fix Poisson's ratio at 1/4, so the longitudinal modulus is ``3 mu``. A fluid
    carries none through bonds."""
    return math.sqrt(3.0 * material.shear_modulus / material.reference_density)


def step_counts(
    *,
    frame_dt: float,
    spacing: float,
    characteristic_speed: float,
    max_wave_speed: float,
    settings: SpaceSimImpactSettings,
) -> tuple[int, int]:
    """``(outer, inner)`` steps per frame: inner is per outer step and always even, so a frame
    starts and ends in the same position and velocity buffers.

    Pure and Warp-free, so compile can size a frame before a device exists. The inner count
    here estimates the bond network's limit from the fastest wave; the solver measures the
    real limit on its own bonds at load and cuts the inner step from that (ADR 0193)."""
    h = settings.smoothing * spacing
    if settings.outer_steps:
        outer = int(settings.outer_steps)
    else:
        v = settings.ejecta_margin * max(characteristic_speed, 0.0)
        outer = max(1, math.ceil(frame_dt * v / (settings.collision_step_factor * h) - 1.0e-9))
    if settings.inner_steps:
        inner = int(settings.inner_steps)
    elif max_wave_speed <= 0.0:
        inner = 1  # all fluid: no bonds to resolve
    else:
        limit = BOND_LIMIT_CROSSINGS * spacing / max_wave_speed
        inner = max(1, math.ceil((frame_dt / outer) / (settings.bond_dt_safety * limit) - 1.0e-9))
    return outer, inner + inner % 2


@dataclass(frozen=True)
class GravityInterval:
    """One gravity interval of a frame: ``planned`` outer steps of the plan's length, so
    ``span`` time units, taken in at most ``budget`` steps when the outer step is adaptive."""

    planned: int
    span: float
    budget: int


def gravity_intervals(
    *, planned: int, dt_outer: float, passes: int, ceiling: int
) -> tuple[GravityInterval, ...]:
    """A frame's gravity intervals (ADR 0199): ``passes`` per frame, spread over the planned
    outer steps, each closed by one force evaluation. Under an adaptive ceiling above the plan
    (ADR 0220) each interval may take up to its share of ``ceiling``, never fewer than its
    planned steps. Pure and Warp-free."""
    planned = max(1, int(planned))
    every = max(1, math.ceil(planned / max(1, int(passes))))
    ceiling = max(int(ceiling), planned)
    out = []
    for first in range(0, planned, every):
        steps = min(every, planned - first)
        budget = max(steps, (ceiling * steps) // planned)
        out.append(GravityInterval(planned=steps, span=steps * dt_outer, budget=budget))
    return tuple(out)


@dataclass(frozen=True)
class FramePlan:
    """The planned frame's step counts and the numbers that follow from them: ``dt_*`` are code
    time, ``vmax`` the speed rail of that outer step and ``heat_decay`` the glow's factor per
    outer step (the same half-life in frames at every count). Under an adaptive ceiling the
    device sizes each outer step itself (ADR 0220); this is then the floor it never goes
    below, the longest step it takes."""

    outer: int
    inner: int
    dt_outer: float
    dt_inner: float
    vmax: float
    heat_decay: float


def morton_order(points: np.ndarray) -> np.ndarray:
    """The order that lays ``points`` (N x 3) along a Z-order curve: 10 bits a axis over their
    common bounding cube, ties kept in input order. Particles near each other in space end up
    near each other in memory, so a bond's or a neighbour's partner is a nearby read."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape[0] < 2:
        return np.arange(p.shape[0])
    low = p.min(axis=0)
    span = float((p.max(axis=0) - low).max()) or 1.0
    q = np.clip(((p - low) / span * 1023.0).astype(np.int64), 0, 1023)

    def spread(v: np.ndarray) -> np.ndarray:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        return (v | (v << 2)) & 0x09249249

    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
    return np.argsort(code, kind="stable")


def fault_cells(points: np.ndarray, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Which fault cell each point of one body is in: the nearest of ``cells`` seeds drawn
    from the body's own points. 0 or 1 cell means the body is one piece."""
    n = points.shape[0]
    if cells <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)
    seeds = points[rng.choice(n, size=min(cells, n), replace=False)]
    best = np.full(n, np.inf)
    owner = np.zeros(n, dtype=np.int32)
    for k, seed in enumerate(seeds):
        d2 = ((points - seed) ** 2).sum(-1)
        closer = d2 < best
        best[closer] = d2[closer]
        owner[closer] = k
    return owner


class SpaceSimImpactSolver:
    """Fixed-count particle state for colliding, fracturing, self-gravitating rock bodies.

    ``positions``/``velocities`` (N x 3), ``masses`` (N), ``body_ids`` (N, int) and
    ``material_ids`` (N, int indexing ``materials``) describe the bodies at their authored
    centres and velocities. ``fault_cell_ids`` (N, int; -1 for none) names each particle's
    fault cell inside its body, and ``fault_strain`` (one per body id, or one for all) is the
    strain a bond crossing two cells breaks at. ``characteristic_speed`` sizes the outer step.

    Any number of solvers can live in one process: each owns its buffers, its neighbour grid
    and its CUDA stream.
    """

    def __init__(
        self,
        positions,
        velocities,
        masses,
        body_ids,
        material_ids,
        materials: tuple[SpaceSimMaterial, ...],
        *,
        spacing: float,
        characteristic_speed: float,
        gravitational_constant: float = 1.0,
        fault_cell_ids=None,
        fault_strain=0.003,
        settings: SpaceSimImpactSettings | None = None,
        device=None,
        use_graph: bool = True,
    ):
        self.settings = s = settings or SpaceSimImpactSettings()
        if s.pressure_iterations < 0:
            raise ValueError("pressure_iterations cannot be negative")
        if s.pressure_min_iterations < 0:
            raise ValueError("pressure_min_iterations cannot be negative")
        if not (math.isfinite(s.pressure_max_error) and s.pressure_max_error >= 0.0):
            raise ValueError("pressure_max_error must be finite and not negative")
        if not materials:
            raise ValueError("at least one material is required")
        for material in materials:
            material.validate()
        self.n = int(len(positions))
        if self.n < 1:
            raise ValueError("a scene needs at least one particle")
        self.n_pad = int(math.ceil(self.n / TILE) * TILE)
        self.spacing = float(spacing)
        if not (math.isfinite(self.spacing) and self.spacing > 0.0):
            raise ValueError("spacing must be positive and finite")
        self.h = s.smoothing * self.spacing
        self.eps2 = (s.softening * self.spacing) ** 2
        #: The summation this solver runs, ``"TREE"`` or ``"DIRECT"`` (ADR 0218).
        self.gravity_model_resolved = resolve_gravity_model(
            s.gravity_model, self.n, float(gravitational_constant), float(s.tree_theta)
        )
        self.materials = tuple(materials)
        self.max_wave_speed = max(wave_speed(mat) for mat in materials)
        self.characteristic_speed = float(characteristic_speed)
        self.outer_steps, self.estimated_inner_steps = step_counts(
            frame_dt=s.frame_dt,
            spacing=self.spacing,
            characteristic_speed=self.characteristic_speed,
            max_wave_speed=self.max_wave_speed,
            settings=s,
        )
        self.inner_steps = self.estimated_inner_steps
        self.dt_outer = s.frame_dt / self.outer_steps
        self.dt_inner = self.dt_outer / self.inner_steps
        self.vmax = s.speed_cap_lengths * self.h / self.dt_outer
        self.iterations = int(s.pressure_iterations) + int(s.pressure_iterations) % 2
        least = int(s.pressure_min_iterations) + int(s.pressure_min_iterations) % 2
        self.min_iterations = min(max(2, least), self.iterations)
        self.heat_decay = 0.5 ** (1.0 / max(s.heat_half_life_frames * self.outer_steps, 1.0e-9))
        self.frame = 0

        masses = np.asarray(masses, dtype=np.float64)
        body_ids = np.asarray(body_ids, dtype=np.int32)
        material_ids = np.asarray(material_ids, dtype=np.int64)
        if masses.shape != (self.n,) or body_ids.shape != (self.n,):
            raise ValueError("masses and body_ids need one entry per particle")
        if material_ids.shape != (self.n,):
            raise ValueError("material_ids needs one entry per particle")
        if material_ids.min() < 0 or material_ids.max() >= len(materials):
            raise ValueError("material_ids must index materials")
        if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
            raise ValueError("masses must be positive and finite")
        self._m_host = masses
        self._body_ids = body_ids
        self._material_ids = material_ids
        self._authored_x = np.asarray(positions, dtype=np.float64)
        self._authored_v = np.asarray(velocities, dtype=np.float64)
        if self._authored_x.shape != (self.n, 3) or self._authored_v.shape != (self.n, 3):
            raise ValueError("positions and velocities need one row of 3 per particle")
        if not (np.all(np.isfinite(self._authored_x)) and np.all(np.isfinite(self._authored_v))):
            raise ValueError("positions and velocities must be finite")
        staged_x = finite_float32(self._authored_x, "positions")
        staged_v = finite_float32(self._authored_v, "velocities")
        staged_m = finite_float32(masses, "masses", positive=True)
        with np.errstate(over="ignore", invalid="ignore"):
            staged_gm = finite_float32(masses * gravitational_constant, "gravity masses")
        faults = (
            np.full(self.n, -1, dtype=np.int32)
            if fault_cell_ids is None
            else np.asarray(fault_cell_ids, dtype=np.int32)
        )
        if faults.shape != (self.n,):
            raise ValueError("fault_cell_ids needs one entry per particle")
        self._fault_ids = faults
        strain = np.asarray(fault_strain, dtype=np.float64)
        if strain.ndim == 0:
            strain = np.full(int(body_ids.max()) + 1, float(strain))
        if strain.ndim != 1 or strain.shape[0] <= int(body_ids.max()):
            raise ValueError("fault_strain needs one value per body id")
        if not np.all(np.isfinite(strain)) or np.any(strain <= 0.0):
            raise ValueError("fault_strain must be positive and finite")
        self._fault_strain = strain
        self.device = wp.get_device(device)
        self.gravity_model_resolved, tree_config = _place_gravity_model(
            s.gravity_model,
            self.gravity_model_resolved,
            bool(self.device.is_cuda),
            self.n,
            float(gravitational_constant),
            float(s.tree_theta),
            math.sqrt(float(self.eps2)),
        )
        # The solver keeps its particles in Z-order (``morton_order``): a bond's and a
        # neighbour's partner is then a nearby read, which the bond pass - most of a frame -
        # is bound by (measured on the shipped Asteroid Impact scene: 11.5 to 9.0 ms a frame).
        # Every public interface stays in the caller's order: the constructor's inputs,
        # ``state``, ``broken_bond_mask``, ``checkpoint``, ``restore`` and
        # ``reference_bond_counts``. The device arrays (``x``, ``v``, ``heat``, ``damage``) are in
        # the solver's order; ``internal_index`` maps a caller index to it.
        order = morton_order(self._authored_x)
        rank = np.empty_like(order)
        rank[order] = np.arange(self.n)
        self._order = order
        self._rank = rank
        masses = masses[order]
        body_ids = body_ids[order]
        material_ids = material_ids[order]
        self._m_host = masses
        self._body_ids = body_ids
        self._material_ids = material_ids
        self._authored_x = self._authored_x[order]
        self._authored_v = self._authored_v[order]
        self._fault_ids = faults[order]

        dev = self.device
        self.stream = wp.Stream(dev) if dev.is_cuda else None

        def arr(a, t):
            return wp.array(np.asarray(a, dtype=np.float32), dtype=t, device=dev)

        self.x = arr(staged_x[order], wp.vec3)
        self.x_alt = wp.zeros_like(self.x)
        self.v = arr(staged_v[order], wp.vec3)
        self.v_alt = wp.zeros_like(self.v)
        self.m = arr(staged_m[order], float)
        self.gm = arr(staged_gm[order], float)
        self.rho = wp.zeros(self.n, dtype=float, device=dev)
        self.rho0 = wp.zeros(self.n, dtype=float, device=dev)
        self.rho_adv = wp.zeros(self.n, dtype=float, device=dev)
        self.aii = wp.zeros(self.n, dtype=float, device=dev)
        self.p = wp.zeros(self.n, dtype=float, device=dev)
        self.p_alt = wp.zeros(self.n, dtype=float, device=dev)
        self.dii = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.dij_pj = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.v_adv = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.a_other = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.a_bond = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.a_grav = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        # Whether ``a_grav`` was computed at the current positions (``_ensure_gravity``).
        self._gravity_fresh = False
        self.a_hydro = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.heat = wp.zeros(self.n, dtype=float, device=dev)
        self.damage = wp.zeros(self.n, dtype=float, device=dev)
        # The thermal energy a particle keeps (ADR 0204): the viscous dissipation and the plastic
        # work of its bonds, J/kg in code units, never cooling. ``heat`` is the glow of the same
        # energy for the Energy colour mode and cools over a few frames; this is what melts.
        self.energy = wp.zeros(self.n, dtype=float, device=dev)
        # What sizes each adaptive outer step on the device (ADR 0220): ``[0]`` the fastest
        # particle now (``_max_speed``, measured before every step) and ``[1]`` the largest
        # pressure acceleration the last pressure solve left (``_max_accel``, kept as state).
        self.speed_meas = wp.zeros(2, dtype=float, device=dev)
        total_mass = float(masses.sum())
        self._v_ref = tuple(
            float(v) for v in (self._authored_v * masses[:, None]).sum(axis=0) / total_mass
        )
        # The same mean on the device, for the step sizing and the speed rail (ADR 0224). It is
        # an array, not a launch constant, so a settle (which runs at rest) can point both at
        # zero without a captured frame keeping the value it was recorded with.
        self._v_ref_dev = wp.array([self._v_ref], dtype=wp.vec3, device=dev)
        # Strength: each particle's cold yield strain, 1 / the energy that melts it, and the
        # yield strain it has now (``_soften``). A bond yields at the smaller of its two ends'.
        # Nothing is hot at load, so the two start equal.
        cold = np.asarray([q.yield_strain for q in materials], dtype=np.float64)
        melts = np.asarray(
            [
                1.0 / q.melt_energy
                if q.is_solid and q.yield_limit > 0.0 and q.melt_energy > 0.0
                else 0.0
                for q in materials
            ],
            dtype=np.float64,
        )
        self.y_cold = arr(cold[material_ids], float)
        self.inv_melt = arr(melts[material_ids], float)
        self.y_eff = arr(cold[material_ids], float)
        self.cap2 = 4.0 * self.h * self.h
        self.min_length = MIN_BOND_LENGTH_SPACINGS * self.spacing
        # Rows are slot-major (CAPACITY x N): slot k of particle i is [k, i].
        self.nbr = wp.full((CAPACITY, self.n), -1, dtype=int, device=dev)
        self.bonds = wp.full((CAPACITY, self.n), -1, dtype=int, device=dev)
        self.bdata = wp.zeros((CAPACITY, self.n), dtype=wp.vec4, device=dev)
        self.count0 = wp.zeros(self.n, dtype=int, device=dev)
        self.pm = wp.zeros(self.n_pad, dtype=wp.vec4, device=dev)
        self.overflow = wp.zeros(1, dtype=int, device=dev)
        self.clamped = wp.zeros(1, dtype=int, device=dev)
        self.bad = wp.zeros(1, dtype=int, device=dev)
        # ADR 0218: the Barnes-Hut tree the N-body solver ships, as this solver's gravity
        # service. It is built on this solver's slot order (the masses it loads are already
        # Z-ordered), so a pass copies ``x`` in and ``a_grav`` out with no permutation. ``None``
        # under DIRECT. ``_tree_bad`` is the tree's sticky overflow flag: every pass folds the
        # tree's overflow state into it on the device, and ``nonfinite`` merges it into ``bad``.
        self._tree = None
        self._tree_bad = None
        if self.gravity_model_resolved == "TREE":
            self._tree_bad = wp.zeros(1, dtype=int, device=dev)
            self._tree = self._build_tree(
                tree_config,
                positions=np.asarray(staged_x[order], dtype=np.float32),
                masses=np.asarray(staged_m[order], dtype=np.float32),
            )
        # The pressure solve's convergence flag and the error it is decided from (ADR 0219), and
        # a count of the sweeps it has run (a counter, not state: a restore does not rewind it).
        self._p_done = wp.zeros(1, dtype=int, device=dev)
        self._p_residual = wp.zeros(1, dtype=wp.int64, device=dev)
        self._p_pushing = wp.zeros(1, dtype=int, device=dev)
        self._p_sweeps = wp.zeros(1, dtype=int, device=dev)
        # The step clock the step kernels read (ADR 0220): (outer dt, inner dt, speed rail, glow
        # decay a step). A unit clock serves the density passes that only need dt = 1.
        self._unit_clock = wp.array(np.ones(4, dtype=np.float32), dtype=float, device=dev)
        self.grid = _CellGrid(self.n, 1.5 * self.h, dev)
        self._use_graph = bool(use_graph and dev.is_cuda)
        self._bond_rows0: np.ndarray | None = None
        self._bond_data0: np.ndarray | None = None
        with self._on_stream():
            self._capture_reference()
        if not s.inner_steps and math.isfinite(self.bond_step_limit):
            # The bond network's own stability limit, measured on the bonds just captured, is
            # what the inner step is cut from; `step_counts` is compile's estimate of it.
            inner = max(
                1,
                math.ceil(self.dt_outer / (s.bond_dt_safety * self.bond_step_limit) - 1.0e-9),
            )
            self.inner_steps = inner + inner % 2
            self.dt_inner = self.dt_outer / self.inner_steps
        self._frame_plan = FramePlan(
            self.outer_steps,
            self.inner_steps,
            self.dt_outer,
            self.dt_inner,
            self.vmax,
            self.heat_decay,
        )
        # Adaptive outer stepping (ADR 0220): the planned count is the floor and
        # ``max_outer_steps`` the ceiling; above the floor each step is sized on the device from
        # the fastest particle, inside one captured graph. A fixed ``outer_steps``, or a ceiling
        # at or below the plan, keeps every frame at the planned count.
        ceiling = int(s.max_outer_steps) if not s.outer_steps else self.outer_steps
        self._adaptive = ceiling > self.outer_steps
        self._intervals = gravity_intervals(
            planned=self.outer_steps,
            dt_outer=self.dt_outer,
            passes=int(s.gravity_passes_per_frame),
            ceiling=ceiling,
        )
        plan = self._frame_plan
        self._clock = wp.array(
            np.asarray([plan.dt_outer, plan.dt_inner, plan.vmax, plan.heat_decay], np.float32),
            dtype=float,
            device=dev,
        )
        self._remaining = wp.zeros(1, dtype=wp.float64, device=dev)
        self._steps_left = wp.zeros(1, dtype=int, device=dev)
        self._more = wp.zeros(1, dtype=wp.int32, device=dev)
        # Step counters (a counter, not state: a restore does not rewind them): ``_tally`` =
        # (the last frame's steps, the most any frame took, frames above the plan, all steps,
        # steps cut short by the budget); ``_hist[k]`` = frames that took k steps.
        self._tally = wp.zeros(5, dtype=int, device=dev)
        most = max(sum(t.budget for t in self._intervals), self.outer_steps)
        self._hist = wp.zeros(most + 1, dtype=int, device=dev)
        self._graph: Any = None
        if self._adaptive and self._use_graph and not wp.is_conditional_graph_supported():
            # A device loop needs CUDA conditional graph nodes (CUDA 12.4); without them the
            # loop runs on the host, which reads its flag each step, so frames run eagerly.
            self._use_graph = False
        with self._on_stream():
            self._refresh_speed()

    def _refresh_speed(self):
        """Measure the fastest particle now, on the device (the end of a frame does the same).
        The pressure acceleration beside it is kept, not measured: only a pressure solve sets it."""
        wp.launch(_zero_float, dim=1, inputs=[0], outputs=[self.speed_meas], device=self.device)
        wp.launch(
            _max_speed,
            dim=self.n,
            inputs=[self.v, self._v_ref_dev],
            outputs=[self.speed_meas],
            device=self.device,
        )

    def _on_stream(self):
        # Everything runs on the solver's stream; on exit the caller's stream waits for it.
        return wp.ScopedStream(self.stream, sync_enter=True, sync_exit=True)

    # -- setup

    def _neighbors(self, x):
        self.grid.build(x)
        wp.launch(
            _build_neighbors,
            dim=self.n,
            inputs=[*self.grid.args(), x, self.h],
            outputs=[self.nbr, self.overflow],
            device=self.device,
        )

    def _capture_reference(self):
        """Rest densities and every particle's bonds.

        The rest density is each particle's own kernel density at load, so a body starts at
        zero pressure. Bonds are the in-body neighbours inside the support, made symmetric and
        cut to 63 per row by dropping the longest (both ends at once), so every bond has two
        ends and one set of numbers."""
        dev = self.device
        self.overflow.zero_()
        self._neighbors(self.x)
        wp.launch(
            _density_dii,
            dim=self.n,
            inputs=[self.x, self.v, self.m, self.h, self.nbr, self.a_other, self._unit_clock],
            outputs=[self.rho, self.dii, self.v_adv],
            device=dev,
        )
        wp.copy(self.rho0, self.rho)
        nbr = self.nbr.numpy().T
        # The reference positions exactly as the device holds them (float32), so the host's
        # bond lengths describe the configuration the springs are at rest in.
        x = self.x.numpy().astype(np.float64)
        rows, cols = np.nonzero(nbr >= 0)
        j = nbr[rows, cols]
        solid = np.asarray([q.is_solid for q in self.materials])[self._material_ids]
        keep = (j != rows) & (self._body_ids[j] == self._body_ids[rows]) & solid[rows] & solid[j]
        i, j = rows[keep], j[keep]
        pairs = np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1), axis=0)
        pairs = pairs.reshape(-1, 2)
        length = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
        degree = np.bincount(pairs.ravel(), minlength=self.n)
        accepted = np.ones(len(pairs), dtype=bool)
        if degree.max(initial=0) > CAPACITY - 1:
            # Cut rows over capacity: drop the longest bonds first, at both ends.
            order = np.argsort(length, kind="stable")
            pairs, length = pairs[order], length[order]
            degree = np.zeros(self.n, dtype=np.int64)
            for e in range(len(pairs)):
                p, q = pairs[e]
                if degree[p] < CAPACITY - 1 and degree[q] < CAPACITY - 1:
                    degree[p] += 1
                    degree[q] += 1
                else:
                    accepted[e] = False
        pairs, length = pairs[accepted], length[accepted]
        self.dropped_bonds = int((~accepted).sum())
        a, b = pairs[:, 0], pairs[:, 1]
        mats = self.materials
        # Break length: the shorter of SpaceSim's 2 h and the bond's current reference length
        # stretched by the break strain - the softer material's critical strain inside a fault
        # cell, the body's fault strain (SpaceSim's activation strain) across two cells. The
        # table holds the ratio ``(1 + strain)^2``; the kernel applies it to the length the bond
        # has now, which plastic flow moves, and applies the 2 h cap itself.
        critical = np.asarray([q.break_strain for q in mats])[self._material_ids]
        fa, fb = self._fault_ids[a], self._fault_ids[b]
        across = (fa >= 0) & (fb >= 0) & (fa != fb)
        intact = np.minimum(critical[a], critical[b])
        # A fault is a weakness: across two cells a bond breaks at the body's fault strain or
        # the material's own critical strain, whichever is smaller.
        strain = np.where(across, np.minimum(self._fault_strain[self._body_ids[a]], intact), intact)
        ratio2 = (1.0 + strain) ** 2
        # Springs at the peridynamic micromodulus c = 18 K / (pi delta^4), K = 5/3 mu (Poisson
        # 1/4), horizon delta = the kernel support. The bond force is c V_i V_j (r - L) / L, with
        # each particle's own volume from its material's density (a surface particle's kernel
        # density reads low and would inflate its springs). A bond between two materials (an
        # iron core under a rock mantle) takes the softer.
        mu = np.asarray([q.shear_modulus for q in mats])[self._material_ids]
        rho_mat = np.asarray([q.reference_density for q in mats])[self._material_ids]
        vol = self._m_host / rho_mat
        horizon = 1.5 * self.h
        micro = 18.0 * (5.0 / 3.0) * np.minimum(mu[a], mu[b]) / (math.pi * horizon**4)
        stiff = micro * vol[a] * vol[b]
        spring = stiff / length
        ma, mb = self._m_host[a], self._m_host[b]
        damp = self.settings.bond_damping * 2.0 * np.sqrt(spring * ma * mb / (ma + mb))
        # The network's exact static stability limit, sqrt(2 m_i / sum k/L). ``pairs.ravel()``
        # is (a0, b0, a1, b1, ...), so each spring weights both its ends.
        load = np.bincount(pairs.ravel(), weights=np.repeat(spring, 2), minlength=self.n)
        loaded = load > 0.0
        self.bond_step_limit = (
            float(np.sqrt(2.0 * self._m_host[loaded] / load[loaded]).min())
            if np.any(loaded)
            else math.inf
        )
        # Row order is partner index, ascending: deterministic, and the same at both ends.
        ends = np.concatenate([pairs, pairs[:, ::-1]])
        data = np.stack([ratio2, length, stiff, damp], axis=1)
        data = np.concatenate([data, data])
        order = np.lexsort((ends[:, 1], ends[:, 0]))
        ends, data = ends[order], data[order]
        fill = np.bincount(ends[:, 0], minlength=self.n)
        first = np.concatenate([[0], np.cumsum(fill)[:-1]])
        slot = np.arange(len(ends)) - first[ends[:, 0]]
        rows = np.full((self.n, CAPACITY), -1, dtype=np.int32)
        table = np.zeros((self.n, CAPACITY, 4), dtype=np.float32)
        rows[ends[:, 0], slot] = ends[:, 1]
        table[ends[:, 0], slot] = data
        self._bond_rows0 = rows
        self._bond_data0 = table
        self._count0_host = fill.astype(np.int32)
        # For every reference bond (i, k), the slot that holds i in its partner's row: a
        # checkpoint's broken bits must agree at both ends, or a spring would act on one end
        # only and stop being equal and opposite.
        ii, kk = np.nonzero(rows >= 0)
        jj = rows[ii, kk].astype(np.int64)
        keys = ii.astype(np.int64) * self.n + jj
        order = np.argsort(keys)
        mirror = order[np.searchsorted(keys[order], jj * self.n + ii)]
        partner = np.full(rows.shape, -1, dtype=np.int64)
        partner[ii, kk] = kk[mirror]
        self._partner_slot0 = partner
        self._upload_rows(rows, table)
        self.count0.assign(self._count0_host)
        self.neighbor_overflow = int(self.overflow.numpy()[0])

    def _upload_rows(self, rows: np.ndarray, table: np.ndarray):
        self.bonds.assign(np.ascontiguousarray(rows.T))
        self.bdata.assign(np.ascontiguousarray(table.transpose(1, 0, 2)))

    def _refresh_yield(self):
        """The yield strain every particle has now, from its persistent thermal energy (a function
        of that energy alone, so a restored frame has exactly the one the live frame had)."""
        wp.launch(
            _soften,
            dim=self.n,
            inputs=[self.energy, self.y_cold, self.inv_melt],
            outputs=[self.y_eff],
            device=self.device,
        )

    def settle(self, centres=None):
        """Damp the bodies into equilibrium under their own gravity, then put each body's centre
        of mass back at ``centres[body]`` (default: where its authored particles had it) with
        its authored velocity. The springs keep their rest lengths: the stress that holds a
        settled body up is kept. ``settle_frames`` 0 (the default) skips it: a body sampled on
        a lattice starts at its bonds' rest lengths and zero pressure."""
        s = self.settings
        with self._on_stream():
            if s.settle_frames > 0:
                self.v.zero_()
                # The bodies settle at rest, so the mean they are sized and railed against is
                # zero until their authored velocity is put back (ADR 0224).
                self._v_ref_dev.zero_()
                self._ensure_gravity()
                for _ in range(s.settle_frames):
                    self._record_frame(damping=s.settle_damping)
                self._v_ref_dev.assign(np.asarray([self._v_ref], dtype=np.float32))
                x = self.x.numpy().astype(np.float64)
                m = self._m_host
                for b in np.unique(self._body_ids):
                    sel = self._body_ids == b
                    if centres is not None:
                        target = np.asarray(centres[int(b)], dtype=np.float64)
                    else:
                        target = (self._authored_x[sel] * m[sel, None]).sum(0) / m[sel].sum()
                    x[sel] += target - (x[sel] * m[sel, None]).sum(0) / m[sel].sum()
                self.x.assign(x.astype(np.float32))
                self._gravity_fresh = False
                self.v.assign(self._authored_v.astype(np.float32))
                self.heat.zero_()
                self.energy.zero_()
                self.damage.zero_()
                self._upload_rows(self._bond_rows0, self._bond_data0)
                self._refresh_yield()
                self._refresh_speed()
                # Settling frames are not played frames: the step counters start at load.
                self._tally.zero_()
                self._hist.zero_()
            self.frame = 0

    # -- stepping

    def _build_tree(self, config: Any, *, positions: Any, masses: Any) -> Any:
        """The Barnes-Hut runtime this solver's gravity passes walk (ADR 0218).

        Loaded once, as a gravity service (no integrator or broadphase kernels), with the
        solver's slot-ordered masses; the positions it holds are overwritten by every pass, so
        the ones given here only size and seed it.
        """

        from .runtime_foundation import DeviceContext
        from .sampling import BodySet
        from .tree import BarnesHutRuntime

        count = config.count
        context = DeviceContext(DeviceDescriptor(required_device=str(self.device)))
        context.initialize()
        tree = BarnesHutRuntime(context, config)
        tree.load_bodies(
            BodySet(
                positions=positions,
                velocities=np.zeros((count, 3), dtype=np.float32),
                masses=masses,
                radii=np.zeros(count, dtype=np.float32),
                stable_ids=np.arange(count, dtype=np.int64),
                active=np.ones(count, dtype=np.int32),
            ),
            gravity_service=True,
        )
        return tree

    def _gravity(self):
        if self._tree is not None:
            # Capture-safe: copies and launches only. The overflow fold is one more launch, so a
            # tree that ran out of nodes or walk stack fails the frame closed through
            # ``nonfinite`` with no readback here.
            self._tree.record_gravity_pass(self.x, self.a_grav)
            self._tree.record_overflow_flag(self._tree_bad)
            return
        wp.launch(
            _pack,
            dim=self.n_pad,
            inputs=[self.x, self.gm, self.n],
            outputs=[self.pm],
            device=self.device,
        )
        wp.launch(
            _gravity,
            dim=self.n_pad,
            inputs=[self.pm, self.n_pad // TILE, self.eps2, self.n],
            outputs=[self.a_grav],
            block_dim=TILE,
            device=self.device,
        )

    def _kick_gravity(self, dt: float, vmax: float):
        wp.launch(
            _gravity_kick,
            dim=self.n,
            inputs=[self.v, self.a_grav, dt, vmax, self._v_ref_dev],
            outputs=[self.clamped],
            device=self.device,
        )

    def _outer_step(self, damping: float = 1.0):
        # One outer step at the step clock's dt (``self._clock``): the plan's, or what
        # ``_plan_step`` set on the device for this step (ADR 0220).
        dev = self.device
        n = self.n
        clock = self._clock
        s = self.settings
        self._neighbors(self.x)
        # The yield strains, from the thermal energy as it stands as this step opens: a function
        # of the state at its start, so a restored state, which has no memory of the step before,
        # runs it exactly as the live one did.
        self._refresh_yield()
        # Gravity's opening half-kick is already in velocity; predict only the bond acceleration.
        wp.launch(
            _bond_acceleration,
            dim=n,
            inputs=[self.x, self.m, self.bonds, self.bdata, self.y_eff],
            outputs=[self.a_bond],
            device=dev,
        )
        wp.launch(
            _density_dii,
            dim=n,
            inputs=[self.x, self.v, self.m, self.h, self.nbr, self.a_bond, clock],
            outputs=[self.rho, self.dii, self.v_adv],
            device=dev,
        )
        wp.launch(
            _predict,
            dim=n,
            inputs=[self.x, self.m, self.h, self.nbr, self.rho, self.dii, self.v_adv, clock],
            outputs=[self.rho_adv, self.aii, self.p],
            device=dev,
        )
        # From the minimum, each even sweep short of the maximum checks the error the pressure
        # it read leaves; once converged, the rest are skipped on the device. Checks fall only
        # on even sweeps, so the answer is always in ``self.p`` (ADR 0219).
        checks = self.min_iterations < self.iterations
        if checks:
            wp.launch(
                _pressure_open,
                dim=1,
                outputs=[self._p_done, self._p_residual, self._p_pushing],
                device=dev,
            )
        src, dst = self.p, self.p_alt
        for sweep in range(1, self.iterations + 1):
            measure = checks and sweep % 2 == 0 and self.min_iterations <= sweep < self.iterations
            wp.launch(
                _dij_pj,
                dim=n,
                inputs=[self.x, self.m, self.h, self.nbr, self.rho, src, clock, self._p_done],
                outputs=[self.dij_pj],
                device=dev,
            )
            wp.launch(
                _pressure_iteration,
                dim=n,
                inputs=[
                    self.x,
                    self.m,
                    self.h,
                    self.nbr,
                    self.rho,
                    self.rho0,
                    self.rho_adv,
                    self.dii,
                    self.aii,
                    self.dij_pj,
                    src,
                    clock,
                    s.omega,
                    1.0 + s.pressure_tolerance,
                    self._p_done,
                    int(measure),
                ],
                outputs=[dst, self._p_residual, self._p_pushing, self._p_sweeps],
                device=dev,
            )
            if measure:
                wp.launch(
                    _pressure_check,
                    dim=1,
                    inputs=[s.pressure_max_error],
                    outputs=[self._p_residual, self._p_pushing, self._p_done],
                    device=dev,
                )
            src, dst = dst, src
        # An even sweep count leaves the result in ``self.p``.
        wp.launch(
            _hydro_impulse,
            dim=n,
            inputs=[
                self.x,
                self.v,
                self.m,
                self.h,
                self.nbr,
                self.rho,
                self.p,
                clock,
                s.alpha_av,
            ],
            outputs=[self.a_hydro, self.heat, self.energy],
            device=dev,
        )
        wp.launch(_zero_float, dim=1, inputs=[1], outputs=[self.speed_meas], device=dev)
        wp.launch(_max_accel, dim=n, inputs=[self.a_hydro], outputs=[self.speed_meas], device=dev)
        xc, xn = self.x, self.x_alt
        vc, vn = self.v, self.v_alt
        for step in range(self.inner_steps):
            # The outer step's pressure and viscosity land as one impulse with the first inner
            # step; gravity is split outside the pressure/bond integration.
            wp.launch(
                _bond_step,
                dim=n,
                inputs=[
                    xc,
                    vc,
                    self.m,
                    self.a_hydro,
                    clock,
                    1 if step == 0 else 0,
                    self.bonds,
                    self.bdata,
                    self.count0,
                    self.y_eff,
                    self.cap2,
                    self.min_length,
                    self.heat,
                    self.energy,
                    self._v_ref_dev,
                ],
                outputs=[vn, self.damage, xn, self.clamped],
                device=dev,
            )
            xc, xn = xn, xc
            vc, vn = vn, vc
        # An even inner count leaves positions and velocities in ``self.x`` and ``self.v``.
        if damping != 1.0:
            wp.launch(_damp, dim=n, inputs=[self.v, damping], device=dev)

    def _adaptive_step(self, damping: float = 1.0):
        # The body of an interval's device loop (ADR 0220): measure the fastest particle, size
        # this step from it on the device, take it. Nothing is read back.
        s = self.settings
        self._refresh_speed()
        wp.launch(
            _plan_step,
            dim=1,
            inputs=[
                self.speed_meas,
                s.collision_step_factor * self.h,
                self.dt_outer,
                self.inner_steps,
                s.speed_cap_lengths * self.h,
                1.0 / max(s.heat_half_life_frames * s.frame_dt, 1.0e-9),
            ],
            outputs=[self._remaining, self._steps_left, self._clock, self._tally, self._more],
            device=self.device,
        )
        self._outer_step(damping)

    def _record_frame(self, damping: float = 1.0):
        # Each kick is central at the positions where it acts. Applying an old acceleration
        # during the intervening drifts would add torque despite reciprocal forces (ADR 0199).
        # The opening acceleration is the one the previous frame closed on: positions have not
        # moved between that frame's last ``_gravity`` and this one, so recomputing it would
        # repeat an all-pairs pass bit for bit. ``_ensure_gravity`` has already run it when
        # something other than a frame moved the particles (load, settle, restore).
        dev = self.device
        plan = self._frame_plan
        wp.launch(_tally_open, dim=1, outputs=[self._tally], device=dev)
        if not self._adaptive:
            for interval in self._intervals:
                half_dt = 0.5 * interval.planned * plan.dt_outer
                self._kick_gravity(half_dt, plan.vmax)
                for _step in range(interval.planned):
                    self._outer_step(damping)
                self._gravity()
                self._kick_gravity(half_dt, plan.vmax)
            fixed = plan.outer
        else:
            # Each gravity interval keeps its span; a device loop covers it in as many steps as
            # the fastest particle needs, at least the plan's and at most its share of the
            # ceiling. The kicks' rail is the one the budget's shortest even step would set.
            cap_length = self.settings.speed_cap_lengths * self.h
            for interval in self._intervals:
                half_dt = 0.5 * interval.span
                rail = cap_length * interval.budget / interval.span
                self._kick_gravity(half_dt, rail)
                wp.launch(
                    _interval_open,
                    dim=1,
                    inputs=[interval.span, interval.budget],
                    outputs=[self._remaining, self._steps_left, self._more],
                    device=dev,
                )
                wp.capture_while(self._more, self._adaptive_step, damping=damping)
                self._gravity()
                self._kick_gravity(half_dt, rail)
            fixed = 0
        wp.launch(
            _tally_close,
            dim=1,
            inputs=[plan.outer, fixed],
            outputs=[self._tally, self._hist],
            device=dev,
        )
        # The yield strains and the fastest particle now: what a restore of this state would
        # rebuild.
        self._refresh_yield()
        self._refresh_speed()
        self._gravity_fresh = True

    def _ensure_gravity(self):
        """Make ``a_grav`` the acceleration at the current positions, if it is not already.

        A frame closes on a gravity pass at the positions it leaves, so the next frame opens
        on a valid one for free. Anything else that moves the particles (the constructor, a
        settle, a restore) marks it stale, and this eager pass, outside any capture, brings
        it back before the frame runs. Replaying a captured frame therefore never reads an
        acceleration from positions other than the ones it was computed at.
        """

        if not self._gravity_fresh:
            self._gravity()
            self._gravity_fresh = True

    def step_frame(self):
        """Advance exactly one displayed frame: one captured graph, its outer steps sized on the
        device as it runs (ADR 0220), with nothing read back."""
        with self._on_stream():
            self._ensure_gravity()
            if not self._use_graph:
                self._record_frame()
            elif self._graph is None:
                # The first frame runs eagerly (it is a real frame, and it warms the sort's
                # scratch so nothing allocates inside the capture that follows, which runs
                # nothing). Every later frame replays that one graph.
                self._record_frame()
                with wp.ScopedCapture(device=self.device, stream=self.stream) as cap:
                    self._record_frame()
                self._graph = cap.graph
            else:
                wp.capture_launch(self._graph, stream=self.stream)
        self.frame += 1

    def _step_tally(self) -> np.ndarray:
        with self._on_stream():
            return self._tally.numpy()

    @property
    def adaptive_frames(self) -> int:
        """Frames that ran more outer steps than planned, since load (one small read)."""
        return int(self._step_tally()[2])

    @property
    def pressure_sweeps(self) -> int:
        """Pressure sweeps run since load, settling included (one four-byte read)."""
        with self._on_stream():
            return int(self._p_sweeps.numpy()[0])

    @property
    def max_outer_steps_used(self) -> int:
        """The most outer steps any frame has run since load (the planned count before any)."""
        tally = self._step_tally()
        return int(tally[1]) if tally[3] > 0 else self.outer_steps

    @property
    def active_outer_steps(self) -> int:
        """The outer count of the last frame (the planned count before any)."""
        tally = self._step_tally()
        return int(tally[0]) if tally[3] > 0 else self.outer_steps

    @property
    def outer_steps_run(self) -> int:
        """Outer steps run by every frame since load."""
        return int(self._step_tally()[3])

    @property
    def budget_bound_steps(self) -> int:
        """Outer steps since load that the step budget sized rather than the collision rule: each
        was longer than the fastest particle's reach, so the speed rail may have clamped it."""
        return int(self._step_tally()[4])

    @property
    def frames_by_outer(self) -> dict[int, int]:
        """Frames run at each outer count since load (a counter, not state: a restore does not
        rewind it)."""
        with self._on_stream():
            hist = self._hist.numpy()
        return {int(k): int(c) for k, c in enumerate(hist) if c}

    # -- read-back

    @property
    def internal_index(self) -> np.ndarray:
        """For each caller index, where that particle sits in the device arrays (a copy)."""
        return self._rank.astype(np.int32)

    def _internal_state(self) -> dict:
        return {
            "positions": self.x.numpy().astype(np.float64),
            "velocities": self.v.numpy().astype(np.float64),
            "heat": self.heat.numpy().astype(np.float64),
            "energy": self.energy.numpy().astype(np.float64),
            "damage": self.damage.numpy().astype(np.float64),
        }

    def state(self) -> dict:
        """Host copies of the particle state (float64), in the caller's order."""
        return {name: values[self._rank] for name, values in self._internal_state().items()}

    def invariants(self) -> dict:
        """Total momentum, and angular momentum about the centre of mass (float64, host)."""
        st = self._internal_state()
        m = self._m_host
        x, v = st["positions"], st["velocities"]
        com = (x * m[:, None]).sum(0) / m.sum()
        p = (v * m[:, None]).sum(0)
        L = (m[:, None] * np.cross(x - com, v - p / m.sum())).sum(0)
        return {"com": com, "momentum": p, "angular_momentum": L}

    def speed_clamps(self) -> int:
        """How many particle-steps the safety rail has clamped since load."""
        return int(self.clamped.numpy()[0])

    def neighbor_overflow_total(self) -> int:
        """Neighbour-row overflows counted since load, the play-time ones included (a row that
        overflowed lost partners beyond its 63rd)."""
        return int(self.overflow.numpy()[0])

    @property
    def reference_bond_counts(self) -> np.ndarray:
        """Each particle's reference bond count at load, in the caller's order (a copy)."""
        return self._count0_host[self._rank].copy()

    @property
    def bond_count(self) -> int:
        """Reference bonds captured at load (each counted once, not once per end)."""
        assert self._bond_rows0 is not None
        return int(self._count0_host.sum()) // 2

    @property
    def submission(self) -> str:
        """``cuda_graph`` when a frame is one captured graph replay, else ``eager``."""
        return "cuda_graph" if self._use_graph else "eager"

    def broken_bond_mask(self) -> np.ndarray:
        """One uint64 per particle, in the caller's order: bit k set when reference bond k of
        its row is broken."""
        return self._internal_broken_mask()[self._rank]

    def _internal_broken_mask(self) -> np.ndarray:
        rows0 = self._bond_rows0
        assert rows0 is not None
        now = self.bonds.numpy().T.copy()
        # A row is its prefix up to the first -1; anything after it is not a bond (the kernel
        # blanks freed slots, and the reader does not rely on it).
        now[~np.cumprod(now >= 0, axis=1).astype(bool)] = -1
        n = np.int64(self.n)
        ii, kk = np.nonzero(rows0 >= 0)
        ref_keys = ii.astype(np.int64) * n + rows0[ii, kk].astype(np.int64)
        ci, ck = np.nonzero(now >= 0)
        cur_keys = ci.astype(np.int64) * n + now[ci, ck].astype(np.int64)
        gone = ~np.isin(ref_keys, cur_keys)
        mask = np.zeros(self.n, dtype=np.uint64)
        np.bitwise_or.at(mask, ii[gone], np.left_shift(np.uint64(1), kk[gone].astype(np.uint64)))
        return mask

    def _internal_plastic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every bond whose reference length has moved: ``(particle, reference slot, length)``
        for the bond's lower-indexed end (in the solver's order), the length as the device holds
        it (float32). Both ends of a bond hold the same length, so one end names it."""
        rows0, data0 = self._bond_rows0, self._bond_data0
        assert rows0 is not None and data0 is not None
        now = self.bonds.numpy().T.copy()
        now[~np.cumprod(now >= 0, axis=1).astype(bool)] = -1
        lengths = self.bdata.numpy().transpose(1, 0, 2)[:, :, 1]
        ci, ck = np.nonzero(now >= 0)
        partner = now[ci, ck].astype(np.int64)
        lower = ci < partner
        ci, ck, partner = ci[lower], ck[lower], partner[lower]
        n = np.int64(self.n)
        ii, kk = np.nonzero(rows0 >= 0)
        reference_keys = ii.astype(np.int64) * n + rows0[ii, kk].astype(np.int64)
        slot = kk[np.searchsorted(reference_keys, ci.astype(np.int64) * n + partner)]
        current = np.ascontiguousarray(lengths[ci, ck], dtype=np.float32)
        reference = np.ascontiguousarray(data0[ci, slot, 1], dtype=np.float32)
        moved = current.view(np.uint32) != reference.view(np.uint32)
        return ci[moved], slot[moved], current[moved]

    def plastic_bonds(self) -> dict:
        """The bonds whose reference length has moved (permanent set), one entry per bond:
        ``particles`` (the caller's index of its lower-indexed end, in the solver's order),
        ``slots`` (the bond's slot in that end's reference row) and ``lengths`` (float32)."""
        particles, slots, lengths = self._internal_plastic()
        return {
            "particles": self._order[particles].astype(np.int64),
            "slots": slots.astype(np.int64),
            "lengths": lengths,
        }

    def checkpoint(self) -> dict:
        """The solver's state beyond positions, velocities and heat: the frame clock, the
        broken bonds (a bit per reference bond), the bonds whose reference length has moved,
        each particle's persistent thermal energy (float32, in the caller's order) and the largest
        pressure acceleration the last pressure solve left, which sizes the next step (ADR 0220)."""
        with self._on_stream():
            pressure_accel = float(self.speed_meas.numpy()[1])
        return {
            "frame": int(self.frame),
            "broken": self.broken_bond_mask(),
            "plastic": self.plastic_bonds(),
            "energy": self.energy.numpy()[self._rank].copy(),
            "pressure_accel": pressure_accel,
        }

    def _restore_plastic(
        self, plastic, keep: np.ndarray, *, slot_of: np.ndarray, table: np.ndarray
    ):
        """Write the moved reference lengths of a checkpoint into the rebuilt ``table``, at both
        ends of each bond. Refuses, before anything is written to the device, a list that is not
        the shape ``plastic_bonds`` gives, names a particle or slot that does not exist, names a
        bond the checkpoint has broken or one end twice, or carries a length no bond can have."""
        if plastic is None:
            return
        try:
            particles = np.asarray(plastic["particles"])
            slots = np.asarray(plastic["slots"])
            lengths = np.asarray(plastic["lengths"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("the plastic-bond list is malformed") from exc
        if not (particles.ndim == slots.ndim == lengths.ndim == 1) or not (
            particles.shape == slots.shape == lengths.shape
        ):
            raise ValueError("the plastic-bond list needs one particle, slot and length per bond")
        if particles.size == 0:
            return
        if particles.dtype.kind not in "iu" or slots.dtype.kind not in "iu":
            raise ValueError("plastic-bond particles and slots must be integers")
        if np.any(particles < 0) or np.any(particles >= self.n):
            raise ValueError("a plastic bond names a particle that does not exist")
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError("a plastic-bond length must be positive and finite")
        # A bond is never longer than 2 h (it is cut there), and a float32 length a hair above
        # is the rounding of the cut itself.
        if np.any(lengths > 2.0 * self.h * (1.0 + 1.0e-3)):
            raise ValueError("a plastic-bond length is beyond the 2 h at which a bond is cut")
        rows0 = self._bond_rows0
        assert rows0 is not None
        own = self._rank[particles.astype(np.int64)]
        slot = slots.astype(np.int64)
        if np.any(slot < 0) or np.any(slot >= self._count0_host[own]):
            raise ValueError("a plastic bond names a slot the reference row does not have")
        if np.unique(own * CAPACITY + slot).size != own.size:
            raise ValueError("a plastic bond is named twice")
        if not np.all(keep[own, slot]):
            raise ValueError("a plastic bond is one the checkpoint has broken")
        partner = rows0[own, slot].astype(np.int64)
        if np.any(partner <= own):
            raise ValueError("a plastic bond must be named by its lower-indexed end")
        far = self._partner_slot0[own, slot]
        value = lengths.astype(np.float32)
        table[own, slot_of[own, slot], 1] = value
        table[partner, slot_of[partner, far], 1] = value

    def restore(self, positions, velocities, heat, checkpoint: dict):
        """Resume from ``state()`` and ``checkpoint()`` values taken together at one frame.

        Bond rows are rebuilt as their reference rows minus the broken bits, in order: exactly
        the rows the stable compaction leaves, so a replay is bit for bit the same. Everything
        is validated before anything on the device is written."""
        frame = int(checkpoint["frame"])
        if frame < 0:
            raise ValueError("a checkpoint frame cannot be negative")
        arrays = (("positions", positions, 3), ("velocities", velocities, 3), ("heat", heat, 0))
        staged = {}
        for name, a, width in arrays:
            shape = np.shape(a)
            if shape != ((self.n, width) if width else (self.n,)):
                raise ValueError(f"{name} has shape {shape}, expected {self.n} rows")
            staged[name] = finite_float32(a, name)
        broken = np.asarray(checkpoint["broken"], dtype=np.uint64)
        if broken.shape != (self.n,):
            raise ValueError("the broken-bond mask does not match the particle count")
        # The persistent thermal energy: absent means none (a checkpoint of a body that never
        # warmed), present must be one finite, non-negative number per particle.
        energy = checkpoint.get("energy")
        if energy is None:
            energy = np.zeros(self.n, dtype=np.float32)
        else:
            if np.shape(energy) != (self.n,):
                raise ValueError(f"energy has shape {np.shape(energy)}, expected {self.n} rows")
            energy = finite_float32(energy, "energy")
            if np.any(energy < 0.0):
                raise ValueError("energy cannot be negative")
        # The pressure acceleration that sizes the first step (ADR 0220): absent means none (a
        # checkpoint taken before any pressure solve), present must be one finite, non-negative
        # float32.
        pressure_accel = checkpoint.get("pressure_accel", 0.0)
        if isinstance(pressure_accel, bool) or not isinstance(pressure_accel, (int, float)):
            raise ValueError("pressure_accel must be a number")
        pressure_accel = float(finite_float32(float(pressure_accel), "pressure_accel"))
        if pressure_accel < 0.0:
            raise ValueError("pressure_accel cannot be negative")
        # Inputs are in the caller's order; the rows and device arrays are in the solver's.
        order = self._order
        positions = staged["positions"][order]
        velocities = staged["velocities"][order]
        heat = staged["heat"][order]
        energy = energy[order]
        broken = broken[order]
        rows0, data0 = self._bond_rows0, self._bond_data0
        assert rows0 is not None and data0 is not None
        c0 = self._count0_host.astype(np.int64)
        slots = np.arange(CAPACITY, dtype=np.uint64)
        bits = ((broken[:, None] >> slots[None, :]) & np.uint64(1)).astype(bool)
        valid = np.arange(CAPACITY)[None, :] < c0[:, None]
        if np.any(bits & ~valid):
            raise ValueError("checkpoint breaks a bond the reference row does not have")
        ii, kk = np.nonzero(valid)
        jj = rows0[ii, kk].astype(np.int64)
        if np.any(bits[ii, kk] != bits[jj, self._partner_slot0[ii, kk]]):
            raise ValueError("checkpoint breaks a bond at one end only")
        keep = valid & ~bits
        kept = keep.sum(axis=1)
        ii, kk = np.nonzero(keep)
        slot = np.cumsum(keep, axis=1)[ii, kk] - 1
        rows = np.full_like(rows0, -1)
        table = np.zeros_like(data0)
        rows[ii, slot] = rows0[ii, kk]
        table[ii, slot] = data0[ii, kk]
        self._restore_plastic(
            checkpoint.get("plastic"), keep, slot_of=np.cumsum(keep, axis=1) - 1, table=table
        )
        # In float32, as the kernel computes it, so a restored frame re-checkpoints identically.
        one = np.float32(1.0)
        damage = np.where(
            c0 > 0,
            one - kept.astype(np.float32) / np.maximum(c0, 1).astype(np.float32),
            np.float32(0.0),
        ).astype(np.float32)
        with self._on_stream():
            self.x.assign(positions)
            self.v.assign(velocities)
            self.heat.assign(heat)
            self.energy.assign(energy)
            self._upload_rows(rows, table)
            self.damage.assign(damage)
            self._refresh_yield()
            self._refresh_speed()
            self.speed_meas.assign(
                np.array([self.speed_meas.numpy()[0], pressure_accel], dtype=np.float32)
            )
            # Gravity is derivative state; rebuild it from the restored positions.
            self._gravity()
            self._gravity_fresh = True
            # So is the density the display publishes: a step only recomputes it at its own
            # opening, so without this the first published frame after a restore would show the
            # density of the state that was replaced. The step recomputes all of it (and the
            # neighbour list, and ``dii``/``v_adv``) before using any of it, so this touches no
            # state a replay depends on.
            self._refresh_density()
            self.frame = frame

    def _refresh_density(self) -> None:
        """Recompute the published density from the current positions (see ``restore``)."""
        self._neighbors(self.x)
        wp.launch(
            _density_dii,
            dim=self.n,
            inputs=[self.x, self.v, self.m, self.h, self.nbr, self.a_other, self._unit_clock],
            outputs=[self.rho, self.dii, self.v_adv],
            device=self.device,
        )

    def nonfinite(self) -> bool:
        """Whether any position or velocity has gone non-finite, or the gravity tree ran out of
        room (ADR 0218): a few launches, one int back. The tree's sticky overflow flag is read
        and cleared here, so the frame that overflowed is the frame that fails closed."""
        with self._on_stream():
            self.bad.zero_()
            wp.launch(
                _flag_nonfinite,
                dim=self.n,
                inputs=[self.x, self.v],
                outputs=[self.bad],
                device=self.device,
            )
            if self._tree_bad is not None:
                wp.launch(
                    _raise_flag,
                    dim=1,
                    inputs=[self._tree_bad],
                    outputs=[self.bad],
                    device=self.device,
                )
                self._tree_bad.zero_()
        return bool(self.bad.numpy()[0])

    def close(self) -> None:
        """Release the gravity tree's device buffers (idempotent); the particle arrays follow
        the object."""

        tree, self._tree = self._tree, None
        self._tree_bad = None
        if tree is not None:
            tree.close()
