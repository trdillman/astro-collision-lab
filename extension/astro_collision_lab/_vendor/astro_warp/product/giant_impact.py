"""Real-time giant impact: position-based-fluid pressure plus tiled all-pairs self-gravity.

A fixed cost per displayed frame, whatever happens in it: one hash-grid pressure solve per
substep, one all-pairs gravity pass per frame, and the whole frame replayed as one captured
CUDA graph. VFX-grade by design - the impact, the debris disk and the moon it clumps into read
right; nothing here is claimed to be a scientific result.

Conservation is structural rather than tuned:

- every pressure push that repels acts along the line joining the pair at the start of the
  substep, so it exerts no torque on a spinning body (``_delta``);
- the velocity smoothing is mass-symmetric and acts only along the pair line, so it moves
  neither momentum nor angular momentum (``_smooth``);
- gravity kicks as half-impulses at frame boundaries, evaluated at the positions it acts on
  (``_record_frame``).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

wp.set_module_options({"fast_math": True})

TILE = 128
_TILE = wp.constant(TILE)

#: Speed the substep is sized for, as a multiple of the scene's characteristic speed: ejecta
#: leave faster than the bodies meet. Speeds beyond it are clamped, so this is a safety margin.
EJECTA_SPEED_MARGIN = 1.5

#: Neighbour-grid table size per axis. Cells wrap modulo this, so any extent is covered; far
#: cells that share a slot only add candidates the distance test rejects.
GRID_DIM = 128
_DIM = wp.constant(GRID_DIM)
_ORIGIN = wp.constant(1 << 20)


# ----------------------------------------------------------------------------- neighbour grid
# The same cell hash, sort and visiting order as ``wp.HashGrid``, with every buffer owned by
# the solver (see ``_CellGrid`` for why ``wp.HashGrid`` itself cannot be used).


@wp.func
def _cell(ix: int, iy: int, iz: int):
    x = wp.max(ix + _ORIGIN, 0)
    y = wp.max(iy + _ORIGIN, 0)
    z = wp.max(iz + _ORIGIN, 0)
    return (z % _DIM) * (_DIM * _DIM) + (y % _DIM) * _DIM + (x % _DIM)


@wp.func
def _span(c: float, h: float, inv: float):
    # Cells a query of radius h around coordinate c touches, never more than one lap of the table.
    lo = int(wp.floor((c - h) * inv))
    return lo, wp.min(int(wp.floor((c + h) * inv)), lo + _DIM - 1)


@wp.kernel
def _cell_keys(
    x: wp.array(dtype=wp.vec3), inv: float, keys: wp.array(dtype=int), ids: wp.array(dtype=int)
):
    i = wp.tid()
    p = x[i]
    keys[i] = _cell(int(wp.floor(p[0] * inv)), int(wp.floor(p[1] * inv)), int(wp.floor(p[2] * inv)))
    ids[i] = i


@wp.kernel
def _cell_ranges(
    keys: wp.array(dtype=int), n: int, start: wp.array(dtype=int), end: wp.array(dtype=int)
):
    t = wp.tid()
    c = keys[t]
    if t == 0:
        start[c] = 0
    else:
        p = keys[t - 1]
        if c != p:
            start[c] = t
            end[p] = t
    if t == n - 1:
        end[c] = t + 1


# ----------------------------------------------------------------------------- kernels


@wp.kernel
def _pack(
    x: wp.array(dtype=wp.vec3),
    gm: wp.array(dtype=float),
    n: int,
    centre: wp.array(dtype=wp.vec3),
    r_acc2: float,
    pm: wp.array(dtype=wp.vec4),
):
    i = wp.tid()
    if i < n:
        p = x[i]
        w = gm[i]
        if wp.length_sq(p - centre[0]) > r_acc2:
            w = -w  # beyond the Roche limit: the sign marks it for the accretion gain
        pm[i] = wp.vec4(p[0], p[1], p[2], w)
    else:
        pm[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _gravity(
    pm: wp.array(dtype=wp.vec4),
    n_tiles: int,
    eps2: float,
    gain: wp.array(dtype=float),
    n: int,
    acc: wp.array(dtype=wp.vec3),
):
    # A pair with both ends beyond the Roche limit attracts `gain` times harder. The boost is
    # symmetric in (i, j) and central, so momentum and angular momentum are still conserved;
    # what it changes is how far a debris clump's pull reaches (its Hill radius grows as the
    # cube root of the gain), which is what lets a disk gather into a moon within a shot.
    i = wp.tid()
    pi = pm[i]
    boost = 0.0
    if pi[3] < 0.0:
        boost = gain[0] - 1.0
    a = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_tiles):
        t = wp.tile_load(pm, shape=_TILE, offset=k * _TILE)
        for j in range(_TILE):
            q = t[j]
            d = wp.vec3(q[0] - pi[0], q[1] - pi[1], q[2] - pi[2])
            r2 = wp.dot(d, d) + eps2
            inv = 1.0 / wp.sqrt(r2)
            w = wp.abs(q[3])
            if q[3] < 0.0:
                w += boost * w
            a += (w * inv * inv * inv) * d
    # The launch is padded to a whole tile; only the first n threads own an acc slot.
    if i < n:
        acc[i] = a


@wp.func
def _w_poly6(r2: float, h2: float, kp: float):
    if r2 >= h2:
        return 0.0
    d = h2 - r2
    return kp * d * d * d


@wp.func
def _grad_spiky(d: wp.vec3, r: float, h: float, ks: float):
    # Gradient w.r.t. x_i of the spiky kernel, d = x_i - x_j.
    if r >= h or r < 1.0e-6:
        return wp.vec3(0.0, 0.0, 0.0)
    t = h - r
    return (-ks * t * t / r) * d


@wp.kernel
def _predict(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    acc: wp.array(dtype=wp.vec3),
    dt: float,
    acc_scale: float,
    vmax: float,
    xp: wp.array(dtype=wp.vec3),
    vp: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    vi = v[i] + acc[i] * (dt * acc_scale)
    s = wp.length(vi)
    if s > vmax:
        vi = vi * (vmax / s)
    vp[i] = vi
    xp[i] = x[i] + vi * dt


@wp.kernel
def _lambda(
    start: wp.array(dtype=int),
    end: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    inv: float,
    xp: wp.array(dtype=wp.vec3),
    invm: wp.array(dtype=float),
    rest: float,
    h: float,
    kp: float,
    ks: float,
    relax: float,
    cohesion: float,
    lam: wp.array(dtype=float),
    dens: wp.array(dtype=float),
    s2out: wp.array(dtype=float),
):
    tid = wp.tid()
    i = ids[tid]
    xi = xp[i]
    h2 = h * h
    # `float(...)`, not a bare literal: Warp treats a literal as a constant and refuses to
    # accumulate into it inside the loop.
    rho = float(0.0)  # noqa: UP018
    s2 = float(0.0)  # noqa: UP018
    gi = wp.vec3(0.0, 0.0, 0.0)
    x0, x1 = _span(xi[0], h, inv)
    y0, y1 = _span(xi[1], h, inv)
    z0, z1 = _span(xi[2], h, inv)
    for cz in range(z0, z1 + 1):
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                cell = _cell(cx, cy, cz)
                for k in range(start[cell], end[cell]):
                    j = ids[k]
                    d = xi - xp[j]
                    r2 = wp.dot(d, d)
                    if r2 < h2:
                        rho += _w_poly6(r2, h2, kp)
                        if j != i:
                            g = _grad_spiky(d, wp.sqrt(r2), h, ks) / rest
                            gi += g
                            s2 += invm[j] * wp.dot(g, g)
    s2 += invm[i] * wp.dot(gi, gi)
    # Under-density below -cohesion is not pulled back: that is what lets a body tear.
    c = wp.max(rho / rest - 1.0, -cohesion)
    lam[i] = -c / (s2 + relax)
    dens[i] = rho
    s2out[i] = s2


@wp.kernel
def _delta(
    start: wp.array(dtype=int),
    end: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    inv: float,
    x: wp.array(dtype=wp.vec3),
    xp: wp.array(dtype=wp.vec3),
    invm: wp.array(dtype=float),
    rest: float,
    lam: wp.array(dtype=float),
    h: float,
    ks: float,
    xout: wp.array(dtype=wp.vec3),
):
    # The velocity is (x_new - x_old)/dt, so its angular momentum is sum m x_old x x_new / dt: a
    # push along the pair line at x_old adds none. Pushing along the predicted line instead adds
    # -sum v_ij x G_ij per substep, which for a spinning body held up by pressure is a spin-up
    # proportional to its own spin. So a repelling pair pushes along its old line, with the
    # strength read off the predicted distance. Two exceptions keep the predicted line: a pair
    # that passed through itself this substep (its old line would drive it further together),
    # and a pulling (cohesive) pair (a pull along a lagging line feeds energy into the pair).
    tid = wp.tid()
    i = ids[tid]
    xi = xp[i]
    xo = x[i]
    li = lam[i]
    h2 = h * h
    dx = wp.vec3(0.0, 0.0, 0.0)
    x0, x1 = _span(xi[0], h, inv)
    y0, y1 = _span(xi[1], h, inv)
    z0, z1 = _span(xi[2], h, inv)
    for cz in range(z0, z1 + 1):
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                cell = _cell(cx, cy, cz)
                for k in range(start[cell], end[cell]):
                    j = ids[k]
                    if j != i:
                        d = xi - xp[j]
                        r2 = wp.dot(d, d)
                        if r2 < h2:
                            r = wp.sqrt(r2)
                            s = li + lam[j]
                            do = xo - x[j]
                            ro = wp.length(do)
                            if s <= 0.0 and ro > 1.0e-6 and wp.dot(do, d) > 0.0:
                                t = h - r
                                dx += s * ((-ks * t * t / ro) * do)
                            else:
                                dx += s * _grad_spiky(d, r, h, ks)
    xout[i] = xi + (invm[i] / rest) * dx


@wp.kernel
def _update(
    x: wp.array(dtype=wp.vec3),
    xp: wp.array(dtype=wp.vec3),
    vp: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    heat: wp.array(dtype=float),
    inv_dt: float,
    heat_gain: float,
    cool: float,
):
    # Heat is the kinetic energy the pressure solve took out of a particle this substep: it
    # glows where bodies hit, then cools. It drives colour only.
    i = wp.tid()
    vn = (xp[i] - x[i]) * inv_dt
    dv = vn - vp[i]
    heat[i] = heat[i] * cool + heat_gain * 0.5 * wp.dot(dv, dv)
    v[i] = vn
    x[i] = xp[i]


@wp.kernel
def _smooth(
    start: wp.array(dtype=int),
    end: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    inv: float,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    dens: wp.array(dtype=float),
    h: float,
    kp: float,
    c_visc: float,
    vout: wp.array(dtype=wp.vec3),
):
    # Pair weight 2 m_j/(m_i+m_j) * 2/(dens_i+dens_j) makes m_i dv_i = -m_j dv_j, so the smoothing
    # moves no momentum; smoothing only the approach speed along the pair line exerts no torque.
    tid = wp.tid()
    i = ids[tid]
    xi = x[i]
    vi = v[i]
    mi = m[i]
    di = dens[i]
    h2 = h * h
    acc = wp.vec3(0.0, 0.0, 0.0)
    x0, x1 = _span(xi[0], h, inv)
    y0, y1 = _span(xi[1], h, inv)
    z0, z1 = _span(xi[2], h, inv)
    for cz in range(z0, z1 + 1):
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                cell = _cell(cx, cy, cz)
                for k in range(start[cell], end[cell]):
                    j = ids[k]
                    if j != i:
                        d = xi - x[j]
                        r2 = wp.dot(d, d)
                        if r2 < h2 and r2 > 1.0e-12:
                            mj = m[j]
                            wgt = (
                                _w_poly6(r2, h2, kp)
                                * (2.0 * mj / (mi + mj))
                                * (2.0 / (di + dens[j]))
                            )
                            acc += ((wp.dot(v[j] - vi, d) / r2) * d) * wgt
    vout[i] = vi + c_visc * acc


@wp.kernel
def _kick(v: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3), s: float):
    i = wp.tid()
    v[i] = v[i] + acc[i] * s


@wp.kernel
def _damp(v: wp.array(dtype=wp.vec3), f: float):
    i = wp.tid()
    v[i] = v[i] * f


# ----------------------------------------------------------------------------- moon guide
# Gravity alone clumps a debris disk into a ring of moonlets over far more orbits than a shot
# lasts. The guide makes the one-moon outcome happen on screen: a moon anchor rides a circular
# orbit read off the debris, and bound debris is steered onto that orbit and swept up by the
# anchor. It acts only through accelerations added to gravity's, and the planet takes the
# reaction, so total momentum is untouched.

#: Uniform slots read by ``_gather`` and ``_react`` (see ``_gather`` for their meaning).
_GP_SIZE = 26
AZIMUTH_BINS = 36


@wp.kernel
def _gather(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    centre: wp.array(dtype=wp.vec3),
    gp: wp.array(dtype=float),
    inner: wp.array(dtype=float),
    acc: wp.array(dtype=wp.vec3),
    react: wp.array(dtype=wp.vec3),
):
    # The anchor is a point on a fixed circular orbit (gp[4..13]); it cannot be torn apart, so
    # what gathers on it cannot lose its identity. Bound debris outside the planet relaxes
    # toward a circular orbit in the anchor's plane; near the anchor it relaxes to the anchor's
    # own velocity instead (inelastic capture). The planet (everything inside gp[14]) takes the
    # reaction.
    #
    # Phase is closed on a rail: debris is held at the anchor's radius and only its speed along
    # the orbit is chosen. Debris a little behind the anchor runs slightly fast to catch it;
    # everything else runs slow and waits for the anchor to come round and sweep it up. Orbital
    # shear alone closes phase far too slowly, which leaves a ring of moonlets. The rail term
    # supplies the part of gravity a held-off-circular speed leaves unbalanced, so a waiting
    # particle neither sinks nor climbs; it is radial, so it exerts no torque.
    #
    # Energy safety is structural: the fastest target is a fraction of circular speed above it
    # and every target is clamped below the local escape speed, so nothing the guide releases
    # can leave. (A tangential catch-up boost to escape speed is what threw debris out of the
    # first anchor designs.)
    #
    # gp: 0 rate, 1 1/approach time, 2 steer cap (fraction of circular speed), 3 capture reach,
    #     4 orbit radius, 5 cos(phase), 6 sin(phase), 7 orbital speed, 8..10 orbit normal,
    #     11..13 in-plane axis, 14 planet edge, 15 full-weight radius, 16 catch-up fraction,
    #     17 target cap (fraction of escape speed), 18 waiting slow-down fraction, 19 catch-up
    #     window (radians behind the anchor), 20 ramp scale, 21 1/capture time,
    #     22..24 barycentre velocity, 25 G * total mass.
    i = wp.tid()
    rate = gp[0]
    if rate <= 0.0:
        return
    p = x[i] - centre[0]
    r3 = wp.length(p)
    r_in = gp[14]
    if r3 <= r_in:
        wp.atomic_add(inner, 0, m[i])
        return
    u = v[i] - wp.vec3(gp[22], gp[23], gp[24])
    mu = gp[25]
    if 0.5 * wp.dot(u, u) - mu / r3 >= 0.0:
        return
    inv_t = gp[1]
    r_cap = gp[3]
    rs = gp[4]
    nrm = wp.vec3(gp[8], gp[9], gp[10])
    e1 = wp.vec3(gp[11], gp[12], gp[13])
    e2 = wp.cross(nrm, e1)
    ers = gp[5] * e1 + gp[6] * e2
    xs = rs * ers
    vs = gp[7] * wp.cross(nrm, ers)
    z = wp.dot(p, nrm)
    pp = p - z * nrm
    r = wp.length(pp)
    if r < 1.0e-6:
        return
    er = pp / r
    ephi = wp.cross(nrm, er)
    vc = wp.sqrt(mu / r)
    vcap = gp[2] * vc
    # alpha > 0: the anchor is ahead (prograde) of this particle.
    alpha = wp.atan2(wp.dot(wp.cross(er, ers), nrm), wp.dot(er, ers))
    phi0 = r_cap / rs  # the anchor's angular reach: inside it the approach eases off
    vt = vc
    if alpha > 0.0 and alpha <= gp[19]:
        vt = vc * (1.0 + gp[16] * wp.min(alpha / phi0, 1.0))
    else:
        lead = -alpha
        if alpha > 0.0:
            lead = 2.0 * wp.pi - alpha
        vt = vc * (1.0 - gp[18] * wp.min(lead / phi0, 1.0))
    steer = er * ((rs - r) * inv_t) - nrm * (z * inv_t)
    sp = wp.length(steer)
    if sp > vcap:
        steer = steer * (vcap / sp)
    target = ephi * vt + steer
    d = wp.length(xs - p)
    w = float(0.0)  # noqa: UP018 - a Warp dynamic variable, reassigned below
    if d < r_cap:
        w = 1.0 - d / r_cap
        w = w * w * (3.0 - 2.0 * w)
        local = (xs - p) * gp[21]
        lp = wp.length(local)
        lcap = gp[2] * gp[7]
        if lp > lcap:
            local = local * (lcap / lp)
        target = target * (1.0 - w) + (vs + local) * w
    vmax = gp[17] * wp.sqrt(2.0 * mu / r3)
    ts = wp.length(target)
    if ts > vmax:
        target = target * (vmax / ts)
    # Weight ramps in from the planet's edge, so the planet's own surface is left alone.
    wr = wp.clamp((r3 - r_in) / (gp[15] - r_in), 0.0, 1.0)
    wr = wr * wr * (3.0 - 2.0 * wr)
    # Capture near the anchor is quicker than the steering elsewhere, so arriving debris stops
    # on the moon instead of passing through it.
    k = rate * (1.0 - w) + gp[20] * gp[21] * w
    a = (target - u) * (k * wr)
    vtf = wp.dot(target, ephi)
    a = a - er * ((vtf * vtf - vc * vc) / r * (gp[20] * wr))
    acc[i] = acc[i] + a
    wp.atomic_add(react, 0, m[i] * a)


@wp.kernel
def _react(
    x: wp.array(dtype=wp.vec3),
    centre: wp.array(dtype=wp.vec3),
    gp: wp.array(dtype=float),
    inner: wp.array(dtype=float),
    react: wp.array(dtype=wp.vec3),
    acc: wp.array(dtype=wp.vec3),
):
    # The planet takes the guide's total push back, shared evenly per unit mass.
    i = wp.tid()
    mi = inner[0]
    if gp[0] > 0.0 and mi > 0.0 and wp.length(x[i] - centre[0]) <= gp[14]:
        acc[i] = acc[i] - react[0] / mi


@wp.kernel
def _debris_moments(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    centre: wp.vec3,
    drift: wp.vec3,
    mu: float,
    r_in: float,
    lsum: wp.array(dtype=wp.vec3),
    msum: wp.array(dtype=float),
):
    # Angular momentum and mass of the bound debris outside the planet.
    i = wp.tid()
    p = x[i] - centre
    r = wp.length(p)
    if r <= r_in:
        return
    u = v[i] - drift
    if 0.5 * wp.dot(u, u) - mu / r >= 0.0:
        return
    wp.atomic_add(lsum, 0, m[i] * wp.cross(p, u))
    wp.atomic_add(msum, 0, m[i])


@wp.kernel
def _debris_azimuth(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    centre: wp.vec3,
    drift: wp.vec3,
    mu: float,
    r_in: float,
    r_far: float,
    nrm: wp.vec3,
    e1: wp.vec3,
    hist: wp.array(dtype=float),
):
    # Mass of the bound debris by azimuth about ``nrm``: bins [0, AZIMUTH_BINS) count all of it,
    # the next AZIMUTH_BINS only what lies beyond ``r_far``, and the last slot is that far mass.
    i = wp.tid()
    p = x[i] - centre
    r = wp.length(p)
    if r <= r_in:
        return
    u = v[i] - drift
    if 0.5 * wp.dot(u, u) - mu / r >= 0.0:
        return
    phi = wp.atan2(wp.dot(p, wp.cross(nrm, e1)), wp.dot(p, e1))
    b = int((phi + wp.pi) / (2.0 * wp.pi) * float(AZIMUTH_BINS)) % AZIMUTH_BINS
    wp.atomic_add(hist, b, m[i])
    if r > r_far:
        wp.atomic_add(hist, AZIMUTH_BINS + b, m[i])
        wp.atomic_add(hist, 2 * AZIMUTH_BINS, m[i])


# ----------------------------------------------------------------------------- sizing


@dataclass(frozen=True)
class MoonSettings:
    """The moon guide (see ``_gather``). Times are in dynamical times and lengths in radii of the
    most massive body, so a setting means the same thing at every scale.

    The timing defaults gather a finished moon about 40 dynamical times after the impact: at
    a tenth of one per frame that is inside the 400 frames the scrub cache holds for the Moon
    Birth template at its default count, so the whole birth can be scrubbed (ADR 0094). The
    slower first calibration (start 30, ramp 10, relax 5, approach 20, capture 1.5, slow 0.5)
    took about 1,600 frames and stalled on a full cache long before its moon."""

    #: When the moon's orbit is read off the debris and the guide starts easing in. By 20 the
    #: splash has fallen back and what is still in orbit is about a moon's worth; earlier the
    #: guide catches splash the planet would have taken back and builds a moon several times too
    #: heavy (5 lunar masses in orbit at 10).
    start: float = 20.0
    #: When the guide starts easing out again; 0 keeps it on. After it the moon orbits unaided.
    end: float = 0.0
    #: How long the guide takes to ease in (and out).
    ramp: float = 3.0
    #: Relaxation time of the steering toward the moon's orbit.
    relax: float = 1.5
    #: Time over which an offset from the orbit is closed (radial and vertical steering).
    approach_time: float = 5.0
    #: Time over which debris near the anchor takes on the anchor's velocity.
    capture_time: float = 0.75
    #: Steering speed cap, as a fraction of the local circular speed.
    steer_cap: float = 0.25
    #: Extra speed debris just behind the anchor runs at to catch it, as a fraction of circular.
    catch_up: float = 0.2
    #: How much slower than circular the rest of the debris runs while it waits for the anchor.
    #: Near 1 it all but stops and the anchor sweeps it up in under one orbit; at 0.5 the anchor
    #: gains on it at half the orbital rate and takes the better part of two.
    slow: float = 0.95
    #: How far behind the anchor (degrees) debris catches up rather than waits.
    window_degrees: float = 60.0
    #: Every steered speed stays below this fraction of the local escape speed.
    escape_cap: float = 0.85
    #: Distance from the anchor within which debris is captured onto it.
    reach: float = 1.0
    #: The planet's edge: nothing inside it is steered, and it takes the reaction.
    planet_edge: float = 1.3
    #: Radius of the moon's orbit; 0 reads it off the debris's angular momentum, clamped to
    #: between 1.3 and 2.2 Roche limits (``accretion_radius``).
    orbit_radius: float = 0.0


@dataclass(frozen=True)
class GiantImpactSettings:
    """Solver knobs. Lengths are multiples of the particle spacing, so a setting means the same
    thing at every particle count."""

    frame_dt: float = 0.1
    #: Pressure substeps per frame; 0 sizes them from the resolution (``substeps_for``).
    substeps: int = 0
    iterations: int = 3
    smoothing: float = 2.0
    softening: float = 1.0
    relaxation: float = 1.0
    cohesion: float = 0.3
    viscosity: float = 0.02
    cfl: float = 0.6
    heat_gain: float = 1.0
    cool_time: float = 8.0
    settle_steps: int = 120
    settle_damping: float = 0.85
    #: Accretion: debris beyond ``accretion_radius`` radii of the most massive body (its Roche
    #: limit) pulls on other such debris ``accretion_gain`` times harder, so the disk gathers
    #: into a moon in a shot's length rather than over months. 1 turns it off. With
    #: ``accretion_gain_final`` > 0 the gain rises geometrically to it between the two
    #: ``accretion_ramp`` times, given in dynamical times of the most massive body.
    accretion_radius: float = 2.9
    accretion_gain: float = 1.0
    accretion_gain_final: float = 0.0
    accretion_ramp: tuple[float, float] = (20.0, 60.0)
    #: The moon guide; None leaves the debris to gravity alone.
    moon: MoonSettings | None = None


def spacing_for(volume: float, count: int) -> float:
    """Particle spacing that fills ``volume`` with about ``count`` particles."""
    return (volume / count) ** (1.0 / 3.0)


def substeps_for(
    frame_dt: float, smoothing_length: float, characteristic_speed: float, cfl: float
) -> int:
    """Pressure substeps per frame so the fastest expected particle crosses at most ``cfl``
    smoothing lengths per substep. Finer particles need more substeps: the count grows as
    N^(1/3), which keeps the impact the same shape at every resolution."""
    v = EJECTA_SPEED_MARGIN * characteristic_speed
    return max(1, math.ceil(frame_dt * v / (cfl * smoothing_length) - 1.0e-9))


def lattice_ball(radius: float, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Particles filling a ball: a jittered, layer-offset lattice (so it does not read as a
    grid)."""
    n = int(math.ceil(radius / spacing)) + 1
    g = np.arange(-n, n + 1, dtype=np.float64) * spacing
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    odd = np.round(p[:, 2] / spacing) % 2
    p[:, 0] += odd * 0.5 * spacing
    p[:, 1] += odd * 0.5 * spacing
    p += rng.normal(scale=0.08 * spacing, size=p.shape)
    return p[np.linalg.norm(p, axis=1) <= radius]


def layered_masses(
    points: np.ndarray, radius: float, mass: float, core_fraction: float, core_density_ratio: float
):
    """Per-particle masses for a body with a denser core, and the material index (0 mantle,
    1 core)."""
    core = np.linalg.norm(points, axis=1) < core_fraction * radius
    w = np.where(core, core_density_ratio, 1.0)
    return w / w.sum() * mass, core.astype(np.int32)


# ----------------------------------------------------------------------------- solver


@wp.kernel
def _flag_nonfinite(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), flag: wp.array(dtype=int)
):
    # One sum per particle: an infinity or a NaN anywhere in it leaves the sum non-finite. Every
    # writer writes the same 1, so the race between them is harmless.
    i = wp.tid()
    p = x[i]
    q = v[i]
    if not wp.isfinite(p[0] + p[1] + p[2] + q[0] + q[1] + q[2]):
        flag[0] = 1


class _CellGrid:
    """Neighbour lists owned by one solver.

    ``wp.HashGrid`` cannot sit inside a captured graph when anything else in the process also
    builds one: its build uploads the grid's descriptor from a single process-wide host slot,
    and a replayed graph re-reads that slot, so it gets whichever grid was built last (Warp
    1.15, ``native/hashgrid.cpp``). Here every buffer belongs to the solver, and the sort runs
    on the solver's own stream, whose radix-sort scratch (cached per stream by Warp) nothing
    else can grow and so move.
    """

    def __init__(self, n: int, cell_width: float, device):
        self.n = n
        self.device = device
        # The reciprocal as wp.HashGrid forms it (float32), so cells match it exactly.
        self.inv = float(np.float32(1.0) / np.float32(cell_width))
        self.keys = wp.zeros(2 * n, dtype=int, device=device)  # the sort needs 2n of each
        self.ids = wp.zeros(2 * n, dtype=int, device=device)
        self.start = wp.zeros(GRID_DIM**3, dtype=int, device=device)
        self.end = wp.zeros(GRID_DIM**3, dtype=int, device=device)

    def build(self, points):
        wp.launch(
            _cell_keys,
            dim=self.n,
            inputs=[points, self.inv],
            outputs=[self.keys, self.ids],
            device=self.device,
        )
        wp.utils.radix_sort_pairs(self.keys, self.ids, self.n)
        self.start.zero_()
        self.end.zero_()
        wp.launch(
            _cell_ranges,
            dim=self.n,
            inputs=[self.keys, self.n],
            outputs=[self.start, self.end],
            device=self.device,
        )

    def args(self) -> list:
        return [self.start, self.end, self.ids, self.inv]


class GiantImpactSolver:
    """Fixed-count particle state for colliding self-gravitating bodies.

    ``positions``, ``velocities`` (N x 3), ``masses`` (N) and ``body_ids`` (N, int) describe the
    bodies at their authored centres and velocities. ``characteristic_speed`` sizes the substep
    (the speed at which the bodies meet is the natural choice).

    Any number of solvers can live in one process: each owns its buffers, its neighbour grid and
    its CUDA stream, and the caller's stream waits for a solver's work before reading it.
    """

    def __init__(
        self,
        positions,
        velocities,
        masses,
        body_ids,
        *,
        spacing: float,
        characteristic_speed: float,
        gravitational_constant: float = 1.0,
        settings: GiantImpactSettings | None = None,
        device=None,
        use_graph: bool = True,
    ):
        self.settings = s = settings or GiantImpactSettings()
        self.device = wp.get_device(device)
        self.n = int(len(positions))
        self.n_pad = int(math.ceil(self.n / TILE) * TILE)
        self.spacing = float(spacing)
        self.h = s.smoothing * spacing
        self.eps2 = (s.softening * spacing) ** 2
        self.kp = 315.0 / (64.0 * math.pi * self.h**9)
        self.ks = 45.0 / (math.pi * self.h**6)
        for name, value in (('kp', self.kp), ('ks', self.ks)):
            # Both are handed to the kernels as float32; an unscaled-SI h can overflow or flush
            # them to zero, which would silently zero the density and the pressure.
            with np.errstate(over='ignore', under='ignore'):
                f32 = float(np.float32(value))
            if not math.isfinite(f32) or f32 == 0.0:
                raise ValueError(
                    f'smoothing length {self.h!r} makes the SPH kernel constant {name} '
                    'unrepresentable in float32; use scaled units'
                )
        self.substeps = s.substeps or substeps_for(s.frame_dt, self.h, characteristic_speed, s.cfl)
        self.dt = s.frame_dt / self.substeps
        self.vmax = s.cfl * self.h / self.dt
        self.frame = 0

        masses = np.asarray(masses, dtype=np.float64)
        self._m_host = masses
        self._body_ids = np.asarray(body_ids, dtype=np.int32)
        self._authored_x = np.asarray(positions, dtype=np.float64)
        self._authored_v = np.asarray(velocities, dtype=np.float64)

        dev = self.device
        self.stream = wp.Stream(dev) if dev.is_cuda else None

        def arr(a, t):
            return wp.array(np.asarray(a, dtype=np.float32), dtype=t, device=dev)

        self.x = arr(positions, wp.vec3)
        self.v = arr(velocities, wp.vec3)
        self.m = arr(masses, float)
        self.gm = arr(masses * gravitational_constant, float)
        self.invm = arr(np.median(masses) / masses, float)
        self.xp = wp.zeros_like(self.x)
        self.xq = wp.zeros_like(self.x)
        self.vp = wp.zeros_like(self.x)
        self.vq = wp.zeros_like(self.x)
        self.acc = wp.zeros_like(self.x)
        self.pm = wp.zeros(self.n_pad, dtype=wp.vec4, device=dev)
        self.lam = wp.zeros(self.n, dtype=float, device=dev)
        self.dens = wp.zeros(self.n, dtype=float, device=dev)
        self.s2 = wp.zeros(self.n, dtype=float, device=dev)
        self.heat = wp.zeros(self.n, dtype=float, device=dev)
        self.grid = _CellGrid(self.n, self.h, dev)

        # The most massive body sets the scene's scales: its radius (from its particle volume,
        # so its settled size) places the Roche limit, and its dynamical time clocks the ramp.
        _, body_index = np.unique(self._body_ids, return_inverse=True)
        body_mass = np.bincount(body_index, weights=masses)
        body_count = np.bincount(body_index)
        big = int(np.argmax(body_mass))
        self.body_radius = (
            3.0 * body_count[big] * self.spacing**3 / (4.0 * math.pi)
        ) ** (1.0 / 3.0)
        self.dynamical_time = math.sqrt(
            self.body_radius**3 / (gravitational_constant * masses.sum())
        )
        self.accretion_r2 = (s.accretion_radius * self.body_radius) ** 2
        # Read by the captured graph from device memory, so they are set afresh each frame.
        self.gain = wp.array([s.accretion_gain], dtype=float, device=dev)
        self.centre = wp.zeros(1, dtype=wp.vec3, device=dev)
        self._centre0 = (self._authored_x * masses[:, None]).sum(0) / masses.sum()
        self._drift = (self._authored_v * masses[:, None]).sum(0) / masses.sum()
        self._mu = float(gravitational_constant * masses.sum())
        # The least debris the guide will gather into a moon: one smoothing kernel's worth of
        # particles, the fewest the pressure solve holds together as a body. Less than that in
        # orbit (a head-on merger, a graze that throws nothing up) and there is no moon to make.
        self._moon_floor_mass = math.ceil(4.0 / 3.0 * math.pi * s.smoothing**3) * float(
            np.median(masses)
        )
        # The moon guide's uniforms (zero rate until it starts) and reduction targets. The
        # orbit is read off the debris once, when the guide starts, and is part of the state.
        self.gp = wp.zeros(_GP_SIZE, dtype=float, device=dev)
        self.inner = wp.zeros(1, dtype=float, device=dev)
        self.react = wp.zeros(1, dtype=wp.vec3, device=dev)
        self.orbit: dict | None = None
        self.bad = wp.zeros(1, dtype=int, device=dev)
        self._use_graph = bool(use_graph and dev.is_cuda)
        self._graph = None
        with self._on_stream():
            self._calibrate()

    def _on_stream(self):
        # Everything runs on the solver's stream; on exit the caller's stream waits for it, so a
        # read made afterwards on the caller's stream sees the finished work.
        return wp.ScopedStream(self.stream, sync_enter=True, sync_exit=True)

    # -- setup

    def _lambda_pass(self, rest: float, relax: float, cohesion: float):
        wp.launch(
            _lambda,
            dim=self.n,
            inputs=[
                *self.grid.args(),
                self.xp,
                self.invm,
                rest,
                self.h,
                self.kp,
                self.ks,
                relax,
                cohesion,
            ],
            outputs=[self.lam, self.dens, self.s2],
            device=self.device,
        )

    def _calibrate(self):
        # Rest number density and constraint stiffness, read off the initial interior.
        wp.copy(self.xp, self.x)
        self.grid.build(self.xp)
        self._lambda_pass(1.0, 1.0, 0.0)
        dens = self.dens.numpy()
        interior = dens > np.percentile(dens, 40)
        self.rest = float(np.median(dens[interior]))
        self._lambda_pass(self.rest, 1.0, 0.0)
        self.relax = float(self.settings.relaxation * np.median(self.s2.numpy()[interior]))

    def settle(self, centres=None):
        """Relax each body into its own hydrostatic shape, then put its centre of mass back at
        ``centres[body]`` (default: where its authored particles had it) with its authored
        velocity. Run once, before the first frame."""
        s = self.settings
        with self._on_stream():
            self.v.zero_()
            for _ in range(s.settle_steps):
                self._substep(apply_gravity=True, damping=s.settle_damping)
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
            self.v.assign(self._authored_v.astype(np.float32))
            self.heat.zero_()
            # The barycentre coasts in a straight line (momentum is conserved), so where it is
            # at any frame is known without a reduction.
            self._centre0 = (x * m[:, None]).sum(0) / m.sum()
            self.frame = 0
            self._set_frame_uniforms()
            self._gravity()  # the first half-kick needs the acceleration at the starting positions

    # -- stepping

    def accretion_gain_at(self, t: float) -> float:
        """The accretion gain at scene time ``t``."""
        s = self.settings
        if s.accretion_gain_final <= 0.0 or s.accretion_gain <= 0.0:
            return s.accretion_gain
        t0, t1 = (r * self.dynamical_time for r in s.accretion_ramp)
        f = min(max((t - t0) / max(t1 - t0, 1.0e-12), 0.0), 1.0)
        return s.accretion_gain * (s.accretion_gain_final / s.accretion_gain) ** f

    def _set_frame_uniforms(self):
        # This frame's gain, and the barycentre at its closing gravity pass. Written before the
        # frame is recorded or replayed, never inside a capture.
        t = self.frame * self.settings.frame_dt
        self.gain.fill_(self.accretion_gain_at(t))
        c = self._centre0 + self._drift * (t + self.settings.frame_dt)
        self.centre.fill_(wp.vec3(float(c[0]), float(c[1]), float(c[2])))
        if self.settings.moon is not None:
            self.gp.assign(self._moon_uniforms(t + self.settings.frame_dt))

    def _moon_uniforms(self, t_end: float) -> np.ndarray:
        # The guide acts at the frame's closing gravity pass, at scene time t_end.
        s = self.settings
        moon = s.moon
        td, rb = self.dynamical_time, self.body_radius
        gp = np.zeros(_GP_SIZE, dtype=np.float32)
        since = t_end - moon.start * td
        if since < 0.0:
            return gp
        if self.orbit is None:
            self.orbit = self._choose_orbit()
        o = self.orbit
        if o["debris_mass"] < self._moon_floor_mass:
            return gp  # too little is in orbit to make a moon from
        ramp = moon.ramp * td
        scale = min(since / ramp, 1.0) if ramp > 0.0 else 1.0
        if moon.end > 0.0:
            after = t_end - moon.end * td
            if after > 0.0:
                scale *= max(1.0 - after / ramp, 0.0) if ramp > 0.0 else 0.0
        th = o["theta0"] + o["omega"] * since
        gp[:] = [
            scale / (moon.relax * td),
            1.0 / (moon.approach_time * td),
            moon.steer_cap,
            moon.reach * rb,
            o["radius"],
            math.cos(th),
            math.sin(th),
            o["speed"],
            *o["normal"],
            *o["e1"],
            moon.planet_edge * rb,
            s.accretion_radius * rb,
            moon.catch_up,
            moon.escape_cap,
            moon.slow,
            math.radians(moon.window_degrees),
            scale,
            1.0 / (moon.capture_time * td),
            *self._drift,
            self._mu,
        ]
        return gp

    def _choose_orbit(self) -> dict:
        """The moon's orbit, read once off the debris at the start of this frame: its plane and
        radius from the bound debris's angular momentum, its phase at the debris's densest
        azimuth. Both are reductions on the device; only their few sums come back."""
        s = self.settings
        moon = s.moon
        dev = self.device
        rb = self.body_radius
        c = self._centre0 + self._drift * (self.frame * s.frame_dt)
        centre = wp.vec3(float(c[0]), float(c[1]), float(c[2]))
        drift = wp.vec3(float(self._drift[0]), float(self._drift[1]), float(self._drift[2]))
        r_in = moon.planet_edge * rb
        roche = s.accretion_radius * rb
        lsum = wp.zeros(1, dtype=wp.vec3, device=dev)
        msum = wp.zeros(1, dtype=float, device=dev)
        wp.launch(
            _debris_moments,
            dim=self.n,
            inputs=[self.x, self.v, self.m, centre, drift, self._mu, r_in],
            outputs=[lsum, msum],
            device=dev,
        )
        lvec = lsum.numpy()[0].astype(np.float64)
        md = float(msum.numpy()[0])
        lnorm = float(np.linalg.norm(lvec))
        if md <= 0.0 or lnorm <= 0.0:
            return {
                "normal": [0.0, 0.0, 1.0],
                "e1": [1.0, 0.0, 0.0],
                "radius": 0.0,
                "radius_from_L": 0.0,
                "theta0": 0.0,
                "speed": 0.0,
                "omega": 0.0,
                "debris_mass": 0.0,
            }
        nrm = lvec / lnorm
        # The circular radius carrying the debris's mean specific angular momentum, clamped to a
        # band just outside the Roche limit where a moon holds together and still fills the frame.
        a_l = (lnorm / md) ** 2 / self._mu
        if moon.orbit_radius > 0.0:
            a = moon.orbit_radius * rb
        else:
            a = float(np.clip(a_l, 1.3 * roche, 2.2 * roche))
        ref = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, nrm)
        e1 /= np.linalg.norm(e1)
        nb = AZIMUTH_BINS
        hist = wp.zeros(2 * nb + 1, dtype=float, device=dev)
        wp.launch(
            _debris_azimuth,
            dim=self.n,
            inputs=[self.x, self.v, self.m, centre, drift, self._mu, r_in, roche,
                    wp.vec3(*(float(q) for q in nrm)), wp.vec3(*(float(q) for q in e1))],
            outputs=[hist],
            device=dev,
        )
        h = hist.numpy().astype(np.float64)
        # Phase at the densest azimuth of the debris beyond the Roche limit, or of all of it
        # when too little lies that far out.
        counts = h[nb : 2 * nb] if h[2 * nb] >= 0.25 * md else h[:nb]
        smooth = sum(
            k * np.roll(counts, shift)
            for k, shift in zip((1.0, 2.0, 3.0, 2.0, 1.0), range(-2, 3), strict=True)
        )
        theta0 = -math.pi + (int(np.argmax(smooth)) + 0.5) * 2.0 * math.pi / nb
        speed = math.sqrt(self._mu / a)
        return {
            "normal": [float(q) for q in nrm],
            "e1": [float(q) for q in e1],
            "radius": float(a),
            "radius_from_L": float(a_l),
            "theta0": float(theta0),
            "speed": speed,
            "omega": speed / a,
            "debris_mass": md,
        }

    def moon_anchor(self):
        """Where the moon guide's anchor is at the current frame (world coordinates), or None
        while there is none."""
        moon = self.settings.moon
        o = self.orbit
        if moon is None or o is None or o["debris_mass"] < self._moon_floor_mass:
            return None
        t = self.frame * self.settings.frame_dt
        th = o["theta0"] + o["omega"] * (t - moon.start * self.dynamical_time)
        e1 = np.asarray(o["e1"])
        e2 = np.cross(np.asarray(o["normal"]), e1)
        orbit = o["radius"] * (math.cos(th) * e1 + math.sin(th) * e2)
        return self._centre0 + self._drift * t + orbit

    def _gravity(self):
        wp.launch(
            _pack,
            dim=self.n_pad,
            inputs=[self.x, self.gm, self.n, self.centre, self.accretion_r2],
            outputs=[self.pm],
            device=self.device,
        )
        wp.launch(
            _gravity,
            dim=self.n_pad,
            inputs=[self.pm, self.n_pad // TILE, self.eps2, self.gain, self.n],
            outputs=[self.acc],
            block_dim=TILE,
            device=self.device,
        )
        if self.settings.moon is not None:
            # Launched every frame so the captured graph is the same before, during and after
            # the guide; while its rate is zero both kernels return at once.
            self.inner.zero_()
            self.react.zero_()
            wp.launch(
                _gather,
                dim=self.n,
                inputs=[self.x, self.v, self.m, self.centre, self.gp],
                outputs=[self.inner, self.acc, self.react],
                device=self.device,
            )
            wp.launch(
                _react,
                dim=self.n,
                inputs=[self.x, self.centre, self.gp, self.inner, self.react],
                outputs=[self.acc],
                device=self.device,
            )

    def _substep(self, apply_gravity: bool, damping: float = 1.0):
        s = self.settings
        dev = self.device
        if apply_gravity:
            self._gravity()
        wp.launch(
            _predict,
            dim=self.n,
            inputs=[self.x, self.v, self.acc, self.dt, 1.0 if apply_gravity else 0.0, self.vmax],
            outputs=[self.xp, self.vp],
            device=dev,
        )
        self.grid.build(self.xp)
        for _ in range(s.iterations):
            self._lambda_pass(self.rest, self.relax, s.cohesion)
            wp.launch(
                _delta,
                dim=self.n,
                inputs=[
                    *self.grid.args(),
                    self.x,
                    self.xp,
                    self.invm,
                    self.rest,
                    self.lam,
                    self.h,
                    self.ks,
                ],
                outputs=[self.xq],
                device=dev,
            )
            wp.copy(self.xp, self.xq)
        wp.launch(
            _update,
            dim=self.n,
            inputs=[
                self.x,
                self.xp,
                self.vp,
                self.v,
                self.heat,
                1.0 / self.dt,
                s.heat_gain,
                math.exp(-self.dt / s.cool_time),
            ],
            device=dev,
        )
        if s.viscosity > 0.0:
            wp.launch(
                _smooth,
                dim=self.n,
                inputs=[
                    *self.grid.args(),
                    self.x,
                    self.v,
                    self.m,
                    self.dens,
                    self.h,
                    self.kp,
                    s.viscosity,
                ],
                outputs=[self.vq],
                device=dev,
            )
            wp.copy(self.v, self.vq)
        if damping != 1.0:
            wp.launch(_damp, dim=self.n, inputs=[self.v, damping], device=dev)

    def _record_frame(self):
        # Gravity (slow, long-range) kicks as half-impulses at the frame boundaries, where every
        # position it was evaluated at is the position it acts on; the pressure solve (fast,
        # short-range) sub-cycles in between with no gravity.
        half = 0.5 * self.settings.frame_dt
        wp.launch(_kick, dim=self.n, inputs=[self.v, self.acc, half], device=self.device)
        for _ in range(self.substeps):
            self._substep(apply_gravity=False)
        self._gravity()
        wp.launch(_kick, dim=self.n, inputs=[self.v, self.acc, half], device=self.device)

    def step_frame(self):
        """Advance exactly one displayed frame."""
        with self._on_stream():
            self._set_frame_uniforms()
            if not self._use_graph:
                self._record_frame()
            elif self._graph is None:
                # The first frame runs eagerly: it is a real frame, and it warms the allocator
                # (the sort's scratch) so nothing allocates inside the capture that follows
                # (capture itself runs nothing).
                self._record_frame()
                with wp.ScopedCapture(device=self.device, stream=self.stream) as cap:
                    self._record_frame()
                self._graph = cap.graph
            else:
                wp.capture_launch(self._graph, stream=self.stream)
        self.frame += 1

    # -- read-back

    def state(self, include_acceleration: bool = False) -> dict:
        """Host copies of the particle state (float64). ``include_acceleration`` adds the
        acceleration the next frame opens with, which ``restore`` needs to resume exactly."""
        out = {
            "positions": self.x.numpy().astype(np.float64),
            "velocities": self.v.numpy().astype(np.float64),
            "heat": self.heat.numpy().astype(np.float64),
        }
        if include_acceleration:
            out["accelerations"] = self.acc.numpy().astype(np.float64)
        return out

    def invariants(self) -> dict:
        """Total momentum, and angular momentum about the centre of mass (float64, host)."""
        st = self.state()
        m = self._m_host
        x, v = st["positions"], st["velocities"]
        com = (x * m[:, None]).sum(0) / m.sum()
        p = (v * m[:, None]).sum(0)
        L = (m[:, None] * np.cross(x - com, v - p / m.sum())).sum(0)
        return {"com": com, "momentum": p, "angular_momentum": L}

    def checkpoint(self) -> dict:
        """The solver's state beyond the particle arrays: what ``restore`` needs alongside
        ``state()`` to resume exactly where this frame left off."""
        return {
            "frame": int(self.frame),
            "orbit": copy.deepcopy(self.orbit),
            "barycentre": [float(c) for c in self._centre0],
        }

    def restore(self, positions, velocities, heat, checkpoint: dict, accelerations=None):
        """Resume from ``state()`` and ``checkpoint()`` values taken together at one frame.

        The acceleration a frame opens with was taken at the close of the frame before it,
        before that frame's last half-kick. The moon guide's share of it depends on the
        velocity it saw then, which the stored (post-kick) velocity is not, so an exact resume
        needs the acceleration itself: pass ``state(include_acceleration=True)["accelerations"]``.
        Without it the acceleration is rebuilt under the previous frame's uniforms, which is
        exact while the guide is idle and off by the guide's velocity dependence over half a
        kick once it runs. The captured graph stays valid: it reads the same arrays, and only
        their contents change."""
        frame = int(checkpoint["frame"])
        if frame < 0:
            raise ValueError("a checkpoint frame cannot be negative")
        arrays = [("positions", positions, 3), ("velocities", velocities, 3), ("heat", heat, 0)]
        if accelerations is not None:
            arrays.append(("accelerations", accelerations, 3))
        for name, a, width in arrays:
            shape = np.shape(a)
            if shape != ((self.n, width) if width else (self.n,)):
                raise ValueError(f"{name} has shape {shape}, expected {self.n} rows")
        with self._on_stream():
            self.x.assign(np.asarray(positions, dtype=np.float32))
            self.v.assign(np.asarray(velocities, dtype=np.float32))
            self.heat.assign(np.asarray(heat, dtype=np.float32))
            self._centre0 = np.asarray(checkpoint["barycentre"], dtype=np.float64)
            self.orbit = copy.deepcopy(checkpoint["orbit"])
            # Frame 0's opening acceleration is the one ``settle`` takes, under frame 0's uniforms.
            self.frame = max(frame - 1, 0)
            self._set_frame_uniforms()
            if accelerations is None:
                self._gravity()
            else:
                self.acc.assign(np.asarray(accelerations, dtype=np.float32))
            # The density the display publishes is only written inside a substep; refresh it
            # from the restored positions so the first frame shown after a restore is not the
            # replaced state's. Scratch only: every substep rebuilds ``xp``, the grid and ``dens``
            # before reading them.
            wp.copy(self.xp, self.x)
            self.grid.build(self.xp)
            self._lambda_pass(self.rest, self.relax, self.settings.cohesion)
            self.frame = frame

    def nonfinite(self) -> bool:
        """Whether any position or velocity has gone non-finite: one launch, and one int read
        back (which waits for the solver's work)."""
        with self._on_stream():
            self.bad.zero_()
            wp.launch(
                _flag_nonfinite,
                dim=self.n,
                inputs=[self.x, self.v],
                outputs=[self.bad],
                device=self.device,
            )
        return bool(self.bad.numpy()[0])
