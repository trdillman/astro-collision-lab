"""CUDA Warp kernels for galaxy models: ring field, frozen-field leapfrog, softened potential.

Both kernels are float64 ports of the numpy reference in ``galaxy_models``
(``ring_field`` and the ``_frozen_accel`` leapfrog of ``relaxed_disk_library``).
Each thread owns one grid point or one tracer and accumulates in a fixed order,
so results are deterministic for a given device. Only summation order and
rounding differ from numpy.
"""
import math

import numpy as np
import warp as wp

PI = wp.constant(wp.float64(math.pi))
AGM_STEPS = wp.constant(12)
M_LIMIT = wp.constant(wp.float64(1.0 - 1e-15))


@wp.func
def _ellip_ke(m: wp.float64):
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    a = one
    b = wp.sqrt(wp.max(one - m, wp.float64(0.0)))
    total = half * m
    power = half
    for _k in range(AGM_STEPS):
        c = half * (a - b)
        mean = half * (a + b)
        b = wp.sqrt(a * b)
        a = mean
        power = power * wp.float64(2.0)
        total = total + power * c * c
    K = PI / (wp.float64(2.0) * a)
    return K, K * (one - total)


@wp.kernel
def _ring_field_kernel(R: wp.array(dtype=wp.float64), Z: wp.array(dtype=wp.float64),
                       ring_r: wp.array(dtype=wp.float64), ring_z: wp.array(dtype=wp.float64),
                       ring_m: wp.array(dtype=wp.float64), chunk: int, G: wp.float64, eps: wp.float64,
                       FR: wp.array2d(dtype=wp.float64), FZ: wp.array2d(dtype=wp.float64)):
    # One thread per (grid point, ring chunk); the host sums chunks in a fixed order.
    i, part = wp.tid()
    r = R[i]
    z = Z[i]
    half = wp.float64(0.5)
    two = wp.float64(2.0)
    safe = wp.max(r, wp.float64(1e-300))
    fr = wp.float64(0.0)
    fz = wp.float64(0.0)
    for k in range(part * chunk, wp.min((part + 1) * chunk, ring_r.shape[0])):
        h = z - ring_z[k]
        base = h * h + eps * eps
        s = r + ring_r[k]
        d = r - ring_r[k]
        D = s * s + base
        near = d * d + base
        K, E = _ellip_ke(wp.min(wp.float64(4.0) * r * ring_r[k] / D, M_LIMIT))
        c = two * G * ring_m[k] / PI / wp.sqrt(D)
        fz = fz - c * h * E / near
        fr = fr + c * (E * D / near * (half / safe - s / D) - K / (two * safe))
    if r > wp.float64(0.0):
        FR[i, part] = fr
    else:
        FR[i, part] = wp.float64(0.0)
    FZ[i, part] = fz


@wp.func
def _cell(g: wp.array(dtype=wp.float64), x: wp.float64):
    # numpy: clip(searchsorted(g, x, 'left') - 1, 0, len(g) - 2)
    lo = int(0)  # noqa: RUF046 - Warp: the cast makes lo a mutable dynamic variable
    hi = g.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if g[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return wp.clamp(lo - 1, 0, g.shape[0] - 2)


@wp.func
def _accel(p: wp.vec3d, Rg: wp.array(dtype=wp.float64), zg: wp.array(dtype=wp.float64),
           FRt: wp.array2d(dtype=wp.float64), FZt: wp.array2d(dtype=wp.float64),
           disk_mass: wp.float64, spheroids: wp.array(dtype=wp.vec3d), G: wp.float64, eps: wp.float64):
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    R = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    az = wp.abs(p[2])
    i = _cell(Rg, R)
    j = _cell(zg, az)
    tx = wp.clamp((R - Rg[i]) / (Rg[i + 1] - Rg[i]), zero, one)
    ty = wp.clamp((az - zg[j]) / (zg[j + 1] - zg[j]), zero, one)
    w00 = (one - tx) * (one - ty)
    w10 = tx * (one - ty)
    w01 = (one - tx) * ty
    w11 = tx * ty
    fr = FRt[i, j] * w00 + FRt[i + 1, j] * w10 + FRt[i, j + 1] * w01 + FRt[i + 1, j + 1] * w11
    fz = FZt[i, j] * w00 + FZt[i + 1, j] * w10 + FZt[i, j + 1] * w01 + FZt[i + 1, j + 1] * w11
    r2 = R * R + az * az
    if R > Rg[Rg.shape[0] - 1] or az > zg[zg.shape[0] - 1]:
        d3 = wp.pow(r2 + eps * eps, wp.float64(1.5))
        fr = -G * disk_mass * R / d3
        fz = -G * disk_mass * az / d3
    r = wp.sqrt(r2)
    for k in range(spheroids.shape[0]):
        mass = spheroids[k][0]
        radius = spheroids[k][1]
        a = spheroids[k][2]
        x = radius / a
        norm = x * x * x / wp.pow(one + x * x, wp.float64(1.5))
        menc = mass
        if r < radius:
            menc = mass / norm * r * r * r / wp.pow(r2 + a * a, wp.float64(1.5))
        g = G * menc / wp.max(r2 + eps * eps, wp.float64(1e-30)) / wp.max(r, wp.float64(1e-12))
        fr = fr - g * R
        fz = fz - g * az
    inv = zero
    if R > zero:
        inv = one / wp.max(R, wp.float64(1e-12))
    sign = zero
    if p[2] > zero:
        sign = one
    elif p[2] < zero:
        sign = -one
    return wp.vec3d(fr * p[0] * inv, fr * p[1] * inv, fz * sign)


@wp.kernel
def _leapfrog_kernel(p0: wp.array(dtype=wp.vec3d), v0: wp.array(dtype=wp.vec3d), steps: int, dt: wp.float64,
                     Rg: wp.array(dtype=wp.float64), zg: wp.array(dtype=wp.float64),
                     FRt: wp.array2d(dtype=wp.float64), FZt: wp.array2d(dtype=wp.float64),
                     disk_mass: wp.float64, spheroids: wp.array(dtype=wp.vec3d), G: wp.float64, eps: wp.float64,
                     p_out: wp.array(dtype=wp.vec3d), v_out: wp.array(dtype=wp.vec3d)):
    t = wp.tid()
    p = p0[t]
    v = v0[t]
    kick = wp.float64(0.5) * dt
    a = _accel(p, Rg, zg, FRt, FZt, disk_mass, spheroids, G, eps)
    for _s in range(steps):
        v = v + kick * a
        p = p + dt * v
        a = _accel(p, Rg, zg, FRt, FZt, disk_mass, spheroids, G, eps)
        v = v + kick * a
    p_out[t] = p
    v_out[t] = v


@wp.kernel
def _potential_kernel(p: wp.array(dtype=wp.vec3), m: wp.array(dtype=wp.float64), eps2: wp.float32,
                      rows: wp.array(dtype=wp.float64)):
    # Row i of -sum_{j != i} m_i m_j / sqrt(|x_i - x_j|^2 + eps^2); float32 positions
    # (the solver's own precision), float64 accumulation in a fixed order.
    i = wp.tid()
    x = p[i]
    total = wp.float64(0.0)
    for j in range(p.shape[0]):
        if j != i:
            d = p[j] - x
            total = total + m[j] / wp.float64(wp.sqrt(wp.dot(d, d) + eps2))
    rows[i] = -m[i] * total


def softened_potential(p, m, G, eps, dev):
    """Exact Plummer-softened potential energy of all particle pairs (each pair once)."""
    n = len(p)
    rows = wp.empty(n, dtype=wp.float64, device=dev)
    wp.launch(_potential_kernel, dim=n,
              inputs=[wp.array(np.ascontiguousarray(p, dtype=np.float32), dtype=wp.vec3, device=dev),
                      _f64(m, dev), float(eps) ** 2], outputs=[rows], device=dev)
    return .5 * float(G) * float(np.sum(rows.numpy()))


def device():
    """The CUDA device the sampler kernels run on, or None when CUDA is unavailable."""
    return "cuda:0" if wp.is_cuda_available() else None


def _f64(x, dev):
    return wp.array(np.ascontiguousarray(x, dtype=np.float64), dtype=wp.float64, device=dev)


def ring_field(R, z, ring_r, ring_z, ring_m, G, eps, dev):
    R = np.asarray(R, float).ravel()
    z = np.broadcast_to(np.asarray(z, float).ravel(), R.shape)
    chunk = 256
    parts = max(1, -(-len(ring_r) // chunk))
    FR = wp.empty((len(R), parts), dtype=wp.float64, device=dev)
    FZ = wp.empty((len(R), parts), dtype=wp.float64, device=dev)
    wp.launch(_ring_field_kernel, dim=(len(R), parts),
              inputs=[_f64(R, dev), _f64(z, dev), _f64(ring_r, dev), _f64(ring_z, dev), _f64(ring_m, dev),
                      chunk, float(G), float(eps)], outputs=[FR, FZ], device=dev)
    return FR.numpy().sum(axis=1), FZ.numpy().sum(axis=1)


def evolve(p, v, steps, dt, field, disk_mass, spheroids, G, eps, dev):
    """Leapfrog every tracer ``steps`` times in the frozen field; returns (p, v) as float64 numpy."""
    Rg, zg, FRt, FZt = field
    sph = np.asarray(spheroids, dtype=np.float64).reshape(-1, 3)
    n = len(p)
    p_out = wp.empty(n, dtype=wp.vec3d, device=dev)
    v_out = wp.empty(n, dtype=wp.vec3d, device=dev)
    wp.launch(_leapfrog_kernel, dim=n,
              inputs=[wp.array(np.ascontiguousarray(p, dtype=np.float64), dtype=wp.vec3d, device=dev),
                      wp.array(np.ascontiguousarray(v, dtype=np.float64), dtype=wp.vec3d, device=dev),
                      int(steps), float(dt), _f64(Rg, dev), _f64(zg, dev),
                      wp.array(np.ascontiguousarray(FRt, dtype=np.float64), dtype=wp.float64, device=dev),
                      wp.array(np.ascontiguousarray(FZt, dtype=np.float64), dtype=wp.float64, device=dev),
                      float(disk_mass), wp.array(sph, dtype=wp.vec3d, device=dev), float(G), float(eps)],
              outputs=[p_out, v_out], device=dev)
    return p_out.numpy().astype(np.float64), v_out.numpy().astype(np.float64)


def warm(dev=None):
    """Compile and run both kernels once so the interactive process loads them from cache."""
    dev = dev or device()
    if dev is None:
        return False
    g = np.array([0., .5, 1.])
    ring_field(g, g, np.array([.5]), np.array([0.]), np.array([1.]), 1., .03, dev)
    table = np.zeros((3, 3))
    evolve(np.array([[.2, .1, .05]]), np.zeros((1, 3)), 2, .01, (g, g, table, table), 1.,
           [(1., 2., .5)], 1., .03, dev)
    softened_potential(np.zeros((2, 3)), np.ones(2), 1., .03, dev)
    wp.synchronize_device(dev)
    return True
