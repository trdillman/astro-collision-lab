"""A local shearing box: Hill's equations with shearing-periodic boundaries (ADR 0202).

A small patch of a differentially rotating disk -- a piece of a planetary ring, a galactic disk,
a protoplanetary disk -- is simulated in a frame that turns with the patch at the angular speed
``Omega`` of its centre, and the patch is a box ``Lx`` by ``Ly`` (and unbounded in z) that
repeats itself forever. This is Hill's approximation, the standard "shearing sheet" of Goldreich
and Lynden-Bell (1965), Hawley, Gammie and Balbus (1995) and Rein and Tremaine (2011). It needs
no central body: the central body's pull is replaced by its tidal field, which is what makes
the box *local*.

**The motion.** With ``x`` radial (outward), ``y`` azimuthal (the direction of the rotation)
and ``z`` vertical, a particle obeys

    x'' =  2 Omega y' + 2 q Omega^2 x + a_x      (the tidal term and the Coriolis term)
    y'' = -2 Omega x'                  + a_y      (the Coriolis term)
    z'' =               - Omega^2 z    + a_z      (the vertical pull of the central body)

where ``a`` is the mutual gravity of the particles and ``q = 3/2`` is the Keplerian shear rate
(``q = -dln(Omega)/dln(r)``), so the tidal term is ``3 Omega^2 x`` and the epicyclic frequency is
``kappa = sqrt(2 (2 - q)) Omega = Omega``. Without ``a`` the equations are linear and solved
exactly (:func:`epicycle_map`); that exact solution is the drift of the integrator, and the
mutual gravity is the kick, which is the symplectic splitting of Rein and Tremaine (2011),
"SEI". The map is exact for any step, so ``Omega * dt`` does not have to be small for stability:
the only error is the kick/drift splitting of the mutual gravity.

**The boundaries.** The background flow is ``v_y = -q Omega x``, so the copy of the box one box
length out in ``x`` slides along ``y`` at ``-q Omega Lx`` relative to this one. A particle that
leaves through ``x > Lx/2`` therefore re-enters through ``x < -Lx/2`` with

    x -= Lx,   y += q Omega Lx t,   v_y += q Omega Lx

and the other way round for ``x < -Lx/2``. ``y`` is plainly periodic. A particle on a circular
orbit (``v_y = -q Omega x``) re-enters on a circular orbit, and the amplitude of an eccentric
one is unchanged: those are the two things the wrap must preserve, and the tests hold it to
them.

**The convention.** The box is centred on the world origin (``x = y = 0``), and every authored
velocity is measured *relative to the circular shear flow* ``(0, -q Omega x, 0)``: a body or a
cloud authored with no velocity is on a circular orbit, so it stays where it is in ``x`` and
drifts along ``y`` with its neighbours, instead of falling out of equilibrium. Any velocity model
composes with that (:func:`astro_warp.product.sampling.expand_bindings` adds the flow once,
after the model has drawn its velocities). A body authored outside the box is the same body in
another copy of it and is brought in, with the shear shift, when the runtime loads.

**The gravity.** The mutual pull of the particles is summed over the box and its nearest images
(by default the 3 x 3 in the plane, ``images = 1``), each image displaced by ``i Lx`` in ``x`` and
by ``-i q Omega Lx t + j Ly`` in ``y`` -- the same sliding the boundary uses, and the identity
that keeps the force continuous when a particle crosses. There is no Ewald sum: the far field
beyond the nearest images is dropped, as it is in REBOUND's shearing box and in every direct-sum
shearing sheet. A particle is not attracted by its own images (they cancel by symmetry).

**What it is not.** No pressure, no viscosity, no collisions (the collision stage assumes an
unwrapped, inertial space), no external fields, emitters or pinned bodies, and no Barnes-Hut
tree: the tree builds one open octree and has no images. Only the Direct solver runs a box.

Everything here is NumPy except :func:`shearing_kernels`, which builds the Warp kernels the Direct
runtime launches. The NumPy functions are the oracle the kernels are tested against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SHEAR_Q = 1.5
"""The Keplerian shear rate ``q = -dln(Omega)/dln(r)``. Fixed: the exact epicycle map below is
the ``q = 3/2`` solution (``kappa = Omega``); another rotation law is another map."""

DEFAULT_IMAGES = 1
"""Image shells summed for the mutual gravity: 1 is the 3 x 3 nearest images."""

MAX_IMAGES = 4
"""The largest shell count accepted, a 9 x 9 sum: the cost grows as ``(2 n + 1) ** 2``."""


@dataclass(frozen=True)
class ShearingBoxSpec:
    """The box a System runs in, in the System's own code units.

    ``omega`` is the angular speed of the frame (1 / code time), ``box_x`` and ``box_y`` the box's
    radial and azimuthal sizes (code length), ``images`` how many shells of box copies the mutual
    gravity is summed over.
    """

    omega: float
    box_x: float
    box_y: float
    images: int = DEFAULT_IMAGES

    def validate(self) -> None:
        for name in ("omega", "box_x", "box_y"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"shearing box {name} must be a finite number above zero, got {value!r}"
                )
        if not isinstance(self.images, int) or isinstance(self.images, bool):
            raise ValueError(f"shearing box images must be a whole number, got {self.images!r}")
        if not 0 <= self.images <= MAX_IMAGES:
            raise ValueError(
                f"shearing box images must be between 0 and {MAX_IMAGES}, got {self.images}"
            )

    @property
    def shear_velocity(self) -> float:
        """``q Omega Lx``: how fast the box one length out in ``x`` slides along ``y``."""

        return SHEAR_Q * self.omega * self.box_x

    def to_parameters(self) -> dict[str, Any]:
        """The graph-node parameters that carry this box (they move ``graph_hash``)."""

        return {
            "omega": float(self.omega),
            "box_x": float(self.box_x),
            "box_y": float(self.box_y),
            "images": int(self.images),
        }

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> ShearingBoxSpec:
        """The box a plan step describes, validated; ``ValueError`` names what is wrong."""

        missing = [key for key in ("omega", "box_x", "box_y") if key not in parameters]
        if missing:
            raise ValueError(
                "the shearing box step is missing " + ", ".join(missing) + "; recompile the context"
            )
        spec = cls(
            omega=_number(parameters["omega"]),
            box_x=_number(parameters["box_x"]),
            box_y=_number(parameters["box_y"]),
            images=parameters.get("images", DEFAULT_IMAGES),
        )
        spec.validate()
        return spec


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


# -- NumPy oracle ---------------------------------------------------------------------------


def _numpy() -> Any:
    import numpy as np

    return np


def epicycle_map(positions: Any, velocities: Any, omega: float, dt: float) -> tuple[Any, Any]:
    """The exact flow of Hill's equations without mutual gravity over ``dt`` (float64).

    Solved by two integrals of the motion: ``v_y + 2 Omega x`` is constant (``y'' = -2 Omega x'``),
    which turns the ``x`` equation into a harmonic oscillator at ``Omega`` about the guiding centre
    ``x_c = 4 x + 2 v_y / Omega``. ``s = sin(Omega dt)`` and ``c = cos(Omega dt)``:

        x'  = (4 - 3 c) x + (s / Omega) v_x + (2 (1 - c) / Omega) v_y
        y'  = y + 6 (s - Omega dt) x - (2 (1 - c) / Omega) v_x + ((4 s - 3 Omega dt) / Omega) v_y
        v_x' = 3 Omega s x + c v_x + 2 s v_y
        v_y' = 6 Omega (c - 1) x - 2 s v_x + (4 c - 3) v_y
        z'  = c z + (s / Omega) v_z,    v_z' = -Omega s z + c v_z

    which is Rein and Tremaine's (2011) eq. for the Keplerian shear. Returns new arrays.
    """

    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    wt = omega * dt
    s, c = math.sin(wt), math.cos(wt)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    new_p = np.stack(
        [
            (4.0 - 3.0 * c) * x + (s / omega) * vx + (2.0 * (1.0 - c) / omega) * vy,
            y
            + 6.0 * (s - wt) * x
            - (2.0 * (1.0 - c) / omega) * vx
            + ((4.0 * s - 3.0 * wt) / omega) * vy,
            c * z + (s / omega) * vz,
        ],
        axis=-1,
    )
    new_v = np.stack(
        [
            3.0 * omega * s * x + c * vx + 2.0 * s * vy,
            6.0 * omega * (c - 1.0) * x - 2.0 * s * vx + (4.0 * c - 3.0) * vy,
            -omega * s * z + c * vz,
        ],
        axis=-1,
    )
    return new_p, new_v


def wrap_offset(spec: ShearingBoxSpec, t: float) -> float:
    """``q Omega Lx t`` folded into ``[-Ly/2, Ly/2)``: how far the ``+Lx`` image has slid in y."""

    return _fold(spec.shear_velocity * t, spec.box_y)


def _fold(value: float, period: float) -> float:
    return value - period * math.floor(value / period + 0.5)


def wrap_into_box(
    positions: Any, velocities: Any, spec: ShearingBoxSpec, offset: float
) -> tuple[Any, Any]:
    """Re-express each particle inside the box at a time whose wrapped y offset is ``offset``.

    ``k`` boxes crossed in ``x`` (``k = floor((x + Lx/2) / Lx)``) subtract ``k Lx`` from ``x``, add
    ``k * offset`` to ``y`` and ``k q Omega Lx`` to ``v_y``; then ``y`` is wrapped plainly into
    ``[-Ly/2, Ly/2)``. Returns new arrays.
    """

    np = _numpy()
    p = np.array(positions, dtype=np.float64)
    v = np.array(velocities, dtype=np.float64)
    k = np.floor((p[..., 0] + 0.5 * spec.box_x) / spec.box_x)
    p[..., 0] -= k * spec.box_x
    p[..., 1] += k * offset
    v[..., 1] += k * spec.shear_velocity
    p[..., 1] -= spec.box_y * np.floor((p[..., 1] + 0.5 * spec.box_y) / spec.box_y)
    return p, v


def image_shifts(spec: ShearingBoxSpec, offset: float) -> list[tuple[float, float]]:
    """The ``(dx, dy)`` of every image copy the mutual gravity is summed over, the box itself first.

    Image ``(i, j)`` sits at ``x + i Lx`` and ``y - i * offset + j Ly``: the ``+Lx`` copy has
    slid ``-offset`` in ``y`` (it moves at ``-q Omega Lx``), which is the inverse of what
    :func:`wrap_into_box` does to a particle that leaves through ``+x``.
    """

    n = spec.images
    shifts = [(0.0, 0.0)]
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if i == 0 and j == 0:
                continue
            shifts.append((i * spec.box_x, -i * offset + j * spec.box_y))
    return shifts


def periodic_accelerations(
    positions: Any,
    masses: Any,
    spec: ShearingBoxSpec,
    offset: float,
    gravitational_constant: float,
    softening: float,
    active: Any = None,
) -> Any:
    """Mutual gravity of the particles summed over the box and its images (float64 oracle).

    The same Plummer softening and ``1e-12`` floor on the squared distance as the Direct solver's
    kernel; a particle is not attracted by its own images (they cancel by symmetry), so the sum
    skips ``i == j`` altogether.
    """

    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    live = np.ones(len(p), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    acc = np.zeros_like(p)
    for dx, dy in image_shifts(spec, offset):
        shift = np.array([dx, dy, 0.0])
        r = p[None, :, :] + shift - p[:, None, :]
        dist2 = np.maximum((r * r).sum(axis=2) + softening * softening, 1.0e-12)
        weight = gravitational_constant * m[None, :] / (dist2 * np.sqrt(dist2))
        weight = np.where(live[None, :] & live[:, None], weight, 0.0)
        np.fill_diagonal(weight, 0.0)
        acc += (weight[:, :, None] * r).sum(axis=1)
    acc[~live] = 0.0
    return acc


def hill_energy_per_mass(positions: Any, velocities: Any, omega: float) -> Any:
    """``v^2/2 - (3/2) Omega^2 x^2 + Omega^2 z^2 / 2`` for each particle: Hill's (Jacobi) energy.

    Conserved along the exact map for a particle with no mutual gravity (no wrap): the tidal
    potential is ``-(3/2) Omega^2 x^2`` and the vertical one ``+Omega^2 z^2 / 2``.
    """

    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    return (
        0.5 * (v * v).sum(axis=-1)
        - 1.5 * omega * omega * p[..., 0] ** 2
        + 0.5 * omega * omega * p[..., 2] ** 2
    )


def guiding_centre(positions: Any, velocities: Any, omega: float) -> tuple[Any, Any]:
    """``(x_c, y_c) = (4 x + 2 v_y / Omega, y - 2 v_x / Omega)``: where the epicycle is centred.

    ``x_c`` is constant along the free flow; ``y_c`` drifts at ``-q Omega x_c``. The ``y`` epicycle
    is ``-2 A sin(Omega t + phi)`` beside the ``x`` one's ``A cos(Omega t + phi)``, and
    ``v_x = -A Omega sin(Omega t + phi)``, so ``y - 2 v_x / Omega`` cancels it.
    """

    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    return 4.0 * p[..., 0] + 2.0 * v[..., 1] / omega, p[..., 1] - 2.0 * v[..., 0] / omega


def epicycle_amplitude(positions: Any, velocities: Any, omega: float) -> Any:
    """The radial excursion about the guiding centre: ``sqrt((x - x_c)^2 + (v_x / Omega)^2)``.

    Unchanged by the exact map, and unchanged by a boundary crossing: the guiding centre moves by
    exactly the ``-Lx`` that ``x`` does.
    """

    np = _numpy()
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    x_c, _ = guiding_centre(p, v, omega)
    return np.hypot(p[..., 0] - x_c, v[..., 0] / omega)


def guiding_centre_invariant(
    positions: Any, velocities: Any, omega: float, t: float, box_y: float
) -> Any:
    """``y_c + q Omega x_c t`` folded into one box length of ``y``: constant along a free orbit.

    The guiding centre drifts as ``dy_c/dt = -q Omega x_c`` while ``x_c`` stays put, so this is
    conserved by the exact map; and it is conserved by the boundary crossing *only if* the crossing
    adds ``k q Omega Lx t`` to ``y``, which makes it the independent check on the wrap's sign and
    its time.
    """

    np = _numpy()
    x_c, y_c = guiding_centre(positions, velocities, omega)
    value = y_c + SHEAR_Q * omega * x_c * t
    return value - box_y * np.floor(value / box_y + 0.5)


# -- Warp kernels ---------------------------------------------------------------------------


def shearing_arrays(wp: Any, device: Any) -> dict[str, Any]:
    """The one device scalar the box needs: the wrapped ``q Omega Lx t`` (float64, both lanes).

    A device array rather than a launch argument because the box's clock advances every substep
    and a captured graph replays the same launch arguments: the offset has to live where the
    graph can read and rewrite it.
    """

    return {"shear_offset": wp.zeros(1, dtype=wp.float64, device=device)}


def shearing_kernels(wp: Any, precision: Any) -> Any:
    """Compile the shearing-box kernels for one state precision (the Direct runtime's lane)."""

    from .nbody import _dtypes, _registrars

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, _ = _registrars(wp, scalar, vector)

    @warp_kernel
    def epicycle_drift(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        omega: scalar,
        dt: scalar,
    ):
        # The exact flow of Hill's equations without mutual gravity (see `epicycle_map`).
        i = wp.tid()
        if active[i] == 0:
            return
        p = positions[i]
        v = velocities[i]
        wt = omega * dt
        s = wp.sin(wt)
        c = wp.cos(wt)
        one = scalar(1.0)
        x = p[0]
        y = p[1]
        z = p[2]
        vx = v[0]
        vy = v[1]
        vz = v[2]
        nx = (
            (scalar(4.0) - scalar(3.0) * c) * x
            + (s / omega) * vx
            + (scalar(2.0) * (one - c) / omega) * vy
        )
        ny = (
            y
            + scalar(6.0) * (s - wt) * x
            - (scalar(2.0) * (one - c) / omega) * vx
            + ((scalar(4.0) * s - scalar(3.0) * wt) / omega) * vy
        )
        nz = c * z + (s / omega) * vz
        nvx = scalar(3.0) * omega * s * x + c * vx + scalar(2.0) * s * vy
        nvy = (
            scalar(6.0) * omega * (c - one) * x
            - scalar(2.0) * s * vx
            + (scalar(4.0) * c - scalar(3.0)) * vy
        )
        nvz = -omega * s * z + c * vz
        positions[i] = vector(nx, ny, nz)
        velocities[i] = vector(nvx, nvy, nvz)

    @warp_kernel
    def advance_shear_offset(
        offset: wp.array(dtype=wp.float64), slide: wp.float64, box_y: wp.float64
    ):
        # One thread: offset <- fold(offset + q Omega Lx dt) into [-Ly/2, Ly/2).
        value = offset[0] + slide
        offset[0] = value - box_y * wp.floor(value / box_y + wp.float64(0.5))

    @warp_kernel
    def shear_wrap(
        positions: wp.array(dtype=vector),
        velocities: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        offset: wp.array(dtype=wp.float64),
        box_x: scalar,
        box_y: scalar,
        shear_velocity: scalar,
    ):
        # See `wrap_into_box`: k boxes out in x re-enter with y += k * offset, v_y += k q Omega Lx.
        i = wp.tid()
        if active[i] == 0:
            return
        p = positions[i]
        v = velocities[i]
        half = scalar(0.5)
        k = wp.floor((p[0] + half * box_x) / box_x)
        x = p[0] - k * box_x
        y = p[1] + k * scalar(offset[0])
        vy = v[1] + k * shear_velocity
        y = y - box_y * wp.floor((y + half * box_y) / box_y)
        positions[i] = vector(x, y, p[2])
        velocities[i] = vector(v[0], vy, v[2])

    @warp_kernel
    def acceleration_images(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        box_x: scalar,
        box_y: scalar,
        images: int,
        offset: wp.array(dtype=wp.float64),
        accelerations: wp.array(dtype=vector),
    ):
        # Mutual gravity over the box and its images (see `periodic_accelerations`): image
        # (a, b) of a source sits at x + a Lx and y - a * offset + b Ly. A body is not pulled
        # by its own images, which cancel by symmetry.
        i = wp.tid()
        if i >= count or active[i] == 0:
            accelerations[i] = vector()
            return
        pi = positions[i]
        slid = scalar(offset[0])
        ai = vector()
        for j in range(count):
            if i != j and active[j] != 0:
                pj = positions[j]
                for a in range(-images, images + 1):
                    for b in range(-images, images + 1):
                        shift = vector(
                            scalar(a) * box_x, scalar(b) * box_y - scalar(a) * slid, scalar(0.0)
                        )
                        r = pj + shift - pi
                        dist2 = wp.max(wp.dot(r, r) + softening * softening, scalar(1.0e-12))
                        inv_dist = scalar(1.0) / wp.sqrt(dist2)
                        ai += r * (
                            gravitational_constant * masses[j] * inv_dist * inv_dist * inv_dist
                        )
        accelerations[i] = ai

    @warp_kernel
    def add_tidal_energy(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        omega: scalar,
        diag: wp.array(dtype=wp.float64),
    ):
        # Hill's tidal and vertical potentials, added to the potential-energy slot the
        # diagnostics kernel filled: m * (-(3/2) Omega^2 x^2 + Omega^2 z^2 / 2).
        i = wp.tid()
        if i >= count or active[i] == 0:
            return
        p = positions[i]
        potential = (
            scalar(0.5) * masses[i] * omega * omega * (p[2] * p[2] - scalar(3.0) * p[0] * p[0])
        )
        wp.atomic_add(diag, 1, wp.float64(potential))

    @dataclass(frozen=True)
    class ShearingKernels:
        epicycle_drift: Any
        advance_shear_offset: Any
        shear_wrap: Any
        acceleration_images: Any
        add_tidal_energy: Any

    return ShearingKernels(
        epicycle_drift=epicycle_drift,
        advance_shear_offset=advance_shear_offset,
        shear_wrap=shear_wrap,
        acceleration_images=acceleration_images,
        add_tidal_energy=add_tidal_energy,
    )
