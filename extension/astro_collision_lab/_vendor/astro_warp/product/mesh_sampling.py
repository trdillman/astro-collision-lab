"""Filling and covering an arbitrary triangle mesh with particles.

`SampleShape.OBJECT_BOX` and `SampleShape.OBJECT_BALL` size a cloud from the bound
object's *dimensions*. That is a convex proxy, and for the meshes an artist actually binds
-- a torus, an L-block, a spiral arm, anything with a hole in it -- the proxy puts a large
fraction of the cloud in space the object does not occupy. This module is the real thing:
`sample_volume` fills the mesh's interior and `sample_surface` scatters over its skin.

**Why the inside test is written the way it is.** Deciding whether a point is inside a
closed mesh is a parity count: fire a ray, count the triangles it crosses, and an odd count
means inside. Written directly that is a test of every point against every triangle, and at
the sizes this product already proves elsewhere -- 10^5 particles, and meshes that reach
10^4 triangles once a subdivision modifier is on -- that product is 10^9 pairs and the
cloud never appears. Instead the ray is fixed along +X and the triangles are binned once
into a uniform grid over the YZ plane, which is the plane the ray projects to a point in.
A query point then only has to be tested against the triangles sharing its YZ cell. The
grid resolution is derived from the median triangle size, so it tracks the mesh's own
feature scale rather than a constant that happens to suit one test fixture.

**Degeneracies.** A ray that grazes exactly along a shared triangle edge is counted twice
and flips the parity to the wrong answer. For randomly drawn interior candidates that is a
measure-zero event and is left alone deliberately -- the alternative, a symbolic tie-break
rule, costs a branch on every one of the millions of candidate pairs to fix an outcome
that does not occur. Callers sampling a *lattice* rather than a random cloud would need
that rule; none does today, and `sample_volume` is the only caller.

**Refusal, not silence.** A mesh with no enclosed interior -- an unclosed surface, a single
triangle, a plane -- cannot be filled. It raises `MeshSamplingError` rather than falling
back to the bounding box, because a silent fallback to a convex proxy is exactly the class
of quiet substitution this module exists to end.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Candidate (point, triangle) pairs are formed in blocks so peak memory tracks this
# constant rather than the product of the point and triangle counts. Four million pairs is
# a few hundred MiB across the index and coordinate arrays the parity test needs at once.
_PAIR_BLOCK = 4_000_000

# The YZ grid is capped in each axis so a mesh whose triangles are tiny next to its extent
# -- a highly subdivided surface -- cannot ask for a grid with more cells than there are
# triangles to put in them.
_MAX_GRID_AXIS = 512

# Rejection sampling draws this many candidates before it is willing to call a mesh
# unfillable. Small enough to stay cheap for the degenerate meshes it is meant to catch,
# large enough that a legitimately thin shape -- a cloud filling 0.1% of its bounding box
# -- still shows a nonzero acceptance rate rather than being wrongly refused.
_ACCEPTANCE_PROBE = 4096

# Each rejection round is sized from the measured acceptance rate with this much headroom,
# so the common case finishes in one round after the probe instead of creeping up on the
# requested count a fraction at a time.
_ROUND_HEADROOM = 1.35

# A ceiling on rejection rounds. Reaching it means the acceptance estimate is not
# converging, which is a broken mesh rather than a slow one.
_MAX_ROUNDS = 64


class MeshSamplingError(RuntimeError):
    """A mesh that cannot be sampled as asked."""


@dataclass(frozen=True)
class MeshSample:
    """One evaluated triangle mesh, in the coordinate frame its particles will live in.

    This is deliberately a triangle soup and not a reference to a host object. The host
    evaluates modifiers, applies the object's transform and triangulates *once*, at compile
    time; everything downstream is pure arrays, so the sampler stays testable without
    Blender and identical between the interactive runtime and a headless bake.

    ``content_key`` is what lets a mesh take part in the graph hash without being embedded
    in a `BindingDescriptor`. The descriptor carries the key -- 64 characters whatever the
    mesh -- and the arrays travel beside it, the same separation `expand_bindings` already
    uses for shell layers.
    """

    vertices: Any
    triangles: Any

    def __post_init__(self) -> None:
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        triangles = np.ascontiguousarray(self.triangles, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise MeshSamplingError(f"vertices must have shape (V, 3), got {vertices.shape}")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise MeshSamplingError(f"triangles must have shape (T, 3), got {triangles.shape}")
        if triangles.size and (
            int(triangles.min()) < 0 or int(triangles.max()) >= vertices.shape[0]
        ):
            raise MeshSamplingError("triangle indexes a vertex the mesh does not have")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def content_key(self) -> str:
        """A stable 64-hex digest of the exact geometry.

        Shapes are folded in as well as bytes so that meshes which happen to share a byte
        run -- a rewound triangle, a vertex list read at a different length -- cannot
        collide onto one key and let a stale sampling survive a recompile.
        """

        digest = hashlib.sha256()
        digest.update(b"astro-warp-mesh-v1")
        digest.update(np.asarray(self.vertices.shape, dtype=np.int64).tobytes())
        digest.update(self.vertices.tobytes())
        digest.update(np.asarray(self.triangles.shape, dtype=np.int64).tobytes())
        digest.update(self.triangles.tobytes())
        return digest.hexdigest()

    def corners(self) -> tuple[Any, Any, Any]:
        """The three vertex positions of every triangle, as ``(T, 3)`` arrays."""

        return (
            self.vertices[self.triangles[:, 0]],
            self.vertices[self.triangles[:, 1]],
            self.vertices[self.triangles[:, 2]],
        )

    def transformed(self, matrix: Any) -> MeshSample:
        """The same triangles in the space ``matrix`` maps this mesh's own space into.

        A row-major 4x4, which is what ``Object.matrix_world`` hands back. The sampler
        works entirely in one space and the host's is world space --- an object's scale,
        rotation and parenting are all in that matrix, so a cloud drawn from mesh-local
        coordinates would ignore every one of them.

        Triangles are carried across untouched. A mirrored or negatively scaled matrix
        reverses their winding, which changes the sign of `volume` and nothing else: the
        inside test counts ray crossings, so it reads an inside-out mesh the same way.
        """

        rows = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        vertices = self.vertices @ rows[:3, :3].T + rows[:3, 3]
        return MeshSample(vertices=vertices, triangles=self.triangles)

    def area(self) -> float:
        """Total surface area, as the sum of the triangles' own areas."""

        a, b, c = self.corners()
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    def volume(self) -> float:
        """The volume the surface encloses.

        The divergence theorem, as the signed sum of the tetrahedra each triangle forms
        with the origin: exact for any closed mesh, convex or not, and O(T) rather than
        the O(T * resolution) a voxel count would cost. Taken absolute because the sign is
        the surface's winding, and a mesh wound inward encloses the same space as one wound
        outward. An unclosed surface has no enclosed volume at all, and the number this
        returns for one is meaningless -- ``sample_volume`` is what refuses that case,
        because refusing it needs the ray test this measure deliberately avoids.
        """

        a, b, c = self.corners()
        return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)

    def bounds(self) -> tuple[Any, Any]:
        """Axis-aligned minimum and maximum corner of the mesh."""

        if self.vertices.shape[0] == 0:
            raise MeshSamplingError("an empty mesh has no bounds")
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def _rng(seed: Sequence[int]) -> Any:
    return np.random.default_rng([int(value) for value in seed])


class _RayGrid:
    """Triangles binned by their footprint in the YZ plane, for +X ray casting.

    The ray direction is a fixed axis rather than a random one on purpose: a fixed axis
    makes the acceleration structure a plain 2D grid that is built once and reused for
    every point, where a per-point direction would need a full spatial hierarchy to do the
    same job.
    """

    def __init__(self, mesh: MeshSample) -> None:
        a, b, c = mesh.corners()
        self._a, self._b, self._c = a, b, c

        low = np.minimum(np.minimum(a, b), c)[:, 1:]
        high = np.maximum(np.maximum(a, b), c)[:, 1:]
        self._origin = low.min(axis=0)
        span = high.max(axis=0) - self._origin
        # A mesh flat in Y or Z has zero span on that axis; one cell is the whole of it.
        span = np.where(span > 0.0, span, 1.0)

        # Match the cell to the median triangle, so a typical triangle lands in about one
        # cell: finer wastes memory on duplicate entries, coarser puts every triangle in
        # every point's candidate list and gives back the dense test this class avoids.
        extent = np.median(high - low, axis=0)
        cell = np.where(extent > 0.0, extent, span)
        axes = np.clip(np.ceil(span / cell).astype(np.int64), 1, _MAX_GRID_AXIS)
        self._axes = axes
        self._cell = span / axes

        lo_cell = self._cell_index(low)
        hi_cell = self._cell_index(high)
        # Each triangle is registered in every cell its YZ box touches. Triangles are
        # enumerated one span-width at a time so the expansion stays vectorised without
        # materialising a ragged list of per-triangle cell sets.
        widths = (hi_cell - lo_cell + 1).prod(axis=1)
        entries = int(widths.sum())
        triangle_ids = np.repeat(np.arange(mesh.triangle_count, dtype=np.int64), widths)
        within = np.arange(entries, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(widths)[:-1]]), widths
        )
        y_width = np.repeat(hi_cell[:, 0] - lo_cell[:, 0] + 1, widths)
        cell_y = np.repeat(lo_cell[:, 0], widths) + within % y_width
        cell_z = np.repeat(lo_cell[:, 1], widths) + within // y_width
        cell_ids = cell_y * axes[1] + cell_z

        order = np.argsort(cell_ids, kind="stable")
        self._triangles_by_cell = triangle_ids[order]
        self._offsets = np.concatenate(
            [[0], np.cumsum(np.bincount(cell_ids, minlength=int(axes.prod())))]
        )

    def _cell_index(self, yz: Any) -> Any:
        index = np.floor((yz - self._origin) / self._cell).astype(np.int64)
        return np.clip(index, 0, self._axes - 1)

    def _candidate_counts(self, cells: Any) -> Any:
        return self._offsets[cells + 1] - self._offsets[cells]

    def inside(self, points: Any) -> Any:
        """Boolean mask of the points enclosed by the mesh."""

        if points.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        cells_yz = self._cell_index(points[:, 1:])
        cells = cells_yz[:, 0] * self._axes[1] + cells_yz[:, 1]
        counts = self._candidate_counts(cells)

        crossings = np.zeros(points.shape[0], dtype=np.int64)
        # Blocks are cut on the cumulative *pair* count, not on a fixed number of points:
        # one point sitting in a dense cell can carry as many candidates as thousands in
        # sparse ones, so a fixed point block would not bound the allocation it is for.
        boundaries = _pair_blocks(counts)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=False):
            block = slice(int(start), int(stop))
            crossings[block] = self._block_crossings(
                points[block], self._offsets[cells[block]], counts[block]
            )
        return (crossings % 2) == 1

    def _block_crossings(self, points: Any, starts: Any, counts: Any) -> Any:
        total = int(counts.sum())
        if total == 0:
            return np.zeros(points.shape[0], dtype=np.int64)
        point_of_pair = np.repeat(np.arange(points.shape[0], dtype=np.int64), counts)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
        )
        triangle_of_pair = self._triangles_by_cell[np.repeat(starts, counts) + within]

        a = self._a[triangle_of_pair]
        b = self._b[triangle_of_pair]
        c = self._c[triangle_of_pair]
        p = points[point_of_pair]

        # The +X ray projects to a single point in YZ, so "does the ray hit this triangle"
        # is the 2D question of whether that point is inside the triangle's YZ shadow.
        e0 = b[:, 1:] - a[:, 1:]
        e1 = c[:, 1:] - a[:, 1:]
        rel = p[:, 1:] - a[:, 1:]
        determinant = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
        # A zero determinant is a triangle seen edge-on: it has no shadow to be inside of,
        # and its two neighbours across the silhouette account for the surface there.
        edge_on = determinant == 0.0
        safe = np.where(edge_on, 1.0, determinant)
        u = (rel[:, 0] * e1[:, 1] - e1[:, 0] * rel[:, 1]) / safe
        v = (e0[:, 0] * rel[:, 1] - rel[:, 0] * e0[:, 1]) / safe
        hits = (~edge_on) & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0)

        # Only crossings *ahead* of the point count: the parity of a full line through a
        # closed surface is always even, and it is the half-line that carries the answer.
        x_hit = a[:, 0] + u * (b[:, 0] - a[:, 0]) + v * (c[:, 0] - a[:, 0])
        hits &= x_hit > p[:, 0]
        return np.bincount(point_of_pair[hits], minlength=points.shape[0])


def _pair_blocks(counts: Any) -> Any:
    """Point-index boundaries cutting ``counts`` into blocks of at most ``_PAIR_BLOCK``."""

    cumulative = np.cumsum(counts)
    total = int(cumulative[-1]) if cumulative.size else 0
    if total <= _PAIR_BLOCK:
        return np.array([0, counts.shape[0]], dtype=np.int64)
    cuts = np.searchsorted(cumulative, np.arange(_PAIR_BLOCK, total, _PAIR_BLOCK), "left") + 1
    return np.unique(np.concatenate([[0], cuts, [counts.shape[0]]])).astype(np.int64)


def sample_surface(mesh: MeshSample, count: int, seed: Sequence[int]) -> Any:
    """``count`` points scattered over the mesh's surface, uniform by area.

    Uniform *by area* rather than by triangle: a mesh is almost never uniformly
    tessellated, and choosing triangles evenly would crowd the cloud onto whichever part
    of the model happened to carry the most geometry.
    """

    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if mesh.triangle_count == 0:
        raise MeshSamplingError("cannot scatter over a mesh with no triangles")

    a, b, c = mesh.corners()
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total_area = float(areas.sum())
    if not total_area > 0.0:
        raise MeshSamplingError("cannot scatter over a mesh of zero surface area")

    rng = _rng(seed)
    cumulative = np.cumsum(areas) / total_area
    chosen = np.searchsorted(cumulative, rng.random(count), side="right")
    chosen = np.clip(chosen, 0, mesh.triangle_count - 1)

    # Uniform barycentric coordinates: draw in the unit square and fold the half that
    # lands outside the triangle back across the diagonal, which is exact rather than a
    # rejection loop with a data-dependent length.
    uv = rng.random((count, 2))
    outside = uv.sum(axis=1) > 1.0
    uv[outside] = 1.0 - uv[outside]
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    base = a[chosen]
    return base + u * (b[chosen] - base) + v * (c[chosen] - base)


def sample_volume(mesh: MeshSample, count: int, seed: Sequence[int]) -> Any:
    """``count`` points uniform in the mesh's enclosed interior.

    Rejection sampling against the bounding box. Uniform-in-the-box survives the rejection
    as uniform-in-the-mesh exactly, with no density correction, which is the property that
    makes the far cheaper alternatives -- shrinking a proxy, or pushing surface points
    inward -- wrong rather than merely approximate.
    """

    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if mesh.triangle_count == 0:
        raise MeshSamplingError("cannot fill a mesh with no triangles")

    low, high = mesh.bounds()
    extent = high - low
    if not float(np.prod(np.where(extent > 0.0, extent, 0.0))) > 0.0:
        raise MeshSamplingError(
            "cannot fill a mesh with no volume: it is flat on at least one axis"
        )

    grid = _RayGrid(mesh)
    rng = _rng(seed)

    def _draw(size: int) -> Any:
        candidates = low + rng.random((int(size), 3)) * extent
        return candidates[grid.inside(candidates)]

    accepted = [_draw(max(_ACCEPTANCE_PROBE, count))]
    found = int(accepted[0].shape[0])
    if found == 0:
        raise MeshSamplingError(
            "no sampled point fell inside the mesh: it encloses no volume, which usually "
            "means the surface is not closed"
        )

    drawn = float(max(_ACCEPTANCE_PROBE, count))
    rounds = 0
    while found < count:
        rounds += 1
        if rounds > _MAX_ROUNDS:
            raise MeshSamplingError(
                f"gave up filling the mesh after {_MAX_ROUNDS} rounds with {found} of "
                f"{count} points: its interior is too small a fraction of its bounds"
            )
        rate = found / drawn
        size = int(np.ceil((count - found) / rate * _ROUND_HEADROOM))
        batch = _draw(size)
        drawn += size
        found += int(batch.shape[0])
        accepted.append(batch)

    return np.concatenate(accepted)[:count]
