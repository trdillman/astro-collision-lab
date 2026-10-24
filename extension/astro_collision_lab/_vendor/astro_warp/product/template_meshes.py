"""Proxy geometry for the Add-menu templates, as pure vertex and face lists.

Three of D8's four templates use Blender's own primitives (UV sphere, icosphere). The
Protoplanetary Disk does not: it needs a *closed annular prism* -- a ring with a hole,
capped top and bottom and walled inside and out -- and Blender has no primitive for that.

Building it here rather than in the operator is the point. ADR 0040 section 5 reads the
disk's outer radius and thickness off the object's **evaluated extents**, so the mesh is not
decoration: `E0 = E1 = 2 * outer_radius` and `E2 = full_thickness` are the contract between
what the template creates and what the compiler later measures. If the mesh is wrong, the
compiled disk is a different disk, and nothing between the two would say so.

Expressed as plain tuples, so those equations are provable on any machine instead of only in
a window. The Blender layer hands the result to ``mesh.from_pydata`` and does no geometry of
its own.
"""

from __future__ import annotations

import math

Vector3 = tuple[float, float, float]
Face = tuple[int, ...]


def annular_prism_mesh(
    *,
    outer_radius: float,
    inner_radius: float,
    full_thickness: float,
    segments: int = 64,
) -> tuple[tuple[Vector3, ...], tuple[Face, ...]]:
    """A closed annular prism centred on the origin, in the local XY plane.

    ``segments`` is D8's 64 by default. Vertices are laid out in four rings of
    ``segments`` each, in this fixed order: outer-bottom, outer-top, inner-bottom,
    inner-top. The order is part of the contract -- a test reads extents off ring slices,
    and the operator's redo path replaces vertices in place.

    Faces are quads wound counter-clockwise seen from outside the solid, so normals point
    away from the material and Blender's own recalculation is a no-op rather than a
    correction. There are ``4 * segments`` of them: outer wall, inner wall, top cap, bottom
    cap.

    The mesh is closed. That matters beyond tidiness: an open shell has no interior, so a
    boolean or a volume query on the proxy would return something arbitrary, and the user
    would have a disk that behaves oddly for reasons invisible in the viewport.
    """

    outer = float(outer_radius)
    inner = float(inner_radius)
    thickness = float(full_thickness)
    if not math.isfinite(outer) or outer <= 0.0:
        raise ValueError(f"outer_radius must be positive and finite, got {outer_radius!r}")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError(
            f"full_thickness must be positive and finite, got {full_thickness!r}"
        )
    if not math.isfinite(inner) or not 0.0 <= inner < outer:
        raise ValueError(
            f"inner_radius must satisfy 0 <= inner < outer, got {inner_radius!r} and "
            f"{outer_radius!r}"
        )
    if isinstance(segments, bool) or not isinstance(segments, int):
        raise ValueError("segments must be an integer")
    if segments < 3:
        raise ValueError(f"segments must be at least 3 to close a ring, got {segments}")

    half = 0.5 * thickness
    angles = [2.0 * math.pi * index / segments for index in range(segments)]

    def ring(radius: float, z: float) -> list[Vector3]:
        return [
            (radius * math.cos(angle), radius * math.sin(angle), z) for angle in angles
        ]

    outer_bottom = ring(outer, -half)
    outer_top = ring(outer, half)
    inner_bottom = ring(inner, -half)
    inner_top = ring(inner, half)

    vertices = tuple(outer_bottom + outer_top + inner_bottom + inner_top)

    ob, ot, ib, it = 0, segments, 2 * segments, 3 * segments
    faces: list[Face] = []
    for index in range(segments):
        following = (index + 1) % segments
        # Outer wall: seen from outside the ring, up is +Z and around is +theta.
        faces.append((ob + index, ob + following, ot + following, ot + index))
        # Inner wall faces the hole, so its winding is the reverse of the outer wall's.
        faces.append((ib + index, it + index, it + following, ib + following))
        # Top cap, seen from +Z.
        faces.append((ot + index, ot + following, it + following, it + index))
        # Bottom cap, seen from -Z, so reversed again.
        faces.append((ob + index, ib + index, ib + following, ob + following))

    return vertices, tuple(faces)


def box_mesh(
    *, width: float, length: float, height: float
) -> tuple[tuple[Vector3, ...], tuple[Face, ...]]:
    """A closed box centred on the origin, ``width`` by ``length`` by ``height`` along X, Y, Z.

    The Shearing sheet's proxy (ADR 0202). Its Object Volume cloud fills exactly this box, so the
    box *is* the shearing box: ``width`` and ``length`` are the radial and azimuthal sizes the
    System is made with. Eight vertices in the order below, six quads wound counter-clockwise
    seen from outside the solid (normals out, so the mesh is closed and has an interior the
    volume sampler's parity test can count crossings of).

    Vertex index = ``x_bit + 2 * y_bit + 4 * z_bit`` with bit 0 the minus side, in Y-then-X order
    within each of the two Z levels: (-,-), (+,-), (-,+), (+,+).
    """

    sizes = (float(width), float(length), float(height))
    for name, value in zip(("width", "length", "height"), sizes, strict=True):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    hx, hy, hz = (0.5 * value for value in sizes)
    vertices: tuple[Vector3, ...] = tuple(
        (hx if index & 1 else -hx, hy if index & 2 else -hy, hz if index & 4 else -hz)
        for index in range(8)
    )
    faces: tuple[Face, ...] = (
        (0, 2, 3, 1),  # bottom, -Z
        (4, 5, 7, 6),  # top, +Z
        (0, 1, 5, 4),  # front, -Y
        (2, 6, 7, 3),  # back, +Y
        (0, 4, 6, 2),  # left, -X
        (1, 3, 7, 5),  # right, +X
    )
    return vertices, faces


def mesh_extents(vertices: tuple[Vector3, ...]) -> Vector3:
    """The local bounding-box spans of a vertex list, in ADR 0040 section 5's order.

    The same `max - min` per axis Blender's ``bound_box`` produces, so a test can state the
    template's extent contract in exactly the terms the compiler will later read it.
    """

    if not vertices:
        raise ValueError("mesh_extents needs at least one vertex")
    return tuple(  # type: ignore[return-value]
        max(vertex[axis] for vertex in vertices)
        - min(vertex[axis] for vertex in vertices)
        for axis in range(3)
    )
