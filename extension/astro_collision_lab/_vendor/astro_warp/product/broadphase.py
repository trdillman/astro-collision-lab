"""Capture-legal fixed-capacity uniform grid for point-collision broad phase."""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .model import Precision

TARGET_HASH_BUCKET_LOAD = 8
DEFAULT_BUCKET_CAPACITY = 128
DEFAULT_ALL_PAIRS_THRESHOLD = 64
DEFAULT_RADIUS_RATIO_THRESHOLD = 64.0

_HASH_X = 73_856_093
_HASH_Y = 19_349_663
_HASH_Z = 83_492_791


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class BroadphaseConfig:
    """Frozen storage and fallback policy for one collision grid."""

    cell_capacity: int = 1
    bucket_capacity: int = DEFAULT_BUCKET_CAPACITY
    all_pairs_threshold: int = DEFAULT_ALL_PAIRS_THRESHOLD
    radius_ratio_threshold: float = DEFAULT_RADIUS_RATIO_THRESHOLD

    @classmethod
    def for_capacity(cls, capacity: int) -> BroadphaseConfig:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        cell_capacity = _next_power_of_two(
            math.ceil(capacity / TARGET_HASH_BUCKET_LOAD)
        )
        return cls(cell_capacity=cell_capacity)

    @property
    def item_capacity(self) -> int:
        return self.cell_capacity * self.bucket_capacity

    def validate(self) -> None:
        if self.cell_capacity < 1 or self.cell_capacity & (self.cell_capacity - 1):
            raise ValueError("cell_capacity must be a positive power of two")
        if self.bucket_capacity < 1:
            raise ValueError("bucket_capacity must be positive")
        if self.all_pairs_threshold < 0:
            raise ValueError("all_pairs_threshold must be non-negative")
        if not math.isfinite(self.radius_ratio_threshold) or self.radius_ratio_threshold <= 1.0:
            raise ValueError("radius_ratio_threshold must be finite and greater than one")


def cell_coordinates(
    position: tuple[float, float, float], cell_width: float
) -> tuple[int, int, int]:
    """Return the signed uniform-grid cell containing ``position``."""

    if not math.isfinite(cell_width) or cell_width <= 0.0:
        raise ValueError("cell_width must be finite and positive")
    return tuple(math.floor(component / cell_width) for component in position)  # type: ignore[return-value]


def neighbor_cells(cell: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    """Yield the 27 cells whose contents can overlap a particle in ``cell``."""

    x, y, z = cell
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yield x + dx, y + dy, z + dz


def bucket_index(cell: tuple[int, int, int], cell_capacity: int) -> int:
    """Map a signed cell coordinate into a power-of-two hash table."""

    if cell_capacity < 1 or cell_capacity & (cell_capacity - 1):
        raise ValueError("cell_capacity must be a positive power of two")
    x, y, z = cell
    return ((x * _HASH_X) ^ (y * _HASH_Y) ^ (z * _HASH_Z)) & (cell_capacity - 1)


def allocate_broadphase_arrays(
    wp: Any,
    device: Any,
    precision: Precision,
    config: BroadphaseConfig,
) -> dict[str, Any]:
    """Allocate every grid array once, before any recorded launch sequence."""

    config.validate()
    scalar = wp.float64 if precision == Precision.FLOAT64 else wp.float32
    return {
        "max_radius": wp.zeros(1, dtype=scalar, device=device),
        "min_radius": wp.zeros(1, dtype=scalar, device=device),
        "cell_counts": wp.zeros(config.cell_capacity, dtype=wp.int32, device=device),
        "cell_items": wp.zeros(config.item_capacity, dtype=wp.int32, device=device),
        # [0] bucket overflow, [1] small-count/radius-spread all-pairs fallback observed.
        "flags": wp.zeros(2, dtype=wp.int32, device=device),
    }


def broadphase_kernels(wp: Any, precision: Precision = Precision.FLOAT32) -> Any:
    """Create the allocation-free kernels used by both collision runtimes."""

    scalar = wp.float64 if precision == Precision.FLOAT64 else wp.float32
    vector = wp.vec3d if precision == Precision.FLOAT64 else wp.vec3
    largest_radius = 1.0e300 if precision == Precision.FLOAT64 else 1.0e30
    namespace = {"wp": wp, "scalar": scalar, "vector": vector}

    def resolve(func: Any) -> Any:
        func.__annotations__ = inspect.get_annotations(func, eval_str=True, locals=namespace)
        return func

    def kernel(func: Any) -> Any:
        return wp.kernel(resolve(func))

    def func(func_: Any) -> Any:
        return wp.func(resolve(func_))

    @func
    def hash_cell(x: int, y: int, z: int, cell_capacity: int) -> int:
        return (
            (x * _HASH_X) ^ (y * _HASH_Y) ^ (z * _HASH_Z)
        ) & (cell_capacity - 1)

    @func
    def coordinate(value: scalar, cell_width: scalar) -> int:
        return int(wp.floor(value / cell_width))

    @kernel
    def clear_grid(
        max_radius: wp.array(dtype=scalar),
        min_radius: wp.array(dtype=scalar),
        cell_counts: wp.array(dtype=wp.int32),
        flags: wp.array(dtype=wp.int32),
    ):
        cell = wp.tid()
        cell_counts[cell] = 0
        if cell == 0:
            max_radius[0] = scalar(0.0)
            min_radius[0] = scalar(largest_radius)
            flags[0] = 0
            flags[1] = 0

    @kernel
    def reduce_max_radius(
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        max_radius: wp.array(dtype=scalar),
        min_radius: wp.array(dtype=scalar),
    ):
        i = wp.tid()
        if i < count and active[i] != 0:
            wp.atomic_max(max_radius, 0, radii[i])
            wp.atomic_min(min_radius, 0, radii[i])

    @kernel
    def bin_particles(
        positions: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        count: int,
        max_radius: wp.array(dtype=scalar),
        cell_capacity: int,
        bucket_capacity: int,
        cell_counts: wp.array(dtype=wp.int32),
        cell_items: wp.array(dtype=wp.int32),
        flags: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0:
            return
        radius = max_radius[0]
        if radius <= scalar(0.0):
            wp.atomic_max(flags, 1, 1)
            return
        cell_width = scalar(2.0) * radius
        p = positions[i]
        x = coordinate(p[0], cell_width)
        y = coordinate(p[1], cell_width)
        z = coordinate(p[2], cell_width)
        cell = hash_cell(x, y, z, cell_capacity)
        slot = wp.atomic_add(cell_counts, cell, 1)
        if slot < bucket_capacity:
            cell_items[cell * bucket_capacity + slot] = i
        else:
            wp.atomic_max(flags, 0, 1)

    @kernel
    def detect_collisions_grid(
        positions: wp.array(dtype=vector),
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        count: int,
        max_radius: wp.array(dtype=scalar),
        min_radius: wp.array(dtype=scalar),
        cell_capacity: int,
        bucket_capacity: int,
        all_pairs_threshold: int,
        radius_ratio_threshold: float,
        cell_counts: wp.array(dtype=wp.int32),
        cell_items: wp.array(dtype=wp.int32),
        flags: wp.array(dtype=wp.int32),
        event_i: wp.array(dtype=wp.int32),
        event_j: wp.array(dtype=wp.int32),
        event_capacity: int,
        idiag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0 or flags[0] != 0:
            return

        radius = max_radius[0]
        minimum = min_radius[0]
        use_all_pairs = (
            count <= all_pairs_threshold
            or radius <= scalar(0.0)
            or minimum <= scalar(0.0)
            or radius > minimum * scalar(radius_ratio_threshold)
        )
        if use_all_pairs:
            wp.atomic_max(flags, 1, 1)
            for j in range(i + 1, count):
                if active[j] != 0:
                    separation = positions[j] - positions[i]
                    limit = radii[i] + radii[j]
                    if wp.dot(separation, separation) <= limit * limit:
                        event_index = wp.atomic_add(idiag, 1, 1)
                        if event_index < event_capacity:
                            event_i[event_index] = i
                            event_j[event_index] = j
                        else:
                            wp.atomic_max(idiag, 2, 1)
            return

        cell_width = scalar(2.0) * radius
        p = positions[i]
        cell_x = coordinate(p[0], cell_width)
        cell_y = coordinate(p[1], cell_width)
        cell_z = coordinate(p[2], cell_width)
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    query_x = cell_x + dx
                    query_y = cell_y + dy
                    query_z = cell_z + dz
                    cell = hash_cell(query_x, query_y, query_z, cell_capacity)
                    slots = wp.min(cell_counts[cell], bucket_capacity)
                    for slot in range(slots):
                        j = cell_items[cell * bucket_capacity + slot]
                        if j <= i or j >= count or active[j] == 0:
                            continue
                        candidate = positions[j]
                        # Hash aliases are conservative only if candidates from unrelated
                        # cells are filtered before the distance test. This also prevents a
                        # pair from being visited twice when two queried cells hash alike.
                        if (
                            coordinate(candidate[0], cell_width) != query_x
                            or coordinate(candidate[1], cell_width) != query_y
                            or coordinate(candidate[2], cell_width) != query_z
                        ):
                            continue
                        separation = candidate - p
                        limit = radii[i] + radii[j]
                        if wp.dot(separation, separation) <= limit * limit:
                            event_index = wp.atomic_add(idiag, 1, 1)
                            if event_index < event_capacity:
                                event_i[event_index] = i
                                event_j[event_index] = j
                            else:
                                wp.atomic_max(idiag, 2, 1)

    @kernel
    def detect_body_collisions_grid(
        positions: wp.array(dtype=vector),
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        body_flags: wp.array(dtype=wp.int32),
        count: int,
        max_radius: wp.array(dtype=scalar),
        min_radius: wp.array(dtype=scalar),
        cell_capacity: int,
        bucket_capacity: int,
        all_pairs_threshold: int,
        radius_ratio_threshold: float,
        cell_counts: wp.array(dtype=wp.int32),
        cell_items: wp.array(dtype=wp.int32),
        flags: wp.array(dtype=wp.int32),
        event_i: wp.array(dtype=wp.int32),
        event_j: wp.array(dtype=wp.int32),
        event_capacity: int,
        idiag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0 or body_flags[i] == 0 or flags[0] != 0:
            return
        radius = max_radius[0]
        minimum = min_radius[0]
        use_all_pairs = (
            count <= all_pairs_threshold
            or radius <= scalar(0.0)
            or minimum <= scalar(0.0)
            or radius > minimum * scalar(radius_ratio_threshold)
        )
        if use_all_pairs:
            wp.atomic_max(flags, 1, 1)
            for j in range(i + 1, count):
                if active[j] != 0 and body_flags[j] != 0:
                    separation = positions[j] - positions[i]
                    limit = radii[i] + radii[j]
                    if wp.dot(separation, separation) <= limit * limit:
                        event_index = wp.atomic_add(idiag, 1, 1)
                        if event_index < event_capacity:
                            event_i[event_index] = i
                            event_j[event_index] = j
                        else:
                            wp.atomic_max(idiag, 2, 1)
            return
        cell_width = scalar(2.0) * radius
        p = positions[i]
        cell_x = coordinate(p[0], cell_width)
        cell_y = coordinate(p[1], cell_width)
        cell_z = coordinate(p[2], cell_width)
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    query_x = cell_x + dx
                    query_y = cell_y + dy
                    query_z = cell_z + dz
                    cell = hash_cell(query_x, query_y, query_z, cell_capacity)
                    slots = wp.min(cell_counts[cell], bucket_capacity)
                    for slot in range(slots):
                        j = cell_items[cell * bucket_capacity + slot]
                        if j <= i or j >= count or active[j] == 0 or body_flags[j] == 0:
                            continue
                        candidate = positions[j]
                        if (
                            coordinate(candidate[0], cell_width) != query_x
                            or coordinate(candidate[1], cell_width) != query_y
                            or coordinate(candidate[2], cell_width) != query_z
                        ):
                            continue
                        separation = candidate - p
                        limit = radii[i] + radii[j]
                        if wp.dot(separation, separation) <= limit * limit:
                            event_index = wp.atomic_add(idiag, 1, 1)
                            if event_index < event_capacity:
                                event_i[event_index] = i
                                event_j[event_index] = j
                            else:
                                wp.atomic_max(idiag, 2, 1)

    @kernel
    def detect_absorption_grid(
        positions: wp.array(dtype=vector),
        radii: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        body_flags: wp.array(dtype=wp.int32),
        stable_ids: wp.array(dtype=wp.int64),
        count: int,
        max_radius: wp.array(dtype=scalar),
        min_radius: wp.array(dtype=scalar),
        cell_capacity: int,
        bucket_capacity: int,
        all_pairs_threshold: int,
        radius_ratio_threshold: float,
        cell_counts: wp.array(dtype=wp.int32),
        cell_items: wp.array(dtype=wp.int32),
        flags: wp.array(dtype=wp.int32),
        targets: wp.array(dtype=wp.int32),
        idiag: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0 or body_flags[i] != 0 or flags[0] != 0:
            return
        radius = max_radius[0]
        minimum = min_radius[0]
        # int(...) forces a mutable dynamic value in Warp's kernel compiler.
        best = int(-1)  # noqa: UP018
        best_d2 = scalar(0.0)
        if (
            count <= all_pairs_threshold
            or radius <= scalar(0.0)
            or minimum <= scalar(0.0)
            or radius > minimum * scalar(radius_ratio_threshold)
        ):
            wp.atomic_max(flags, 1, 1)
            for j in range(count):
                if body_flags[j] != 0 and active[j] != 0:
                    separation = positions[i] - positions[j]
                    d2 = wp.dot(separation, separation)
                    if d2 <= radii[j] * radii[j]:
                        if best < 0 or d2 < best_d2:
                            best = j
                            best_d2 = d2
                        elif d2 == best_d2 and stable_ids[j] < stable_ids[best]:
                            best = j
            if best >= 0:
                targets[i] = best
                wp.atomic_add(idiag, 1, 1)
            return
        cell_width = scalar(2.0) * radius
        p = positions[i]
        cell_x = coordinate(p[0], cell_width)
        cell_y = coordinate(p[1], cell_width)
        cell_z = coordinate(p[2], cell_width)
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    query_x = cell_x + dx
                    query_y = cell_y + dy
                    query_z = cell_z + dz
                    cell = hash_cell(query_x, query_y, query_z, cell_capacity)
                    slots = wp.min(cell_counts[cell], bucket_capacity)
                    for slot in range(slots):
                        j = cell_items[cell * bucket_capacity + slot]
                        if j >= count or body_flags[j] == 0 or active[j] == 0:
                            continue
                        candidate = positions[j]
                        if (
                            coordinate(candidate[0], cell_width) != query_x
                            or coordinate(candidate[1], cell_width) != query_y
                            or coordinate(candidate[2], cell_width) != query_z
                        ):
                            continue
                        separation = p - candidate
                        d2 = wp.dot(separation, separation)
                        if d2 <= radii[j] * radii[j]:
                            if best < 0 or d2 < best_d2:
                                best = j
                                best_d2 = d2
                            elif d2 == best_d2 and stable_ids[j] < stable_ids[best]:
                                best = j
        if best >= 0:
            targets[i] = best
            wp.atomic_add(idiag, 1, 1)

    @dataclass(frozen=True)
    class Kernels:
        clear_grid: Any
        reduce_max_radius: Any
        bin_particles: Any
        detect_collisions_grid: Any
        detect_body_collisions_grid: Any
        detect_absorption_grid: Any

    return Kernels(
        clear_grid=clear_grid,
        reduce_max_radius=reduce_max_radius,
        bin_particles=bin_particles,
        detect_collisions_grid=detect_collisions_grid,
        detect_body_collisions_grid=detect_body_collisions_grid,
        detect_absorption_grid=detect_absorption_grid,
    )
