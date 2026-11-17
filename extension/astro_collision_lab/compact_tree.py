"""Reachable, compact Barnes-Hut storage, with occupancy-driven device allocation.

SPDX-License-Identifier: LicenseRef-Private-Internal
Moment and depth-first walk kernels below are derived from
astro_warp/product/tree.py, vendor commit e5ebdc970a3633bc61cfe8909e28923f97ca2cfc.
Force, softening, opening criterion and
integration remain its implementation. This adapter deliberately submits eagerly:
one scalar occupancy read precedes any pool growth, so no captured graph retains
an obsolete device pointer and a frame is never integrated twice to retry a build.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from ._vendor.astro_warp.product.tree import (
    BarnesHutConfig, BarnesHutRuntime, MAX_TREE_NODES, _quad_dtype,
)
from ._vendor.astro_warp.product.nbody import Precision, _dtypes, _registrars
from .galaxy_layout import make_fill_nodes, record_walk
from .galaxy_tree import configure_galaxy_walk, record_radix_sort


def compact_tree_config(config: BarnesHutConfig) -> BarnesHutConfig:
    """Validate the vendor ABI with a minimal startup pool, before occupancy exists."""
    result = replace(config, node_capacity=config.max_depth + 1)
    result.validate()
    return result


def reachable_node_bound(count: int, max_depth: int, leaf_capacity: int) -> int:
    """Proven bound, including unary chains; this is not an allocation estimate.

    Each internal node has at least K+1 bodies. Internal nodes at a given level
    have disjoint ranges, so at most min(8**level, N//(K+1)) exist. Every nonroot
    node has an internal parent; leaves partition the bodies. Independently, each
    level has at most min(8**level,N) occupied prefixes.
    """
    if count < 1 or max_depth < 1 or leaf_capacity < 1:
        raise ValueError('count, max_depth and leaf_capacity must be positive')
    if count <= leaf_capacity:
        return 1
    prefixes = 1 + sum(min(8**level, count) for level in range(1, max_depth + 1))
    reachable = count + sum(min(8**level, count // (leaf_capacity + 1))
                            for level in range(max_depth))
    return min(prefixes, reachable)


def grown_node_capacity(current: int, required: int) -> int:
    """First occupancy is exact; subsequent growth keeps 25 percent spare capacity."""
    if required < 1 or required > MAX_TREE_NODES:
        raise ValueError(f'compact tree requires {required} nodes; supported maximum is {MAX_TREE_NODES}')
    if required <= current:
        return current
    return min(MAX_TREE_NODES, max(required, math.ceil(current * 1.25)))


class CompactBarnesHutRuntime(BarnesHutRuntime):
    """Vendor-compatible tree runtime whose node pool follows reachable occupancy."""

    def __init__(self, device_context, config: BarnesHutConfig):
        super().__init__(device_context, compact_tree_config(config))
        self._compact_kernels = None
        self._compact_actual = 0
        self._compact_growths = 0
        self._compact_radix = False

    def rebuild_tree(self):
        configure_galaxy_walk(self)
        # The classification buffer is idle during sorting. Warp requires two
        # int64 keys and two int32 indices per padded slot (six int32 words).
        self._compact_radix = (self.config.count * self.config.max_depth >= 6 * self._sort_size
                               and self._stable_ids_increasing)
        super().rebuild_tree()

    def _record_bounds_sort(self):
        if not self._compact_radix:
            return super()._record_bounds_sort()
        self._ensure_compact_scratch()
        wp = self._require_wp()
        device = self.device_context.device
        address = self._arrays["compact_flags"].ptr
        length = 2 * self._sort_size
        keys = wp.array(ptr=address, shape=(length,), dtype=wp.int64, device=device)
        indices = wp.array(ptr=address + 8 * length, shape=(length,), dtype=wp.int32, device=device)
        record_radix_sort(self, keys, indices)

    def diagnostics(self):
        report = super().diagnostics()
        return replace(report, sort_algorithm="stable_morton_radix") if self._compact_radix else report

    def _submit(self, name, record):
        record()

    def _require_compact(self):
        if self._compact_kernels is None:
            wp = self._require_wp()
            use_cutoff = self.config.precision.value == 'float32' and wp.get_device(self.device_context.device).is_cuda
            self._compact_kernels = _compact_kernels(wp, self.config.precision, use_cutoff)
        return self._compact_kernels

    def storage_status(self):
        scalar_bytes = 8 if self.config.precision == Precision.FLOAT64 else 4
        return dict(storage='COMPACT', actual_nodes=self._compact_actual,
                    allocated_nodes=self._total_nodes, allocation_growths=self._compact_growths,
                    node_storage_bytes=self._total_nodes * (40 + 21 * scalar_bytes +
                        (4 if 'walk_storage' in self._arrays else 0)),
                    submission='eager', occupancy_readback_bytes=4)

    def _grow_node_pool(self, required):
        target = grown_node_capacity(self._total_nodes, required)
        if self._compact_growths == 0 and required > self._total_nodes:
            target = required
        if target == self._total_nodes:
            return
        wp = self._require_wp()
        arrays = self._arrays
        names = [name for name in arrays if name.startswith('node_') and name != 'node_cursor']
        fields = ('walk_links', 'walk_cells', 'walk_mass_diagonal', 'walk_off_diagonal')
        device = self.device_context.device
        packed = self.config.precision == Precision.FLOAT32 and wp.get_device(device).is_cuda
        scalar_bytes = 8 if self.config.precision == Precision.FLOAT64 else 4
        # Group four aligned fields per cell. The final vec3 has one padding word.
        separate_bytes = 40 + 21 * scalar_bytes
        bytes_per_node = separate_bytes + (4 if packed else 0)
        try:
            self._preflight_device_bytes(target * bytes_per_node)
        except MemoryError:
            # Drop spare capacity first, then the optional alignment padding.
            target = required
            if packed:
                try:
                    self._preflight_device_bytes(target * bytes_per_node)
                except MemoryError:
                    packed = False
            if not packed:
                self._preflight_device_bytes(target * separate_bytes)
        if not packed:
            names.extend(fields)
        # Allocate transactionally. A failed allocation leaves every existing pointer
        # intact; no particle state is reallocated or copied during topology growth.
        replacements = {name: wp.zeros(target, dtype=arrays[name].dtype,
                                      device=self.device_context.device) for name in names}
        if packed:
            storage = wp.zeros((target, 4), dtype=wp.vec4, device=device)
            replacements['walk_storage'] = storage
            for index, name in enumerate(fields):
                replacements[name] = wp.array(ptr=storage.ptr + index * 16, dtype=arrays[name].dtype,
                                              shape=(target,), strides=(64,), device=device)
        arrays.update(replacements)
        if not packed:
            arrays.pop('walk_storage', None)
        self._total_nodes = target
        self._compact_growths += 1
        self._graphs.clear()

    def _preflight_device_bytes(self, additional):
        """Account for the temporary overlap with the old pool before replacing it."""
        device = self._require_wp().get_device(self.device_context.device)
        if device.is_cuda:
            free = device.free_memory
            reserve = 64 * 1024 * 1024
            if additional + reserve > free:
                raise MemoryError(f'compact tree allocation needs {additional} additional GPU bytes '
                                  f'plus {reserve} reserve; only {free} are free')

    def _ensure_compact_scratch(self):
        if 'compact_flags' not in self._arrays:
            wp = self._require_wp()
            device = self.device_context.device
            slots = self.config.count * self.config.max_depth
            self._preflight_device_bytes(4 * slots + 4)
            flags = wp.zeros(slots, dtype=wp.int32, device=device)
            total = wp.zeros(1, dtype=wp.int32, device=device)
            self._arrays.update(compact_flags=flags, compact_total=total)

    def _record_topology_clear_build(self):
        wp = self._require_wp()
        kernels = self._require_kernels()
        compact = self._require_compact()
        arrays = self._arrays
        config = self.config
        device = self.device_context.device
        slots = config.count * config.max_depth
        self._ensure_compact_scratch()
        # slot_nodes row zero is the root. Subsequent rows are a level-major
        # inclusive prefix scan; each reachable prefix head receives a dense ID.
        mapped = wp.array(ptr=arrays['slot_nodes'].ptr + 4 * config.count,
                          shape=(slots,), dtype=wp.int32, device=device)
        wp.launch(compact.classify, dim=slots,
                  inputs=[arrays['sorted_keys'], config.count, config.max_depth,
                          config.leaf_capacity, arrays['compact_flags']], device=device)
        from warp.utils import array_scan
        array_scan(arrays['compact_flags'], mapped, inclusive=True)
        wp.copy(arrays['compact_total'], mapped, src_offset=slots - 1, count=1)
        required = int(arrays['compact_total'].numpy()[0]) + 1
        self._grow_node_pool(required)
        self._compact_actual = required
        wp.launch(kernels.clear_tree_flags, dim=8, inputs=[arrays['diag']], device=device)
        moment_names = ['node_mass', 'node_moment_x', 'node_moment_y', 'node_moment_z',
                        'node_qxx', 'node_qxy', 'node_qxz', 'node_qyy', 'node_qyz', 'node_qzz']
        wp.launch(kernels.clear_tree, dim=self._total_nodes,
                  inputs=[arrays['node_prefix'], arrays['node_levels'],
                          *[arrays[name] for name in moment_names], arrays['node_count'],
                          arrays['node_start'], arrays['node_end'], self._sort_size], device=device)
        wp.launch(compact.build, dim=slots,
                  inputs=[arrays['sorted_keys'], config.count, config.max_depth,
                          arrays['compact_flags'], arrays['slot_nodes'], arrays['node_prefix'],
                          arrays['node_levels'], arrays['node_start'], arrays['node_end'],
                          arrays['node_count'], arrays['node_cursor'], required], device=device)
        wp.launch(compact.prune_mapping, dim=slots,
                  inputs=[config.count, arrays['slot_nodes'], arrays['node_levels'],
                          arrays['node_end']], device=device)

    def _record_node_moments(self):
        wp = self._require_wp()
        compact = self._require_compact()
        arrays = self._arrays
        config = self.config
        moments = [arrays[name] for name in ['node_mass', 'node_moment_x', 'node_moment_y',
                  'node_moment_z', 'node_qxx', 'node_qxy', 'node_qxz', 'node_qyy',
                  'node_qyz', 'node_qzz']]
        wp.launch(compact.leaf_moments, dim=self._compact_actual,
                  inputs=[arrays['positions'], self._tree_masses, arrays['active'],
                          arrays['sorted_indices'], config.count, 0, arrays['node_start'],
                          arrays['node_end'], *moments, int(config.quadrupoles), arrays['node_count'],
                          arrays['node_levels'], config.max_depth, config.leaf_capacity],
                  device=self.device_context.device)
        for level in range(config.max_depth - 1, -1, -1):
            wp.launch(compact.reduce_moments, dim=self._compact_actual,
                      inputs=[config.count, level, 0, arrays['slot_nodes'], arrays['node_start'],
                              arrays['node_end'], *moments, int(config.quadrupoles),
                              arrays['node_count'], arrays['node_levels'], config.leaf_capacity],
                      device=self.device_context.device)
        self._record_node_quadrupole_finalize()

    def _record_tree_diagnostic_collection(self):
        wp = self._require_wp()
        wp.launch(self._require_kernels().clear_tree_report, dim=8,
                  inputs=[self._arrays['diag']], device=self.device_context.device)
        wp.launch(self._require_compact().collect_diag, dim=self._compact_actual,
                  inputs=[self._arrays['node_count'], self._arrays['node_levels'],
                          self.config.max_depth, self.config.leaf_capacity, self._arrays['diag']],
                  device=self.device_context.device)


    def _record_walk(self, output="accelerations"):
        layout = self._require_compact()
        record_walk(self, layout.prepare_walk_slots, layout.fill_walk_nodes, self._arrays["slot_nodes"], output)


def _compact_kernels(wp, precision, use_cutoff=False):
    scalar, vector, _ = _dtypes(wp, precision)
    quad = _quad_dtype(wp, precision)
    warp_kernel, warp_func = _registrars(wp, scalar, vector, quad=quad)
    zero = scalar(0.0)
    sentinel = wp.int64(9223372036854775807)

    @warp_func
    def lower_prefix(keys: wp.array(dtype=wp.int64), count: int, prefix: wp.int64, shift: int):
        low = wp.int32(0)
        high = count
        while low < high:
            middle = (low + high) // 2
            if (keys[middle] >> wp.int64(shift)) < prefix:
                low = middle + 1
            else:
                high = middle
        return low

    @warp_kernel
    def classify(keys: wp.array(dtype=wp.int64), count: int, depth: int, leaf_capacity: int,
                 flags: wp.array(dtype=wp.int32)):
        index = wp.tid()
        level = index // count + 1
        slot = index % count
        key = keys[slot]
        value = wp.int32(0)
        shift = 3 * (depth - level)
        if key != sentinel:
            prefix = key >> wp.int64(shift)
            head = wp.int32(1)
            if slot > 0:
                if (keys[slot - 1] >> wp.int64(shift)) == prefix:
                    head = 0
            if head != 0:
                parent_shift = shift + 3
                parent_prefix = key >> wp.int64(parent_shift)
                # Find the start of this parent, clipped to the preceding K
                # slots. A run longer than K includes start+K even if clipped.
                low = wp.max(0, slot - leaf_capacity)
                high = slot
                while low < high:
                    middle = (low + high) // 2
                    if (keys[middle] >> wp.int64(parent_shift)) < parent_prefix:
                        low = middle + 1
                    else:
                        high = middle
                if leaf_capacity < count - low and (keys[low + leaf_capacity] >> wp.int64(parent_shift)) == parent_prefix:
                    value = 1
        flags[index] = value

    @warp_kernel
    def build(keys: wp.array(dtype=wp.int64), count: int, depth: int,
              flags: wp.array(dtype=wp.int32), mapping: wp.array(dtype=wp.int32),
              prefixes: wp.array(dtype=wp.int64), levels: wp.array(dtype=wp.int32),
              starts: wp.array(dtype=wp.int32), ends: wp.array(dtype=wp.int32),
              counts: wp.array(dtype=wp.int32), cursor: wp.array(dtype=wp.int32), total: int):
        index = wp.tid()
        if index == 0:
            active_count = lower_prefix(keys, count, wp.int64(1), 3 * depth)
            prefixes[0] = wp.int64(0)
            levels[0] = 0
            starts[0] = 0
            ends[0] = active_count - 1
            counts[0] = active_count
            cursor[0] = total
        level = index // count + 1
        slot = index % count
        shift = 3 * (depth - level)
        prefix = keys[slot] >> wp.int64(shift)
        if flags[index] != 0:
            end = lower_prefix(keys, count, prefix + wp.int64(1), shift)
            node = mapping[count + index]
            prefixes[node] = prefix
            levels[node] = level
            starts[node] = slot
            ends[node] = end - 1
            counts[node] = end - slot

    @warp_kernel
    def prune_mapping(count: int, mapping: wp.array(dtype=wp.int32),
                      levels: wp.array(dtype=wp.int32), ends: wp.array(dtype=wp.int32)):
        index = wp.tid()
        node = mapping[count + index]
        # Scanning repeats the last head's ID. A slot at or after that
        # head belongs to it exactly when the level matches and the run
        # hasn't ended. build has published every reachable range by now.
        if levels[node] != index // count + 1 or ends[node] < index % count:
            mapping[count + index] = -1

    @warp_kernel
    def collect_diag(counts: wp.array(dtype=wp.int32), levels: wp.array(dtype=wp.int32),
                     depth: int, leaf_capacity: int, diag: wp.array(dtype=wp.int64)):
        node = wp.tid()
        if node == 0:
            diag[6] = wp.int64(counts[0])
        if counts[node] > 0:
            wp.atomic_add(diag, 0, wp.int64(1))
            wp.atomic_max(diag, 1, wp.int64(levels[node]))
            if counts[node] <= leaf_capacity or levels[node] == depth:
                wp.atomic_add(diag, 5, wp.int64(1))

    @warp_kernel
    def leaf_moments(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        node_base: int,
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        use_quadrupoles: int,
        node_count: wp.array(dtype=wp.int32),
        node_levels: wp.array(dtype=wp.int32),
        max_depth: int,
        leaf_capacity: int,
    ):
        node = node_base + wp.tid()
        if node_count[node] > leaf_capacity and node_levels[node] < max_depth:
            return
        start = node_start[node]
        end = node_end[node]
        if start < 0 or end < start:
            return
        total_mass = scalar(0.0)
        total_x = scalar(0.0)
        total_y = scalar(0.0)
        total_z = scalar(0.0)
        bodies = wp.int32(0)
        for slot in range(start, end + 1):
            if slot < count:
                source = sorted_indices[slot]
                if source >= 0 and active[source] != 0:
                    mass = masses[source]
                    p = positions[source]
                    total_mass += mass
                    total_x += mass * p[0]
                    total_y += mass * p[1]
                    total_z += mass * p[2]
                    bodies += 1
        node_mass[node] = total_mass
        moment_x[node] = total_x
        moment_y[node] = total_y
        moment_z[node] = total_z
        node_count[node] = bodies

        if use_quadrupoles != 0 and total_mass > zero:
            inverse_mass = scalar(1.0) / total_mass
            center = vector(
                total_x * inverse_mass, total_y * inverse_mass, total_z * inverse_mass
            )
            sxx = scalar(0.0)
            sxy = scalar(0.0)
            sxz = scalar(0.0)
            syy = scalar(0.0)
            syz = scalar(0.0)
            szz = scalar(0.0)
            for slot in range(start, end + 1):
                if slot < count:
                    source = sorted_indices[slot]
                    if source >= 0 and active[source] != 0:
                        mass = masses[source]
                        s = positions[source] - center
                        sxx += mass * s[0] * s[0]
                        sxy += mass * s[0] * s[1]
                        sxz += mass * s[0] * s[2]
                        syy += mass * s[1] * s[1]
                        syz += mass * s[1] * s[2]
                        szz += mass * s[2] * s[2]
            qxx[node] = sxx
            qxy[node] = sxy
            qxz[node] = sxz
            qyy[node] = syy
            qyz[node] = syz
            qzz[node] = szz

    @warp_kernel
    def reduce_moments(
        count: int,
        level: int,
        node_base: int,
        slot_nodes: wp.array(dtype=wp.int32),
        node_start: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        node_mass: wp.array(dtype=scalar),
        moment_x: wp.array(dtype=scalar),
        moment_y: wp.array(dtype=scalar),
        moment_z: wp.array(dtype=scalar),
        qxx: wp.array(dtype=scalar),
        qxy: wp.array(dtype=scalar),
        qxz: wp.array(dtype=scalar),
        qyy: wp.array(dtype=scalar),
        qyz: wp.array(dtype=scalar),
        qzz: wp.array(dtype=scalar),
        use_quadrupoles: int,
        node_count: wp.array(dtype=wp.int32),
        node_levels: wp.array(dtype=wp.int32),
        leaf_capacity: int,
    ):
        node = node_base + wp.tid()
        if node_levels[node] != level or node_count[node] <= leaf_capacity:
            return
        start = node_start[node]
        end = node_end[node]
        if start < 0 or end < start:
            return
        child_row = (level + 1) * count
        total_mass = scalar(0.0)
        total_x = scalar(0.0)
        total_y = scalar(0.0)
        total_z = scalar(0.0)
        bodies = wp.int32(0)
        # Walk the children by jumping over each one's slot range, the same enumeration
        # the traversal uses. A node has at most eight children, so this loop is short and
        # uniform across the launch.
        cursor = start
        while cursor <= end:
            # A slot with no node at the child level (one `build_sparse_topology` skipped,
            # or one it could not place) is stepped over rather than ending the walk, so a
            # gap cannot silently truncate a parent's children. Landing anywhere inside a
            # child's range returns that child and jumps past its end, so the walk is
            # self-correcting and always advances.
            child = slot_nodes[child_row + cursor]
            next_cursor = cursor + 1
            if child >= 0:
                child_end = node_end[child]
                if child_end >= cursor:
                    next_cursor = child_end + 1
                    total_mass += node_mass[child]
                    total_x += moment_x[child]
                    total_y += moment_y[child]
                    total_z += moment_z[child]
                    bodies += node_count[child]
            cursor = next_cursor
        node_mass[node] = total_mass
        moment_x[node] = total_x
        moment_y[node] = total_y
        moment_z[node] = total_z
        node_count[node] = bodies

        # Parallel-axis shift, child centre of mass to parent centre of mass. The warning
        # on the body-level pass above is about shifting from the ORIGIN, where the offset
        # is the distance to the scene origin and the subtraction eats the whole moment.
        # Here the offset is bounded by the parent cell, so `m d_a d_b` is the same order
        # as the S it is added to -- the identical conditioning the leaf pass already has
        # when it forms `positions[source] - center`.
        if use_quadrupoles != 0 and total_mass > zero:
            inverse_mass = scalar(1.0) / total_mass
            center = vector(
                total_x * inverse_mass, total_y * inverse_mass, total_z * inverse_mass
            )
            sxx = scalar(0.0)
            sxy = scalar(0.0)
            sxz = scalar(0.0)
            syy = scalar(0.0)
            syz = scalar(0.0)
            szz = scalar(0.0)
            cursor = start
            while cursor <= end:
                child = slot_nodes[child_row + cursor]
                next_cursor = cursor + 1
                if child >= 0:
                    child_end = node_end[child]
                    if child_end >= cursor:
                        next_cursor = child_end + 1
                        child_mass = node_mass[child]
                        if child_mass > zero:
                            child_inverse = scalar(1.0) / child_mass
                            d = vector(
                                moment_x[child] * child_inverse - center[0],
                                moment_y[child] * child_inverse - center[1],
                                moment_z[child] * child_inverse - center[2],
                            )
                            sxx += qxx[child] + child_mass * d[0] * d[0]
                            sxy += qxy[child] + child_mass * d[0] * d[1]
                            sxz += qxz[child] + child_mass * d[0] * d[2]
                            syy += qyy[child] + child_mass * d[1] * d[1]
                            syz += qyz[child] + child_mass * d[1] * d[2]
                            szz += qzz[child] + child_mass * d[2] * d[2]
                cursor = next_cursor
            qxx[node] = sxx
            qxy[node] = sxy
            qxz[node] = sxz
            qyy[node] = syy
            qyz[node] = syz
            qzz[node] = szz

    @warp_kernel
    def prepare_walk_slots(
        positions: wp.array(dtype=vector),
        masses: wp.array(dtype=scalar),
        active: wp.array(dtype=wp.int32),
        sorted_keys: wp.array(dtype=wp.int64),
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        max_depth: int,
        bounds: wp.array(dtype=scalar),
        node_count: wp.array(dtype=wp.int32),
        leaf_capacity: int,
        node_capacity: int,
        walk_bodies: wp.array(dtype=quad),
        walk_first_level: wp.array(dtype=wp.int32),
        walk_node_counts: wp.array(dtype=wp.int32),
        slot_nodes: wp.array(dtype=wp.int32),
    ):
        slot = wp.tid()
        if slot >= count:
            return
        i = sorted_indices[slot]
        first = max_depth + 1
        body = quad()
        key = sorted_keys[slot]
        if i >= 0:
            if active[i] != 0:
                p = positions[i]
                body = quad(p[0], p[1], p[2], masses[i])
            if slot == 0:
                first = 0
            else:
                previous = sorted_keys[slot - 1]
                for level in range(1, max_depth + 1):
                    shift = wp.int64(3 * (max_depth - level))
                    if first > max_depth and (key >> shift) != (previous >> shift):
                        first = level
        # ADR 0228: lay out only the nodes a walk can reach. The walk never goes below a node
        # it sums as a leaf (at most `leaf_capacity` bodies, or at `max_depth`), so a node
        # whose parent is such a leaf is never visited, and neither is anything under it.
        # Laying those out cost about ten entries per body at 500,000 bodies, nine of them
        # dead, and the walk read its live entries scattered among them. A node's count only
        # shrinks with depth, so the reachable nodes a slot starts are one run of levels: from
        # `first`, if its parent is not a leaf, down to the first one that is a leaf itself.
        emitted = wp.int32(0)
        if first <= max_depth:
            reachable = wp.int32(1)
            if first > 0:
                parent_level = first - 1
                parent = wp.int32(0)
                if parent_level > 0:
                    # The parent was started by the first slot sharing its prefix.
                    shift = wp.int64(3 * (max_depth - parent_level))
                    prefix = key >> shift
                    low = wp.int32(0)
                    high = slot
                    while low < high:
                        middle = (low + high) // 2
                        if (sorted_keys[middle] >> shift) < prefix:
                            low = middle + 1
                        else:
                            high = middle
                    parent = slot_nodes[parent_level * count + low]
                # A parent the topology could not allocate is laid out below as it always
                # was; the walk fails closed on reaching it.
                if parent < 0:
                    reachable = 0
                elif parent < node_capacity and node_count[parent] <= leaf_capacity:
                    reachable = 0
            if reachable != 0:
                for level in range(first, max_depth + 1):
                    emitted += 1
                    node = wp.int32(0)
                    if level > 0:
                        node = slot_nodes[level * count + slot]
                    if node < 0 or node >= node_capacity:
                        break
                    if node_count[node] <= leaf_capacity:
                        break
        walk_bodies[slot] = body
        walk_first_level[slot] = first
        walk_node_counts[slot] = emitted

    fill_walk_nodes = make_fill_nodes(wp, precision, compact=True, use_cutoff=use_cutoff)
    return SimpleNamespace(classify=classify, build=build, prune_mapping=prune_mapping, collect_diag=collect_diag,
                           leaf_moments=leaf_moments, reduce_moments=reduce_moments,
                           prepare_walk_slots=prepare_walk_slots, fill_walk_nodes=fill_walk_nodes)
