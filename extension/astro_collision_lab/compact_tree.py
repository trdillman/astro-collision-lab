"""Reachable, compact Barnes-Hut storage, with occupancy-driven device allocation.

GPL-3.0-or-later. Moment and depth-first walk kernels below are derived from
astro_warp/product/tree.py, vendor commit e5ebdc970a3633bc61cfe8909e28923f97ca2cfc.
The original vendor module is unmodified. Force, softening, opening criterion and
integration remain its implementation. This adapter deliberately submits eagerly:
one scalar occupancy read precedes any pool growth, so no captured graph retains
an obsolete device pointer and a frame is never integrated twice to retry a build.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from ._vendor.astro_warp.product.nbody import Precision, _dtypes, _registrars
from ._vendor.astro_warp.product.tree import (
    MAX_TREE_NODES,
    BarnesHutConfig,
    BarnesHutRuntime,
    _quad_dtype,
)


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
        self._sort_scratch = None
        self._stable_order = None
        self._compact_kernels = None
        self._compact_actual = 0
        self._compact_growths = 0

    def _submit(self, name, record):
        record()

    def _require_compact(self):
        if self._compact_kernels is None:
            self._compact_kernels = _compact_kernels(self._require_wp(), self.config.precision)
        return self._compact_kernels

    def storage_status(self):
        scalar_bytes = 8 if self.config.precision == Precision.FLOAT64 else 4
        return {'storage': 'COMPACT', 'actual_nodes': self._compact_actual,
                    'allocated_nodes': self._total_nodes, 'allocation_growths': self._compact_growths,
                    'node_storage_bytes': self._total_nodes * (40 + 21 * scalar_bytes),
                    'submission': 'eager', 'occupancy_readback_bytes': 4}

    def _grow_node_pool(self, required):
        target = grown_node_capacity(self._total_nodes, required)
        if self._compact_growths == 0 and required > self._total_nodes:
            target = required
        if target == self._total_nodes:
            return
        wp = self._require_wp()
        arrays = self._arrays
        names = [name for name in arrays if name.startswith('node_') and name != 'node_cursor']
        names += ['walk_links', 'walk_cells', 'walk_mass_diagonal', 'walk_off_diagonal']
        scalar_bytes = 8 if self.config.precision == Precision.FLOAT64 else 4
        bytes_per_node = 40 + 21 * scalar_bytes
        try:
            self._preflight_device_bytes(target * bytes_per_node)
        except MemoryError:
            if target == required:
                raise
            # Spare growth capacity is optional. Preserve the same reserve and
            # overlap accounting while admitting the exact live topology.
            target = required
            self._preflight_device_bytes(target * bytes_per_node)
        # Allocate transactionally. A failed allocation leaves every existing pointer
        # intact; no particle state is reallocated or copied during topology growth.
        replacements = {name: wp.zeros(target, dtype=arrays[name].dtype,
                                      device=self.device_context.device) for name in names}
        arrays.update(replacements)
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

    def _record_topology_clear_build(self):
        wp = self._require_wp()
        kernels = self._require_kernels()
        compact = self._require_compact()
        arrays = self._arrays
        config = self.config
        device = self.device_context.device
        slots = config.count * config.max_depth
        if 'compact_flags' not in arrays:
            self._preflight_device_bytes(4 * slots + 4)
            arrays['compact_flags'] = wp.zeros(slots, dtype=wp.int32, device=device)
            arrays['compact_total'] = wp.zeros(1, dtype=wp.int32, device=device)
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


    def _record_walk(self, output: str = "accelerations") -> None:
        """ADR 0132: lay the tree out depth-first, then walk it with no stack."""

        wp = self._require_wp()
        kernels = self._require_kernels()
        compact = self._require_compact()
        device = self.device_context.device
        count = self.config.count
        arrays = self._arrays
        wp.launch(
            compact.prepare_walk_slots,
            dim=count,
            inputs=[
                arrays["positions"],
                self._tree_masses,
                arrays["active"],
                arrays["sorted_keys"],
                arrays["sorted_indices"],
                count,
                self.config.max_depth,
                arrays["bounds"],
                arrays["node_count"],
                self.config.leaf_capacity,
                self._total_nodes,
                arrays["walk_bodies"],
                arrays["walk_first_level"],
                arrays["walk_node_counts"],
                arrays["slot_nodes"],
            ],
            device=device,
        )
        wp.launch(
            kernels.total_walk_chunks,
            dim=self._walk_chunks,
            inputs=[
                arrays["walk_node_counts"],
                count,
                self._walk_chunk,
                arrays["walk_chunk_totals"],
            ],
            device=device,
        )
        wp.launch(
            kernels.offset_walk_slots,
            dim=self._walk_chunks,
            inputs=[
                arrays["walk_node_counts"],
                count,
                self._walk_chunk,
                arrays["walk_chunk_totals"],
                arrays["walk_offsets"],
            ],
            device=device,
        )
        wp.launch(
            compact.fill_walk_nodes,
            dim=count,
            inputs=[
                count,
                self.config.max_depth,
                self._total_nodes,
                self._total_nodes,
                arrays["walk_first_level"],
                arrays["walk_offsets"],
                arrays["node_mass"],
                arrays["node_moment_x"],
                arrays["node_moment_y"],
                arrays["node_moment_z"],
                arrays["node_qxx"],
                arrays["node_qxy"],
                arrays["node_qxz"],
                arrays["node_qyy"],
                arrays["node_qyz"],
                arrays["node_qzz"],
                arrays["node_count"],
                arrays["node_end"],
                arrays["sorted_keys"],
                arrays["bounds"],
                self.config.theta,
                arrays["walk_links"],
                arrays["walk_cells"],
                arrays["walk_mass_diagonal"],
                arrays["walk_off_diagonal"],
                arrays["diag"],
                arrays["slot_nodes"],
            ],
            device=device,
        )
        wp.launch(
            compact.acceleration_walk,
            dim=count,
            inputs=[
                arrays["sorted_indices"],
                count,
                self.config.gravitational_constant,
                self.config.softening,
                self.config.theta,
                self.config.max_depth,
                self.config.leaf_capacity,
                self._total_nodes,
                int(self.config.quadrupoles),
                arrays["walk_bodies"],
                arrays["walk_offsets"],
                arrays["walk_links"],
                arrays["walk_cells"],
                arrays["walk_mass_diagonal"],
                arrays["walk_off_diagonal"],
                arrays[output],
                arrays["diag"],
            ],
            device=device,
        )

    def load_bodies(self, bodies, **kwargs):
        self._stable_order = None
        return super().load_bodies(bodies, **kwargs)

    def restore_snapshot(self, snapshot):
        self._stable_order = None
        return super().restore_snapshot(snapshot)

    def _sort_arrays(self):
        wp = self._require_wp()
        device = self.device_context.device
        count = self.config.count
        if self._sort_scratch is None or self._sort_scratch[0].shape[0] < 2 * count:
            self._sort_scratch = (wp.zeros(2 * count, dtype=wp.int64, device=device),
                                  wp.zeros(2 * count, dtype=wp.int32, device=device),
                                  wp.zeros(count, dtype=wp.int32, device=device))
            self._stable_order = None
        return self._sort_scratch

    def _record_bounds_sort(self) -> None:
        """The vendor Morton order (key, then stable id) from one stable radix sort.

        Stable ids change only in ``load_bodies`` and ``restore_snapshot``, so their order
        is sorted once there; each rebuild then radix-sorts the Morton keys of bodies laid
        out in stable-id order. A stable sort keeps that order within equal keys, which is
        exactly the vendor bitonic network's tie-break, in O(N) instead of O(N log^2 N).
        """
        wp = self._require_wp()
        kernels = self._require_kernels()
        compact = self._require_compact()
        device = self.device_context.device
        count = self.config.count
        arrays = self._arrays
        wp.launch(kernels.clear_bounds, dim=6, inputs=[arrays["bounds"]], device=device)
        wp.launch(kernels.reduce_bounds, dim=count,
                  inputs=[arrays["positions"], arrays["active"], count, arrays["bounds"]],
                  device=device)
        wp.launch(kernels.finalize_bounds, dim=1, inputs=[arrays["bounds"]], device=device)
        wp.launch(kernels.compute_morton, dim=self._sort_size,
                  inputs=[arrays["positions"], arrays["active"], count, self._sort_size,
                          self.config.max_depth, arrays["bounds"], arrays["body_keys"],
                          arrays["sorted_keys"], arrays["sorted_indices"]],
                  device=device)
        if count < 2:
            return
        keys, values, order = self._sort_arrays()
        if self._stable_order is None:
            wp.launch(compact.stable_id_keys, dim=count,
                      inputs=[arrays["stable_ids"], count, keys, values], device=device)
            wp.utils.radix_sort_pairs(keys, values, count)
            wp.copy(order, values, count=count)
            self._stable_order = order
        # Morton keys use 3 bits per level; inactive bodies sort after every real key.
        key_bits = 3 * self.config.max_depth
        wp.launch(compact.gather_sort_pairs, dim=count,
                  inputs=[arrays["body_keys"], order, count, wp.int64(1 << key_bits), keys, values],
                  device=device)
        wp.utils.radix_sort_pairs(keys, values, count, 0, key_bits + 1)
        wp.launch(compact.scatter_sorted_pairs, dim=count,
                  inputs=[keys, values, count, wp.int64(1 << key_bits),
                          arrays["sorted_keys"], arrays["sorted_indices"]],
                  device=device)


# Minimum of `value` over the CUDA lanes converged at the call (one warp at most).
# On CPU a thread is its own group. sm_80+ has a single-instruction reduction.
_CONVERGED_MIN = """
#if defined(__CUDA_ARCH__)
    unsigned mask = __activemask();
#if __CUDA_ARCH__ >= 800
    return __reduce_min_sync(mask, value);
#else
    int result = value;
    unsigned rest = mask;
    while (rest) {
        int lane = __ffs(rest) - 1;
        int other = __shfl_sync(mask, value, lane);
        result = other < result ? other : result;
        rest &= rest - 1;
    }
    return result;
#endif
#else
    return value;
#endif
"""


def _compact_kernels(wp, precision):
    scalar, vector, _ = _dtypes(wp, precision)
    quad = _quad_dtype(wp, precision)
    warp_kernel, warp_func = _registrars(wp, scalar, vector, quad=quad)
    zero = scalar(0.0)
    half_one = scalar(0.5)
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
                start = lower_prefix(keys, count, parent_prefix, parent_shift)
                end = lower_prefix(keys, count, parent_prefix + wp.int64(1), parent_shift)
                # Sentinel >> shift remains above every valid prefix, including root.
                if end - start > leaf_capacity:
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
        # Inclusive scan repeats the preceding ID through pruned prefixes. Mark
        # those ranges explicitly absent before the force layout can consult them.
        head = lower_prefix(keys, count, prefix, shift)
        if keys[slot] == sentinel:
            mapping[count + index] = -1
        elif flags[(level - 1) * count + head] == 0:
            mapping[count + index] = -1
        if flags[index] != 0:
            end = lower_prefix(keys, count, prefix + wp.int64(1), shift)
            node = mapping[count + index]
            prefixes[node] = prefix
            levels[node] = level
            starts[node] = slot
            ends[node] = end - 1
            counts[node] = end - slot

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

    @warp_kernel
    def fill_walk_nodes(
        count: int,
        max_depth: int,
        node_capacity: int,
        walk_capacity: int,
        walk_first_level: wp.array(dtype=wp.int32),
        walk_offsets: wp.array(dtype=wp.int32),
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
        node_count: wp.array(dtype=wp.int32),
        node_end: wp.array(dtype=wp.int32),
        sorted_keys: wp.array(dtype=wp.int64),
        bounds: wp.array(dtype=scalar),
        theta: scalar,
        walk_links: wp.array(dtype=wp.vec4i),
        walk_cells: wp.array(dtype=quad),
        walk_mass_diagonal: wp.array(dtype=quad),
        walk_off_diagonal: wp.array(dtype=vector),
        diag: wp.array(dtype=wp.int64),
        slot_nodes: wp.array(dtype=wp.int32),
    ):
        # One writer per entry: slot s writes exactly the nodes it starts.
        slot = wp.tid()
        if slot >= count:
            return
        first = walk_first_level[slot]
        base = walk_offsets[slot]
        # ADR 0228: the slot's run of reachable levels; it may stop above max_depth.
        last = first + (walk_offsets[slot + 1] - base) - 1
        # ADR 0226: the nodes a slot starts are the ancestors of its own body, so one descent
        # along the slot's key gives every one of their geometric centres in O(depth), where
        # rebuilding each from its prefix cost O(level) per node. The same halvings and
        # signed steps run in the same order, so each centre is the same number to the bit.
        key = sorted_keys[slot]
        center = vector(
            (bounds[0] + bounds[3]) * half_one,
            (bounds[1] + bounds[4]) * half_one,
            (bounds[2] + bounds[5]) * half_one,
        )
        half = (bounds[3] - bounds[0]) * half_one
        for level in range(0, max_depth + 1):
            if level > last:
                break
            if level > 0:
                shift = wp.int64(3 * (max_depth - level))
                child_code = wp.int32((key >> shift) & wp.int64(7))
                half *= half_one
                if (child_code & 1) != 0:
                    center[0] += half
                else:
                    center[0] -= half
                if (child_code & 2) != 0:
                    center[1] += half
                else:
                    center[1] -= half
                if (child_code & 4) != 0:
                    center[2] += half
                else:
                    center[2] -= half
            if level < first:
                continue
            entry = base + level - first
            if entry >= walk_capacity:
                wp.atomic_max(diag, 3, wp.int64(1))
            else:
                node = wp.int32(0)
                if level > 0:
                    node = slot_nodes[level * count + slot]
                if node < 0 or node >= node_capacity:
                    # A node the topology could not allocate; `build_sparse_topology` has
                    # already raised diag[7]. A walk that reaches it fails closed.
                    walk_links[entry] = wp.vec4i(slot, slot, entry + 1, -1)
                else:
                    end = node_end[node]
                    mass = node_mass[node]
                    walk_links[entry] = wp.vec4i(
                        slot,
                        end,
                        walk_offsets[end + 1],
                        node_count[node] | (level << 24),
                    )
                    com = vector(
                        moment_x[node] / mass,
                        moment_y[node] / mass,
                        moment_z[node] / mass,
                    )
                    # ADR 0147: a cell reaches as far as its edge plus theta times the offset
                    # of its mass from its centre (Barnes 1994), so a body just outside a cell
                    # whose mass sits on the far side still opens it.
                    reach = (half + half) + theta * wp.length(com - center)
                    walk_cells[entry] = quad(com[0], com[1], com[2], reach)
                    walk_mass_diagonal[entry] = quad(mass, qxx[node], qyy[node], qzz[node])
                    walk_off_diagonal[entry] = vector(qxy[node], qxz[node], qyz[node])

    @wp.func_native(_CONVERGED_MIN)
    def converged_min(value: int) -> int:
        ...

    @warp_kernel
    def acceleration_walk(
        sorted_indices: wp.array(dtype=wp.int32),
        count: int,
        gravitational_constant: scalar,
        softening: scalar,
        theta: scalar,
        max_depth: int,
        leaf_capacity: int,
        walk_capacity: int,
        use_quadrupoles: int,
        walk_bodies: wp.array(dtype=quad),
        walk_offsets: wp.array(dtype=wp.int32),
        walk_links: wp.array(dtype=wp.vec4i),
        walk_cells: wp.array(dtype=quad),
        walk_mass_diagonal: wp.array(dtype=quad),
        walk_off_diagonal: wp.array(dtype=vector),
        accelerations: wp.array(dtype=vector),
        diag: wp.array(dtype=wp.int64),
    ):
        # The vendor stackless walk (tree_acceleration_walk) with the same decisions and
        # the same arithmetic in the same order for every body. What changes is how lanes
        # share the list: each lane keeps its own next entry (`mine`), and the lanes that
        # are converged step together to the smallest of them, so every load is one
        # broadcast per warp instead of one scattered sector per lane. A lane only acts on
        # an entry equal to its own next entry; at any other it waits. Any grouping of
        # lanes is correct (a group's minimum never passes a member's next entry), so
        # convergence affects speed only. On CPU each thread is its own group.
        slot = wp.tid()
        if slot >= count:
            return
        i = sorted_indices[slot]
        if i < 0:
            return
        total = wp.min(walk_offsets[count], walk_capacity)
        body = walk_bodies[slot]
        p = vector(body[0], body[1], body[2])
        a = vector()
        interactions = wp.int32(0)
        overflow = wp.int32(0)
        mine = wp.int32(0)
        entry = wp.int32(0)
        while mine < total:
            if entry == mine:
                link = walk_links[entry]
                cell = walk_cells[entry]
                moments = walk_mass_diagonal[entry]
                oq = walk_off_diagonal[entry]
                start = link[0]
                end = link[1]
                skip = link[2]
                packed = link[3]
                count_node = packed & 16777215
                level = packed >> 24
                if packed < 0:
                    overflow = 1
                    mine = total
                elif count_node == 0:
                    mine = skip
                elif level == max_depth or count_node <= leaf_capacity:
                    cursor = start
                    while cursor <= end:
                        if cursor != slot:
                            source = walk_bodies[cursor]
                            delta = vector(source[0], source[1], source[2]) - p
                            dist2 = wp.dot(delta, delta) + softening * softening
                            inv = scalar(1.0) / wp.sqrt(dist2)
                            a += delta * (gravitational_constant * source[3] * inv * inv * inv)
                            interactions += 1
                        cursor += 1
                    mine = skip
                else:
                    r = vector(cell[0], cell[1], cell[2]) - p
                    d2 = wp.dot(r, r) + softening * softening
                    distance = wp.sqrt(d2)
                    contains_target = slot >= start and slot <= end
                    reach = cell[3]
                    if (not contains_target) and reach / distance < theta:
                        mass = moments[0]
                        inv = scalar(1.0) / distance
                        a += r * (gravitational_constant * mass * inv * inv * inv)
                        if use_quadrupoles != 0:
                            qx = moments[1] * r[0] + oq[0] * r[1] + oq[1] * r[2]
                            qy = oq[0] * r[0] + moments[2] * r[1] + oq[2] * r[2]
                            qz = oq[1] * r[0] + oq[2] * r[1] + moments[3] * r[2]
                            r_q_r = r[0] * qx + r[1] * qy + r[2] * qz
                            inv2 = inv * inv
                            inv5 = inv2 * inv2 * inv
                            inv7 = inv5 * inv2
                            a += vector(qx, qy, qz) * (-gravitational_constant * inv5)
                            a += r * (gravitational_constant * scalar(2.5) * r_q_r * inv7)
                        interactions += 1
                        mine = skip
                    else:
                        mine = entry + 1
            entry = converged_min(mine)

        if overflow != 0:
            accelerations[i] = vector()
            wp.atomic_max(diag, 3, wp.int64(1))
        else:
            accelerations[i] = a
        wp.atomic_add(diag, 4, wp.int64(interactions))

    @warp_kernel
    def stable_id_keys(stable_ids: wp.array(dtype=wp.int64), count: int,
                       keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.int32)):
        i = wp.tid()
        if i < count:
            keys[i] = stable_ids[i]
            values[i] = i

    @warp_kernel
    def gather_sort_pairs(body_keys: wp.array(dtype=wp.int64), order: wp.array(dtype=wp.int32),
                          count: int, inactive_key: wp.int64, keys: wp.array(dtype=wp.int64),
                          values: wp.array(dtype=wp.int32)):
        j = wp.tid()
        if j >= count:
            return
        i = order[j]
        keys[j] = wp.min(body_keys[i], inactive_key)
        values[j] = i

    @warp_kernel
    def scatter_sorted_pairs(keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.int32),
                             count: int, inactive_key: wp.int64,
                             sorted_keys: wp.array(dtype=wp.int64),
                             sorted_indices: wp.array(dtype=wp.int32)):
        j = wp.tid()
        if j >= count:
            return
        key = keys[j]
        if key >= inactive_key:
            sorted_keys[j] = sentinel
            sorted_indices[j] = -1
        else:
            sorted_keys[j] = key
            sorted_indices[j] = values[j]

    return SimpleNamespace(classify=classify, build=build, collect_diag=collect_diag,
                           leaf_moments=leaf_moments, reduce_moments=reduce_moments,
                           prepare_walk_slots=prepare_walk_slots, fill_walk_nodes=fill_walk_nodes,
                           acceleration_walk=acceleration_walk,
                           stable_id_keys=stable_id_keys, gather_sort_pairs=gather_sort_pairs,
                           scatter_sorted_pairs=scatter_sorted_pairs)
