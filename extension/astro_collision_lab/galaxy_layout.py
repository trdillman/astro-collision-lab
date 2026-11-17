"""Pack occupied cells for the galaxy force walk in either storage layout.

Derived from astro_warp/product/tree.py at e5ebdc970a3633bc61cfe8909e28923f97ca2cfc.
SPDX-License-Identifier: LicenseRef-Private-Internal
"""
from __future__ import annotations

from ._vendor.astro_warp.product.nbody import _dtypes, _registrars
from ._vendor.astro_warp.product.tree import _quad_dtype


def record_walk(runtime, prepare_slots, fill_nodes, mapping, output):
    """ADR 0132: lay the tree out depth-first, then walk it with no stack."""

    wp = runtime._require_wp()
    kernels = runtime._require_kernels()
    device = runtime.device_context.device
    count = runtime.config.count
    arrays = runtime._arrays
    wp.launch(
        prepare_slots,
        dim=count,
        inputs=[
            arrays["positions"],
            runtime._tree_masses,
            arrays["active"],
            arrays["sorted_keys"],
            arrays["sorted_indices"],
            count,
            runtime.config.max_depth,
            arrays["bounds"],
            arrays["node_count"],
            runtime.config.leaf_capacity,
            runtime._total_nodes,
            arrays["walk_bodies"],
            arrays["walk_first_level"],
            arrays["walk_node_counts"],
            *([mapping] if mapping is not None else []),
        ],
        device=device,
    )
    wp.launch(
        kernels.total_walk_chunks,
        dim=runtime._walk_chunks,
        inputs=[
            arrays["walk_node_counts"],
            count,
            runtime._walk_chunk,
            arrays["walk_chunk_totals"],
        ],
        device=device,
    )
    wp.launch(
        kernels.offset_walk_slots,
        dim=runtime._walk_chunks,
        inputs=[
            arrays["walk_node_counts"],
            count,
            runtime._walk_chunk,
            arrays["walk_chunk_totals"],
            arrays["walk_offsets"],
        ],
        device=device,
    )
    wp.launch(
        fill_nodes,
        dim=count,
        inputs=[
            count,
            runtime.config.max_depth,
            runtime._total_nodes,
            runtime._total_nodes,
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
            runtime.config.theta,
            arrays["walk_links"],
            arrays["walk_cells"],
            arrays["walk_mass_diagonal"],
            arrays["walk_off_diagonal"],
            arrays["diag"],
            mapping,
            runtime.config.leaf_capacity,
        ],
        device=device,
    )
    wp.launch(
        kernels.tree_acceleration_walk,
        dim=count,
        block_dim=runtime.walk_block_dim,
        inputs=[
            arrays["sorted_indices"],
            count,
            runtime.config.gravitational_constant,
            runtime.config.softening,
            runtime.config.theta,
            runtime.config.max_depth,
            runtime.config.leaf_capacity,
            runtime._total_nodes,
            int(runtime.config.quadrupoles),
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


def make_fill_nodes(wp, precision, *, compact, use_cutoff):
    from .galaxy_walk import opening_cutoff
    scalar, vector, _ = _dtypes(wp, precision)
    quad = _quad_dtype(wp, precision)
    warp_kernel, _ = _registrars(wp, scalar, vector, quad=quad)
    half_one = scalar(0.5)

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
        leaf_capacity: int,
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
                    if wp.static(compact):
                        node = slot_nodes[level * count + slot]
                    else:
                        node = 1 + (level - 1) * count + slot
                if node < 0 or node >= node_capacity:
                    # A node the topology could not allocate; `build_sparse_topology` has
                    # already raised diag[7]. A walk that reaches it fails closed.
                    walk_links[entry] = wp.vec4i(slot, slot, entry + 1, -1)
                else:
                    end = node_end[node]
                    walk_links[entry] = wp.vec4i(
                        slot,
                        end,
                        walk_offsets[end + 1],
                        node_count[node] | (level << 24),
                    )
                    # Leaf visits read sorted bodies directly; only internal cells need moments.
                    if level == max_depth or node_count[node] <= leaf_capacity:
                        continue
                    mass = node_mass[node]
                    com = vector(
                        moment_x[node] / mass,
                        moment_y[node] / mass,
                        moment_z[node] / mass,
                    )
                    # ADR 0147: a cell reaches as far as its edge plus theta times the offset
                    # of its mass from its centre (Barnes 1994), so a body just outside a cell
                    # whose mass sits on the far side still opens it.
                    reach = (half + half) + theta * wp.length(com - center)
                    if wp.static(use_cutoff):
                        reach = opening_cutoff(reach, theta)
                    walk_cells[entry] = quad(com[0], com[1], com[2], reach)
                    walk_mass_diagonal[entry] = quad(mass, qxx[node], qyy[node], qzz[node])
                    walk_off_diagonal[entry] = vector(qxy[node], qxz[node], qyz[node])

    return fill_walk_nodes
