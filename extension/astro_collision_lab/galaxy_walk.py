"""Float32 CUDA force walk for galaxy playback.

Derived from astro_warp/product/tree.py at e5ebdc970a3633bc61cfe8909e28923f97ca2cfc.
SPDX-License-Identifier: LicenseRef-Private-Internal

The opening test uses the reference distance. Once an interaction is selected,
evaluate its inverse distance with the single-precision CUDA reciprocal square root.
"""
import warp as wp

wp.set_module_options({"block_dim": 512, "enable_backward": False})

scalar = wp.float32
vector = wp.vec3
quad = wp.vec4


@wp.func_native("""
#ifdef __CUDA_ARCH__
return rsqrtf(x);
#else
return 1.0f / sqrtf(x);
#endif
""")
def inverse_root(x: float) -> float:
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
return (int)__activemask();
#else
return 0;
#endif
""")
def live_lanes() -> int:
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
return __reduce_min_sync((unsigned)mask, entry);
#else
return entry;
#endif
""")
def earliest_entry(mask: int, entry: int) -> int:
    ...


@wp.kernel
def tree_acceleration_walk(
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
    entry = wp.int32(0)
    # Neighboring Morton slots usually visit the same cells. Schedule the
    # earliest outstanding cell together, preserving each target's own skip
    # decisions, force terms and summation order. Completed lanes participate
    # in the vote until every live lane is done, including partial final warps.
    mask = live_lanes()
    while True:
        shared_entry = earliest_entry(mask, entry)
        if shared_entry >= total:
            break
        if entry != shared_entry:
            continue
        link = walk_links[entry]
        start = link[0]
        end = link[1]
        skip = link[2]
        packed = link[3]
        if packed < 0:
            overflow = 1
            entry = total
            continue
        count_node = packed & 16777215
        level = packed >> 24
        if count_node == 0:
            entry = skip
            continue
        if level == max_depth or count_node <= leaf_capacity:
            # Every slot in a node's range holds a sorted body, so the self test is
            # the slot; an inactive source was stored with zero mass.
            cursor = start
            while cursor <= end:
                if cursor != slot:
                    source = walk_bodies[cursor]
                    delta = vector(source[0], source[1], source[2]) - p
                    dist2 = wp.dot(delta, delta) + softening * softening
                    inv = inverse_root(dist2)
                    a += delta * (gravitational_constant * source[3] * inv * inv * inv)
                    interactions += 1
                cursor += 1
            entry = skip
            continue
        cell = walk_cells[entry]
        r = vector(cell[0], cell[1], cell[2]) - p
        d2 = wp.dot(r, r) + softening * softening
        distance = wp.sqrt(d2)
        contains_target = slot >= start and slot <= end
        reach = cell[3]
        if (not contains_target) and reach / distance < theta:
            moments = walk_mass_diagonal[entry]
            mass = moments[0]
            inv = inverse_root(d2)
            a += r * (gravitational_constant * mass * inv * inv * inv)
            if use_quadrupoles != 0:
                oq = walk_off_diagonal[entry]
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
            entry = skip
        else:
            entry += 1

    if overflow != 0:
        accelerations[i] = vector()
        wp.atomic_max(diag, 3, wp.int64(1))
    else:
        accelerations[i] = a
    wp.atomic_add(diag, 4, wp.int64(interactions))
