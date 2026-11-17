"""Stable Morton radix sorting for the add-on's ordered galaxy particle IDs."""
from dataclasses import replace
from itertools import pairwise

from ._vendor.astro_warp.product.tree import BarnesHutRuntime


class RadixTreeRuntime(BarnesHutRuntime):
    def rebuild_tree(self):
        # load_bodies has installed fresh device arrays here, before graph capture.
        arrays = self._arrays
        wp = self._require_wp()
        device = self.device_context.device
        if "radix_keys" not in arrays and all(
            a < b for a, b in pairwise(self._stable_id_order)
        ):
            keys = wp.zeros(2 * self._sort_size, dtype=wp.int64, device=device)
            indices = wp.zeros(2 * self._sort_size, dtype=wp.int32, device=device)
            # Prime Warp's device-sort workspace outside CUDA graph capture.
            wp.utils.radix_sort_pairs(keys, indices, self._sort_size)
            arrays.update(radix_keys=keys, radix_indices=indices)
        super().rebuild_tree()

    def _record_bounds_sort(self):
        arrays = self._arrays
        if "radix_keys" not in arrays:
            return super()._record_bounds_sort()
        wp = self._require_wp()
        kernels = self._require_kernels()
        device = self.device_context.device
        count = self.config.count
        bounds = arrays["bounds"]
        wp.launch(kernels.clear_bounds, dim=6, inputs=[bounds], device=device)
        wp.launch(kernels.reduce_bounds, dim=count,
                  inputs=[arrays["positions"], arrays["active"], count, bounds], device=device)
        wp.launch(kernels.finalize_bounds, dim=1, inputs=[bounds], device=device)
        keys, indices = arrays["radix_keys"], arrays["radix_indices"]
        wp.launch(kernels.compute_morton, dim=self._sort_size,
                  inputs=[arrays["positions"], arrays["active"], count, self._sort_size,
                          self.config.max_depth, bounds, arrays["body_keys"], keys, indices],
                  device=device)
        wp.utils.radix_sort_pairs(keys, indices, self._sort_size)
        wp.copy(arrays["sorted_keys"], keys, count=self._sort_size)
        wp.copy(arrays["sorted_indices"], indices, count=self._sort_size)

    def diagnostics(self):
        report = super().diagnostics()
        if "radix_keys" in self._arrays:
            return replace(report, sort_algorithm="stable_morton_radix")
        return report


class GalaxyTreeRuntime(RadixTreeRuntime):
    def rebuild_tree(self):
        wp = self._require_wp()
        if self.config.precision.value == "float32" and wp.get_device(self.device_context.device).is_cuda:
            from .galaxy_walk import tree_acceleration_walk
            if self._kernels.tree_acceleration_walk is not tree_acceleration_walk:
                self._kernels = replace(self._kernels, tree_acceleration_walk=tree_acceleration_walk)
                self.walk_block_dim = 512
        super().rebuild_tree()
