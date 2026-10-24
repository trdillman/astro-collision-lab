"""Device-only packing of a retained DeviceFrame into a CUDA/GL display buffer.

Only explicitly authored metadata and ramp knots are ever uploaded by this module.
``write`` never reads particle arrays onto the host, synchronizes, or allocates a
particle-sized host buffer. All launches use the device's current Warp stream, as
required by RegisteredGLBuffer's map/unmap contract.

The default four-stop ramp and material palette match product/display.py at
e5ebdc97. Equal/inverted bounds fit finite device min/max; this is NOT the host's
grow-only percentile AutoRange. Pass that tracker's bounds to preserve its semantics.
DISTANCE measures the explicit reference_point, not an implicitly read-back centroid.
"""

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_RAMP = (
    (0.267, 0.005, 0.329, 1.0),
    (0.188, 0.408, 0.557, 1.0),
    (0.208, 0.718, 0.473, 1.0),
    (0.993, 0.906, 0.144, 1.0),
)
MATERIAL_PALETTE = (
    (0.902, 0.624, 0.000, 1.0),
    (0.337, 0.706, 0.914, 1.0),
    (0.000, 0.620, 0.451, 1.0),
    (0.941, 0.894, 0.259, 1.0),
    (0.000, 0.447, 0.698, 1.0),
    (0.835, 0.369, 0.000, 1.0),
    (0.800, 0.475, 0.655, 1.0),
    (0.600, 0.600, 0.600, 1.0),
    (0.451, 0.800, 0.200, 1.0),
    (0.700, 0.200, 0.200, 1.0),
)
_SOURCES = {
    "FLAT": (0, None),
    "FRAME": (1, "colors"),
    "SPEED": (2, "velocities"),
    "DENSITY": (3, "densities"),
    "DAMAGE": (3, "damage"),
    "ENERGY": (3, "internal_energies"),
    "DISTANCE": (4, None),
    "MATERIAL": (5, "material_ids"),
    "SOURCE_OBJECT": (6, "stable_ids"),
    "OBJECT_COLOR": (6, "stable_ids"),
}
_KERNELS = {}


class UnsupportedAppearance(ValueError):
    """The requested appearance cannot be served without inventing/reading data."""


@dataclass(frozen=True)
class Appearance:
    size_source: str = "UNIFORM"
    radius_multiplier: float = 1.0
    uniform_radius: float = 0.0
    color_source: str = "FLAT"
    flat_color: tuple = (0.25, 0.75, 1.0, 1.0)
    ramp_low: float = 0.0
    ramp_high: float = 0.0
    ramp_knots: tuple = DEFAULT_RAMP
    reference_point: tuple = (0.0, 0.0, 0.0)
    highlight_rgb: tuple = (1.0, 0.63, 0.15)


def _finite_tuple(values, size, label):
    result = tuple(float(v) for v in values)
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} must have {size} finite components")
    return result


def _kernels(wp, position_type, velocity_type, size_type, scalar_type):
    key = (position_type, velocity_type, size_type, scalar_type)
    if key in _KERNELS:
        return _KERNELS[key]

    @wp.func
    def scalar_value(
        positions: wp.array(dtype=position_type),
        velocities: wp.array(dtype=velocity_type),
        scalars: wp.array(dtype=scalar_type),
        mode: int,
        reference: wp.vec3f,
        i: int,
    ) -> wp.float32:
        result = wp.float32(0.0)
        if mode == 2:
            v = velocities[i]
            result = wp.float32(wp.length(v))
        elif mode == 3:
            result = wp.float32(scalars[i])
        elif mode == 4:
            p = positions[i]
            pos = wp.vec3f(wp.float32(p[0]), wp.float32(p[1]), wp.float32(p[2]))
            result = wp.length(pos - reference)
        return result

    @wp.func
    def metadata_row(ids: wp.array(dtype=wp.int64), count: int, key: wp.int64) -> int:
        low = int(0)
        high = count
        while low < high:
            middle = (low + high) // 2
            if ids[middle] < key:
                low = middle + 1
            else:
                high = middle
        result = int(-1)
        if low < count:
            if ids[low] == key:
                result = low
        return result

    @wp.kernel
    def reset_bounds(bounds: wp.array(dtype=wp.float32)):
        bounds[0] = wp.float32(3.402823e38)
        bounds[1] = wp.float32(-3.402823e38)

    @wp.kernel
    def reduce_bounds(
        positions: wp.array(dtype=position_type),
        velocities: wp.array(dtype=velocity_type),
        scalars: wp.array(dtype=scalar_type),
        mode: int,
        reference: wp.vec3f,
        bounds: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        value = scalar_value(positions, velocities, scalars, mode, reference, i)
        if wp.isfinite(value):
            wp.atomic_min(bounds, 0, value)
            wp.atomic_max(bounds, 1, value)

    @wp.kernel
    def pack(
        positions: wp.array(dtype=position_type),
        velocities: wp.array(dtype=velocity_type),
        sizes: wp.array(dtype=size_type),
        frame_colors: wp.array(dtype=wp.vec4f),
        scalars: wp.array(dtype=scalar_type),
        stable_ids: wp.array(dtype=wp.int64),
        material_ids: wp.array(dtype=wp.int64),
        have_velocity: int,
        have_ids: int,
        physical: int,
        multiplier: wp.float32,
        uniform_radius: wp.float32,
        mode: int,
        flat_color: wp.vec4f,
        low: wp.float32,
        high: wp.float32,
        bounds: wp.array(dtype=wp.float32),
        knots: wp.array(dtype=wp.vec4f),
        knot_count: int,
        palette: wp.array(dtype=wp.vec4f),
        reference: wp.vec3f,
        meta_ids: wp.array(dtype=wp.int64),
        meta_colors: wp.array(dtype=wp.vec4f),
        meta_floors: wp.array(dtype=wp.float32),
        meta_highlight: wp.array(dtype=wp.int32),
        meta_count: int,
        highlight: wp.vec3f,
        output: wp.array2d(dtype=wp.float32),
    ):
        i = wp.tid()
        p = positions[i]
        velocity = wp.vec3f(0.0)
        if have_velocity != 0:
            v = velocities[i]
            velocity = wp.vec3f(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))
        radius = uniform_radius * multiplier
        if physical != 0:
            radius = wp.float32(sizes[i]) * multiplier
        if not wp.isfinite(radius):
            radius = wp.float32(0.0)
        radius = wp.max(radius, wp.float32(0.0))
        color = flat_color
        if mode == 1:
            color = frame_colors[i]
        elif mode >= 2 and mode <= 4:
            value = scalar_value(positions, velocities, scalars, mode, reference, i)
            lower = low
            upper = high
            if lower >= upper:
                lower = bounds[0]
                upper = bounds[1]
            t = wp.float32(0.0)
            if upper > lower and wp.isfinite(value):
                t = wp.clamp((value - lower) / (upper - lower), wp.float32(0.0), wp.float32(1.0))
            scaled = t * wp.float32(knot_count - 1)
            segment = wp.min(int(wp.floor(scaled)), knot_count - 2)
            color = wp.lerp(knots[segment], knots[segment + 1], scaled - wp.float32(segment))
        elif mode == 5:
            label = material_ids[i]
            color = wp.vec4f(0.35, 0.35, 0.35, 1.0)
            if label >= wp.int64(0):
                color = palette[int(label % wp.int64(10))]
        row = int(-1)
        if have_ids != 0 and meta_count > 0:
            row = metadata_row(meta_ids, meta_count, stable_ids[i])
        if mode == 6:
            color = wp.vec4f(0.35, 0.35, 0.35, 1.0)
            if row >= 0:
                color = meta_colors[row]
        if row >= 0:
            radius = wp.max(radius, meta_floors[row])
            if meta_highlight[row] != 0:
                color = wp.vec4f(highlight[0], highlight[1], highlight[2], color[3])
        for j in range(3):
            output[i, j] = wp.float32(p[j])
            output[i, j + 8] = velocity[j]
        output[i, 3] = radius
        for j in range(4):
            output[i, j + 4] = color[j]
        output[i, 11] = wp.float32(0.0)

    result = {"reset": reset_bounds, "reduce": reduce_bounds, "pack": pack}
    _KERNELS[key] = result
    return result


class DeviceAppearanceWriter:
    """Reusable GPU writer; CPU execution requires explicit test-only opt-in."""

    def __init__(self, device, *, allow_cpu=False):
        import warp as wp

        self._wp = wp
        self.device = wp.get_device(device)
        if not self.device.is_cuda and not allow_cpu:
            raise RuntimeError("appearance packing requires a CUDA device")
        self._dummy = {}
        self._bounds = wp.empty(2, dtype=wp.float32, device=self.device)
        self._palette = wp.array(MATERIAL_PALETTE, dtype=wp.vec4f, device=self.device)
        self._metadata_key = None
        self._metadata_revision = None
        self._meta_count = 0
        self._meta_has_colors = False
        self._metadata = None
        self._knots_key = None
        self._knots = None
        self._metadata_uploads = 0
        self._appearance_uploads = 1
        self._metadata_bytes = 0
        self._writes = 0
        self._specializations = set()

    def set_metadata(
        self, *, stable_ids=(), colors=None, floor_radii=None, highlighted=None, revision=None
    ):
        """Upload retained CPU metadata only on change; sort opaque signed ID bits.

        SOURCE_OBJECT/OBJECT_COLOR both use these direct colors. The caller builds
        the appropriate source palette/object colors from its retained descriptor.
        A supplied revision is an authoritative invalidation token; unchanged tokens
        skip even host comparisons. Never pass frame-varying particle arrays here.
        """
        if revision is not None and revision == self._metadata_revision:
            return False
        ids = np.asarray(stable_ids).reshape(-1).astype(np.int64, copy=False)
        count = ids.size
        color_array = (
            np.zeros((count, 4), dtype=np.float32)
            if colors is None
            else np.asarray(colors, dtype=np.float32)
        )
        floors = (
            np.zeros(count, dtype=np.float32)
            if floor_radii is None
            else np.asarray(floor_radii, dtype=np.float32).reshape(-1)
        )
        selected = (
            np.zeros(count, dtype=np.int32)
            if highlighted is None
            else np.asarray(highlighted, dtype=np.int32).reshape(-1)
        )
        if color_array.shape != (count, 4) or floors.size != count or selected.size != count:
            raise ValueError("metadata columns must match stable_ids")
        if (
            not np.isfinite(color_array).all()
            or not np.isfinite(floors).all()
            or (floors < 0).any()
        ):
            raise ValueError("metadata colors/floors must be finite; floors must be nonnegative")
        order = np.argsort(ids, kind="stable")
        ids = np.ascontiguousarray(ids[order])
        if count > 1 and (ids[1:] == ids[:-1]).any():
            raise ValueError("metadata stable_ids must be unique")
        arrays = (
            ids,
            np.ascontiguousarray(color_array[order]),
            np.ascontiguousarray(floors[order]),
            np.ascontiguousarray(selected[order]),
        )
        key = (colors is not None, *(a.tobytes() for a in arrays))
        if key == self._metadata_key:
            self._metadata_revision = revision
            return False
        wp = self._wp
        self._metadata = tuple(
            wp.array(a, dtype=dtype, device=self.device) if count else self._empty(dtype)
            for a, dtype in zip(arrays, (wp.int64, wp.vec4f, wp.float32, wp.int32), strict=True)
        )
        self._metadata_key = key
        self._metadata_revision = revision
        self._meta_count = count
        self._meta_has_colors = colors is not None
        self._metadata_bytes = sum(a.nbytes for a in arrays)
        self._metadata_uploads += 1
        return True

    def _empty(self, dtype):
        if dtype not in self._dummy:
            self._dummy[dtype] = self._wp.empty(1, dtype=dtype, device=self.device)
        return self._dummy[dtype]

    def validate(self, frame, appearance, *, count=None):
        """Reject unsupported modes/channels/types before the caller maps GL storage."""
        wp = self._wp
        channels = frame.channels
        positions = channels.get("positions")
        if positions is None:
            raise UnsupportedAppearance("positions channel is required")
        raw_count = getattr(frame, "count", positions.shape[0]) if count is None else count
        if isinstance(raw_count, bool) or int(raw_count) != raw_count:
            raise ValueError("count must be an integer")
        count = int(raw_count)
        if not 0 <= count <= positions.shape[0]:
            raise ValueError("count exceeds positions")
        source = appearance.color_source
        if source not in _SOURCES:
            raise UnsupportedAppearance("unsupported color source " + source)
        _, field = _SOURCES[source]
        if field is not None and field not in channels:
            raise UnsupportedAppearance("requested color requires " + field)
        if appearance.size_source not in ("UNIFORM", "PHYSICAL"):
            raise UnsupportedAppearance("unsupported size source " + appearance.size_source)
        if appearance.size_source == "PHYSICAL" and "sizes" not in channels:
            raise UnsupportedAppearance("physical size requires sizes")
        if source in ("SOURCE_OBJECT", "OBJECT_COLOR") and not self._meta_has_colors:
            raise UnsupportedAppearance("source color requires authored color metadata")
        if self._meta_count and "stable_ids" not in channels:
            raise UnsupportedAppearance("authored metadata requires stable_ids")
        _finite_tuple(appearance.flat_color, 4, "flat_color")
        _finite_tuple(appearance.reference_point, 3, "reference_point")
        _finite_tuple(appearance.highlight_rgb, 3, "highlight_rgb")
        if not 2 <= len(appearance.ramp_knots) <= 5:
            raise ValueError("ramp requires 2 to 5 knots")
        for knot in appearance.ramp_knots:
            _finite_tuple(knot, 4, "ramp knot")
        for field in ("radius_multiplier", "uniform_radius", "ramp_low", "ramp_high"):
            if not math.isfinite(float(getattr(appearance, field))):
                raise ValueError(field + " must be finite")
        if appearance.radius_multiplier < 0 or appearance.uniform_radius < 0:
            raise ValueError("radii and multiplier must be nonnegative")
        used = {"positions"}
        if field := _SOURCES[source][1]:
            used.add(field)
        used.update(name for name in ("velocities", "stable_ids") if name in channels)
        if appearance.size_source == "PHYSICAL":
            used.add("sizes")
        expected = {
            "positions": (wp.vec3f, wp.vec3d),
            "velocities": (wp.vec3f, wp.vec3d),
            "colors": (wp.vec4f,),
            "stable_ids": (wp.int64,),
            "material_ids": (wp.int64,),
        }
        for name in used:
            array = channels[name]
            if array.ndim != 1 or array.shape[0] < count or array.device != self.device:
                raise UnsupportedAppearance(name + " must be a same-device array with count rows")
            if array.dtype not in expected.get(name, (wp.float32, wp.float64)):
                raise UnsupportedAppearance("unsupported dtype for " + name)
        return count

    def write(self, frame, mapped, appearance, *, count=None):
        count = self.validate(frame, appearance, count=count)
        wp = self._wp
        if (
            mapped.ndim != 2
            or mapped.shape[1] != 12
            or mapped.shape[0] < count
            or mapped.dtype != wp.float32
            or mapped.device != self.device
        ):
            raise ValueError(
                "mapped output must be same-device float32 (capacity,12) with count<=capacity"
            )
        if not count:
            return
        ch = frame.channels
        positions = ch["positions"]
        velocities = ch.get("velocities", self._empty(positions.dtype))
        sizes = ch["sizes"] if appearance.size_source == "PHYSICAL" else self._empty(wp.float32)
        mode, field = _SOURCES[appearance.color_source]
        scalars = ch[field] if mode == 3 else self._empty(wp.float32)
        kernels = _kernels(wp, positions.dtype, velocities.dtype, sizes.dtype, scalars.dtype)
        self._specializations.add((positions.dtype, velocities.dtype, sizes.dtype, scalars.dtype))
        knots_key = tuple(_finite_tuple(k, 4, "ramp knot") for k in appearance.ramp_knots)
        if knots_key != self._knots_key:
            self._knots = wp.array(knots_key, dtype=wp.vec4f, device=self.device)
            self._knots_key = knots_key
            self._appearance_uploads += 1
        if self._metadata is None:
            self._metadata = tuple(
                self._empty(t) for t in (wp.int64, wp.vec4f, wp.float32, wp.int32)
            )
        reference = wp.vec3f(*appearance.reference_point)
        if mode in (2, 3, 4) and appearance.ramp_low >= appearance.ramp_high:
            wp.launch(kernels["reset"], dim=1, inputs=[self._bounds], device=self.device)
            wp.launch(
                kernels["reduce"],
                dim=count,
                inputs=[positions, velocities, scalars, mode, reference, self._bounds],
                device=self.device,
            )
        inputs = [
            positions,
            velocities,
            sizes,
            ch["colors"] if mode == 1 else self._empty(wp.vec4f),
            scalars,
            ch.get("stable_ids", self._empty(wp.int64)),
            ch["material_ids"] if mode == 5 else self._empty(wp.int64),
            int("velocities" in ch),
            int("stable_ids" in ch),
            int(appearance.size_source == "PHYSICAL"),
            float(appearance.radius_multiplier),
            float(appearance.uniform_radius),
            mode,
            wp.vec4f(*appearance.flat_color),
            float(appearance.ramp_low),
            float(appearance.ramp_high),
            self._bounds,
            self._knots,
            len(knots_key),
            self._palette,
            reference,
            *self._metadata,
            self._meta_count,
            wp.vec3f(*appearance.highlight_rgb),
            mapped,
        ]
        wp.launch(kernels["pack"], dim=count, inputs=inputs, device=self.device)
        self._writes += 1

    def report(self):
        scratch = (
            8
            + 160
            + sum(a.size * self._wp.types.type_size_in_bytes(a.dtype) for a in self._dummy.values())
        )
        return {
            "device": str(self.device),
            "writes": self._writes,
            "host_particle_readbacks": 0,
            "metadata_uploads": self._metadata_uploads,
            "metadata_rows": self._meta_count,
            "metadata_device_bytes": self._metadata_bytes,
            "appearance_uploads": self._appearance_uploads,
            "scratch_device_bytes": scratch,
            "ramp_device_bytes": 0 if self._knots_key is None else len(self._knots_key) * 16,
            "dtype_specializations": len(self._specializations),
            "auto_range": "finite-device-minmax",
        }
