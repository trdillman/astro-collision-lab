"""Bounded cache of full-count, object-local authoring preview geometry.

This cache is for CPU initial-condition previews only. A caller applies the
current rigid transform when drawing; neither evolved particles nor solver
qualification is stored here.
"""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json

import numpy as np

from . import galaxy_models
from .simulation import IDENTITY


class PreviewCache:
    def __init__(self, limit=32):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('preview cache limit must be a positive integer')
        self.limit = limit
        self._entries = OrderedDict()

    def sample(self, body, config):
        """Return ``(geometry_key, local_points)``; cached points are read-only.

        Names, membership, motion, equilibrium support, and appearance do not
        affect the fixed preview density geometry. All authored particles are
        retained; there is no resolution reduction on this path.
        """
        count = body.particle_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('preview particle count must be a positive integer')
        geometry = (config.mode.upper(), config.seed, body.kind, count,
                    body.radius, body.scale_length, body.thickness, body.seed)
        try:
            encoded = json.dumps(geometry, allow_nan=False, separators=(',', ':'))
        except (ValueError, TypeError) as exc:
            raise ValueError('preview geometry parameters must be finite and serializable') from exc
        key = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        if key in self._entries:
            self._entries.move_to_end(key)
            return key, self._entries[key]

        # Density geometry is mass-independent. Unit mass also avoids needless
        # roundoff differences in the normalized cumulative density calculation.
        local = replace(body, position=(0., 0., 0.), rotation=IDENTITY, mass=1.)
        points = np.ascontiguousarray(
            galaxy_models.preview((local,), config, limit=None)[0], dtype=float)
        if points.shape != (count, 3) or not np.isfinite(points).all():
            raise ValueError('preview sampler must return the full finite particle count')
        points.setflags(write=False)
        self._entries[key] = points
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return key, points

    def clear(self):
        self._entries.clear()
