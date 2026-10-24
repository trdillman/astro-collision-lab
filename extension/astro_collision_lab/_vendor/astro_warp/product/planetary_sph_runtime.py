"""Standalone unguided compressible runtime bridge.

Numerical stages come from the project at 4b1f2b4100abd39f16ec42ec95eceb7da82f8495.
This bridge changes no old graph identities: the independent Lab drives arrays directly.
"""
from .sph_runtime import SphProductRuntime

class PlanetarySphRuntime(SphProductRuntime):
    BACKEND = "planetary_sph"

    def display_channels(self):
        channels = super().display_channels()
        if not hasattr(self, "_display_identity"):
            wp = self.device_context.wp
            dev = self.device_context.device
            bodies = self._require_bodies()
            self._display_identity = {
                "stable_ids": wp.array(bodies.stable_ids, dtype=wp.int64, device=dev),
                "active": wp.array(bodies.active, dtype=wp.int32, device=dev),
            }
        return {**channels, **self._display_identity}

    def close(self):
        self._display_identity = {}
        super().close()
