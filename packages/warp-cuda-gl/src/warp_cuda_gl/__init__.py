"""Zero-copy particle display: NVIDIA Warp writes straight into an OpenGL vertex buffer.

``cuda_gl`` owns a CUDA-registered OpenGL buffer and draws it as instanced billboards.
``device_appearance`` fills that buffer on the GPU (position, radius, colour) from Warp
arrays. No particle data is copied back to the CPU. Both files are kept byte-identical to
the Astro Collision Lab add-on (see ``tools/sync_packages.py``).

Importing this package is safe anywhere; creating a ``CudaGLBatch`` needs Windows, a CUDA
GPU and a current OpenGL 4.5 context (or 4.0+ with
GL_ARB_clip_control) on the calling thread.
"""
from . import cuda_gl, device_appearance
from .cuda_gl import CudaGLBatch, current_context_key
from .device_appearance import (
                                DEFAULT_RAMP,
                                MATERIAL_PALETTE,
                                Appearance,
                                DeviceAppearanceWriter,
                                UnsupportedAppearance,
)

#: Floats per point in the shared buffer: x, y, z, radius, r, g, b, a, and 4 extra slots.
STRIDE = 12

__version__ = "0.1.0"
__all__ = ["DEFAULT_RAMP", "MATERIAL_PALETTE", "STRIDE", "Appearance", "CudaGLBatch",
           "DeviceAppearanceWriter", "UnsupportedAppearance", "cuda_gl", "current_context_key",
           "device_appearance"]
