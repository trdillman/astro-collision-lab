"""Strict owned-buffer CUDA/OpenGL presentation for Windows Blender.

No particle data enters Python. Call construction/update/draw/close only with the
owning viewport OpenGL context current. Importing this module creates no context.
The CUDA driver owns the GL registration; Warp only aliases a checked mapped
pointer. This avoids Warp 1.15's map/unmap wrappers, which swallow driver errors.
"""

import ctypes as C
import math
import os
import threading
from contextlib import contextmanager


def current_context_key():
    """Current WGL context handle, or zero on unsupported/no-context surfaces."""
    if os.name != "nt":
        return 0
    library = C.WinDLL("opengl32.dll")
    get = library.wglGetCurrentContext
    get.argtypes, get.restype = [], C.c_void_p
    return int(get() or 0)


_VERTEX = r"""#version 330 core
layout(location=0) in vec4 particle;
layout(location=1) in vec4 color;
layout(location=2) in vec4 velocity;
uniform mat4 u_vp;
uniform mat4 u_view;
uniform vec2 u_viewport;
uniform float u_min_diameter;
uniform float u_stretch;
uniform int u_alignment;
uniform vec3 u_fixed_normal;
uniform vec3 u_lock_axis;
out vec2 v_uv;
out vec4 v_color;
out vec3 v_center;
out vec3 v_right;
out vec3 v_up;
out vec3 v_normal;
out float v_radius;
vec3 safe_normalize(vec3 v, vec3 fallback) {
    float d = dot(v,v); return d > 1e-12 ? v*inversesqrt(d) : fallback;
}
void main() {
    vec2 corners[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
    vec2 uv = corners[gl_VertexID];
    mat3 inv_view = transpose(mat3(u_view));
    vec3 right = inv_view[0], up = inv_view[1], normal = inv_view[2];
    if (u_alignment == 1) {
        vec3 along = velocity.xyz - normal*dot(velocity.xyz,normal);
        right = safe_normalize(along, right);
        up = safe_normalize(cross(normal,right),up);
    } else if (u_alignment == 2) {
        normal = safe_normalize(u_fixed_normal,normal);
        vec3 helper = abs(normal.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
        right = safe_normalize(cross(helper,normal),right);
        up = cross(normal,right);
    } else if (u_alignment == 3) {
        up = safe_normalize(u_lock_axis,up);
        right = safe_normalize(cross(up,normal),right);
        normal = safe_normalize(cross(right,up),normal);
    }
    vec4 center_clip = u_vp * vec4(particle.xyz,1);
    float radius = max(particle.w,0.0);
    // Minimum diameter has screen-space meaning, without GL point-size limits.
    vec4 edge_clip = u_vp * vec4(particle.xyz + right*radius,1);
    float px_radius = length((edge_clip.xy/max(abs(edge_clip.w),1e-8)
                           - center_clip.xy/max(abs(center_clip.w),1e-8))*u_viewport*0.5);
    float scale = max(1.0, (u_min_diameter*0.5)/max(px_radius,1e-8));
    vec4 clip;
    if (radius > 0.0) {
        radius *= scale;
        clip = u_vp * vec4(particle.xyz + radius*(right*uv.x*u_stretch + up*uv.y),1);
    } else {
        clip = center_clip;
        clip.xy += uv * vec2(u_stretch,1) * u_min_diameter/u_viewport * clip.w;
    }
    gl_Position = clip;
    v_uv=uv; v_color=color; v_center=particle.xyz; v_right=right; v_up=up;
    v_normal=normal; v_radius=radius;
}
"""

_FRAGMENT = r"""#version 330 core
uniform mat4 u_vp;
uniform int u_shaded;
uniform float u_stretch;
in vec2 v_uv;
in vec4 v_color;
in vec3 v_center;
in vec3 v_right;
in vec3 v_up;
in vec3 v_normal;
in float v_radius;
out vec4 frag_color;
void main() {
    float rr=dot(v_uv,v_uv);
    if (rr>1.0) discard;
    gl_FragDepth=gl_FragCoord.z;
    float light=1.0;
    if (u_shaded != 0) {
        float nz=sqrt(max(0.0,1.0-rr));
        vec3 n=normalize(vec3(v_uv,nz));
        light=0.22 + 0.78*max(dot(n,normalize(vec3(-0.4,0.6,1.0))),0.0);
        if (v_radius>0.0) {
            vec3 world=v_center+v_radius*(v_right*v_uv.x*u_stretch+v_up*v_uv.y+v_normal*nz);
            vec4 clip=u_vp*vec4(world,1);
            float window_depth=0.5*(clip.z/clip.w)+0.5;
            gl_FragDepth=mix(gl_DepthRange.near,gl_DepthRange.far,window_depth);
        }
    }
    frag_color=vec4(v_color.rgb*light,v_color.a);
}
"""


class _CUDA:
    """Checked CUDA driver interop. All calls require the caller's CUDA guard.

    A resource here is CUgraphicsResource itself, not the pointer-to-resource
    allocation used internally by Warp's RegisteredGLBuffer implementation.
    """

    def __init__(self):
        self.library = C.WinDLL("nvcuda.dll")
        resource_pointer = C.POINTER(C.c_void_p)
        signatures = {
            "cuGraphicsGLRegisterBuffer": (resource_pointer, C.c_uint, C.c_uint),
            "cuGraphicsMapResources": (C.c_uint, resource_pointer, C.c_void_p),
            "cuGraphicsResourceGetMappedPointer_v2": (
                C.POINTER(C.c_uint64), C.POINTER(C.c_size_t), C.c_void_p
            ),
            "cuGraphicsUnmapResources": (C.c_uint, resource_pointer, C.c_void_p),
            "cuGraphicsUnregisterResource": (C.c_void_p,),
            "cuCtxSynchronize": (),
        }
        self.functions = {}
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = list(arguments), C.c_int
            self.functions[name] = function

    def _checked(self, name, *arguments):
        status = self.functions[name](*arguments)
        if status != 0:
            raise RuntimeError("CUDA/GL %s failed (CUresult %d)" % (name, status))

    def register_buffer(self, buffer):
        resource = C.c_void_p()
        # Flags 0 preserves buffer contents; no copying fallback exists.
        self._checked("cuGraphicsGLRegisterBuffer", C.byref(resource), buffer, 0)
        if not resource.value:
            raise RuntimeError("CUDA/GL registration returned a null resource")
        return resource.value

    def map(self, resource, stream):
        value = C.c_void_p(resource)
        self._checked("cuGraphicsMapResources", 1, C.byref(value), C.c_void_p(stream))

    def pointer_and_size(self, resource):
        pointer, size = C.c_uint64(), C.c_size_t()
        self._checked(
            "cuGraphicsResourceGetMappedPointer_v2",
            C.byref(pointer), C.byref(size), C.c_void_p(resource),
        )
        return pointer.value, size.value

    def unmap(self, resource, stream):
        value = C.c_void_p(resource)
        self._checked("cuGraphicsUnmapResources", 1, C.byref(value), C.c_void_p(stream))

    def unregister(self, resource):
        self._checked("cuGraphicsUnregisterResource", C.c_void_p(resource))

    def synchronize(self):
        self._checked("cuCtxSynchronize")


class _GL:
    """Narrow ctypes loader. All names belong to the caller's current context."""

    def __init__(self):
        if os.name != "nt" or not current_context_key():
            raise RuntimeError(
                "owned CUDA/GL presentation requires a current Windows OpenGL context"
            )
        self.lib = C.WinDLL("opengl32.dll")
        self.get_proc = self.lib.wglGetProcAddress
        self.get_proc.argtypes, self.get_proc.restype = [C.c_char_p], C.c_void_p
        self.functions = {}

    def current_context(self):
        return current_context_key()

    def _fn(self, name, result, *args):
        if name not in self.functions:
            addr = self.get_proc(name.encode("ascii"))
            if not addr or addr in (1, 2, 3, C.c_void_p(-1).value):
                try:
                    fn = getattr(self.lib, name)
                    fn.argtypes, fn.restype = list(args), result
                except AttributeError as error:
                    raise RuntimeError("OpenGL function unavailable: " + name) from error
            else:
                fn = C.WINFUNCTYPE(result, *args)(addr)
            self.functions[name] = fn
        return self.functions[name]

    def _call(self, name, result, args, *values):
        return self._fn(name, result, *args)(*values)

    def cuda_devices(self):
        cuda = C.WinDLL("nvcuda.dll")
        fn = cuda.cuGLGetDevices
        fn.argtypes = [C.POINTER(C.c_uint), C.POINTER(C.c_int), C.c_uint, C.c_uint]
        fn.restype = C.c_int
        count, devices = C.c_uint(), (C.c_int * 32)()
        status = fn(C.byref(count), devices, 32, 1)  # CU_GL_DEVICE_LIST_ALL
        if status:
            raise RuntimeError("CUDA/GL device matching failed (cuGLGetDevices=%d)" % status)
        return list(devices[: count.value])

    def _integer(self, enum):
        value = C.c_int()
        self._call("glGetIntegerv", None, (C.c_uint, C.POINTER(C.c_int)), enum, C.byref(value))
        return value.value

    def _enabled(self, enum):
        return bool(self._call("glIsEnabled", C.c_ubyte, (C.c_uint,), enum))

    def _enable(self, enum, enabled):
        self._call("glEnable" if enabled else "glDisable", None, (C.c_uint,), enum)

    def _integer_indexed(self, enum, index):
        value = C.c_int()
        self._call(
            "glGetIntegeri_v", None, (C.c_uint, C.c_uint, C.POINTER(C.c_int)),
            enum, index, C.byref(value),
        )
        return value.value

    def _enabled_indexed(self, enum, index):
        return bool(self._call("glIsEnabledi", C.c_ubyte, (C.c_uint, C.c_uint), enum, index))

    def _enable_indexed(self, enum, index, enabled):
        self._call(
            "glEnablei" if enabled else "glDisablei", None, (C.c_uint, C.c_uint), enum, index
        )

    @contextmanager
    def _state(self):
        # Only query/restore states this renderer mutates; VAO owns attrib state.
        keys = (
            0x8B8D,
            0x85B5,
            0x8894,
            0x0B74,
            0x0B72,
            0x935C,
            0x935D,
        )
        state = {k: self._integer(k) for k in keys}
        enabled = {k: self._enabled(k) for k in (0x0B71, 0x0B44)}
        # This shader writes color attachment 0 only. Non-indexed blend setters
        # mutate every draw buffer; preserving only index 0 would corrupt MRT state.
        blend_keys = (0x80C9, 0x80C8, 0x80CB, 0x80CA, 0x8009, 0x883D)
        blend = {k: self._integer_indexed(k, 0) for k in blend_keys}
        blend_enabled = self._enabled_indexed(0x0BE2, 0)
        try:
            yield
        finally:
            self._call("glUseProgram", None, (C.c_uint,), state[0x8B8D])
            self._call("glBindVertexArray", None, (C.c_uint,), state[0x85B5])
            self._call("glBindBuffer", None, (C.c_uint, C.c_uint), 0x8892, state[0x8894])
            self._call("glDepthFunc", None, (C.c_uint,), state[0x0B74])
            self._call("glDepthMask", None, (C.c_ubyte,), state[0x0B72])
            self._call(
                "glBlendFuncSeparatei",
                None,
                (C.c_uint,) * 5,
                0,
                blend[0x80C9],
                blend[0x80C8],
                blend[0x80CB],
                blend[0x80CA],
            )
            self._call(
                "glBlendEquationSeparatei", None, (C.c_uint,) * 3,
                0, blend[0x8009], blend[0x883D],
            )
            self._enable_indexed(0x0BE2, 0, blend_enabled)
            self._call("glClipControl", None, (C.c_uint, C.c_uint), state[0x935C], state[0x935D])
            for enum, value in enabled.items():
                self._enable(enum, value)

    def _shader(self, kind, source):
        shader = self._call("glCreateShader", C.c_uint, (C.c_uint,), kind)
        try:
            data = C.c_char_p(source.encode("utf-8"))
            self._call(
                "glShaderSource",
                None,
                (C.c_uint, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_int)),
                shader,
                1,
                C.byref(data),
                None,
            )
            self._call("glCompileShader", None, (C.c_uint,), shader)
            status = C.c_int()
            self._call(
                "glGetShaderiv",
                None,
                (C.c_uint, C.c_uint, C.POINTER(C.c_int)),
                shader,
                0x8B81,
                C.byref(status),
            )
            if not status.value:
                log = C.create_string_buffer(8192)
                self._call(
                    "glGetShaderInfoLog",
                    None,
                    (C.c_uint, C.c_int, C.POINTER(C.c_int), C.c_void_p),
                    shader,
                    8192,
                    None,
                    log,
                )
                raise RuntimeError(
                    "owned GL shader compile failed: " + log.value.decode("utf-8", "replace")
                )
            return shader
        except BaseException:
            self._call("glDeleteShader", None, (C.c_uint,), shader)
            raise

    def create(self, capacity):
        vao, vbo, program = 0, 0, 0
        shaders = []
        try:
            with self._state():
                shaders = [self._shader(0x8B31, _VERTEX)]
                shaders.append(self._shader(0x8B30, _FRAGMENT))
                program = self._call("glCreateProgram", C.c_uint, ())
                for shader in shaders:
                    self._call("glAttachShader", None, (C.c_uint, C.c_uint), program, shader)
                self._call("glLinkProgram", None, (C.c_uint,), program)
                status = C.c_int()
                self._call(
                    "glGetProgramiv",
                    None,
                    (C.c_uint, C.c_uint, C.POINTER(C.c_int)),
                    program,
                    0x8B82,
                    C.byref(status),
                )
                if not status.value:
                    log = C.create_string_buffer(8192)
                    self._call(
                        "glGetProgramInfoLog",
                        None,
                        (C.c_uint, C.c_int, C.POINTER(C.c_int), C.c_void_p),
                        program,
                        8192,
                        None,
                        log,
                    )
                    raise RuntimeError(
                        "owned GL program link failed: " + log.value.decode("utf-8", "replace")
                    )
                name = C.c_uint()
                self._call(
                    "glGenVertexArrays", None, (C.c_int, C.POINTER(C.c_uint)), 1, C.byref(name)
                )
                vao = name.value
                self._call("glGenBuffers", None, (C.c_int, C.POINTER(C.c_uint)), 1, C.byref(name))
                vbo = name.value
                if not vao or not vbo:
                    raise RuntimeError("OpenGL buffer allocation failed")
                self._call("glBindVertexArray", None, (C.c_uint,), vao)
                self._call("glBindBuffer", None, (C.c_uint, C.c_uint), 0x8892, vbo)
                self._call(
                    "glBufferData",
                    None,
                    (C.c_uint, C.c_ssize_t, C.c_void_p, C.c_uint),
                    0x8892,
                    capacity * 48,
                    None,
                    0x88E8,
                )
                for attribute in range(3):
                    self._call("glEnableVertexAttribArray", None, (C.c_uint,), attribute)
                    self._call(
                        "glVertexAttribPointer",
                        None,
                        (C.c_uint, C.c_int, C.c_uint, C.c_ubyte, C.c_int, C.c_void_p),
                        attribute,
                        4,
                        0x1406,
                        0,
                        48,
                        C.c_void_p(attribute * 16),
                    )
                    self._call("glVertexAttribDivisor", None, (C.c_uint, C.c_uint), attribute, 1)
            return vao, vbo, program
        except BaseException:
            self.delete(vao, vbo, program)
            raise
        finally:
            for shader in shaders:
                self._call("glDeleteShader", None, (C.c_uint,), shader)

    def delete(self, vao, vbo, program):
        for name, function in ((vao, "glDeleteVertexArrays"), (vbo, "glDeleteBuffers")):
            if name:
                value = C.c_uint(name)
                self._call(function, None, (C.c_int, C.POINTER(C.c_uint)), 1, C.byref(value))
        if program:
            self._call("glDeleteProgram", None, (C.c_uint,), program)

    def draw(self, vao, program, count, **options):
        with self._state():
            # Blender 5.1 can use ZERO_TO_ONE and inject z remapping into its
            # shaders. Our shader receives the Python/OpenGL [-w,w] matrix.
            # Isolate this convention and restore Blender's actual clip state.
            self._call("glClipControl", None, (C.c_uint, C.c_uint), 0x8CA1, 0x935E)
            self._call("glUseProgram", None, (C.c_uint,), program)
            self._call("glBindVertexArray", None, (C.c_uint,), vao)
            self._enable(0x0B44, False)
            self._enable(0x0B71, True)
            self._call("glDepthFunc", None, (C.c_uint,), 0x0203)
            blend = options["blend_mode"]
            self._enable_indexed(0x0BE2, 0, blend != "OPAQUE")
            self._call("glDepthMask", None, (C.c_ubyte,), blend == "OPAQUE")
            self._call("glBlendEquationSeparatei", None, (C.c_uint,) * 3, 0, 0x8006, 0x8006)
            dst = 1 if blend == "ADDITIVE" else 0x0303
            self._call("glBlendFuncSeparatei", None, (C.c_uint,) * 5, 0, 0x0302, dst, 1, dst)

            def location(name):
                return self._call(
                    "glGetUniformLocation",
                    C.c_int,
                    (C.c_uint, C.c_char_p),
                    program,
                    name.encode("ascii"),
                )

            for name, values in (
                ("u_vp", options["view_projection"]),
                ("u_view", options["view_matrix"]),
            ):
                values = (C.c_float * 16)(*values)
                self._call(
                    "glUniformMatrix4fv",
                    None,
                    (C.c_int, C.c_int, C.c_ubyte, C.POINTER(C.c_float)),
                    location(name),
                    1,
                    0,
                    values,
                )
            for name, value in (
                ("u_min_diameter", options["min_diameter_px"]),
                ("u_stretch", options["stretch"]),
            ):
                self._call("glUniform1f", None, (C.c_int, C.c_float), location(name), value)
            for name, value in (
                ("u_shaded", int(options["shaded"])),
                ("u_alignment", options["alignment"]),
            ):
                self._call("glUniform1i", None, (C.c_int, C.c_int), location(name), value)
            self._call(
                "glUniform2f",
                None,
                (C.c_int, C.c_float, C.c_float),
                location("u_viewport"),
                *options["viewport_size"],
            )
            for name, value in (
                ("u_fixed_normal", options["fixed_normal"]),
                ("u_lock_axis", options["lock_axis"]),
            ):
                self._call(
                    "glUniform3f",
                    None,
                    (C.c_int, C.c_float, C.c_float, C.c_float),
                    location(name),
                    *value,
                )
            self._call(
                "glDrawArraysInstanced",
                None,
                (C.c_uint, C.c_int, C.c_int, C.c_int),
                0x0005,
                0,
                4,
                count,
            )


def _floats(values, length, name):
    values = tuple(float(v) for v in values)
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError(name + " must contain %d finite values" % length)
    return values


class CudaGLBatch:
    """One context-owned, strict CUDA/GL instanced billboard batch."""

    def __init__(self, device, capacity):
        import warp as wp

        if isinstance(capacity, bool) or int(capacity) != capacity or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._wp = wp
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError("CUDA device required")
        self._gl = _GL()
        self.owning_context = self._gl.current_context()
        if not self.owning_context:
            raise RuntimeError("current OpenGL context required")
        self._thread = threading.get_ident()
        if self.device.ordinal not in self._gl.cuda_devices():
            raise RuntimeError("CUDA device does not match the current OpenGL context")
        self._cuda = _CUDA()
        self.vao, self.vbo, self.program = self._gl.create(self.capacity)
        self._resource = None
        self._closed = False
        self._mapped = False
        self._mapping_uncertain = False
        self._fault = None
        self._updates, self._draws = 0, 0
        try:
            with self.device.context_guard:
                self._resource = self._cuda.register_buffer(self.vbo)
        except BaseException:
            self._gl.delete(self.vao, self.vbo, self.program)
            self.vao = self.vbo = self.program = 0
            self._closed = True
            raise

    def _check(self, allow_faulted=False):
        if self._closed:
            raise RuntimeError("CUDA/GL batch is closed")
        if (
            threading.get_ident() != self._thread
            or self._gl.current_context() != self.owning_context
        ):
            raise RuntimeError("owning OpenGL context must be current on its original thread")
        if self._fault is not None and not allow_faulted:
            raise RuntimeError("CUDA/GL batch is faulted: " + self._fault)

    def update(self, writer):
        """Pack on the current Warp stream; writer receives (capacity,12) float32.

        The writer must submit to that stream (or join any other writer stream
        before returning), and must not retain the array after this call. Only
        a successful unmap transfers ownership back to OpenGL.
        """
        self._check()
        if self._mapped:
            raise RuntimeError("CUDA/GL batch is already mapped")
        with self.device.context_guard:
            stream = self._wp.get_stream(self.device)
            stream_handle = stream.cuda_stream
            with self._wp.ScopedStream(stream):
                try:
                    self._cuda.map(self._resource, stream_handle)
                except BaseException as error:
                    # Driver errors can surface earlier asynchronous failures.
                    # Ownership is unproven: do not draw, unmap speculatively,
                    # unregister or delete the GL allocation after a failed map.
                    self._mapping_uncertain = True
                    self._fault = str(error)
                    raise
                self._mapped = True
                original_error = None
                try:
                    pointer, size = self._cuda.pointer_and_size(self._resource)
                    if not pointer or size < self.capacity * 48:
                        raise RuntimeError(
                            "CUDA/GL mapped allocation has invalid pointer or size: "
                            "pointer=%d, bytes=%d, required=%d"
                            % (pointer, size, self.capacity * 48)
                        )
                    # Warp 1.15's ptr path aliases external memory; deleter=None
                    # means the wrapper neither owns nor frees the GL allocation.
                    mapped = self._wp.array(
                        ptr=pointer, capacity=size, dtype=self._wp.float32,
                        shape=(self.capacity, 12), device=self.device,
                        copy=False, deleter=None,
                    )
                    writer(mapped)
                except BaseException as error:
                    original_error = error
                    self._fault = str(error)
                    raise
                finally:
                    try:
                        self._cuda.unmap(self._resource, stream_handle)
                    except BaseException as cleanup_error:
                        self._fault = str(cleanup_error)
                        if original_error is not None:
                            original_error.add_note("CUDA/GL unmap failed: " + str(cleanup_error))
                        else:
                            raise
                    else:
                        self._mapped = False
        self._updates += 1

    def draw(
        self,
        count,
        *,
        view_projection,
        view_matrix,
        viewport_size,
        min_diameter_px=4.0,
        shaded=True,
        blend_mode="OPAQUE",
        alignment="VIEW",
        stretch=1.0,
        fixed_normal=(0, 0, 1),
        lock_axis=(0, 0, 1),
    ):
        self._check()
        if self._mapped:
            raise RuntimeError("cannot draw a CUDA-mapped GL buffer")
        if isinstance(count, bool) or int(count) != count or not 0 <= count <= self.capacity:
            raise ValueError("count must be between zero and capacity")
        vp = _floats(view_projection, 16, "view_projection")
        view = _floats(view_matrix, 16, "view_matrix")
        viewport = _floats(viewport_size, 2, "viewport_size")
        if min(viewport) <= 0:
            raise ValueError("viewport_size must be positive")
        diameter = float(min_diameter_px)
        stretch = float(stretch)
        if (
            not math.isfinite(diameter)
            or diameter <= 0
            or not math.isfinite(stretch)
            or stretch <= 0
        ):
            raise ValueError("diameter and stretch must be finite and positive")
        blend = str(blend_mode).upper()
        blend = {"ALPHA": "ALPHA", "ALPHA_BLEND": "ALPHA", "ADD": "ADDITIVE"}.get(blend, blend)
        if blend not in ("OPAQUE", "ALPHA", "ADDITIVE"):
            raise ValueError("unsupported blend_mode " + blend)
        align = str(alignment).upper()
        align = {
            "CAMERA": "VIEW",
            "SCREEN": "VIEW",
            "FIXED_NORMAL": "FIXED",
            "LOCK_AXIS": "AXIS",
            "AXIS_LOCK": "AXIS",
        }.get(align, align)
        alignments = {"VIEW": 0, "VELOCITY": 1, "FIXED": 2, "AXIS": 3}
        if align not in alignments:
            raise ValueError("unsupported alignment " + align)
        normal = _floats(fixed_normal, 3, "fixed_normal")
        axis = _floats(lock_axis, 3, "lock_axis")
        if count:
            self._gl.draw(
                self.vao,
                self.program,
                int(count),
                view_projection=vp,
                view_matrix=view,
                viewport_size=viewport,
                min_diameter_px=diameter,
                shaded=bool(shaded),
                blend_mode=blend,
                alignment=alignments[align],
                stretch=stretch,
                fixed_normal=normal,
                lock_axis=axis,
            )
            self._draws += 1

    def close(self):
        """Explicitly release; wrong context raises without discarding ownership."""
        if self._closed:
            return
        self._check(allow_faulted=True)
        if self._mapped or self._mapping_uncertain:
            raise RuntimeError("cannot close CUDA/GL storage with mapped or uncertain ownership")
        try:
            with self.device.context_guard:
                self._cuda.synchronize()
                if self._resource is not None:
                    self._cuda.unregister(self._resource)
                    self._resource = None
        except BaseException as error:
            self._fault = str(error)
            raise
        self._gl.delete(self.vao, self.vbo, self.program)
        self.vao = self.vbo = self.program = 0
        self._closed = True

    def report(self):
        return {
            "backend": "CUDA_OPENGL_OWNED_BUFFER",
            "strict_interop": True,
            "fallback_to_copy": False,
            "host_particle_copies": 0,
            "host_copy_counter_instrumented": False,
            "copy_accounting_scope": "owned_buffer_module_only",
            "device": str(self.device.alias),
            "capacity": self.capacity,
            "bytes": self.capacity * 48,
            "owning_context": self.owning_context,
            "vao": self.vao,
            "vbo": self.vbo,
            "program": self.program,
            "mapped": self._mapped,
            "mapping_uncertain": self._mapping_uncertain,
            "fault": self._fault,
            "updates": self._updates,
            "draws": self._draws,
            "closed": self._closed,
        }
