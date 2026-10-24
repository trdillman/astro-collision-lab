"""CPU contract tests: fake foreign APIs because these tests cannot create a GL context."""

import ctypes
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def load_module():
    path = Path(__file__).parents[1] / "extension" / "astro_collision_lab" / "cuda_gl.py"
    assert path.exists(), "strict owned GL renderer has not been implemented"
    spec = importlib.util.spec_from_file_location("cuda_gl_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGL:
    context = 41
    devices = [0]

    def __init__(self):
        self.calls = []

    def current_context(self):
        return self.context

    def cuda_devices(self):
        return self.devices

    def create(self, capacity):
        self.calls.append(("create", capacity))
        return (11, 12, 13)

    def delete(self, vao, vbo, program):
        self.calls.append(("delete", vao, vbo, program))

    def draw(self, *args, **kwargs):
        self.calls.append(("draw", args, kwargs))


class DriverRig:
    """The production CUDA loader calls a status-returning foreign-ABI double.

    Only the OS/GPU boundary is substituted; driver status checking and the batch
    ownership state machine run from the actual renderer module.
    """

    def __init__(self):
        self.module = load_module()
        self.gl = FakeGL()
        self.events = []
        self.statuses = {}
        self.native_mapped = False
        self.pointer = 0x4000
        self.byte_size = 192
        self.array_error = None
        self.arrays = []
        self.guards = 0
        self.stream_handle = 0x1234

        @contextmanager
        def guard():
            self.guards += 1
            try:
                yield
            finally:
                self.guards -= 1

        class Device:
            is_cuda, ordinal, alias = True, 0, "cuda:0"

            @property
            def context_guard(_self):
                return guard()

        self.device = Device()
        self.stream = types.SimpleNamespace(cuda_stream=self.stream_handle)

        def alias_array(**kwargs):
            if self.array_error:
                raise self.array_error
            self.arrays.append(kwargs)
            return types.SimpleNamespace(**kwargs)

        self.wp = types.SimpleNamespace(
            get_device=lambda device: self.device,
            get_stream=lambda device: self.stream,
            ScopedStream=lambda stream: guard(),
            float32="float32",
            array=alias_array,
        )
        self.module._GL = lambda: self.gl
        self.library = types.SimpleNamespace()
        names = (
            "cuGraphicsGLRegisterBuffer",
            "cuGraphicsMapResources",
            "cuGraphicsResourceGetMappedPointer_v2",
            "cuGraphicsUnmapResources",
            "cuGraphicsUnregisterResource",
            "cuCtxSynchronize",
        )
        for name in names:
            setattr(self.library, name, self.function(name))

    def function(self, name):
        def call(*args):
            self.events.append((name, args))
            assert self.guards, "CUDA calls need the owning CUDA context guard"
            statuses = self.statuses.get(name, [])
            status = statuses.pop(0) if statuses else 0
            if status:
                return status
            if name == "cuGraphicsGLRegisterBuffer":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 88
            elif name == "cuGraphicsMapResources":
                self.native_mapped = True
            elif name == "cuGraphicsResourceGetMappedPointer_v2":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_uint64))[0] = self.pointer
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_size_t))[0] = self.byte_size
            elif name == "cuGraphicsUnmapResources":
                self.native_mapped = False
            return 0

        return call

    def activate(self):
        return patch.multiple(self.module.C, WinDLL=lambda name: self.library, create=True)

    def batch(self):
        sys.modules["warp"] = self.wp
        with self.activate():
            return self.module.CudaGLBatch("cuda:0", 4)

    def names(self):
        return [name for name, args in self.events]


class NativeStatusContracts(unittest.TestCase):
    def setUp(self):
        self.previous_warp = sys.modules.get("warp")

    def tearDown(self):
        if self.previous_warp is None:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = self.previous_warp

    def test_map_status_failure_never_calls_writer_or_draws(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.statuses["cuGraphicsMapResources"] = [999]
        writes = []
        with self.assertRaisesRegex(RuntimeError, "cuGraphicsMapResources.*999"):
            batch.update(lambda array: writes.append(array))
        self.assertEqual(writes, [])
        self.assertEqual(batch.report()["updates"], 0)
        with self.assertRaisesRegex(RuntimeError, "fault|poison"):
            batch.draw(1, view_projection=[0] * 16, view_matrix=[0] * 16, viewport_size=(10, 10))
        self.assertFalse(any(c[0] == "draw" for c in rig.gl.calls))

    def test_failed_unmap_keeps_ownership_and_blocks_draw_update_and_close(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.statuses["cuGraphicsUnmapResources"] = [999]
        with self.assertRaisesRegex(RuntimeError, "cuGraphicsUnmapResources.*999"):
            batch.update(lambda array: None)
        self.assertTrue(batch.report()["mapped"])
        self.assertEqual(batch.report()["updates"], 0)
        with self.assertRaises(RuntimeError):
            batch.update(lambda array: None)
        with self.assertRaises(RuntimeError):
            batch.draw(1, view_projection=[0] * 16, view_matrix=[0] * 16, viewport_size=(10, 10))
        with self.assertRaises(RuntimeError):
            batch.close()
        self.assertNotIn("cuGraphicsUnregisterResource", rig.names())
        self.assertFalse(any(c[0] in ("delete", "draw") for c in rig.gl.calls))

    def test_mapped_pointer_query_status_failure_unmaps_without_publishing(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.statuses["cuGraphicsResourceGetMappedPointer_v2"] = [400]
        writes = []
        with self.assertRaisesRegex(RuntimeError, "cuGraphicsResourceGetMappedPointer_v2.*400"):
            batch.update(lambda array: writes.append(array))
        self.assertEqual(writes, [])
        self.assertEqual(rig.names()[-1], "cuGraphicsUnmapResources")
        self.assertFalse(batch.report()["mapped"])
        batch.close()

    def test_short_or_null_mapped_allocation_cannot_be_wrapped_or_written(self):
        for pointer, byte_size in ((0, 192), (0x4000, 191)):
            with self.subTest(pointer=pointer, byte_size=byte_size):
                rig = DriverRig()
                batch = rig.batch()
                rig.pointer, rig.byte_size = pointer, byte_size
                with self.assertRaisesRegex(
                    RuntimeError, "mapped.*(pointer|size|bytes|allocation)"
                ):
                    batch.update(lambda array: self.fail("writer must not run"))
                self.assertFalse(rig.arrays)
                self.assertFalse(batch.report()["mapped"])
                batch.close()

    def test_failed_unregister_retains_handles_and_never_deletes_gl_storage(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.statuses["cuGraphicsUnregisterResource"] = [400]
        with self.assertRaisesRegex(RuntimeError, "cuGraphicsUnregisterResource.*400"):
            batch.close()
        self.assertFalse(batch.report()["closed"])
        self.assertEqual(batch.report()["vbo"], 12)
        self.assertFalse(any(c[0] == "delete" for c in rig.gl.calls))

    def test_failed_device_synchronization_never_unregisters_or_deletes(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.statuses["cuCtxSynchronize"] = [700]
        with self.assertRaisesRegex(RuntimeError, "cuCtxSynchronize.*700"):
            batch.close()
        self.assertNotIn("cuGraphicsUnregisterResource", rig.names())
        self.assertFalse(any(c[0] == "delete" for c in rig.gl.calls))
        self.assertEqual(rig.guards, 0)

    def test_driver_registration_failure_releases_only_the_owned_gl_objects(self):
        rig = DriverRig()
        rig.statuses["cuGraphicsGLRegisterBuffer"] = [304]
        with self.assertRaisesRegex(RuntimeError, "cuGraphicsGLRegisterBuffer.*304"):
            rig.batch()
        self.assertEqual(rig.gl.calls[-1], ("delete", 11, 12, 13))
        self.assertNotIn("cuGraphicsUnregisterResource", rig.names())

    def test_success_aliases_driver_memory_and_unmaps_on_the_same_stream(self):
        rig = DriverRig()
        batch = rig.batch()
        batch.update(lambda array: None)
        self.assertEqual(rig.arrays[0]["ptr"], 0x4000)
        self.assertEqual(rig.arrays[0].get("capacity"), 192)
        self.assertEqual(rig.arrays[0]["shape"], (4, 12))
        self.assertIsNone(rig.arrays[0].get("deleter"))
        calls = [
            (name, args)
            for name, args in rig.events
            if name in ("cuGraphicsMapResources", "cuGraphicsUnmapResources")
        ]
        self.assertEqual(
            [name for name, _ in calls], ["cuGraphicsMapResources", "cuGraphicsUnmapResources"]
        )
        self.assertEqual(
            [getattr(args[-1], "value", args[-1]) for _, args in calls], [0x1234, 0x1234]
        )
        self.assertEqual(batch.report()["updates"], 1)
        batch.close()
        batch.close()
        self.assertEqual(rig.names().count("cuGraphicsUnregisterResource"), 1)

    def test_writer_error_is_preserved_and_successful_unmap_allows_safe_close(self):
        rig = DriverRig()
        batch = rig.batch()

        def fail(array):
            raise ValueError("bad writer")

        with self.assertRaisesRegex(ValueError, "bad writer"):
            batch.update(fail)
        self.assertFalse(batch.report()["mapped"])
        self.assertIn("cuGraphicsUnmapResources", rig.names())
        batch.close()

    def test_writer_error_and_failed_unmap_preserve_both_failures_and_ownership(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.statuses["cuGraphicsUnmapResources"] = [999]

        def fail(array):
            raise ValueError("writer failed first")

        with self.assertRaisesRegex(ValueError, "writer failed first") as caught:
            batch.update(fail)
        self.assertIn("cuGraphicsUnmapResources", " ".join(caught.exception.__notes__))
        self.assertTrue(batch.report()["mapped"])
        self.assertEqual(rig.guards, 0)
        with self.assertRaises(RuntimeError):
            batch.close()
        self.assertFalse(any(c[0] == "delete" for c in rig.gl.calls))

    def test_draw_preserves_independent_blend_states_of_other_attachments(self):
        module = load_module()
        gl = object.__new__(module._GL)
        state = {0x8B8D: 91, 0x85B5: 92, 0x8894: 93, 0x0B74: 0x0201,
                 0x0B72: 1, 0x935C: 0x8CA2, 0x935D: 0x935F}
        enabled = {0x0B71: False, 0x0B44: True}
        blend0 = {0x0BE2: True, 0x80C9: 1, 0x80C8: 0, 0x80CB: 1,
                  0x80CA: 0, 0x8009: 0x800A, 0x883D: 0x800B}
        blend1 = {0x0BE2: False, 0x80C9: 0x0302, 0x80C8: 0x0303, 0x80CB: 0,
                  0x80CA: 1, 0x8009: 0x8006, 0x883D: 0x8006}
        indexed = [dict(blend0), dict(blend1)]
        original = [dict(b) for b in indexed]

        def call(name, result, args, *values):
            if name == "glGetIntegerv":
                ctypes.cast(values[1], ctypes.POINTER(ctypes.c_int))[0] = (
                    indexed[0][values[0]] if values[0] in indexed[0] else state[values[0]])
            elif name == "glGetIntegeri_v":
                ctypes.cast(values[2], ctypes.POINTER(ctypes.c_int))[0] = indexed[values[1]][values[0]]
            elif name == "glIsEnabled":
                return indexed[0][values[0]] if values[0] == 0x0BE2 else enabled[values[0]]
            elif name == "glIsEnabledi":
                return indexed[values[1]][values[0]]
            elif name in ("glEnable", "glDisable"):
                if values[0] == 0x0BE2:
                    for b in indexed:
                        b[0x0BE2] = name == "glEnable"
                else:
                    enabled[values[0]] = name == "glEnable"
            elif name in ("glEnablei", "glDisablei"):
                indexed[values[1]][values[0]] = name == "glEnablei"
            elif name.startswith("glBlendFuncSeparate"):
                targets = [indexed[values[0]]] if name.endswith("i") else indexed
                vals = values[1:] if name.endswith("i") else values
                for b in targets:
                    b.update(zip((0x80C9, 0x80C8, 0x80CB, 0x80CA), vals))
            elif name.startswith("glBlendEquationSeparate"):
                targets = [indexed[values[0]]] if name.endswith("i") else indexed
                vals = values[1:] if name.endswith("i") else values
                for b in targets:
                    b.update(zip((0x8009, 0x883D), vals))
            elif name == "glGetUniformLocation":
                return 0
            elif name == "glDrawArraysInstanced":
                raise RuntimeError("draw failed")

        gl._call = call
        with self.assertRaisesRegex(RuntimeError, "draw failed"):
            gl.draw(11, 13, 4, view_projection=[0] * 16, view_matrix=[0] * 16,
                    viewport_size=(800, 600), min_diameter_px=4, stretch=1,
                    shaded=True, alignment=0, blend_mode="ALPHA",
                    fixed_normal=(0, 0, 1), lock_axis=(0, 0, 1))
        self.assertEqual(indexed, original)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.previous_warp = sys.modules.get("warp")

    def tearDown(self):
        if self.previous_warp is None:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = self.previous_warp

    def test_strict_registration_and_finally_unmap(self):
        rig = DriverRig()
        batch = rig.batch()
        assert batch.report()["fallback_to_copy"] is False

        def fail(mapped):
            assert mapped.ptr == 0x4000
            raise ValueError("writer failed")

        with self.assertRaisesRegex(ValueError, "writer failed"):
            batch.update(fail)
        assert rig.names()[-1] == "cuGraphicsUnmapResources"
        assert rig.arrays[0]["shape"] == (4, 12)
        batch.close()

    def test_successful_update_and_explicit_idempotent_close(self):
        rig = DriverRig()
        batch = rig.batch()
        batch.update(lambda mapped: None)
        assert batch.report()["updates"] == 1
        batch.close()
        batch.close()
        assert rig.names().count("cuGraphicsUnregisterResource") == 1
        assert rig.guards == 0
        assert sum(e[0] == "delete" for e in rig.gl.calls) == 1
        assert batch.report()["closed"] is True

    def test_context_mismatch_preserves_resources_for_deferred_cleanup(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.gl.context = 42
        with self.assertRaisesRegex(RuntimeError, "context"):
            batch.close()
        assert batch.report()["closed"] is False
        assert not any(e[0] == "delete" for e in rig.gl.calls)
        rig.gl.context = 41
        batch.close()

    def test_device_mismatch_rejected_before_allocating(self):
        rig = DriverRig()
        rig.gl.devices = [1]
        with self.assertRaisesRegex(RuntimeError, "device"):
            rig.batch()
        assert not rig.gl.calls

    def test_draw_validates_count_and_passes_column_major_matrices(self):
        rig = DriverRig()
        batch = rig.batch()
        identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "count"):
            batch.draw(5, view_projection=identity, view_matrix=identity, viewport_size=(100, 100))
        batch.draw(4, view_projection=identity, view_matrix=identity, viewport_size=(100, 100))
        assert rig.gl.calls[-1][0] == "draw"
        assert batch.report()["draws"] == 1

    def test_registration_failure_releases_owned_gl_objects(self):
        rig = DriverRig()

        def reject(*args, **kwargs):
            raise RuntimeError("interop unavailable")

        rig.library.cuGraphicsGLRegisterBuffer = reject
        with self.assertRaisesRegex(RuntimeError, "interop unavailable"):
            rig.batch()
        assert rig.gl.calls[-1] == ("delete", 11, 12, 13)

    def test_native_draw_restores_foreign_state_when_draw_raises(self):
        module = load_module()
        gl = object.__new__(module._GL)
        state = {
            0x8B8D: 91,
            0x85B5: 92,
            0x8894: 93,
            0x0B74: 0x0201,
            0x0B72: 1,
            0x80C9: 1,
            0x80C8: 0,
            0x80CB: 1,
            0x80CA: 0,
            0x8009: 0x800A,
            0x883D: 0x800B,
            0x935C: 0x8CA2,
            0x935D: 0x935F,
        }
        enabled = {0x0BE2: True, 0x0B71: False, 0x0B44: True}
        calls = []
        gl._integer = lambda enum: state[enum]
        gl._integer_indexed = lambda enum, index: state[enum]
        gl._enabled = lambda enum: enabled[enum]
        gl._enabled_indexed = lambda enum, index: enabled[enum]
        gl._enable = lambda enum, value: calls.append(("enable", enum, value))
        gl._enable_indexed = lambda enum, index, value: calls.append(("enable_i", enum, index, value))

        def call(name, result, args, *values):
            calls.append((name, *values))
            if name == "glGetUniformLocation":
                return 0
            if name == "glDrawArraysInstanced":
                raise RuntimeError("driver draw failure")

        gl._call = call
        identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "driver draw failure"):
            gl.draw(
                11,
                13,
                4,
                view_projection=identity,
                view_matrix=identity,
                viewport_size=(800, 600),
                min_diameter_px=4.0,
                stretch=1.0,
                shaded=True,
                alignment=0,
                blend_mode="ALPHA",
                fixed_normal=(0, 0, 1),
                lock_axis=(0, 0, 1),
            )
        assert ("glUseProgram", 91) in calls
        assert ("glBindVertexArray", 92) in calls
        assert ("glBindBuffer", 0x8892, 93) in calls
        assert ("glDepthMask", 1) in calls
        assert ("glDepthFunc", 0x0201) in calls
        assert ("glBlendEquationSeparatei", 0, 0x800A, 0x800B) in calls
        assert ("glBlendFuncSeparatei", 0, 1, 0, 1, 0) in calls
        assert ("glClipControl", 0x8CA1, 0x935E) in calls
        assert ("glClipControl", 0x8CA2, 0x935F) in calls
        for key, value in enabled.items():
            if key == 0x0BE2:
                assert ("enable_i", key, 0, value) in calls
            else:
                assert ("enable", key, value) in calls

    def test_partial_map_failure_attempts_unmap_and_preserves_error(self):
        rig = DriverRig()
        batch = rig.batch()
        rig.array_error = ValueError("mapped pointer wrapper failed")
        with self.assertRaisesRegex(ValueError, "mapped pointer wrapper failed"):
            batch.update(lambda mapped: None)
        assert rig.names()[-1] == "cuGraphicsUnmapResources"
        assert not batch.report()["mapped"]
        batch.close()

    def test_non_finite_matrix_rejected_before_gl_draw(self):
        rig = DriverRig()
        batch = rig.batch()
        matrix = [0.0] * 16
        matrix[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            batch.draw(4, view_projection=matrix, view_matrix=[0.0] * 16, viewport_size=(10, 10))
        assert not any(c[0] == "draw" for c in rig.gl.calls)


if __name__ == "__main__":
    unittest.main()
