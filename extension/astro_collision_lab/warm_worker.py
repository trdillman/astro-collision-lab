"""Owned background Blender process for bounded kernel probes or explicit validation."""

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path


def kernel_probe(mode):
    """Return a deterministic, bounded sample that touches a mode's kernels.

    This is a compilation probe only: it neither represents authored physics
    nor qualifies stability. The full authored scene is initialized once in the
    interactive process after this code/runtime receipt is available.
    """
    from .simulation import BodySpec, SimulationConfig

    mode = str(mode).upper()
    if mode not in {"GALAXY", "IMPACT", "COMPRESSIBLE"}:
        raise ValueError(f"unsupported kernel probe mode: {mode}")
    config = SimulationConfig(mode=mode, particle_count=256)
    body = BodySpec(name=f"{mode.title()} kernel probe", kind="SPHERE",
                    particle_count=256, mass=1.0, radius=1.0,
                    material="ROCK", scale_length=.1)
    config.validate()
    return config, (body,)


def main(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    sys.path.insert(0, request["dependency_path"])
    # Preserve the installed module name: Warp's cache key includes it. Compiling
    # an unqualified source import would leave the GUI's installed kernels cold.
    name = request["module_name"]
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            namespace = types.ModuleType(parent)
            namespace.__path__ = []
            sys.modules[parent] = namespace
    package = Path(request["package"])
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from types import SimpleNamespace

    import warp as wp

    simulation = importlib.import_module(name + ".simulation")
    appearance = importlib.import_module(name + ".device_appearance")
    contract = importlib.import_module(name + ".interaction")
    import bpy

    wp.config.kernel_cache_dir = request["cache"]
    config = simulation.SimulationConfig(**request["config"])
    bodies = tuple(simulation.BodySpec(**body) for body in request["bodies"])
    simulation.validate_scene(config, bodies)
    task = request.get("task", "KERNELS")
    code = contract.code_fingerprint(package)
    runtime = contract.runtime_identity(wp, bpy.app, name)
    if task == "VALIDATE_ISOLATION":
        actual_identity = contract.warmup_identity(
            signature=contract.equilibrium_signature(config, bodies),
            code_fingerprint=code, runtime=runtime,
        )
    elif task == "KERNELS":
        actual_identity = contract.kernel_identity(
            config, code_fingerprint=code, runtime=runtime,
        )
    else:
        raise ValueError(f"unknown preparation task: {task}")
    if actual_identity != request["identity"]:
        raise RuntimeError("Preparation inputs, code or runtime changed; request a fresh run")
    if task == "VALIDATE_ISOLATION":
        equilibrium = importlib.import_module(name + ".equilibrium")
        result = equilibrium.qualify(config, bodies, str(request["marker"]) + ".isolation.json")
        marker = Path(request["marker"])
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({**contract.ready_marker(actual_identity),
                                         "validation": result}), encoding="utf-8")
        temporary.replace(marker)
        return
    sim = simulation.AstroCollisionSimulation(config)
    try:
        sim.initialize([simulation.BodySpec(**body) for body in request["bodies"]])
        if config.mode.upper() == "GALAXY":
            # The sphere probe never samples a disk; compile the disk sampler's
            # kernels here so authored disks do not compile on the UI thread.
            importlib.import_module(name + "._galaxy_kernels").warm()
        sim.step()
        channels = sim.display_channels()
        frame = SimpleNamespace(channels=channels, count=sim.count)
        writer = appearance.DeviceAppearanceWriter(channels["positions"].device)
        out = wp.empty((sim.count, 12), dtype=wp.float32, device=channels["positions"].device)
        writer.write(frame, out, appearance.Appearance())
        writer.write(frame, out, appearance.Appearance(color_source="SPEED"))
        wp.synchronize_device(channels["positions"].device)
        marker = Path(request["marker"])
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({**contract.ready_marker(actual_identity),
                                         "stats": sim.stats}), encoding="utf-8")
        temporary.replace(marker)
    finally:
        sim.close()


if __name__ == "__main__":
    request_path = sys.argv[sys.argv.index("--")+1]
    try:
        main(request_path)
    except Exception as error:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from preparation_result import write_failure
            write_failure(request_path, error)
        except Exception:  # noqa: BLE001 - failure reporting must never replace the original error
            # Failure reporting must never replace the original traceback.
            import traceback
            traceback.print_exc()
        raise
