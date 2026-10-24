"""Owned background Blender process: compile kernels before the live GUI uses CUDA."""

import json
import importlib
import importlib.util
import sys
import types
from pathlib import Path


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
    import warp as wp
    from types import SimpleNamespace

    simulation = importlib.import_module(name + ".simulation")
    appearance = importlib.import_module(name + ".device_appearance")
    contract = importlib.import_module(name + ".interaction")
    import bpy

    wp.config.kernel_cache_dir = request["cache"]
    config = simulation.SimulationConfig(**request["config"])
    bodies = tuple(simulation.BodySpec(**body) for body in request["bodies"])
    simulation.validate_scene(config, bodies)
    actual_identity = contract.warmup_identity(
        signature=contract.equilibrium_signature(config, bodies) if config.mode == "GALAXY"
                  else contract.physical_signature(config, bodies),
        code_fingerprint=contract.code_fingerprint(package),
        runtime=contract.runtime_identity(wp, bpy.app, name),
    )
    if actual_identity != request["identity"]:
        raise RuntimeError("Preparation inputs or runtime changed; request a fresh rebuild")
    if config.mode == "GALAXY":
        equilibrium=importlib.import_module(name+".equilibrium")
        equilibrium.qualify(config,tuple(simulation.BodySpec(**b) for b in request["bodies"]),
                            str(request["marker"])+".isolation.json")
    sim = simulation.AstroCollisionSimulation(config)
    try:
        sim.initialize([simulation.BodySpec(**body) for body in request["bodies"]])
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
    main(sys.argv[sys.argv.index("--")+1])
