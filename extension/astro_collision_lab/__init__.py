"""Astro Collision Lab: independent primitive authoring and live CUDA/GL display."""

_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    from . import dependencies, runtime, ui

    dependencies.register()
    ui.register()
    runtime.register_handlers()
    from . import authoring
    authoring.register()
    _REGISTERED = True


def unregister():
    global _REGISTERED
    from . import dependencies, runtime, ui

    dependencies.unregister()
    from . import authoring
    authoring.unregister()
    runtime.shutdown()
    runtime.unregister_handlers()
    ui.unregister()
    _REGISTERED = False
