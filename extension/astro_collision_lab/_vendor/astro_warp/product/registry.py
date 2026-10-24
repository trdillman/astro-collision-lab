"""Process-local multi-context registry for product descriptors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .compiler import GraphCompiler
from .model import ContextState, SimulationDescriptor, new_context_id


class ContextRegistry:
    """Owns independent simulation descriptors for one process."""

    def __init__(self, compiler: GraphCompiler | None = None) -> None:
        self._compiler = compiler or GraphCompiler()
        self._contexts: dict[str, SimulationDescriptor] = {}
        self.active_context_id: str | None = None

    def create(self, name: str, **kwargs) -> SimulationDescriptor:
        descriptor = SimulationDescriptor(context_id=new_context_id(), name=name, **kwargs)
        descriptor.validate()
        self._contexts[descriptor.context_id] = descriptor
        self.active_context_id = descriptor.context_id
        return descriptor

    def put(self, descriptor: SimulationDescriptor) -> SimulationDescriptor:
        descriptor.validate()
        self._contexts[descriptor.context_id] = descriptor
        if self.active_context_id is None:
            self.active_context_id = descriptor.context_id
        return descriptor

    def get(self, context_id: str) -> SimulationDescriptor:
        return self._contexts[context_id]

    def active(self) -> SimulationDescriptor | None:
        if self.active_context_id is None:
            return None
        return self._contexts.get(self.active_context_id)

    def set_active(self, context_id: str) -> SimulationDescriptor:
        descriptor = self._contexts[context_id]
        self.active_context_id = context_id
        return descriptor

    def compile(self, context_id: str) -> SimulationDescriptor:
        descriptor = self._compiler.compile(self._contexts[context_id])
        self._contexts[context_id] = descriptor
        return descriptor

    def mark_unloaded_after_reopen(self) -> None:
        for key, descriptor in list(self._contexts.items()):
            state = ContextState.UNLOADED
            cache_root = descriptor.cache_root
            # A `//`-relative cache_root is Blender's blend-file-relative notation, not a
            # literal filesystem path; probing it with Path.exists() would resolve against
            # the wrong root, so leave the state as UNLOADED rather than guess.
            if (
                cache_root
                and not str(cache_root).startswith("//")
                and not Path(cache_root).exists()
            ):
                state = ContextState.MISSING_CACHE
            self._contexts[key] = replace(descriptor, state=state, runtime_plan=None)

    def close(self, context_id: str) -> None:
        self._contexts[context_id] = replace(self._contexts[context_id], state=ContextState.CLOSED)
        if self.active_context_id == context_id:
            self.active_context_id = None

    def close_all(self) -> None:
        for key, descriptor in list(self._contexts.items()):
            self._contexts[key] = replace(descriptor, state=ContextState.CLOSED)
        self.active_context_id = None

    def descriptors(self) -> tuple[SimulationDescriptor, ...]:
        return tuple(self._contexts.values())
