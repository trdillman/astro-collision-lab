"""Where the geometry a mesh-shaped cloud is sampled from actually lives.

`SimulationDescriptor.to_dict` is `asdict`, so every field a descriptor carries is written
into the `.blend`, into the bake journal and into the compiler's hashed payload. A mesh
belongs in none of those: a subdivided character or a scanned asteroid is megabytes of
triangles, it is duplicated once per binding that names the same object, and it would make
the descriptor's size a function of the *scene's* geometry --- the exact property
`BindingDescriptor`'s comment on `sample_extent` says the descriptor must not have.

So the descriptor carries `sample_mesh_key`, a fixed 64-character digest, and the triangles
live here under that same digest.

Content addressing, rather than a side table keyed by binding ID, is what makes this safe.
A key *is* the hash of the geometry it names, so the store and the descriptor cannot drift
into disagreement the way two parallel channels would: edit the mesh and the key changes,
which means the old key resolves either to the geometry it always named or to nothing at
all. There is no third outcome in which a current key quietly yields stale triangles.

Lifetime is the process. The host registers a binding's evaluated mesh when it compiles
that binding, which is the only moment the evaluated geometry exists, and every compile
re-registers. A descriptor read back from JSON therefore arrives with a key the store may
not hold --- `resolve` raises for it. That is the correct answer and not a gap to paper
over: the mesh must come from the scene, and silently filling a bounding box instead is the
substitution this whole subsystem exists to end.
"""

from __future__ import annotations

from .mesh_sampling import MeshSample, MeshSamplingError

__all__ = [
    "MissingMeshError",
    "clear",
    "contains",
    "register",
    "resolve",
    "size",
]


class MissingMeshError(MeshSamplingError):
    """A `sample_mesh_key` named geometry this process does not hold."""


_MESHES: dict[str, MeshSample] = {}


def register(mesh: MeshSample) -> str:
    """Store ``mesh`` and return the key a descriptor should carry for it.

    Idempotent by construction: the key is the mesh's own content digest, so registering
    equal geometry twice --- two bindings on one object, or the same object recompiled ---
    writes one entry and returns one key.
    """

    key = mesh.content_key
    _MESHES.setdefault(key, mesh)
    return key


def resolve(key: str) -> MeshSample:
    """The geometry ``key`` names.

    Raises rather than returning ``None``. Every caller of this function is about to draw a
    cloud, and the only thing it could do with ``None`` is fall back to a proxy volume ---
    which is a wrong answer that looks like a right one in the viewport.
    """

    try:
        return _MESHES[key]
    except KeyError:
        raise MissingMeshError(
            f"no mesh registered for key {key!r}; the bound object's geometry must be "
            "re-evaluated by the host before this cloud can be sampled"
        ) from None


def contains(key: str) -> bool:
    return key in _MESHES


def size() -> int:
    return len(_MESHES)


def clear() -> None:
    """Drop every entry. For tests, and for a host tearing a session down."""

    _MESHES.clear()
