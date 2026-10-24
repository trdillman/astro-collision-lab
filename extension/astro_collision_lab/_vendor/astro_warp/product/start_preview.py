"""The start-frame preview: where a scene's particles are, before any physics exists (ADR 0162).

Where a particle starts is a pure function of the authored scene. `sampling.expand_bindings`
draws it from the binding alone -- seed, shape, extent, count and the object's pose -- with no
solver and no device. This module holds those draws, one block per binding, and keeps them
current as the artist edits:

* ``build`` samples a scene's blocks, **reusing** every block whose sampling key did not change
  so an edit to one object re-samples that object only and leaves every other block as the
  same array, byte for byte;
* ``follow`` moves a block's already-drawn positions with the rigid delta between the
  object's old and new world matrix, with **no** sampling at all, which is what a drag costs.

It is numpy only (no ``bpy``, no Warp), so the whole behaviour is unit-testable on a CPU, and
it never runs a physics step. ADR 0049 ("templates only author, never run") holds because this
draws and does not run: it creates no plan, runtime or device buffer.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .model import BindingDescriptor, BindingRole, BodyRepresentation, SampleShape
from .sampling import binding_body_count, expand_bindings

__all__ = [
    "PREVIEW_SETTLE_S",
    "PreviewBlock",
    "RefreshReport",
    "StartPreview",
    "build",
    "follow",
    "previewable",
    "sample_positions",
    "sampling_key",
]

#: How long a drag must be still before the preview is brought back to exactly what a compile
#: would sample for the final pose. Short enough that the picture settles as the mouse stops;
#: long enough that a drag, which delivers an update every few milliseconds, never pays it.
PREVIEW_SETTLE_S = 0.15

#: Binding fields that cannot move a particle, so editing one is not a reason to re-sample.
#: Every other field is in the key: a new field is a re-sample until someone names it here,
#: which is the safe default (the same discipline as ``invalidation.EXCLUDED_BINDING_FIELDS``).
#: Velocity inputs only reach the second expansion pass; mass reaches masses only; the
#: Emitter and Force surfaces belong to roles the preview does not draw.
_NOT_POSITIONAL = frozenset(
    {
        "object_name",
        "render_object_name",
        "orbital_elements",
        "mass",
        "velocity",
        "velocity_model",
        "support_fraction",
        "keplerian_central_binding_id",
        "keplerian_companion_binding_id",
        "keplerian_counter_clockwise",
        "eccentricity_jitter",
        "mass_jitter",
        "expansion_speed",
        "expansion_proportional",
        "sample_frame_provenance",
        "spin",
        "spin_rest",
        "pinned",
        "rate",
        "active_range",
        "emitter_capacity",
        "force_kind",
        "strength",
    }
)

#: Shapes whose compile samples in the object's own frame, so a rotation moves the particles
#: (the annulus has a plane; a mesh shape is sampled from geometry the host evaluated in world
#: space). Every other shape is drawn about the object's origin with a size read from its
#: dimensions and **no rotation**, so a follow that rotated it would show a cloud the compile
#: will not make.
_ROTATES_WITH_OBJECT = frozenset({SampleShape.DISK_ANNULUS})


def previewable(binding: BindingDescriptor) -> bool:
    """Whether this binding has particles to show at the start frame.

    Only a Body-role binding (an analytic point mass or a sampled cloud) has a step-zero
    picture. An Emitter's slots all start dead, a Force has no body, and an inactive binding is
    not in the run.
    """

    return binding.active and binding.role == BindingRole.BODY and binding_body_count(binding) > 0


def point_body(binding: BindingDescriptor) -> bool:
    """A binding that is one solver body drawn as one point (a Star, a planet), not a cloud."""

    return binding.role != BindingRole.FIELD and binding.representation in (
        BodyRepresentation.ANALYTICAL,
        BodyRepresentation.RIGID_EXTENDED,
    )


def authored_body_radii(
    bindings: Any, half_extents: dict[str, float]
) -> dict[str, float]:
    """The radius each point Body is drawn at, by binding ID (ADR 0238).

    The larger of the body's physical radius and the sphere it was authored as
    (``half_extents[object_name]``, half its largest world dimension). The Add menu authors a
    Star or planet as a sphere far larger than its physical radius, so it can be seen and
    picked (ADR 0233); drawing the physical radius once Play starts made it shrink to a dot.
    This is a display size only: gravity and collisions keep the physical radius.
    """

    out: dict[str, float] = {}
    for binding in bindings:
        if not point_body(binding):
            continue
        radius = max(
            float(binding.radius or 0.0), float(half_extents.get(binding.object_name, 0.0))
        )
        if math.isfinite(radius) and radius > 0.0:
            out[binding.binding_id] = radius
    return out


def sampling_key(binding: BindingDescriptor) -> tuple[str, ...]:
    """Everything that decides where this binding's particles are, as a hashable key."""

    return tuple(
        f"{f.name}={getattr(binding, f.name)!r}"
        for f in dataclasses.fields(binding)
        if f.name not in _NOT_POSITIONAL
    )


def sample_positions(binding: BindingDescriptor) -> Any:
    """The binding's particle positions, float64 ``(n, 3)``, exactly as a compile draws them.

    One binding at a time is the same as the whole System at once: each binding's stream is
    seeded from its own seed and ID, so none depends on what else is in the scene.
    """

    bodies = expand_bindings(
        (binding,),
        gravitational_constant=1.0,
        positions_only=True,
    )
    positions = bodies.positions
    positions.setflags(write=False)
    return positions


def _flat(matrix: Any) -> tuple[float, ...]:
    return tuple(float(value) for row in matrix for value in row)


@dataclass(frozen=True)
class PreviewBlock:
    """One binding's drawn particles and the pose they were drawn (or last followed) at."""

    binding_id: str
    key: tuple[str, ...]
    positions: Any  # float64 (n, 3), read-only
    #: The object's 4x4 world matrix, row-major flattened, that these positions match; ``None``
    #: when the caller supplied none, in which case the block cannot follow.
    matrix: tuple[float, ...] | None
    rotates: bool

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class RefreshReport:
    """What a ``build`` did, by binding ID, so a caller (and a test) can see what was re-made."""

    resampled: tuple[str, ...] = ()
    reused: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    #: A binding whose sample was refused (a mesh that is gone, a disk with no plane): it is
    #: left out of the picture and the reason is here, so one bad object never blanks the rest.
    refused: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StartPreview:
    """Every previewed block of one System, in binding order."""

    blocks: tuple[PreviewBlock, ...] = ()
    revision: int = 0
    report: RefreshReport = field(default_factory=RefreshReport)

    @property
    def count(self) -> int:
        return sum(block.count for block in self.blocks)

    @cached_property
    def positions(self) -> Any:
        """The display array: float32 ``(N, 3)``, C-contiguous, what the overlay uploads."""

        import numpy as np

        if not self.blocks:
            return np.empty((0, 3), dtype=np.float32)
        out = np.ascontiguousarray(
            np.concatenate([block.positions for block in self.blocks]), dtype=np.float32
        )
        out.setflags(write=False)
        return out

    def row_owners(self, names: dict[str, str]) -> Any:
        """The owner of each row of ``positions``: ``names[binding_id]`` once per particle.

        The preview's rows carry no stable ids (a runtime assigns those), but they are laid
        out one block per binding in order, so a row's owner is the block it falls in. An
        object array, so 100,000 rows hold 100,000 references to a handful of strings and not
        100,000 padded copies of them. A binding with no name maps to ``""``.
        """

        import numpy as np

        owners = np.array([names.get(block.binding_id, "") for block in self.blocks], dtype=object)
        counts = np.asarray([block.count for block in self.blocks], dtype=np.intp)
        return np.repeat(owners, counts)

    def row_sizes(self, radii: dict[str, float]) -> Any:
        """Each row's display radius: ``radii[binding_id]`` once per particle (ADR 0238).

        A point Body is drawn at its authored size before Play, as it is after; a binding not
        in ``radii`` (a sampled cloud) draws ``0``, which the overlay's minimum size covers.
        float32, read-only, one entry per row of ``positions``.
        """

        import numpy as np

        values = np.asarray(
            [float(radii.get(block.binding_id, 0.0)) for block in self.blocks], dtype=np.float32
        )
        counts = np.asarray([block.count for block in self.blocks], dtype=np.intp)
        out = np.repeat(values, counts)
        out.setflags(write=False)
        return out

    def block(self, binding_id: str) -> PreviewBlock | None:
        for block in self.blocks:
            if block.binding_id == binding_id:
                return block
        return None


def build(
    bindings: tuple[BindingDescriptor, ...],
    matrices: dict[str, Any] | None = None,
    previous: StartPreview | None = None,
) -> StartPreview:
    """The preview for ``bindings``, re-sampling only what changed since ``previous``.

    ``matrices`` maps a binding ID to its object's 4x4 world matrix (any 4x4 nested sequence),
    recorded on the block so a later ``follow`` has the pose to move from. A block whose key is
    unchanged is carried over as the very same object, so its positions are byte-for-byte what
    they were. The key holds the pose (``position``, ``sample_basis``, ``sample_extent``), so a
    block that was only *followed* to a new pose has a stale key and is re-sampled on the
    settle: that is what brings the picture to exactly what a compile will draw.
    """

    matrices = matrices or {}
    carried = {block.binding_id: block for block in (previous.blocks if previous else ())}
    blocks: list[PreviewBlock] = []
    resampled: list[str] = []
    reused: list[str] = []
    refused: list[tuple[str, str]] = []
    seen: set[str] = set()
    for binding in bindings:
        if not previewable(binding):
            continue
        binding_id = binding.binding_id
        seen.add(binding_id)
        key = sampling_key(binding)
        old = carried.get(binding_id)
        if old is not None and old.key == key:
            blocks.append(old)
            reused.append(binding_id)
            continue
        try:
            positions = sample_positions(binding)
        except (ValueError, KeyError, RuntimeError) as exc:  # a refusal is data, not a crash
            refused.append((binding_id, str(exc)))
            continue
        matrix = matrices.get(binding_id)
        blocks.append(
            PreviewBlock(
                binding_id=binding_id,
                key=key,
                positions=positions,
                matrix=None if matrix is None else _flat(matrix),
                rotates=binding.sample_shape in _ROTATES_WITH_OBJECT
                or binding.sample_shape.needs_mesh,
            )
        )
        resampled.append(binding_id)
    dropped = tuple(sorted(set(carried) - seen))
    return StartPreview(
        blocks=tuple(blocks),
        revision=(previous.revision + 1) if previous else 1,
        report=RefreshReport(
            resampled=tuple(resampled),
            reused=tuple(reused),
            dropped=dropped,
            refused=tuple(refused),
        ),
    )


def _inverse_or_none(matrix: tuple[float, ...]) -> Any:
    import numpy as np

    array = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    linear = array[:3, :3]
    if not math.isfinite(float(np.linalg.det(linear))) or abs(np.linalg.det(linear)) < 1e-12:
        return None
    return np.linalg.inv(array)


def _scale_of(matrix: Any) -> float:
    """The largest column length of the matrix's linear part: how big the object is."""

    import numpy as np

    return float(np.max(np.linalg.norm(matrix[:3, :3], axis=0)))


def follow(
    preview: StartPreview, matrices: dict[str, Any]
) -> tuple[StartPreview, tuple[str, ...], tuple[str, ...]]:
    """Move blocks with their objects. Returns ``(preview, moved, unfollowable)``.

    No sampling happens here. A block whose matrix changed gets the delta ``M_new * M_old^-1``
    applied to its positions (full for a shape that turns with its object; translation plus
    uniform scale for one the compile does not rotate). A block with no recorded matrix, or
    whose old matrix is singular (an object scaled to nothing has no delta to give), is
    reported as unfollowable and left alone, so the caller can refresh instead. Blocks whose
    matrix did not change are the same objects in the result.
    """

    import numpy as np

    moved: list[str] = []
    stuck: list[str] = []
    blocks: list[PreviewBlock] = []
    for block in preview.blocks:
        new = matrices.get(block.binding_id)
        if new is None or block.matrix is None:
            blocks.append(block)
            continue
        new_flat = _flat(new)
        if new_flat == block.matrix:
            blocks.append(block)
            continue
        inverse = _inverse_or_none(block.matrix)
        new_matrix = np.asarray(new_flat, dtype=np.float64).reshape(4, 4)
        if inverse is None or not np.isfinite(new_matrix).all():
            blocks.append(block)
            stuck.append(block.binding_id)
            continue
        old_matrix = np.asarray(block.matrix, dtype=np.float64).reshape(4, 4)
        points = block.positions
        if block.rotates:
            delta = new_matrix @ inverse
            moved_points = points @ delta[:3, :3].T + delta[:3, 3]
        else:
            ratio = _scale_of(new_matrix) / _scale_of(old_matrix)
            old_origin = old_matrix[:3, 3]
            new_origin = new_matrix[:3, 3]
            moved_points = (points - old_origin) * ratio + new_origin
        moved_points.setflags(write=False)
        blocks.append(dataclasses.replace(block, positions=moved_points, matrix=new_flat))
        moved.append(block.binding_id)
    if not moved:
        return preview, (), tuple(stuck)
    return (
        StartPreview(
            blocks=tuple(blocks),
            revision=preview.revision + 1,
            report=RefreshReport(reused=preview.report.reused, resampled=()),
        ),
        tuple(moved),
        tuple(stuck),
    )
