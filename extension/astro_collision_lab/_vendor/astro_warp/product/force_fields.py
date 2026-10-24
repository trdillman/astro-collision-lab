"""External force fields: the ``astro.force.external_field`` module surface (ADR 0045).

A Force member of a System compiles to a ``BindingRole.FIELD`` binding, and every FIELD
binding in a compiled descriptor becomes one :class:`ExternalFieldSpec` here -- the
host-side record the orbital runtimes read to apply the field. The stage is an
acceleration accumulation launched after the gravity kernel of whichever backend is
running, adding into the same per-body ``accelerations`` array, so within the FORCE
family the steps commute and plan order carries no physics.

Two kinds (:class:`~astro_warp.product.model.ForceKind`):

- ``UNIFORM``: constant acceleration of magnitude ``strength`` along ``axis`` -- the
  bound object's local +Z (its ``sample_basis`` row 2), matching Blender's Wind
  convention. Position-independent.
- ``POINT_ATTRACTOR``: acceleration of magnitude ``strength / r^2`` toward ``position``;
  ``strength`` plays the role of GM, so a negative value repels.

An external field makes the System open: momentum and energy conservation diagnostics
will legitimately drift under one, which ADR 0045 documents so the drift is read as
physics rather than as an integrator bug.

The device tier lives here too: :func:`field_kernels` compiles the per-precision
accumulation kernel with the same registrars the direct solver uses, and
:func:`field_arrays` bakes a spec tuple into device arrays -- baked rather than passed
per launch so the single launch both orbital lanes append is legal inside CUDA graph
capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .model import BindingDescriptor, BindingRole, ForceKind

# The integers the device kernel branches on. One code per enum member, pinned by test:
# the kernel cannot read a StrEnum, and an unmapped kind must fail loudly on the host
# (KeyError at bake time) rather than silently on the device.
FIELD_KIND_CODES: dict[ForceKind, int] = {
    ForceKind.UNIFORM: 0,
    ForceKind.POINT_ATTRACTOR: 1,
}


@dataclass(frozen=True)
class ExternalFieldSpec:
    """One FIELD binding resolved for a runtime: what to apply, how hard, from where.

    ``position`` and ``axis`` are resolved by the compile bridge from the bound object's
    transform (translation and normalized local +Z respectively); the runtime consumes
    them as plain code-unit vectors and never touches the scene.
    """

    kind: ForceKind
    strength: float
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def validate(self) -> None:
        """Refuse numbers the kernel would turn into silent NaN trajectories.

        Sign is meaning, not an error: negative ``strength`` points a uniform push down
        the axis and turns an attractor into a repulsor, so only finiteness is checked.
        """

        if not math.isfinite(self.strength):
            raise ValueError(f"external field strength must be finite, got {self.strength!r}")
        if not all(math.isfinite(component) for component in self.position):
            raise ValueError(f"external field position must be finite, got {self.position!r}")
        norm = math.sqrt(sum(component * component for component in self.axis))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError(
                f"external field axis must be finite and non-zero, got {self.axis!r}"
            )


def specs_from_bindings(
    bindings: tuple[BindingDescriptor, ...],
) -> tuple[ExternalFieldSpec, ...]:
    """Every active FIELD binding as a runtime spec, in binding order.

    The push axis is the binding's ``sample_basis`` row 2 -- the bound object's
    transformed local +Z -- normalized here so a scaled Blender object scales its
    geometry but never its force. A degenerate row refuses loudly: it means the
    bridge recorded a zero basis, and silently substituting world +Z would apply a
    force the scene never authored. Inactive fields contribute nothing, the same
    way an inactive body leaves the solver population.
    """

    specs: list[ExternalFieldSpec] = []
    for binding in bindings:
        if binding.role != BindingRole.FIELD or not binding.active:
            continue
        row = binding.sample_basis[2]
        norm = math.sqrt(sum(component * component for component in row))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError(
                f"binding {binding.binding_id!r} has a degenerate force axis "
                f"(sample_basis row 2 = {row!r}); the field's push direction is undefined"
            )
        specs.append(
            ExternalFieldSpec(
                kind=binding.force_kind,
                strength=float(binding.strength),
                position=tuple(float(component) for component in binding.position),
                axis=tuple(float(component) / norm for component in row),
            )
        )
    return tuple(specs)


def field_arrays(
    wp: Any,
    precision: Any,
    specs: tuple[ExternalFieldSpec, ...],
    device: Any,
) -> dict[str, Any]:
    """Bake a spec tuple into the device arrays the field kernel reads.

    Struct-of-arrays for the same reason the body state is: Warp kernels take typed
    arrays, not host objects. Baked once at load rather than passed per launch so the
    launch is capture-legal -- the arrays are stable pointers for the life of the
    runtime, because the config that carries the specs is frozen.
    """

    import numpy as np

    from .nbody import _dtypes

    scalar, vector, state_dtype = _dtypes(wp, precision)
    kinds = np.array([FIELD_KIND_CODES[spec.kind] for spec in specs], dtype=np.int32)
    strengths = np.array([spec.strength for spec in specs], dtype=state_dtype)
    positions = np.array([spec.position for spec in specs], dtype=state_dtype)
    axes = np.array([spec.axis for spec in specs], dtype=state_dtype)
    return {
        "field_kinds": wp.array(kinds, dtype=wp.int32, device=device),
        "field_strengths": wp.array(strengths, dtype=scalar, device=device),
        "field_positions": wp.array(positions, dtype=vector, device=device),
        "field_axes": wp.array(axes, dtype=vector, device=device),
    }


def specs_with_strengths(
    specs: tuple[ExternalFieldSpec, ...], strengths: tuple[float, ...]
) -> tuple[ExternalFieldSpec, ...]:
    """The same fields with new strengths (ADR 0165); pure, so it can validate before any write.

    Only the strength moves: kind, position and axis size and aim the field, so they stay
    as they were baked. A different field count is refused, because it resizes the arrays.
    """

    if len(strengths) != len(specs):
        raise ValueError(
            f"{len(strengths)} field strengths given for {len(specs)} fields; "
            "adding or removing a Force changes the sizes the runtime was built with"
        )
    values = [float(value) for value in strengths]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("field strengths must be finite")
    return tuple(
        ExternalFieldSpec(
            kind=spec.kind,
            strength=value,
            position=spec.position,
            axis=spec.axis,
        )
        for spec, value in zip(specs, values, strict=True)
    )


def write_field_strengths(
    wp: Any, arrays: dict[str, Any], specs: tuple[ExternalFieldSpec, ...]
) -> None:
    """Rewrite the strengths in the device array in place (ADR 0165).

    The kernel reads ``field_strengths`` on every launch, so the next step sees the new
    values and a captured graph that already reads the array needs no re-record. ``arrays``
    may be empty (bodies not loaded yet): the specs on the config are what the next load
    bakes.
    """

    import numpy as np

    target = arrays.get("field_strengths")
    if target is None or not specs:
        return
    host = np.asarray([spec.strength for spec in specs], dtype=target.numpy().dtype)
    target.assign(wp.array(host, dtype=target.dtype, device=target.device))


def field_kernels(wp: Any, precision: Any) -> Any:
    """Compile the field accumulation kernel for one state precision.

    One kernel over bodies with the (few) fields iterated inside, rather than one
    launch per field: field count is authored-member count, so the loop is short and
    a single launch keeps the captured CUDA graph one node longer, not N.
    """

    from .nbody import _dtypes, _registrars

    scalar, vector, _ = _dtypes(wp, precision)
    warp_kernel, _ = _registrars(wp, scalar, vector)

    @warp_kernel
    def apply_external_fields(
        positions: wp.array(dtype=vector),
        active: wp.array(dtype=wp.int32),
        count: int,
        field_kinds: wp.array(dtype=wp.int32),
        field_strengths: wp.array(dtype=scalar),
        field_positions: wp.array(dtype=vector),
        field_axes: wp.array(dtype=vector),
        field_count: int,
        softening: scalar,
        accelerations: wp.array(dtype=vector),
    ):
        i = wp.tid()
        if i >= count or active[i] == 0:
            return
        ai = vector()
        for f in range(field_count):
            strength = field_strengths[f]
            if field_kinds[f] == 0:
                # UNIFORM: constant push along the field's axis, position-independent.
                ai += field_axes[f] * strength
            else:
                # POINT_ATTRACTOR: strength / r^2 toward the field position. Same
                # softening and same r=0 clamp floor as the gravity kernels (ADR 0025),
                # so a body passing through the attractor degrades the same way a body
                # passing through another body does.
                r = field_positions[f] - positions[i]
                dist2 = wp.max(wp.dot(r, r) + softening * softening, scalar(1.0e-12))
                inv_dist = scalar(1.0) / wp.sqrt(dist2)
                ai += r * (strength * inv_dist * inv_dist * inv_dist)
        # Accumulate rather than assign: gravity already wrote this slot, and within
        # the FORCE family every step is a commutative addition into the same array.
        accelerations[i] = accelerations[i] + ai

    @dataclass(frozen=True)
    class FieldKernels:
        apply_external_fields: Any

    return FieldKernels(apply_external_fields=apply_external_fields)
