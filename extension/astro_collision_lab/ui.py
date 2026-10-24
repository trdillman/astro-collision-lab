"""Standalone Astro Collision Lab authoring UI.

This module uses Blender primitives and a scene-local source collection. It has
no dependency on the Astro Warp extension UI. Simulation controls are delegated
to this addon's runtime; authoring operators do not start or step a simulation.

Do not enable postponed annotations: Blender must evaluate RNA property calls.
"""

import math
import threading
import uuid

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, Menu

from .preset_specs import galaxy_collision, moon_collision


def _source_changed(self, context):
    from .authoring import changed
    changed(self, context)


def _appearance_changed(self, context):
    """Presentation changes redraw the current state without rebuilding physics."""
    if threading.current_thread() is not threading.main_thread():
        from .authoring import appearance_changed
        appearance_changed()
        return
    from .viewport import request_redraw
    request_redraw()


def _balanced_sampling_counts(radii, total):
    """Volume-proportional largest remainders, minimum one particle/source."""
    radii=tuple(float(r) for r in radii)
    if not radii or any(not math.isfinite(r) or r<=0 for r in radii):
        raise ValueError("Every source needs a positive finite world radius")
    if isinstance(total,bool) or not isinstance(total,int) or total<len(radii):
        raise ValueError("The total must retain at least one particle per source")
    largest=max(radii)
    weights=tuple((r/largest)**3 for r in radii)
    weight_sum=math.fsum(weights)
    quotas=tuple(total*w/weight_sum for w in weights)
    counts=[max(1,math.floor(q)) for q in quotas]
    remaining=total-sum(counts)
    if remaining>0:
        order=sorted(range(len(counts)),key=lambda i:(-(quotas[i]-counts[i]),i))
        for i in order[:remaining]:
            counts[i]+=1
    while sum(counts)>total:
        candidates=(i for i,count in enumerate(counts) if count>1)
        index=max(candidates,key=lambda i:(counts[i]-quotas[i],-i))
        counts[index]-=1
    return tuple(counts)


def _sampling_spacing_ratio(radii, counts):
    radii=tuple(radii)
    largest=max(radii)
    spacing=tuple((radius/largest)/count**(1.0/3.0)
        for radius,count in zip(radii,counts,strict=True))
    return max(spacing)/min(spacing)


def _scene_collections(scene):
    """Enumerate only collections actually linked into this scene."""
    pending = [scene.collection]
    seen = set()
    while pending:
        collection = pending.pop()
        pointer = collection.as_pointer()
        if pointer in seen:
            continue
        seen.add(pointer)
        yield collection
        pending.extend(collection.children)


def _source_collection_poll(settings, collection):
    scene = settings.id_data
    return any(item == collection for item in _scene_collections(scene))


def source_objects(scene):
    """Return enabled mesh sources from this scene's explicitly selected group."""
    collection = scene.acl_settings.source_collection
    if collection is None:
        return ()
    if not any(item == collection for item in _scene_collections(scene)):
        return ()
    return tuple(
        obj for obj in collection.all_objects
        if obj.type == "MESH" and obj.acl_body.enabled
    )


def _source_in_active_collection(scene, obj):
    """UI membership includes Galaxy Empty roots as well as mesh sources."""
    collection = scene.acl_settings.source_collection
    return (collection is not None
            and any(item == collection for item in _scene_collections(scene))
            and obj in tuple(collection.all_objects))


def _new_source_collection(scene, name):
    if not scene.get("acl_scene_id"):
        scene["acl_scene_id"] = uuid.uuid4().hex
    collection = bpy.data.collections.new(f"AstroCollide Â· {name}")
    collection["acl_source_group"] = True
    collection["acl_owner_scene"] = scene["acl_scene_id"]
    scene.collection.children.link(collection)
    scene.acl_settings.source_collection = collection
    return collection


def _set_active_sources(context, objects):
    for obj in context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    if objects:
        context.view_layer.objects.active = objects[0]


def _create_body(context, collection, spec):
    if spec.kind == "DISK":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=spec.radius,
            depth=0.14 * spec.radius,
            end_fill_type="NGON",
            location=spec.location,
            rotation=spec.rotation,
        )
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32,
            ring_count=16,
            radius=spec.radius,
            location=spec.location,
            rotation=spec.rotation,
        )
    obj = context.object
    obj.name = spec.name
    if collection not in obj.users_collection:
        collection.objects.link(obj)
    # Only the newly created guide is relinked. Existing user objects are never
    # unlinked, removed, or reused by a preset.
    for old_collection in tuple(obj.users_collection):
        if old_collection != collection:
            old_collection.objects.unlink(obj)
    obj.display_type = "WIRE"
    obj.show_in_front = False
    obj.color = spec.color
    body = obj.acl_body
    body.enabled = True
    body.kind = spec.kind
    body.mass = spec.mass
    body.particle_count = spec.count
    body.velocity = spec.velocity
    body.spin = spec.spin
    body.material = spec.material
    body.is_mesh_proxy = False
    body.native_primitive = True
    body.radius = spec.radius
    body.scale_length = spec.scale_length or spec.radius / (5 if spec.kind == "DISK" else 10)
    body.thickness = spec.thickness or .14 * spec.radius
    body.seed = spec.seed
    body.rotation_fraction = spec.rotation_fraction
    obj["acl_guide_radius"] = spec.radius
    obj["acl_guide_thickness"] = body.thickness
    return obj


def _author_preset(context, spec):
    scene = context.scene
    collection = _new_source_collection(scene, spec.name)
    settings = scene.acl_settings
    settings.mode = spec.mode
    settings.blend_mode = "ADDITIVE" if spec.mode=="GALAXY" else "OPAQUE"
    settings.seconds_per_frame = spec.seconds_per_frame
    settings.gravitational_constant = 1.0
    settings.theta = 0.6
    settings.softening = 0.03
    objects = [_create_body(context, collection, body) for body in spec.bodies]
    _set_active_sources(context, objects)
    return objects


def _runtime_action(operator, context, action):
    try:
        from . import runtime
        result = getattr(runtime, action)(context)
        if isinstance(result, set) and ("CANCELLED" in result or "FINISHED" in result):
            return result
    except Exception as exc:
        operator.report({"ERROR"}, f"{action.replace('_', ' ').title()}: {exc}")
        return {"CANCELLED"}
    return {"FINISHED"}


def _runtime_status():
    try:
        from . import runtime
        controller = getattr(runtime, "controller", None)
        if controller is None:
            return {"state": "Not initialized"}
        return controller.status()
    except Exception as exc:
        return {"error": str(exc)}


def _draw_status(layout):
    status = _runtime_status()
    box = layout.box()
    box.label(text="Runtime status", icon="INFO")
    if isinstance(status, dict):
        if not status:
            box.label(text="No runtime status available")
        for key, value in status.items():
            if value is None or isinstance(value, (list, tuple, dict, set)):
                continue
            if isinstance(value, float):
                value = f"{value:.5g}"
            row = box.row()
            row.alert = key.lower() in {"error", "fault"} and bool(value)
            row.label(text=f"{str(key).replace('_', ' ').capitalize()}: {value}")
    else:
        box.label(text=str(status))


class ACLBodySettings(PropertyGroup):
    enabled: BoolProperty(
        update=_source_changed,
        name="Simulation source", default=False,
        description="Include this mesh in the active source collection's initial conditions",
    )
    kind: EnumProperty(
        update=_source_changed,
        name="Sampled shape",
        items=(
            ("SPHERE", "Sphere", "Sample a spherical volume from this object's radius"),
            ("DISK", "Disk", "Sample a galactic disk in the object's local XY plane"),
            ("HALO", "Dark matter halo", "Concentrated Plummer halo; isolated validation required"),
            ("BULGE", "Stellar bulge", "Concentrated stellar Plummer spheroid"),
        ),
        default="SPHERE",
    )
    mass: FloatProperty(
        update=_source_changed,
        name="Total mass", default=1.0, min=0.000001, precision=5,
        description="Total source mass in normalized simulation units; divided over its particles",
    )
    particle_count: IntProperty(
        update=_source_changed,
        name="Particles", default=50_000, min=1,
        description="Requested particles for this source; scene total is the sum of enabled sources",
    )
    velocity: FloatVectorProperty(
        update=_source_changed,
        name="Bulk velocity", size=3, default=(0.0, 0.0, 0.0),
        subtype="XYZ", precision=4,
        description="World-space bulk velocity in normalized length per simulation-time unit",
    )
    spin: FloatVectorProperty(
        update=_source_changed,
        name="Angular velocity", size=3, default=(0.0, 0.0, 0.0),
        subtype="XYZ", precision=4,
        description="World-space angular velocity in radians per simulation-time unit",
    )
    material: EnumProperty(
        update=_source_changed,
        name="Material",
        items=(
            ("ROCK", "Synthetic solid", "Normalized solid; density comes from authored mass and radius"),
            ("IRON", "Synthetic solid (iron tag)", "Same normalized solid law; iron is a provenance tag, not a physical EOS table"),
            ("FLUID", "Fluid", "Fluid material without persistent solid cohesion"),
            ("TILLOTSON", "Tillotson Â· Synthetic", "Normalized synthetic condensed-material EOS parameters"),
        ),
        default="ROCK",
    )
    is_mesh_proxy: BoolProperty(
        update=_source_changed,
        name="Bounding-sphere proxy", default=False, options={"HIDDEN"},
        description="This source was created from an existing mesh and uses a spherical proxy",
    )

    native_primitive: BoolProperty(default=False, options={"HIDDEN"})
    is_galaxy: BoolProperty(default=False, options={"HIDDEN"})
    galaxy_type: EnumProperty(name="Galaxy type", items=(
        ("DISK", "Disk / spiral progenitor", "Axisymmetric disk; spiral structure must emerge dynamically"),
        ("LENTICULAR", "Lenticular", "Thicker disk and larger bulge"),
        ("ELLIPTICAL", "Elliptical / spheroidal", "Pressure-supported stellar spheroid and dark halo"),
        ("DWARF", "Dwarf", "Halo-dominated faint disk")), update=_source_changed)
    radius: FloatProperty(name="Outer radius",default=1.,min=.0001,update=_source_changed)
    scale_length: FloatProperty(name="Scale length",default=.2,min=.0001,update=_source_changed)
    thickness: FloatProperty(name="Disk thickness",default=.14,min=.0001,update=_source_changed)
    toomre_q: FloatProperty(name="Toomre Q",default=1.5,min=1.,max=3.,update=_source_changed)
    rotation_fraction: FloatProperty(name="Rotation fraction",default=1.,min=-1.,max=1.,update=_source_changed)
    disk_fraction: FloatProperty(name="Disk mass fraction",default=.2,min=0.,max=1.,update=_source_changed)
    bulge_fraction: FloatProperty(name="Bulge mass fraction",default=0.,min=0.,max=1.,update=_source_changed)
    seed: IntProperty(name="Sampling seed",default=0,min=0,update=_source_changed)


class ACLSceneSettings(PropertyGroup):
    source_collection: PointerProperty(
        update=_source_changed,
        name="Source collection", type=bpy.types.Collection,
        poll=_source_collection_poll,
        description="Only enabled sources in this scene collection participate; other presets remain independent",
    )
    mode: EnumProperty(
        update=_source_changed,
        name="Dynamics",
        items=(
            ("IMPACT", "Rock Impact Â· IISPH", "Unguided incompressible particle impact with fracture"),
            ("GALAXY", "Galaxy Â· Collisionless", "Approximate collisionless tree gravity; no gas or star formation"),
            ("COMPRESSIBLE", "Condensed SPH Â· Experimental", "Experimental compressible condensed-material SPH"),
        ),
        default="IMPACT",
    )
    seconds_per_frame: FloatProperty(
        update=_source_changed,
        name="Simulation time / step", default=0.005, min=0.000001,
        precision=5,
        description="Normalized time advanced per simulation step; independent of Blender timeline frames and measured viewport FPS",
    )
    theta: FloatProperty(
        update=_source_changed,
        name="Tree opening angle", default=0.6, min=0.05, max=0.7999999523162842, precision=3,
        description="Barnes-Hut opening criterion; smaller values request more accurate gravity",
    )
    softening: FloatProperty(
        update=_source_changed,
        name="Gravity softening", default=0.03, min=0.000001, precision=4,
        description="Gravity softening length in normalized simulation units",
    )
    gravitational_constant: FloatProperty(
        update=_source_changed,
        name="Gravity constant G", default=1.0, min=0.000001, precision=4,
        description="Gravitational constant in normalized simulation units",
    )
    blend_mode: EnumProperty(update=_appearance_changed, name="Particle blending",items=(
        ("OPAQUE","Opaque","Solid point surfaces; source alpha is ignored"),
        ("ALPHA","Alpha","Use source opacity"),
        ("ADDITIVE","Additive","Emissive stars and translucent diagnostic halo")),default="OPAQUE")
    particle_size_px: FloatProperty(
        update=_appearance_changed,
        name="Particle size (px)", default=2.0, min=0.1, max=128.0,
        description="Viewport particle size in screen pixels",
    )
    physical_radius: BoolProperty(
        update=_appearance_changed,
        name="Use physical particle radius", default=False,
        description="Use simulation-space particle radius instead of constant screen size",
    )
    color_by: EnumProperty(
        update=_appearance_changed,
        name="Color by",
        items=(
            ("SOURCE", "Source body", "Color each authored source separately"),
            ("SPEED", "Speed", "Color by particle speed"),
            ("ENERGY", "Energy", "Color by the active solver's energy channel"),
            ("DENSITY", "Density", "Color by the active solver's density channel"),
        ),
        default="SOURCE",
    )
    ramp_min: FloatProperty(
        update=_appearance_changed,
        name="Ramp minimum", default=0.0,
        description="Color range minimum; both bounds at zero request the runtime's automatic range",
    )
    ramp_max: FloatProperty(
        update=_appearance_changed,
        name="Ramp maximum", default=0.0,
        description="Color range maximum; both bounds at zero request the runtime's automatic range",
    )

    authoring_status: bpy.props.StringProperty(name="Preparation", default="Preview")
    substeps: IntProperty(name="Gravity substeps",default=1,min=1,max=32,update=_source_changed)
    seed: IntProperty(name="Scene seed",default=0,min=0,update=_source_changed)
    bulk_modulus: FloatProperty(name="Bulk modulus",default=10.,min=.0001,update=_source_changed)
    shear_modulus: FloatProperty(name="Shear modulus",default=.5,min=0.,update=_source_changed)
    yield_stress: FloatProperty(name="Yield stress",default=.02,min=0.,update=_source_changed)
    melt_energy: FloatProperty(name="Melt energy",default=5.,min=0.,update=_source_changed)
    specific_internal_energy: FloatProperty(name="Initial specific energy",default=0.,min=0.,update=_source_changed)
    courant_number: FloatProperty(name="Courant number",default=.3,min=.001,max=1.,update=_source_changed)


class ACL_OT_install_dependencies(Operator):
    bl_idname="acl.install_dependencies"
    bl_label="Install certified Warp 1.15.0"
    def execute(self,context):
        from .dependencies import install
        install()
        return {"FINISHED"}


class ACLPreferences(bpy.types.AddonPreferences):
    bl_idname=__package__
    def draw(self,context):
        from .dependencies import status
        self.layout.label(text=status())
        self.layout.operator("acl.install_dependencies")


class ACL_OT_add_source(Operator):
    bl_idname="acl.add_source"
    bl_label="Add Astro Primitive"
    bl_options={"REGISTER","UNDO"}
    kind: EnumProperty(items=tuple((k,n,"") for k,n in (("GALAXY","Galaxy"),("SPHERE","Planet / Impactor"),("DISK","Stellar Disk"),("BULGE","Stellar Bulge"),("HALO","Dark Matter Halo"))))

    def execute(self,context):
        from .authoring import add_galaxy,add_component
        if self.kind=="GALAXY":
            add_galaxy(context)
            context.scene.acl_settings.mode="GALAXY"
            context.scene.acl_settings.blend_mode="ADDITIVE"
        else:
            obj=add_component(context,self.kind)
            if self.kind!="SPHERE":context.scene.acl_settings.mode="GALAXY"
            ui_objects=[obj]
            _set_active_sources(context,ui_objects)
        return {"FINISHED"}


class ACL_MT_add(Menu):
    bl_idname="ACL_MT_add"
    bl_label="Astro Collision Lab"
    def draw(self,context):
        for kind,label in (("GALAXY","Galaxy"),("DISK","Stellar Disk"),("BULGE","Stellar Bulge"),("HALO","Dark Matter Halo"),("SPHERE","Planet / Impactor")):
            self.layout.operator("acl.add_source",text=label).kind=kind


def _add_menu(self,context):
    self.layout.menu("ACL_MT_add",icon="OUTLINER_OB_POINTCLOUD")


class ACL_OT_moon_preset(Operator):
    bl_idname = "acl.moon_preset"
    bl_label = "Moon Collision"
    bl_description = "Create a new 100,000-particle impact source group from two spheres"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene is not None and context.mode == "OBJECT"

    def execute(self, context):
        try:
            _author_preset(context, moon_collision())
        except Exception as exc:
            self.report({"ERROR"}, f"Could not author impact: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Authored 100,000 requested particles across two spheres. Edit with G / R / S, then Start.")
        return {"FINISHED"}


class ACL_OT_galaxy_preset(Operator):
    bl_idname = "acl.galaxy_preset"
    bl_label = "Galaxy Collision"
    bl_description = "Create a new 100,000-particle collisionless source group with two disks and two halos"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene is not None and context.mode == "OBJECT"

    def execute(self, context):
        try:
            from .authoring import add_galaxy,_sync_galaxy
            recipe = galaxy_collision()
            c=_new_source_collection(context.scene,recipe.name)
            c["acl_preset"] = "GALAXY"
            settings=context.scene.acl_settings
            settings.mode=recipe.mode;settings.seconds_per_frame=recipe.seconds_per_frame;settings.blend_mode="ADDITIVE"
            roots=[]
            groups = {}
            for spec in recipe.bodies:
                groups.setdefault(spec.galaxy_id, []).append(spec)
            for components in groups.values():
                disk = next(spec for spec in components if spec.kind == "DISK")
                root=add_galaxy(context,collection=c);root.name=disk.name.rsplit(" ",1)[0]
                root.location=disk.location
                body=root.acl_body
                body.mass=sum(spec.mass for spec in components)
                body.particle_count=sum(spec.count for spec in components)
                body.disk_fraction=disk.mass/body.mass
                body.bulge_fraction=sum(spec.mass for spec in components if spec.kind=="BULGE")/body.mass
                body.radius=disk.radius;body.scale_length=disk.scale_length;body.thickness=disk.thickness
                body.velocity=disk.velocity;body.spin=disk.spin;body.seed=disk.seed
                body.rotation_fraction=disk.rotation_fraction
                _sync_galaxy(root,context.scene)
                colors = {spec.kind:spec.color for spec in components}
                for child in root.children:
                    child.color=colors[child.acl_body.kind]
                roots.append(root)
            context.view_layer.update()
            _set_active_sources(context,roots)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not author galaxies: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Authored 100,000 requested particles across four sources. Preparation will validate each galaxy in isolation.")
        return {"FINISHED"}


class ACL_OT_build_from_selection(Operator):
    bl_idname = "acl.build_from_selection"
    bl_label = "Build From Selection"
    bl_description = "Use selected meshes as bounding-sphere proxies in a new source group; mesh surfaces are not simulated"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        objects = [obj for obj in context.selected_objects if obj.type == "MESH"]
        if not objects:
            self.report({"WARNING"}, "Select at least one mesh in Object Mode")
            return {"CANCELLED"}
        collection = _new_source_collection(context.scene, "Selection")
        # Allocate exactly 100,000 particles for new, unconfigured meshes.
        # Preserve any already-authored source settings instead of silently
        # overwriting user mass, velocity, spin, or particle counts.
        quotient, remainder = divmod(100_000, len(objects))
        for index, obj in enumerate(objects):
            if obj.name not in collection.objects:
                collection.objects.link(obj)
            body = obj.acl_body
            if not body.enabled and not body.is_galaxy:
                body.kind = "SPHERE"
                body.particle_count = max(1, quotient + (index < remainder))
                body.is_mesh_proxy = True
            body.enabled = True
        count = sum(obj.acl_body.particle_count for obj in objects)
        self.report({"INFO"}, f"Source group: {count:,} particles. Newly configured meshes use bounding-sphere proxies.")
        return {"FINISHED"}


class ACL_OT_start(Operator):
    bl_idname = "acl.start"
    bl_label = "Start"
    bl_description = "Initialize or resume simulation from the active source collection"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(source_objects(context.scene))

    def execute(self, context):
        return _runtime_action(self, context, "start")


class ACL_OT_balance_sampling(Operator):
    bl_idname="acl.balance_sampling"
    bl_label="Balance Sampling"
    bl_description=("Redistribute the existing particle total over active impact spheres by world volume; changes counts only")
    bl_options={"REGISTER","UNDO"}

    @classmethod
    def poll(cls,context):
        return (context.scene is not None and context.mode=="OBJECT"
                and context.scene.acl_settings.mode=="IMPACT"
                and bool(source_objects(context.scene)))

    def execute(self,context):
        try:
            from . import runtime
            config,bodies,objects=runtime.authored_scene(context.scene, validate=False)
            if config.mode!="IMPACT" or any(body.kind!="SPHERE" for body in bodies):
                raise ValueError("Balance Sampling requires an IISPH impact group of spheres")
            if any(not getattr(obj,"is_editable",True) for obj in objects):
                raise ValueError("Every source must be editable")
            radii=tuple(body.radius for body in bodies)
            total=sum(body.particle_count for body in bodies)
            counts=_balanced_sampling_counts(radii,total)
            ratio=_sampling_spacing_ratio(radii,counts)
            if ratio>1.01:
                raise ValueError(f"Spacing ratio {ratio:.4g} exceeds 1%; increase total or reduce size contrast")
            previous=tuple(obj.acl_body.particle_count for obj in objects)
            try:
                for obj,count in zip(objects,counts,strict=True):
                    obj.acl_body.particle_count=count
            except Exception:
                for obj,count in zip(objects,previous,strict=True):
                    obj.acl_body.particle_count=count
                raise
        except Exception as exc:
            self.report({"ERROR"},f"Balance Sampling: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"},f"Sampling updated; {total:,} total retained; spacing ratio {ratio:.5f}. Playback rebuilds automatically if active.")
        return {"FINISHED"}


class ACL_OT_pause(Operator):
    bl_idname = "acl.pause"
    bl_label = "Pause"
    bl_description = "Pause the active simulation"

    def execute(self, context):
        return _runtime_action(self, context, "pause")


class ACL_OT_reset(Operator):
    bl_idname = "acl.reset"
    bl_label = "Reset"
    bl_description = "Reset the runtime so current authored sources can be rebuilt"

    def execute(self, context):
        return _runtime_action(self, context, "reset")


class ACL_OT_step(Operator):
    bl_idname = "acl.step"
    bl_label = "Single Step"
    bl_description = "Advance one simulation frame using the configured simulation time per frame"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(source_objects(context.scene))

    def execute(self, context):
        return _runtime_action(self, context, "step")


def _draw_body(layout, context):
    obj = context.object
    if obj is None or obj.type not in {"MESH", "EMPTY"}:
        layout.label(text="Select an Astro source", icon="MESH_DATA")
        return
    body = obj.acl_body
    managed = bool(obj.parent and obj.parent.acl_body.is_galaxy)
    if managed:
        layout.label(text="Edit structure on parent: " + obj.parent.name, icon="INFO")
    physical = layout.column()
    physical.enabled = not managed
    if not body.is_galaxy:
        physical.prop(body, "enabled")
    if not body.enabled and not body.is_galaxy and not managed:
        layout.operator("acl.build_from_selection", icon="ADD")
        return
    if not _source_in_active_collection(context.scene, obj):
        warning = layout.box()
        warning.alert = True
        warning.label(text="Outside active source collection", icon="INFO")
    if body.is_galaxy:
        for key in ("galaxy_type", "mass", "particle_count", "radius", "scale_length",
                    "disk_fraction", "bulge_fraction", "velocity", "spin", "seed"):
            physical.prop(body, key)
        if body.disk_fraction > 0:
            for key in ("thickness", "toomre_q", "rotation_fraction"):
                physical.prop(body, key)
        layout.label(text="Components inherit root structure and motion")
        layout.label(text="Physical edits require isolation preparation")
        return
    physical.prop(body, "kind")
    if context.scene.acl_settings.mode == "GALAXY":
        physical.label(text="Fixed exponential / sech-squared disk" if body.kind == "DISK"
                       else "Fixed spherical Plummer profile")
    for key in ("particle_count", "mass", "velocity", "spin"):
        physical.prop(body, key)
    if body.native_primitive:
        physical.prop(body, "radius")
    if context.scene.acl_settings.mode == "GALAXY":
        for key in ("scale_length", "rotation_fraction", "seed"):
            physical.prop(body, key)
        if body.kind == "DISK":
            physical.prop(body, "thickness")
            physical.prop(body, "toomre_q")
    else:
        physical.prop(body, "material")
    layout.prop(obj, "color", text="Source color")
    layout.separator()
    if managed:
        layout.label(text="Move / Rotate / Scale the parent Galaxy")
    else:
        layout.label(text="Move / Rotate / Scale: G / R / S")
    if body.kind == "DISK":
        layout.label(text="Disk plane: local XY; axis: local Z")
    else:
        layout.label(text="Spherical proxy; largest scale sets radius")
    if body.is_mesh_proxy:
        layout.label(text="Mesh surface is not simulation geometry", icon="INFO")
    layout.label(text="Physical edits rebuild the initial state")


class ACL_PT_main(Panel):
    bl_label = "Astro Collision Lab"
    bl_idname = "ACL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AstroCollide"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.acl_settings
        presets = layout.column(align=True)
        presets.operator("acl.moon_preset", icon="MESH_UVSPHERE")
        presets.operator("acl.galaxy_preset", icon="FORCE_FORCE")
        layout.operator("acl.build_from_selection", icon="MESH_DATA")
        layout.separator()
        layout.prop(settings, "source_collection")
        objects = source_objects(context.scene)
        total = sum(obj.acl_body.particle_count for obj in objects)
        layout.label(text=f"{len(objects)} sources Â· {total:,} requested particles")
        layout.prop(settings, "mode")
        info = layout.column(align=True)
        if settings.mode == "IMPACT":
            info.label(text="Unguided IISPH + fracture")
            info.label(text="Incompressible rock-collision preview")
            layout.operator("acl.balance_sampling", icon="MOD_PARTICLES")
        elif settings.mode == "GALAXY":
            info.label(text="Collisionless gravity; approximate initial state")
        else:
            info.label(text="Experimental condensed-material SPH")
            info.label(text="No vapor or phase-change model")
        layout.separator()
        row = layout.row(align=True)
        row.operator("acl.start", icon="PLAY")
        row.operator("acl.pause", icon="PAUSE")
        row = layout.row(align=True)
        row.operator("acl.reset", icon="FILE_REFRESH")
        row.operator("acl.step", icon="FRAME_NEXT")
        layout.label(text="Physical edits rebuild; resume if playing")
        layout.label(text="Appearance edits redraw immediately")
        layout.label(text=settings.authoring_status)
        _draw_status(layout)


class ACL_PT_simulation(Panel):
    bl_label = "Simulation"
    bl_idname = "ACL_PT_simulation"
    bl_parent_id = "ACL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AstroCollide"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.acl_settings
        layout.label(text="Normalized units Â· default G = 1")
        layout.prop(settings, "seconds_per_frame")
        layout.prop(settings, "gravitational_constant")
        layout.prop(settings, "theta")
        layout.prop(settings, "softening")
        layout.prop(settings,"seed")
        if settings.mode=="GALAXY":layout.prop(settings,"substeps")
        else:
            for key in (("shear_modulus","yield_stress","melt_energy") if settings.mode=="IMPACT" else ("bulk_modulus","shear_modulus","yield_stress","melt_energy","specific_internal_energy","courant_number")):
                layout.prop(settings,key)
        layout.label(text="Play / Step are independent of the timeline")
        layout.label(text="Step time is separate from measured FPS")


class ACL_PT_appearance(Panel):
    bl_label = "Particle Viewport"
    bl_idname = "ACL_PT_appearance"
    bl_parent_id = "ACL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AstroCollide"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.acl_settings
        layout.prop(settings, "blend_mode")
        layout.prop(settings, "physical_radius")
        row = layout.row()
        row.enabled = not settings.physical_radius
        row.prop(settings, "particle_size_px")
        layout.prop(settings, "color_by")
        if settings.color_by != "SOURCE":
            row = layout.row(align=True)
            row.prop(settings, "ramp_min", text="Min")
            row.prop(settings, "ramp_max", text="Max")
            layout.label(text="Both zero: automatic color range")
            if settings.ramp_max != 0.0 and settings.ramp_max <= settings.ramp_min:
                warning = layout.row()
                warning.alert = True
                warning.label(text="Maximum must be greater than minimum", icon="ERROR")
        layout.label(text="Preview: source color, screen size and blending")
        layout.label(text="Diagnostics / physical sizes require simulation")
        if settings.color_by == "ENERGY":
            layout.label(text="Energy channel depends on active dynamics")


class ACL_PT_source(Panel):
    bl_label = "Selected Source"
    bl_idname = "ACL_PT_source"
    bl_parent_id = "ACL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AstroCollide"

    def draw(self, context):
        _draw_body(self.layout, context)


class ACL_PT_object(Panel):
    bl_label = "Astro Collision Source"
    bl_idname = "ACL_PT_object"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type in {"MESH", "EMPTY"}

    def draw(self, context):
        _draw_body(self.layout, context)


_CLASSES = (
    ACLBodySettings,
    ACL_OT_install_dependencies,
    ACLPreferences,
    ACL_OT_add_source,
    ACL_MT_add,
    ACLSceneSettings,
    ACL_OT_moon_preset,
    ACL_OT_galaxy_preset,
    ACL_OT_build_from_selection,
    ACL_OT_start,
    ACL_OT_balance_sampling,
    ACL_OT_pause,
    ACL_OT_reset,
    ACL_OT_step,
    ACL_PT_main,
    ACL_PT_simulation,
    ACL_PT_appearance,
    ACL_PT_source,
    ACL_PT_object,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.acl_settings = PointerProperty(type=ACLSceneSettings)
    bpy.types.Object.acl_body = PointerProperty(type=ACLBodySettings)
    bpy.types.VIEW3D_MT_add.append(_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_add_menu)
    if hasattr(bpy.types.Object, "acl_body"):
        del bpy.types.Object.acl_body
    if hasattr(bpy.types.Scene, "acl_settings"):
        del bpy.types.Scene.acl_settings
    for cls in reversed(_CLASSES):
        if hasattr(cls, "bl_rna"):
            bpy.utils.unregister_class(cls)
