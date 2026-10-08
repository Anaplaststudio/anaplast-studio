"""Live (interactive) displacement: a Geometry Nodes modifier stays on the prototype while you drag the
sliders, and is baked into the mesh when you are happy. UV is computed inside the nodes from position,
so Width / Shift / Rotate update instantly even on a million-face prototype."""
import os
import math
import shutil
import bpy
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep

MOD = "Anaplast_XYZ_Live"
NG = "Anaplast_XYZ_Live_NG"


def maps_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")


def bundled_maps(self, context):
    items = []
    d = maps_dir()
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in (".png", ".tif", ".tiff", ".exr", ".jpg", ".psd"):
                items.append((f, os.path.splitext(f)[0], os.path.join(d, f)))
    return items or [("NONE", "(no maps added yet)", "")]


def axes(p, src):
    n = mu.average_normal(src)
    up = Vector((0, 0, 1))
    if abs(n.dot(up)) > 0.9:
        up = Vector((0, 1, 0))
    u = up.cross(n).normalized()
    v = n.cross(u).normalized()
    a = math.radians(p.xyz_rotation)
    return u * math.cos(a) + v * math.sin(a), -u * math.sin(a) + v * math.cos(a)


def build_group(image):
    ng = bpy.data.node_groups.get(NG)
    if ng:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(NG, 'GeometryNodeTree')
    I = ng.interface
    I.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for name, default in (("U axis", (1.0, 0.0, 0.0)), ("V axis", (0.0, 1.0, 0.0)), ("Centre", (0.0, 0.0, 0.0))):
        sk = I.new_socket(name, in_out='INPUT', socket_type='NodeSocketVector'); sk.default_value = default
    sk = I.new_socket("Stamp centre", in_out='INPUT', socket_type='NodeSocketVector'); sk.default_value = (0.0, 0.0, 0.0)
    for name, default in (("Width", 180.0), ("Shift U", 0.0), ("Shift V", 0.0), ("Mid", 0.5),
                          ("Depth R", 0.15), ("Depth G", 0.0), ("Depth B", 0.0), ("Stamp radius", 0.0)):
        sk = I.new_socket(name, in_out='INPUT', socket_type='NodeSocketFloat'); sk.default_value = default
    I.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = ng.nodes, ng.links
    gi = nodes.new("NodeGroupInput"); go = nodes.new("NodeGroupOutput")
    pos = nodes.new("GeometryNodeInputPosition")
    rel = nodes.new("ShaderNodeVectorMath"); rel.operation = 'SUBTRACT'
    links.new(pos.outputs["Position"], rel.inputs[0]); links.new(gi.outputs["Centre"], rel.inputs[1])
    def coord(axis_socket, shift_socket):
        dot = nodes.new("ShaderNodeVectorMath"); dot.operation = 'DOT_PRODUCT'
        links.new(rel.outputs["Vector"], dot.inputs[0]); links.new(gi.outputs[axis_socket], dot.inputs[1])
        div = nodes.new("ShaderNodeMath"); div.operation = 'DIVIDE'
        links.new(dot.outputs["Value"], div.inputs[0]); links.new(gi.outputs["Width"], div.inputs[1])
        add = nodes.new("ShaderNodeMath"); add.operation = 'ADD'; add.inputs[1].default_value = 0.5
        links.new(div.outputs[0], add.inputs[0])
        add2 = nodes.new("ShaderNodeMath"); add2.operation = 'ADD'
        links.new(add.outputs[0], add2.inputs[0]); links.new(gi.outputs[shift_socket], add2.inputs[1])
        return add2
    cu, cv = coord("U axis", "Shift U"), coord("V axis", "Shift V")
    comb = nodes.new("ShaderNodeCombineXYZ")
    links.new(cu.outputs[0], comb.inputs["X"]); links.new(cv.outputs[0], comb.inputs["Y"])
    tex = nodes.new("GeometryNodeImageTexture"); tex.inputs["Image"].default_value = image
    tex.extension = 'REPEAT'; tex.interpolation = 'Cubic'
    links.new(comb.outputs["Vector"], tex.inputs["Vector"])
    sep = nodes.new("FunctionNodeSeparateColor"); sep.mode = 'RGB'
    links.new(tex.outputs["Color"], sep.inputs["Color"])
    total = None
    for ch, sock in (("Red", "Depth R"), ("Green", "Depth G"), ("Blue", "Depth B")):
        sub = nodes.new("ShaderNodeMath"); sub.operation = 'SUBTRACT'
        links.new(sep.outputs[ch], sub.inputs[0]); links.new(gi.outputs["Mid"], sub.inputs[1])
        mul = nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'
        links.new(sub.outputs[0], mul.inputs[0]); links.new(gi.outputs[sock], mul.inputs[1])
        if total is None:
            total = mul
        else:
            add = nodes.new("ShaderNodeMath"); add.operation = 'ADD'
            links.new(total.outputs[0], add.inputs[0]); links.new(mul.outputs[0], add.inputs[1]); total = add
    mask = nodes.new("GeometryNodeInputNamedAttribute"); mask.data_type = 'FLOAT'; mask.inputs["Name"].default_value = "anaplast_outer"
    gate = nodes.new("ShaderNodeMath"); gate.operation = 'MULTIPLY'
    links.new(total.outputs[0], gate.inputs[0]); links.new(mask.outputs["Attribute"], gate.inputs[1])
    # stamp falloff: full inside 65% of the radius, fading to nothing at the rim; radius 0 = whole surface
    dist = nodes.new("ShaderNodeVectorMath"); dist.operation = 'DISTANCE'
    links.new(pos.outputs["Position"], dist.inputs[0]); links.new(gi.outputs["Stamp centre"], dist.inputs[1])
    inner = nodes.new("ShaderNodeMath"); inner.operation = 'MULTIPLY'; inner.inputs[1].default_value = 0.65
    links.new(gi.outputs["Stamp radius"], inner.inputs[0])
    fall = nodes.new("ShaderNodeMapRange"); fall.clamp = True
    fall.inputs["To Min"].default_value = 1.0; fall.inputs["To Max"].default_value = 0.0
    links.new(dist.outputs["Value"], fall.inputs["Value"])
    links.new(inner.outputs[0], fall.inputs["From Min"]); links.new(gi.outputs["Stamp radius"], fall.inputs["From Max"])
    on = nodes.new("FunctionNodeCompare"); on.data_type = 'FLOAT'; on.operation = 'GREATER_THAN'
    links.new(gi.outputs["Stamp radius"], on.inputs[0]); on.inputs[1].default_value = 0.001
    onf = nodes.new("ShaderNodeMath"); onf.operation = 'SUBTRACT'; onf.inputs[0].default_value = 1.0
    links.new(on.outputs["Result"], onf.inputs[1])                      # 1 - on
    part = nodes.new("ShaderNodeMath"); part.operation = 'MULTIPLY'
    links.new(on.outputs["Result"], part.inputs[0]); links.new(fall.outputs["Result"], part.inputs[1])
    weight = nodes.new("ShaderNodeMath"); weight.operation = 'ADD'
    links.new(onf.outputs[0], weight.inputs[0]); links.new(part.outputs[0], weight.inputs[1])
    gate2 = nodes.new("ShaderNodeMath"); gate2.operation = 'MULTIPLY'
    links.new(gate.outputs[0], gate2.inputs[0]); links.new(weight.outputs[0], gate2.inputs[1])
    nrm = nodes.new("GeometryNodeInputNormal")
    scale = nodes.new("ShaderNodeVectorMath"); scale.operation = 'SCALE'
    links.new(nrm.outputs["Normal"], scale.inputs[0]); links.new(gate2.outputs[0], scale.inputs["Scale"])
    setp = nodes.new("GeometryNodeSetPosition")
    links.new(gi.outputs["Geometry"], setp.inputs["Geometry"]); links.new(scale.outputs["Vector"], setp.inputs["Offset"])
    links.new(setp.outputs["Geometry"], go.inputs["Geometry"])
    return ng


def sync(context):
    """Push the current slider values into the live modifier (called on every property change)."""
    from .shell import current_prototype, source_surface
    p = context.scene.anaplast
    obj = current_prototype(p) or source_surface(p)
    src = source_surface(p)
    if obj is None or src is None:
        return
    mod = obj.modifiers.get(MOD)
    if mod is None or mod.node_group is None:
        return
    u, v = axes(p, obj)
    lo, hi = mu.world_bbox(obj)
    c = (lo + hi) * 0.5
    alpha = (p.xyz_mode == 'ALPHA')
    vals = {"U axis": u[:], "V axis": v[:], "Centre": c[:], "Width": p.xyz_width,
            "Stamp centre": tuple(p.stamp_centre), "Stamp radius": (p.stamp_radius if p.stamp_mode else 0.0),
            "Shift U": p.xyz_off_u, "Shift V": p.xyz_off_v, "Mid": p.xyz_midlevel,
            "Depth R": (p.xyz_depth if alpha else p.xyz_depth_r),
            "Depth G": (0.0 if alpha else p.xyz_depth_g), "Depth B": (0.0 if alpha else p.xyz_depth_b)}
    for item in mod.node_group.interface.items_tree:
        if getattr(item, "in_out", "") == 'INPUT' and item.name in vals:
            try:
                mod[item.identifier] = vals[item.name]
            except Exception:
                pass
    obj.update_tag()


def live_update(self, context):
    sync(context)
    try:
        update_ring(context)
    except Exception:
        pass


class ANAPLAST_OT_xyz_live(bpy.types.Operator):
    """Start live displacement: the map is applied through a modifier you can adjust with the sliders in real time"""
    bl_idname = "anaplast.xyz_live"
    bl_label = "Start live preview"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.psd;*.tif;*.tiff;*.exr;*.png;*.jpg", options={'HIDDEN'})
    use_bundled: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        return (current_prototype(p) or source_surface(p)) is not None

    def invoke(self, context, event):
        p = context.scene.anaplast
        if self.use_bundled:
            if p.xyz_bundled and p.xyz_bundled != "NONE":
                self.filepath = os.path.join(maps_dir(), p.xyz_bundled)
                return self.execute(context)
            rep(self, {'ERROR'}, "No map in the module library yet — use 'Add map to library'")
            return {'CANCELLED'}
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj, src = (current_prototype(p) or source_surface(p)), source_surface(p)
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            rep(self, {'ERROR'}, f"Could not load map ({e})")
            return {'CANCELLED'}
        if img.size[0] == 0:
            bpy.data.images.remove(img)
            rep(self, {'ERROR'}, "Map is empty (layered PSD?) — flatten it or save as 16-bit TIFF")
            return {'CANCELLED'}
        p.xyz_image = img.name
        if obj.data.attributes.get("anaplast_outer") is None:
            mu.restore_detail(obj, src, max_dist=max(0.6, 4.0 * p.voxel_size))   # also writes the outer-skin mask
        if p.xyz_subdivide and len(obj.data.vertices) < 1200000:
            def setup(m):
                m.levels = 1; m.render_levels = 1; m.subdivision_type = 'SIMPLE'
            mu.apply_modifier(obj, 'SUBSURF', setup)
            mu.restore_detail(obj, src, max_dist=max(0.6, 4.0 * p.voxel_size))
        old = obj.modifiers.get(MOD)
        if old:
            obj.modifiers.remove(old)
        mod = obj.modifiers.new(MOD, 'NODES')
        mod.node_group = build_group(img)
        sync(context)
        bpy.ops.anaplast.display_mode(mode='CLAY')
        rep(self, {'INFO'}, f"{img.name} live on {obj.name} — drag Depth / Width / Shift / Rotate and watch it update. Then Bake")
        return {'FINISHED'}


class ANAPLAST_OT_xyz_bake(bpy.types.Operator):
    """Bake the live displacement into the mesh (or Discard to remove it)"""
    bl_idname = "anaplast.xyz_bake"
    bl_label = "Bake"
    bl_options = {'REGISTER', 'UNDO'}

    discard: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj = current_prototype(p) or source_surface(p)
        return obj is not None and obj.modifiers.get(MOD) is not None

    def execute(self, context):
        from .shell import current_prototype, source_surface
        obj = current_prototype(context.scene.anaplast) or source_surface(context.scene.anaplast)
        mod = obj.modifiers.get(MOD)
        if self.discard:
            obj.modifiers.remove(mod)
            rep(self, {'INFO'}, "Live displacement removed")
            return {'FINISHED'}
        dg = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        new_me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
        obj.modifiers.remove(mod)
        old = obj.data
        obj.data = new_me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        for pg in obj.data.polygons:
            pg.use_smooth = True
        rep(self, {'INFO'}, f"Displacement baked into {obj.name} ({len(obj.data.polygons):,} faces)")
        return {'FINISHED'}


class ANAPLAST_OT_xyz_add_to_library(bpy.types.Operator):
    """Copy a displacement map into the module's own maps folder so it is always available"""
    bl_idname = "anaplast.xyz_add_to_library"
    bl_label = "Add map to library"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.psd;*.tif;*.tiff;*.exr;*.png;*.jpg", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        d = maps_dir()
        os.makedirs(d, exist_ok=True)
        try:
            dst = os.path.join(d, os.path.basename(self.filepath))
            shutil.copyfile(self.filepath, dst)
        except OSError as e:
            rep(self, {'ERROR'}, f"Could not copy: {e}")
            return {'CANCELLED'}
        context.scene.anaplast.xyz_bundled = os.path.basename(self.filepath)
        rep(self, {'INFO'}, f"{os.path.basename(dst)} added to the module library ({os.path.getsize(dst) / 1e6:.0f} MB)")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_xyz_live, ANAPLAST_OT_xyz_bake, ANAPLAST_OT_xyz_add_to_library)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


RING = "Anaplast_Stamp_Ring"


def update_ring(context):
    """A bright ring on the surface showing exactly where and how big the stamp is."""
    import bmesh
    from mathutils import Matrix
    from .shell import current_prototype, source_surface
    p = context.scene.anaplast
    old = bpy.data.objects.get(RING)
    if old:
        mu.delete_object(old)
    if not p.stamp_mode:
        return
    obj = current_prototype(p) or source_surface(p)
    if obj is None:
        return
    centre = Vector(p.stamp_centre)
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromObject(obj, context.evaluated_depsgraph_get())
    loc, nrm, idx, dist = tree.find_nearest(obj.matrix_world.inverted() @ centre)
    n = (obj.matrix_world.to_3x3() @ nrm).normalized() if nrm is not None else Vector((0, 0, 1))
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=False, segments=64, radius=p.stamp_radius)
    ring = mu.new_mesh_object(RING, bm, mu.get_collection(context.scene, "Anaplast_Slice"))
    ring.matrix_world = Matrix.Translation(centre + n * 0.4) @ n.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    ring.color = (1.0, 0.35, 0.1, 1.0)
    ring.show_in_front = True
    ring.hide_select = True
    ring.display_type = 'WIRE'


class ANAPLAST_OT_stamp_here(bpy.types.Operator):
    """Click on the sculpt to place the stamp centre — the map is previewed only inside the stamp circle. Enter finishes"""
    bl_idname = "anaplast.stamp_here"
    bl_label = "Place stamp (click)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype
        return current_prototype(context.scene.anaplast) is not None and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        context.scene.anaplast.stamp_mode = True
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set("Click where the stamp should sit · scroll wheel still zooms · ENTER finish · ESC cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        from .shell import current_prototype
        p = context.scene.anaplast
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            from bpy_extras import view3d_utils
            obj = current_prototype(p)
            region, rv3d = context.region, context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            inv = obj.matrix_world.inverted()
            ok, loc, nrm, idx = obj.ray_cast(inv @ origin, (inv.to_3x3() @ direction).normalized(), depsgraph=context.evaluated_depsgraph_get())
            if ok:
                p.stamp_centre = (obj.matrix_world @ loc)[:]
                sync(context)
                update_ring(context)
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            context.area.header_text_set(None); context.window.cursor_modal_restore()
            rep(self, {'INFO'}, "Stamp placed — adjust Radius and Depth, then Bake")
            return {'FINISHED'}
        if event.type == 'ESC':
            context.area.header_text_set(None); context.window.cursor_modal_restore()
            p.stamp_mode = False
            sync(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


_old_reg2, _old_unreg2 = register, unregister


def register():
    _old_reg2()
    bpy.utils.register_class(ANAPLAST_OT_stamp_here)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_stamp_here)
    _old_unreg2()


class ANAPLAST_OT_alpha_brush(bpy.types.Operator):
    """Sculpt with the loaded Texturing.xyz / AlphaSkin map as a Draw brush in drag-dot mode:
click and drag to place one stamp of the map, exactly like ZBrush DragRect"""
    bl_idname = "anaplast.alpha_brush"
    bl_label = "Alpha brush (drag dot)"

    invert: bpy.props.BoolProperty(name="Carve inward", default=True)

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        return (current_prototype(p) or source_surface(p)) is not None

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj = current_prototype(p) or source_surface(p)
        img = bpy.data.images.get(p.xyz_image) if p.xyz_image else None
        if img is None and p.xyz_bundled and p.xyz_bundled != "NONE":
            try:
                img = bpy.data.images.load(os.path.join(maps_dir(), p.xyz_bundled), check_existing=True)
                p.xyz_image = img.name
            except RuntimeError:
                img = None
        if img is None:
            rep(self, {'ERROR'}, "Load a map first (1 · Load map, or Use library map)")
            return {'CANCELLED'}
        tex = bpy.data.textures.get("Anaplast_AlphaBrush") or bpy.data.textures.new("Anaplast_AlphaBrush", 'IMAGE')
        tex.image = img
        tex.extension = 'CLIP'
        br = bpy.data.brushes.get("Anaplast Alpha") or bpy.data.brushes.new("Anaplast Alpha", mode='SCULPT')
        try:
            br.sculpt_tool = 'DRAW'
        except Exception:
            pass
        br.texture = tex
        br.texture_slot.map_mode = 'VIEW_PLANE'          # the map faces you, like a ZBrush alpha
        br.strength = p.brush_strength
        br.use_frontface = True
        try:
            br.stroke_method = 'DRAG_DOT'                 # one stamp per drag
        except Exception:
            pass
        try:
            br.direction = 'SUBTRACT' if self.invert else 'ADD'
        except (TypeError, AttributeError):
            pass
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.hide_set(False); obj.select_set(True); context.view_layer.objects.active = obj
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='SCULPT')
        try:
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.draw")
        except Exception:
            pass
        # Blender 4.3+ keeps the active brush in tool_settings.sculpt.brush_asset_reference; assigning
        # .brush can silently fail, so configure whatever brush is actually active as well.
        try:
            context.tool_settings.sculpt.brush = br
        except Exception:
            pass
        active = getattr(context.tool_settings.sculpt, "brush", None)
        if active is not None:
            active.texture = tex
            active.texture_slot.map_mode = 'VIEW_PLANE'
            active.strength = p.brush_strength
            try:
                active.stroke_method = 'DRAG_DOT'
            except Exception:
                pass
            try:
                active.direction = 'SUBTRACT' if self.invert else 'ADD'
            except (TypeError, AttributeError):
                pass
        rep(self, {'INFO'}, f"Alpha brush ready on {obj.name}: click-drag to place one stamp · F size · Shift+F strength · Ctrl flips in/out")
        return {'FINISHED'}


_reg_ab, _unreg_ab = register, unregister


def register():
    _reg_ab()
    bpy.utils.register_class(ANAPLAST_OT_alpha_brush)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_alpha_brush)
    _unreg_ab()
