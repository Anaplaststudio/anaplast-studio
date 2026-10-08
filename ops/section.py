"""Cut-away using a Geometry Nodes modifier that deletes geometry on one side of a plane.
Viewport clip planes proved unreliable across builds; this always shows."""
import bpy
from ..utils import mesh as mu
from .report import rep

MOD = "Anaplast_CutAway"
NG = "Anaplast_CutAway_NG"
AXES = {"SAG": (1.0, 0.0, 0.0), "COR": (0.0, 1.0, 0.0), "AXI": (0.0, 0.0, 1.0)}


def build_group():
    ng = bpy.data.node_groups.get(NG)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(NG, 'GeometryNodeTree')
    I = ng.interface
    I.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    sk = I.new_socket("Axis", in_out='INPUT', socket_type='NodeSocketVector'); sk.default_value = (1.0, 0.0, 0.0)
    sk = I.new_socket("Position", in_out='INPUT', socket_type='NodeSocketFloat'); sk.default_value = 0.0
    sk = I.new_socket("Slab", in_out='INPUT', socket_type='NodeSocketFloat'); sk.default_value = 0.0
    I.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    n, lk = ng.nodes, ng.links
    gi = n.new("NodeGroupInput"); go = n.new("NodeGroupOutput")
    pos = n.new("GeometryNodeInputPosition")
    dot = n.new("ShaderNodeVectorMath"); dot.operation = 'DOT_PRODUCT'
    lk.new(pos.outputs["Position"], dot.inputs[0]); lk.new(gi.outputs["Axis"], dot.inputs[1])
    # in front of the plane
    cmp = n.new("FunctionNodeCompare"); cmp.data_type = 'FLOAT'; cmp.operation = 'GREATER_THAN'
    lk.new(dot.outputs["Value"], cmp.inputs[0]); lk.new(gi.outputs["Position"], cmp.inputs[1])
    # slab mode: also delete anything further behind than Position - Slab (Slab = 0 disables via the switch)
    low = n.new("ShaderNodeMath"); low.operation = 'SUBTRACT'
    lk.new(gi.outputs["Position"], low.inputs[0]); lk.new(gi.outputs["Slab"], low.inputs[1])
    cmp2 = n.new("FunctionNodeCompare"); cmp2.data_type = 'FLOAT'; cmp2.operation = 'LESS_THAN'
    lk.new(dot.outputs["Value"], cmp2.inputs[0]); lk.new(low.outputs[0], cmp2.inputs[1])
    slab_on = n.new("FunctionNodeCompare"); slab_on.data_type = 'FLOAT'; slab_on.operation = 'GREATER_THAN'
    lk.new(gi.outputs["Slab"], slab_on.inputs[0]); slab_on.inputs[1].default_value = 0.0
    behind = n.new("FunctionNodeBooleanMath"); behind.operation = 'AND'
    lk.new(cmp2.outputs["Result"], behind.inputs[0]); lk.new(slab_on.outputs["Result"], behind.inputs[1])
    both = n.new("FunctionNodeBooleanMath"); both.operation = 'OR'
    lk.new(cmp.outputs["Result"], both.inputs[0]); lk.new(behind.outputs[0], both.inputs[1])
    dele = n.new("GeometryNodeDeleteGeometry"); dele.domain = 'FACE'; dele.mode = 'ALL'
    lk.new(gi.outputs["Geometry"], dele.inputs["Geometry"]); lk.new(both.outputs[0], dele.inputs["Selection"])
    lk.new(dele.outputs["Geometry"], go.inputs["Geometry"])
    return ng


def targets(context):
    from .shell import current_prototype, source_surface
    p = context.scene.anaplast
    return [o for o in (p.face_scan_obj, p.cast_obj, p.cbct_obj, current_prototype(p), source_surface(p)) if o is not None]


def apply_cutaway(context):
    p = context.scene.anaplast
    ax = AXES[p.section_axis]
    sign = -1.0 if p.section_flip else 1.0
    n_on = 0
    for o in targets(context):
        mod = o.modifiers.get(MOD)
        if not p.section_on:
            if mod:
                o.modifiers.remove(mod)
            continue
        if mod is None:
            mod = o.modifiers.new(MOD, 'NODES')
            mod.node_group = build_group()
        ids = {}
        for item in mod.node_group.interface.items_tree:
            if getattr(item, "in_out", "") == 'INPUT':
                ids[item.name] = item.identifier
        try:
            mod[ids["Axis"]] = tuple(sign * a for a in ax)
            mod[ids["Position"]] = sign * p.section_pos
            mod[ids["Slab"]] = p.slice_slab
        except Exception:
            pass
        o.update_tag()
        n_on += 1
    return n_on


OUTLINE = "Anaplast_Slice_Outline"


def build_outline(context):
    """A real cross-section contour: bisect a decimated copy of each object with the plane and keep the curve."""
    import bmesh
    from mathutils import Vector
    p = context.scene.anaplast
    old = bpy.data.objects.get(OUTLINE)
    if old:
        mu.delete_object(old)
    if not p.section_on or not p.slice_outline:
        return 0
    ax = Vector(AXES[p.section_axis])
    plane_co = ax * p.section_pos
    out = bmesh.new()
    for o in targets(context):
        if o.name == OUTLINE:
            continue
        tmp = mu.duplicate_object(o, "anaplast_slice_tmp", o.users_collection[0])
        for m in list(tmp.modifiers):
            tmp.modifiers.remove(m)
        nf = len(tmp.data.polygons)
        if nf > 400000:
            def dec(m):
                m.decimate_type = 'COLLAPSE'; m.ratio = 400000 / nf; m.use_collapse_triangulate = True
            mu.apply_modifier(tmp, 'DECIMATE', dec)
        bm = bmesh.new()
        bm.from_mesh(tmp.data)
        bm.transform(tmp.matrix_world)
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=plane_co, plane_no=ax, clear_inner=True, clear_outer=True)
        me = bpy.data.meshes.new("anaplast_slice_part")
        bm.to_mesh(me); bm.free()
        out.from_mesh(me)
        bpy.data.meshes.remove(me)
        mu.delete_object(tmp)
    me = bpy.data.meshes.new(OUTLINE)
    out.to_mesh(me)
    n = len(me.edges)
    out.free()
    obj = bpy.data.objects.new(OUTLINE, me)
    obj.color = (1.0, 0.35, 0.1, 1.0)
    obj.show_in_front = True
    obj.hide_select = True
    mu.get_collection(context.scene, "Anaplast_Slice").objects.link(obj)
    return n


class ANAPLAST_OT_slice_outline(bpy.types.Operator):
    """Draw the cross-section contour where the plane cuts the surfaces (orange). Click again to refresh after moving the plane"""
    bl_idname = "anaplast.slice_outline"
    bl_label = "Draw cut outline"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.anaplast
        if not p.section_on:
            rep(self, {'ERROR'}, "Turn the slice viewer on first")
            return {'CANCELLED'}
        p.slice_outline = True
        n = build_outline(context)
        rep(self, {'INFO'}, f"Cut outline drawn ({n:,} segments)" if n else "Plane misses the objects — slide Position into the scan")
        return {'FINISHED'}


class ANAPLAST_OT_cutaway(bpy.types.Operator):
    """Cut the face and the sculpt open along a sagittal / coronal / axial plane so you can look inside. Slide Position to move the cut; nothing is deleted from the meshes"""
    bl_idname = "anaplast.cutaway"
    bl_label = "Slice viewer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.anaplast
        p.section_on = not p.section_on
        n = apply_cutaway(context)
        if p.section_on and n == 0:
            p.section_on = False
            rep(self, {'ERROR'}, "Nothing to cut — import a scan first")
            return {'CANCELLED'}
        if not p.section_on:
            o = bpy.data.objects.get(OUTLINE)
            if o:
                mu.delete_object(o)
        rep(self, {'INFO'}, f"Slice viewer ON across {n} object(s) — slide Position; Slab > 0 shows a slice of that thickness" if p.section_on else "Slice viewer OFF")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_cutaway)
    bpy.utils.register_class(ANAPLAST_OT_slice_outline)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_slice_outline)
    bpy.utils.unregister_class(ANAPLAST_OT_cutaway)
