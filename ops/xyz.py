"""Texturing.xyz multi-channel displacement onto the built prototype's outer skin.
Planar projection along the sculpt' own facing direction (no seams on a region), per-channel
strengths (R = coarse, G = medium, B = fine), applied through Geometry Nodes so 16-bit maps keep
their precision. Only the outer skin moves."""
import os
import math
import numpy as np
import bpy
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep


def _planar_uv(obj, source, width_mm, off_u, off_v, rot_deg):
    """UV layer 'xyz' = planar projection of the outer skin along -average_normal(source)."""
    n = mu.average_normal(source)
    up = Vector((0, 0, 1))
    if abs(n.dot(up)) > 0.9:
        up = Vector((0, 1, 0))
    u_axis = up.cross(n).normalized()
    v_axis = n.cross(u_axis).normalized()
    a = math.radians(rot_deg)
    u_r = u_axis * math.cos(a) + v_axis * math.sin(a)
    v_r = -u_axis * math.sin(a) + v_axis * math.cos(a)
    me = obj.data
    lo, hi = mu.world_bbox(obj)
    c = (lo + hi) * 0.5
    uv = me.uv_layers.get("xyz") or me.uv_layers.new(name="xyz")
    mw = obj.matrix_world
    for poly in me.polygons:
        for li in poly.loop_indices:
            w = mw @ me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = (((w - c).dot(u_r)) / width_mm + 0.5 + off_u, ((w - c).dot(v_r)) / width_mm + 0.5 + off_v)
    return uv.name


def _outer_mask(obj, source):
    from mathutils.bvhtree import BVHTree
    me = obj.data
    attr = me.attributes.get("xyz_mask") or me.attributes.new("xyz_mask", 'FLOAT', 'POINT')
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(source, dg)
    s_inv = source.matrix_world.inverted()
    mw = obj.matrix_world
    vals = np.zeros(len(me.vertices))
    for i, v in enumerate(me.vertices):
        loc, nn, idx, dist = tree.find_nearest(s_inv @ (mw @ v.co), 0.6)
        if loc is not None and dist < 0.3:
            vals[i] = 1.0
    attr.data.foreach_set("value", vals)
    return int(vals.sum())


def _displace_group(image, uv_name, strengths, midlevel, alpha=False):
    ng = bpy.data.node_groups.get("Anaplast_XYZ_Displace")
    if ng:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new("Anaplast_XYZ_Displace", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = ng.nodes, ng.links
    gi = nodes.new("NodeGroupInput"); go = nodes.new("NodeGroupOutput")
    uv = nodes.new("GeometryNodeInputNamedAttribute"); uv.data_type = 'FLOAT_VECTOR'; uv.inputs["Name"].default_value = uv_name
    mask = nodes.new("GeometryNodeInputNamedAttribute"); mask.data_type = 'FLOAT'; mask.inputs["Name"].default_value = "xyz_mask"
    tex = nodes.new("GeometryNodeImageTexture"); tex.inputs["Image"].default_value = image; tex.extension = 'REPEAT'; tex.interpolation = 'Cubic'
    links.new(uv.outputs["Attribute"], tex.inputs["Vector"])
    sep = nodes.new("FunctionNodeSeparateColor"); sep.mode = 'RGB'
    links.new(tex.outputs["Color"], sep.inputs["Color"])
    total = None
    channels = (("Red",), (strengths[0],)) if alpha else (("Red", "Green", "Blue"), strengths)
    for ch, k in zip(*channels):
        sub = nodes.new("ShaderNodeMath"); sub.operation = 'SUBTRACT'; sub.inputs[1].default_value = midlevel
        links.new(sep.outputs[ch], sub.inputs[0])
        mul = nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = k
        links.new(sub.outputs[0], mul.inputs[0])
        if total is None:
            total = mul
        else:
            add = nodes.new("ShaderNodeMath"); add.operation = 'ADD'
            links.new(total.outputs[0], add.inputs[0]); links.new(mul.outputs[0], add.inputs[1]); total = add
    masked = nodes.new("ShaderNodeMath"); masked.operation = 'MULTIPLY'
    links.new(total.outputs[0], masked.inputs[0]); links.new(mask.outputs["Attribute"], masked.inputs[1])
    nrm = nodes.new("GeometryNodeInputNormal")
    scale = nodes.new("ShaderNodeVectorMath"); scale.operation = 'SCALE'
    links.new(nrm.outputs["Normal"], scale.inputs[0]); links.new(masked.outputs[0], scale.inputs["Scale"])
    setp = nodes.new("GeometryNodeSetPosition")
    links.new(gi.outputs["Geometry"], setp.inputs["Geometry"]); links.new(scale.outputs["Vector"], setp.inputs["Offset"])
    links.new(setp.outputs["Geometry"], go.inputs["Geometry"])
    return ng


class ANAPLAST_OT_xyz_displace(bpy.types.Operator):
    """Apply a Texturing.xyz (or any RGB multi-channel) displacement map to the outer skin of the built prototype"""
    bl_idname = "anaplast.xyz_displace"
    bl_label = "Apply Texturing.xyz Map"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.psd;*.tif;*.tiff;*.exr;*.png;*.jpg", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        return current_prototype(p) is not None and source_surface(p) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj, src = current_prototype(p), source_surface(p)
        img = bpy.data.images.get(p.xyz_image) if (p.xyz_image and not self.filepath) else None
        if img is None:
            try:
                img = bpy.data.images.load(self.filepath, check_existing=True)
            except RuntimeError as e:
                rep(self, {'ERROR'}, f"Could not load map ({e}). Use 1 · Load map first, or flatten the PSD and export 16-bit TIFF")
                return {'CANCELLED'}
            p.xyz_image = img.name
        if img.size[0] == 0:
            bpy.data.images.remove(img)
            rep(self, {'ERROR'}, "Map loaded empty — Blender reads only flattened PSDs. Export it as 16-bit TIFF from Photoshop (File → Save a Copy → TIFF)")
            return {'CANCELLED'}
        if "anaplast_home" in obj:
            bpy.ops.anaplast.lift_prototype()
        if p.xyz_subdivide:
            def setup(m):
                m.levels = 1; m.render_levels = 1; m.subdivision_type = 'SIMPLE'
            mu.apply_modifier(obj, 'SUBSURF', setup)
        n_outer = _outer_mask(obj, src)
        uv_name = _planar_uv(obj, src, p.xyz_width, p.xyz_off_u, p.xyz_off_v, p.xyz_rotation)
        alpha = (p.xyz_mode == 'ALPHA')
        depths = (p.xyz_depth, 0.0, 0.0) if alpha else (p.xyz_depth_r, p.xyz_depth_g, p.xyz_depth_b)
        ng = _displace_group(img, uv_name, depths, p.xyz_midlevel, alpha=alpha)
        before = np.empty(len(obj.data.vertices) * 3); obj.data.vertices.foreach_get("co", before)
        def setup(m):
            m.node_group = ng
        mu.apply_modifier(obj, 'NODES', setup)
        after = np.empty(len(obj.data.vertices) * 3); obj.data.vertices.foreach_get("co", after)
        d = np.linalg.norm((after - before).reshape(-1, 3), axis=1)
        for pg in obj.data.polygons:
            pg.use_smooth = True
        rep(self, {'INFO'}, f"{os.path.basename(self.filepath)} on {n_outer:,} outer vertices: rms {np.sqrt((d[d > 0] ** 2).mean()) * 1000 if (d > 0).any() else 0:.0f} µm, max {d.max() * 1000:.0f} µm; {len(obj.data.polygons):,} faces")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_xyz_displace)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_xyz_displace)


def _viewport(context, shading_type, matcap_cavity=False):
    if bpy.app.background:
        return
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            sh = area.spaces.active.shading
            sh.type = shading_type
            if shading_type == 'SOLID' and matcap_cavity:
                sh.light = 'MATCAP'; sh.show_cavity = True; sh.cavity_type = 'BOTH'; sh.color_type = 'MATERIAL'


class ANAPLAST_OT_xyz_load(bpy.types.Operator):
    """Step 1 — load the Texturing.xyz map and show where it lands on the sculpt (adjust Width / Shift / Rotate, click again to refresh)"""
    bl_idname = "anaplast.xyz_load"
    bl_label = "1 · Load map & preview placement"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.psd;*.tif;*.tiff;*.exr;*.png;*.jpg", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        return current_prototype(p) is not None and source_surface(p) is not None

    def invoke(self, context, event):
        p = context.scene.anaplast
        if p.xyz_image and bpy.data.images.get(p.xyz_image):
            return self.execute(context)          # already loaded: just refresh the preview
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj, src = current_prototype(p), source_surface(p)
        img = bpy.data.images.get(p.xyz_image) if p.xyz_image else None
        if img is None:
            try:
                img = bpy.data.images.load(self.filepath, check_existing=True)
            except RuntimeError as e:
                rep(self, {'ERROR'}, f"Could not load map ({e}). Flatten the PSD and save as 16-bit TIFF")
                return {'CANCELLED'}
            if img.size[0] == 0:
                bpy.data.images.remove(img)
                rep(self, {'ERROR'}, "Map loaded empty — layered PSD. Flatten it, or export 16-bit TIFF")
                return {'CANCELLED'}
            p.xyz_image = img.name
        uv_name = _planar_uv(obj, src, p.xyz_width, p.xyz_off_u, p.xyz_off_v, p.xyz_rotation)
        mat = bpy.data.materials.get("Anaplast_XYZ_Preview")
        if mat is None:
            mat = bpy.data.materials.new("Anaplast_XYZ_Preview"); mat.use_nodes = True
            nt = mat.node_tree
            tex = nt.nodes.new("ShaderNodeTexImage"); tex.name = "xyz_tex"
            uvn = nt.nodes.new("ShaderNodeUVMap"); uvn.uv_map = uv_name
            nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
            nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        mat.node_tree.nodes["xyz_tex"].image = img
        me = obj.data
        if not me.materials:
            me.materials.append(mat)
        else:
            me.materials[0] = mat
        _viewport(context, 'MATERIAL')
        rep(self, {'INFO'}, f"{img.name} ({img.size[0]}×{img.size[1]}) shown on the prosthesis. Adjust Width / Shift / Rotate and click again; then 2 · Apply")
        return {'FINISHED'}


_old_exec = ANAPLAST_OT_xyz_displace.execute
_old_invoke = ANAPLAST_OT_xyz_displace.invoke


def _invoke2(self, context, event):
    p = context.scene.anaplast
    if p.xyz_image and bpy.data.images.get(p.xyz_image):
        self.filepath = bpy.data.images[p.xyz_image].filepath
        return self.execute(context)
    return _old_invoke(self, context, event)


def _exec2(self, context):
    r = _old_exec(self, context)
    if 'FINISHED' in r:
        from .shell import current_prototype
        obj = current_prototype(context.scene.anaplast)
        grey = mu.get_material("Anaplast_grey", (0.75, 0.75, 0.75))
        if obj.data.materials:
            obj.data.materials[0] = grey
        _viewport(context, 'SOLID', matcap_cavity=True)
    return r


ANAPLAST_OT_xyz_displace.invoke = _invoke2
ANAPLAST_OT_xyz_displace.execute = _exec2
ANAPLAST_OT_xyz_displace.bl_label = "2 · Apply displacement"

_old_reg_xyz, _old_unreg_xyz = register, unregister


def register():
    _old_reg_xyz()
    bpy.utils.register_class(ANAPLAST_OT_xyz_load)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_xyz_load)
    _old_unreg_xyz()
