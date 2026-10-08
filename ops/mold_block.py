import bpy
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep


class ANAPLAST_OT_mold_block(bpy.types.Operator):
    """Tool 2 — build the mold block around the sculpt and subtract the cavity"""
    bl_idname = "anaplast.mold_block"
    bl_label = "Make Mold Block"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.anaplast.prosthesis_obj is not None

    def execute(self, context):
        p = context.scene.anaplast
        col = mu.get_collection(context.scene)
        pros = p.prosthesis_obj

        # remove a previous block/parts for this case
        for o in list(col.objects):
            if o.name.startswith("Mold_"):
                mu.delete_object(o)

        lo, hi = mu.world_bbox(pros)
        off = Vector((p.block_offset,) * 3)
        if p.block_shape == 'BOX':
            block = mu.make_box("Mold_Block", lo - off, hi + off, col)
        else:
            center = (lo + hi) * 0.5
            r = max(hi.x - lo.x, hi.y - lo.y) * 0.5 + p.block_offset
            h = (hi.z - lo.z) + 2 * p.block_offset
            block = mu.make_cylinder("Mold_Block", center, r, h, col)

        cutter = pros
        temp_cutter = False
        if p.simplify_prosthesis and len(pros.data.polygons) > p.boolean_max_faces:
            cutter = mu.duplicate_object(pros, "anaplast_pros_lite", col)
            ratio = p.boolean_max_faces / len(pros.data.polygons)
            def dec(m):
                m.decimate_type = 'COLLAPSE'; m.ratio = ratio; m.use_collapse_triangulate = True
            mu.apply_modifier(cutter, 'DECIMATE', dec)
            temp_cutter = True
        mu.apply_boolean(block, cutter, 'DIFFERENCE', solver=p.boolean_solver)
        if temp_cutter:
            mu.delete_object(cutter)
        if p.include_cast and p.cast_obj is not None:
            solid, temp = mu.solid_copy(p.cast_obj, p.scan_thickness, p.scan_flip, roi=mu.world_bbox(block), max_faces=p.boolean_max_faces)
            mu.apply_boolean(block, solid, 'DIFFERENCE', solver=p.boolean_solver)
            if temp:
                mu.delete_object(solid)

        mu.cleanup_mesh(block)
        block["anaplast_part"] = "BLOCK"
        n, nonman, boundary, loose = mu.mesh_stats(block)
        if nonman or boundary:
            rep(self, {'WARNING'}, f"Block has {nonman} non-manifold / {boundary} boundary edges — check the prosthesis mesh is watertight")
        else:
            rep(self, {'INFO'}, f"Mold block created ({n} faces)")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_mold_block)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_block)
