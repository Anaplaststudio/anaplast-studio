"""Surgical guide — extraoral implant planning on the aligned CBCT / cast, and a sleeve guide."""
import bpy
import bmesh
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep

IMPLANT_COLLECTION = "Anaplast_Implants"


def implants():
    col = bpy.data.collections.get(IMPLANT_COLLECTION)
    if not col:
        return []
    return sorted((o for o in col.objects if o.name.startswith("Implant_")), key=lambda o: int(o.name.split("_")[-1]))


def support_surface(p):
    """Surface the implant is placed on: bone if we have it, else the tissue."""
    return p.cbct_obj or p.cast_obj or p.face_scan_obj


def implant_axis(obj):
    """World-space (top point, outward unit axis) of an implant object."""
    mw = obj.matrix_world
    return mw.translation.copy(), (mw.to_3x3() @ Vector((0, 0, 1))).normalized()


class ANAPLAST_OT_add_implant(bpy.types.Operator):
    """Place an implant at the 3D cursor, axis along the surface normal (bone if CBCT is assigned, else tissue). Rotate/move it afterwards if needed"""
    bl_idname = "anaplast.add_implant"
    bl_label = "Add Implant at Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return support_surface(context.scene.anaplast) is not None

    def execute(self, context):
        p = context.scene.anaplast
        surf = support_surface(p)
        cur = context.scene.cursor.location.copy()
        tree = mu.oriented_scan_bvh(surf, p.scan_flip, roi=(cur - Vector((30,) * 3), cur + Vector((30,) * 3)))
        loc, nrm, idx, dist = tree.find_nearest(cur)
        if loc is None:
            rep(self, {'ERROR'}, "Cursor is not near the support surface")
            return {'CANCELLED'}
        col = mu.get_collection(context.scene, IMPLANT_COLLECTION)
        idx = len(implants()) + 1
        d, L = p.implant_diameter, p.implant_length
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=d * 0.5, radius2=d * 0.5, depth=L)
        bmesh.ops.translate(bm, vec=(0, 0, -L * 0.5), verts=bm.verts)          # top at origin, body along -Z (into bone)
        obj = mu.new_mesh_object(f"Implant_{idx}", bm, col)
        obj.matrix_world = Matrix.Translation(loc) @ nrm.to_track_quat('Z', 'Y').to_matrix().to_4x4()
        obj["anaplast_part"] = "IMPLANT"
        obj["diameter"] = d
        obj["length"] = L
        obj.show_in_front = True
        obj.data.materials.append(mu.get_material(f"Anaplast_{mu.PALETTE[3]}", mu.PALETTE[3]))
        rep(self, {'INFO'}, f"Implant_{idx}: Ø{d:.2f} × {L:.1f} mm on {surf.name}. Rotate (R) / move (G) if the axis needs adjusting")
        return {'FINISHED'}


class ANAPLAST_OT_make_guide(bpy.types.Operator):
    """Build a sleeve guide: a shell of the chosen support surface around the implants, with a drill sleeve hole along each implant axis"""
    bl_idname = "anaplast.make_guide"
    bl_label = "Make Surgical Guide"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(implants()) > 0

    def execute(self, context):
        p = context.scene.anaplast
        surf = p.cbct_obj if (p.guide_support == 'BONE' and p.cbct_obj) else (p.cast_obj or p.face_scan_obj)
        if surf is None:
            rep(self, {'ERROR'}, "No support surface assigned (CBCT bone, cast, or face scan)")
            return {'CANCELLED'}
        imps = implants()
        tops = [implant_axis(o)[0] for o in imps]
        lo = Vector((min(t.x for t in tops), min(t.y for t in tops), min(t.z for t in tops)))
        hi = Vector((max(t.x for t in tops), max(t.y for t in tops), max(t.z for t in tops)))
        R = p.guide_radius
        bm = mu.scan_region_bmesh(surf, (lo, hi), margin=R + 2.0, flip=p.scan_flip)
        far = [f for f in bm.faces if min((f.calc_center_median() - t).length for t in tops) > R]
        bmesh.ops.delete(bm, geom=far, context='FACES')
        if len(bm.faces) < 10:
            bm.free()
            rep(self, {'ERROR'}, "Support surface has no geometry near the implants")
            return {'CANCELLED'}
        mu.keep_largest_island(bm)
        col = mu.get_collection(context.scene, "Anaplast_Guide")
        old = bpy.data.objects.get("Surgical_Guide")
        if old:
            mu.delete_object(old)
        guide = mu.new_mesh_object("Surgical_Guide", bm, col)
        # shell grows OUTWARD from the surface (normals point out of the head): offset +1
        def setup(m):
            m.thickness = p.guide_thickness
            m.offset = 1.0
            m.use_even_offset = True
            m.use_rim = True
            m.use_quality_normals = True
        mu.apply_modifier(guide, 'SOLIDIFY', setup)
        mu.cleanup_mesh(guide)
        # sleeve holes along each implant axis
        for o in imps:
            top, ax = implant_axis(o)
            h = p.guide_thickness * 2 + 20.0
            cyl = mu.make_cylinder("anaplast_sleeve", Vector((0, 0, 0)), p.sleeve_diameter * 0.5, h, col, segments=48)
            cyl.matrix_world = Matrix.Translation(top + ax * (p.guide_thickness * 0.5)) @ ax.to_track_quat('Z', 'Y').to_matrix().to_4x4()
            mu.apply_boolean(guide, cyl, 'DIFFERENCE')
            mu.delete_object(cyl)
        mu.cleanup_mesh(guide)
        mu.heal_small_holes(guide)
        guide["anaplast_part"] = "GUIDE"
        n, nonman, boundary, loose = mu.mesh_stats(guide)
        rep(self, {'INFO'}, f"Surgical_Guide: {len(imps)} sleeves Ø{p.sleeve_diameter:.1f} mm, {p.guide_thickness:.1f} mm shell on {surf.name} ({n} faces, {'closed' if boundary == 0 else str(boundary) + ' open edges'})")
        return {'FINISHED'}


class ANAPLAST_OT_align_cbct_setup(bpy.types.Operator):
    """Set Moving = CBCT bone and Target = the scan, ready for point pairs + ICP in this panel"""
    bl_idname = "anaplast.align_cbct_setup"
    bl_label = "Set Up CBCT → Scan Alignment"

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.cbct_obj is not None and (p.face_scan_obj or p.cast_obj)

    def execute(self, context):
        p = context.scene.anaplast
        p.align_moving = p.cbct_obj
        p.align_target = p.face_scan_obj or p.cast_obj
        rep(self, {'INFO'}, f"Moving = {p.cbct_obj.name}, Target = {p.align_target.name}. Place 3–5 point pairs on matching bony/skin landmarks, Align by Points, then Refine")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_add_implant, ANAPLAST_OT_make_guide, ANAPLAST_OT_align_cbct_setup)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
