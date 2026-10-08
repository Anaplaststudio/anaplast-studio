"""Bar & substructure — retention parts on the planned implants."""
import bpy
import bmesh
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep
from .guide import implants, implant_axis

BAR_COLLECTION = "Anaplast_Bar"


def _cyl_between(name, a, b, radius, col, segments=32):
    d = b - a
    L = d.length
    cyl = mu.make_cylinder(name, Vector((0, 0, 0)), radius, L, col, segments=segments)
    cyl.matrix_world = Matrix.Translation((a + b) * 0.5) @ d.normalized().to_track_quat('Z', 'Y').to_matrix().to_4x4()
    return cyl


class ANAPLAST_OT_make_bar(bpy.types.Operator):
    """Round bar joining the implant abutments, plus abutment posts; clip housings or magnet keepers per the retention choice"""
    bl_idname = "anaplast.make_bar"
    bl_label = "Make Bar / Retention"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(implants()) >= 1

    def execute(self, context):
        p = context.scene.anaplast
        imps = implants()
        col = mu.get_collection(context.scene, BAR_COLLECTION)
        for o in list(col.objects):
            mu.delete_object(o)
        posts, tops = [], []
        for o in imps:
            top, ax = implant_axis(o)
            hi = top + ax * p.bar_height
            posts.append(_cyl_between("anaplast_post", top - ax * 0.5, hi, p.abutment_diameter * 0.5, col))
            tops.append(hi)
        parts = posts[:]
        if p.retention == 'BAR' and len(tops) >= 2:
            # bar along the implant order (left→right by X)
            order = sorted(range(len(tops)), key=lambda i: tops[i].x)
            for i, j in zip(order[:-1], order[1:]):
                a, b = tops[i], tops[j]
                parts.append(_cyl_between("anaplast_bar", a, b, p.bar_diameter * 0.5, col))
                # sphere joints so the segments blend
                parts.append(mu.make_sphere("anaplast_joint", a, p.bar_diameter * 0.5, col, subdiv=2))
                parts.append(mu.make_sphere("anaplast_joint", b, p.bar_diameter * 0.5, col, subdiv=2))
                # clip housing at the midpoint of each span (a block the clip sits in; goes into the acrylic substructure)
                mid = (a + b) * 0.5
                d = (b - a).normalized()
                w = p.bar_diameter + 2 * p.housing_wall
                hb = mu.make_box(f"Bar_Housing_{len(parts)}", Vector((-p.housing_length * 0.5, -w * 0.5, -w * 0.5)), Vector((p.housing_length * 0.5, w * 0.5, w * 0.5)), col)
                hb.matrix_world = Matrix.Translation(mid) @ d.to_track_quat('X', 'Z').to_matrix().to_4x4()
                hb["anaplast_part"] = "HOUSING"
                hb.data.materials.append(mu.get_material(f"Anaplast_{mu.PALETTE[1]}", mu.PALETTE[1]))
        elif p.retention == 'MAGNET':
            for hi_pt, o in zip(tops, imps):
                top, ax = implant_axis(o)
                mag = _cyl_between(f"Magnet_{o.name.split('_')[-1]}", hi_pt, hi_pt + ax * p.magnet_height, p.magnet_diameter * 0.5, col)
                mag["anaplast_part"] = "MAGNET"
                mag.data.materials.append(mu.get_material(f"Anaplast_{mu.PALETTE[4]}", mu.PALETTE[4]))
        # union bar + posts into one printable/castable part
        base = parts[0]
        for extra in parts[1:]:
            mu.apply_boolean(base, extra, 'UNION')
            mu.delete_object(extra)
        mu.cleanup_mesh(base)
        base.name = "Bar_Assembly"
        base.data.name = "Bar_Assembly"
        base["anaplast_part"] = "BAR"
        base.data.materials.append(mu.get_material("Anaplast_grey", (0.75, 0.75, 0.75)))
        n, nonman, boundary, loose = mu.mesh_stats(base)
        what = "bar + posts + housings" if p.retention == 'BAR' else "posts + magnet keepers"
        rep(self, {'INFO'}, f"Bar_Assembly ({what}): {n} faces, {'closed' if boundary == 0 else str(boundary) + ' open edges'}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_make_bar)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_make_bar)
