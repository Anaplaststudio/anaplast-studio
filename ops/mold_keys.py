import bpy
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep
from .mold_split import split_planes, mold_parts


def _candidates(plane, block, pros, radius, grid=17):
    """Grid of points on the plane, inside the block footprint, clear of the cavity."""
    lo, hi = mu.world_bbox(block)
    half = (hi - lo).length * 0.5
    pts = []
    mw = plane.matrix_world
    inset = radius * 1.5
    for i in range(grid):
        for j in range(grid):
            u = (i / (grid - 1) - 0.5) * 2 * half
            v = (j / (grid - 1) - 0.5) * 2 * half
            w = mw @ Vector((u, v, 0.0))
            if not all(lo[k] + inset < w[k] < hi[k] - inset for k in range(3)):
                continue
            if mu.distance_to_surface(pros, w) < radius * 2.0:
                continue
            pts.append(w)
    return pts


def _pick_spread(points, n, pros):
    """Greedy farthest-point selection, seeded by the point farthest from the cavity."""
    if not points:
        return []
    chosen = [max(points, key=lambda q: mu.distance_to_surface(pros, q))]
    while len(chosen) < n and len(chosen) < len(points):
        best = max(points, key=lambda q: min((q - c).length for c in chosen))
        if best in chosen:
            break
        chosen.append(best)
    return chosen


class ANAPLAST_OT_mold_keys(bpy.types.Operator):
    """Tool 4 — add spherical registration keys on every split plane (bump on + side, socket on − side)"""
    bl_idname = "anaplast.mold_keys"
    bl_label = "Add Registration Keys"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Mold_A") is not None

    def execute(self, context):
        from .explode import ensure_collapsed
        ensure_collapsed(context.scene)
        p = context.scene.anaplast
        col = mu.get_collection(context.scene)
        block = bpy.data.objects["Mold_Block"]
        pros = p.prosthesis_obj
        parts = mold_parts(col)
        r = p.key_diameter * 0.5
        total = 0

        for plane in split_planes(col):
            n = plane.matrix_world.to_3x3() @ Vector((0, 0, 1))
            n.normalize()
            pts = _pick_spread(_candidates(plane, block, pros, r), p.key_count, pros)
            for pt in pts:
                plus = [o for o in parts if mu.point_inside(o, pt + n * (r * 0.5))]
                minus = [o for o in parts if mu.point_inside(o, pt - n * (r * 0.5))]
                if not plus or not minus:
                    continue
                bump = mu.make_sphere("anaplast_key", pt, r, col)
                socket = mu.make_sphere("anaplast_key_s", pt, r + p.key_clearance, col)
                mu.apply_boolean(plus[0], bump, 'UNION')
                mu.apply_boolean(minus[0], socket, 'DIFFERENCE')
                mu.delete_object(bump)
                mu.delete_object(socket)
                total += 1

        for o in parts:
            mu.cleanup_mesh(o)
        rep(self, {'INFO'}, f"Placed {total} keys")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_mold_keys)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_keys)
