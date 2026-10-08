"""A cap over any closed object — a scanned base with a wax-up, a printed part — from the viewing direction."""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep


class ANAPLAST_OT_cap_from_object(bpy.types.Operator):
    """Make a cap over the selected CLOSED object (e.g. a scan of a base with the wax-up on it). Select the object — or, in Edit Mode, select the faces of its inner surface (land + sculpt): the cap then covers exactly that region and takes its full impression. Look from the direction the cap should lift off, then click. Cap = block − object, at the mold resolution"""
    bl_idname = "anaplast.cap_from_object"
    bl_label = "Cap over selected object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH' and not o.name.startswith("Mold_")

    def execute(self, context):
        from .mold_auto import view_axes, convex_hull_2d, offset_polygon, prism, cut
        p = context.scene.anaplast
        src = context.active_object
        if not mu.is_closed(src):
            rep(self, {'ERROR'}, f"{src.name} is not watertight — the cap needs a closed object to cast over")
            return {'CANCELLED'}
        col = mu.get_collection(context.scene, mu.MOLD_COLLECTION)
        old = bpy.data.objects.get("Mold_Cap")
        if old:
            mu.delete_object(old)
        up, right, fwd = view_axes(context)
        frame = (up, right, fwd)
        me = src.data; n = len(me.vertices)
        co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        M = np.array(src.matrix_world); W = co @ M[:3, :3].T + M[:3, 3]
        U, R, F = np.array(up), np.array(right), np.array(fwd)
        # if faces are selected on the object, THAT is the inner surface the cap must take the impression of:
        # the footprint is the selection's outline and the cap reaches down to its lowest point
        if context.mode == 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='OBJECT')
        sel = np.zeros(n, dtype=bool)
        flags = np.empty(len(me.polygons), dtype=bool); me.polygons.foreach_get("select", flags)
        if flags.any() and not flags.all():
            for pg in me.polygons:
                if pg.select:
                    sel[pg.vertices] = True
        self._selected = bool(sel.any())
        used = W[sel] if self._selected else W
        h = used @ U; x = used @ R; y = used @ F
        step = max(1, len(used) // 6000)
        hull = convex_hull_2d(list(zip(x[::step], y[::step])))
        outline = offset_polygon(hull, 2.0)                                    # the cap follows the (selected) outline
        z_lo = float(h.min()) - 1.0 if self._selected else float(np.percentile(h, 2))
        z_hi = float(h.max()) + p.m_cap_height
        cap = prism("Mold_Cap", outline, z_lo, z_hi, frame, col)
        obj_copy = mu.duplicate_object(src, "Mold_CapSrc_tmp", col)
        v_src = mu.voxel_remesh(obj_copy, p.m_voxel)                            # one clean surface before the boolean
        cut(cap, obj_copy, 'DIFFERENCE')
        mu.delete_object(obj_copy)
        mu.keep_main_body(cap)
        v_cap = mu.voxel_remesh(cap, p.m_voxel, adaptivity=0.5 * p.m_voxel)
        for pg in cap.data.polygons:
            pg.use_smooth = True
        cap["anaplast_part"] = "MOLD"; cap["mold_up"] = list(up); cap["mold_right"] = list(right)
        cap.color = (0.35, 0.55, 0.85, 1.0)
        rep(self, {'INFO'}, f"Cap over {src.name}{' (selected inner surface)' if self._selected else ''}: {len(cap.data.polygons):,} faces ({'closed' if mu.is_closed(cap) else 'OPEN'}), remeshed @ {v_cap:.2f} mm. Add a spout with the click tool if needed")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_cap_from_object)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_cap_from_object)
