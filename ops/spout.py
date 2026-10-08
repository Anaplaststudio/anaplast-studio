"""Click on the cap (or base) to add a spout: a round channel from the cavity out through the wall."""
import bpy
import bmesh
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep


class ANAPLAST_OT_mold_spout(bpy.types.Operator):
    """Click where the spout should enter the mold: a channel (Spout Ø) is cut straight through the part at that spot, along the surface normal. Esc cancels"""
    bl_idname = "anaplast.mold_spout"
    bl_label = "Add spout (click)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (bpy.data.objects.get("Mold_Cap") is not None or bpy.data.objects.get("Mold_Base") is not None) and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set("Spout: click on the mold where it should enter · ESC cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        from .mold_auto import cut
        p = context.scene.anaplast
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            from bpy_extras import view3d_utils
            region, rv3d = context.region, context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            w, wn, obj = mu.ray_cast_clipped(context, origin, direction)
            if w is None or obj is None or not obj.name.startswith("Mold_"):
                return {'RUNNING_MODAL'}
            # the sprue runs along the mold's opening axis: from the cavity out through the top of the cap
            # (or the bottom of the base) — narrow at the cavity, opening to 2.5 Ø at the outside. Which face
            # you clicked does not matter; the axis comes from the mold, not the clicked surface.
            up_m = Vector(obj.get("mold_up", (0.0, 0.0, 1.0))).normalized()
            axis = up_m if obj.name == "Mold_Cap" else -up_m
            lo, hi = mu.world_bbox(obj)
            length = abs((hi - lo).dot(axis)) + 12.0
            bm = bmesh.new()
            bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=p.m_spout_diameter * 0.5, radius2=p.m_spout_outer * 0.5, depth=length)
            bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, 0.0, length * 0.5 - 3.0))      # narrow end 3 mm past the click, into the cavity
            bmesh.ops.triangulate(bm, faces=bm.faces)
            col = obj.users_collection[0] if obj.users_collection else context.scene.collection
            cyl = mu.new_mesh_object("Mold_Spout_tmp", bm, col)
            cyl.matrix_world = Matrix.Translation(w) @ axis.to_track_quat('Z', 'Y').to_matrix().to_4x4()
            from . import mold_volume
            volume_ledger, before_volume = mold_volume.before_channel_edit(context.scene,obj)
            cut(obj, cyl, 'DIFFERENCE')
            mold_volume.after_channel_edit(context.scene,obj,volume_ledger,before_volume)
            mu.delete_object(cyl)
            self._finish(context)
            rep(self, {'INFO'}, f"Sprue cut through {obj.name}: Ø {p.m_spout_diameter:.1f} mm at the cavity, Ø {p.m_spout_diameter * 2.5:.1f} mm at the outside")
            return {'FINISHED'}
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


def register():
    bpy.utils.register_class(ANAPLAST_OT_mold_spout)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_spout)
