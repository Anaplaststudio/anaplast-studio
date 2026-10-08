"""Park the moving object beside the target, facing the same way, so pairs are easy to click."""
import bpy
from mathutils import Matrix, Vector
from ..utils import mesh as mu
from .report import rep
from .align import _points

KEY = "anaplast_park"


class ANAPLAST_OT_side_by_side(bpy.types.Operator):
    """Move the Moving object next to the Target (same orientation, clear gap) for marking pairs — click again to put it back, points included"""
    bl_idname = "anaplast.side_by_side"
    bl_label = "Side by side"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype
        p = context.scene.anaplast
        return (p.align_moving or current_prototype(p)) is not None and (p.align_target or p.face_scan_obj) is not None

    def execute(self, context):
        p = context.scene.anaplast
        from .shell import current_prototype
        mov = p.align_moving or current_prototype(p)
        tgt = p.align_target or p.face_scan_obj
        if KEY in mov:
            back = Matrix.Translation(-Vector(mov[KEY]))
            mov.matrix_world = back @ mov.matrix_world
            for o in _points("AL_S_"):
                o.matrix_world = back @ o.matrix_world
            del mov[KEY]
            mu.frame_view(context, tgt)
            rep(self, {'INFO'}, "Back in place — source points moved with it. Now Align by Points, then Refine")
            return {'FINISHED'}
        lo_t, hi_t = mu.world_bbox(tgt)
        lo_m, hi_m = mu.world_bbox(mov)
        gap = 0.25 * max(20.0, (hi_t - lo_t).x)
        shift = Vector(((hi_t.x - lo_m.x) + gap, 0.0, 0.0))      # park to the patient's right of the target
        off = Matrix.Translation(shift)
        mov.matrix_world = off @ mov.matrix_world
        for o in _points("AL_S_"):
            o.matrix_world = off @ o.matrix_world
        mov[KEY] = list(shift)
        for o in context.view_layer.objects:
            o.select_set(False)
        mov.select_set(True); tgt.select_set(True)
        context.view_layer.objects.active = tgt
        if not bpy.app.background:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    if region:
                        with context.temp_override(area=area, region=region):
                            bpy.ops.view3d.view_selected()
        rep(self, {'INFO'}, f"{mov.name} parked beside {tgt.name} — click matching pairs, then press this again to put it back")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_side_by_side)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_side_by_side)
