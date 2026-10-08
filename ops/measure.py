"""Measure between two clicked points (on any meshes)."""
import bpy
import bmesh
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep

COL = "Anaplast_Measure"


class ANAPLAST_OT_measure(bpy.types.Operator):
    """Click two points on any surfaces (they can be on different meshes): a line and the distance in mm are drawn. Esc cancels"""
    bl_idname = "anaplast.measure"
    bl_label = "Measure (click 2 points)"
    bl_options = {'REGISTER', 'UNDO'}

    _first = None

    def invoke(self, context, event):
        self._first = None
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set("Measure: click the FIRST point · ESC cancel")
        return {'RUNNING_MODAL'}

    def _hit(self, context, event):
        from bpy_extras import view3d_utils
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        w, wn, obj = mu.ray_cast_clipped(context, origin, direction)
        return w

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._hit(context, event)
            if hit is None:
                return {'RUNNING_MODAL'}
            if self._first is None:
                self._first = hit.copy()
                context.area.header_text_set("Measure: click the SECOND point · ESC cancel")
                return {'RUNNING_MODAL'}
            a, b = self._first, hit
            d = (b - a).length
            col = mu.get_collection(context.scene, COL)
            n = len([o for o in col.objects if o.name.startswith("Measure_")]) + 1
            bm = bmesh.new()
            v1, v2 = bm.verts.new(a), bm.verts.new(b)
            bm.edges.new((v1, v2))
            line = mu.new_mesh_object(f"Measure_{n}", bm, col)
            line.color = (1.0, 0.85, 0.1, 1.0)
            line.show_in_front = True
            line.hide_select = True
            cu = bpy.data.curves.new(f"Measure_{n}_label", type='FONT')
            cu.body = f"{d:.2f} mm"
            cu.size = max(2.0, d * 0.08)
            cu.align_x = 'CENTER'
            txt = bpy.data.objects.new(f"Measure_{n}_label", cu)
            txt.location = (a + b) * 0.5 + Vector((0, 0, cu.size * 0.6))
            txt.show_in_front = True
            txt.hide_select = True
            txt.color = (1.0, 0.85, 0.1, 1.0)
            col.objects.link(txt)
            # face the current view
            rv3d = context.region_data
            if rv3d is not None:
                txt.rotation_euler = rv3d.view_rotation.to_euler()
            self._finish(context)
            rep(self, {'INFO'}, f"Distance: {d:.2f} mm")
            return {'FINISHED'}
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


class ANAPLAST_OT_measure_tool(bpy.types.Operator):
    """Blender's own Measure tool, with snapping to surfaces on by default: click-drag to measure, Ctrl for angles, Delete removes a ruler"""
    bl_idname = "anaplast.measure_tool"
    bl_label = "Measure (Blender tool)"

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        ts = context.scene.tool_settings
        # snapping defaults: to faces, projecting onto the surface you point at
        ts.use_snap = True
        try:
            ts.snap_elements = {'FACE'}
        except Exception:
            try:
                ts.snap_elements_base = {'FACE'}
            except Exception:
                pass
        for attr, val in (("use_snap_self", True), ("use_snap_align_rotation", False), ("snap_target", 'CLOSEST'),
                          ("use_snap_translate", True)):
            try:
                setattr(ts, attr, val)
            except Exception:
                pass
        ok = False
        for area in context.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            with context.temp_override(area=area, region=region):
                try:
                    bpy.ops.wm.tool_set_by_id(name="builtin.measure")
                    ok = True
                except Exception:
                    pass
            break
        rep(self, {'INFO'}, "Measure tool active with face snapping — click-drag between two points; Ctrl snaps angles; hover a ruler and press Delete to remove it"
            if ok else "Could not switch tools — hover the 3D view and try again")
        return {'FINISHED'}


class ANAPLAST_OT_measure_clear(bpy.types.Operator):
    """Remove all measurements"""
    bl_idname = "anaplast.measure_clear"
    bl_label = "Clear measurements"

    def execute(self, context):
        col = bpy.data.collections.get(COL)
        n = 0
        import json
        for o in list(context.scene.objects):
            if (col and o.name in col.objects) or COL in json.loads(o.get('anaplast_legacy_collections','[]')):
                bpy.data.objects.remove(o, do_unlink=True); n += 1
        # Blender's rulers live in the scene's annotation layer "RulerData3D"
        gp = context.scene.grease_pencil
        rulers = 0
        if gp is not None:
            for layer in list(gp.layers):
                if layer.info == "RulerData3D":
                    rulers = sum(len(fr.strokes) for fr in layer.frames)
                    gp.layers.remove(layer)
        rep(self, {'INFO'}, f"Cleared {rulers} ruler(s)" + (f" and {n} old measurement object(s)" if n else ""))
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_measure)
    bpy.utils.register_class(ANAPLAST_OT_measure_tool)
    bpy.utils.register_class(ANAPLAST_OT_measure_clear)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_measure_clear)
    bpy.utils.unregister_class(ANAPLAST_OT_measure_tool)
    bpy.utils.unregister_class(ANAPLAST_OT_measure)
