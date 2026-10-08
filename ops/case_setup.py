import bpy
from .report import rep


class ANAPLAST_OT_case_setup(bpy.types.Operator):
    """Set scene units to millimetres (1 Blender unit = 1 mm) and organize existing case objects"""
    bl_idname = "anaplast.case_setup"
    bl_label = "Set Up Case"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import time
        if not context.scene.anaplast.case_date:
            context.scene.anaplast.case_date = time.strftime("%Y-%m-%d")
        us = context.scene.unit_settings
        us.system = 'METRIC'
        us.scale_length = 0.001
        us.length_unit = 'MILLIMETERS'
        try:
            context.preferences.inputs.use_auto_perspective = False
        except Exception:
            pass
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                ov=area.spaces.active.overlay
                ov.show_stats=True; ov.show_text=True
                ov.show_axis_x=False; ov.show_axis_y=False
                ov.show_object_origins=False; ov.show_object_origins_all=False
        from .scene_helpers import organize
        removed = 0
        for name, kind in (("Cube", 'MESH'), ("Camera", 'CAMERA'), ("Light", 'LIGHT')):
            o = bpy.data.objects.get(name)
            if o and o.type == kind:
                bpy.data.objects.remove(o)
                removed += 1
        organize(context.scene)
        if not bpy.app.background:
            try:
                bpy.ops.anaplast.no_auto_perspective()
            except Exception:
                pass
        from .overlay import disable_crosshair
        disable_crosshair(context)
        rep(self, {'INFO'}, f"Units set to mm; default objects removed ({removed}); statistics on; axes and object origins hidden")
        return {'FINISHED'}


class ANAPLAST_OT_preview_solid(bpy.types.Operator):
    """Make a visible thickened copy of the cast to check it grows INTO the head (delete it afterwards)"""
    bl_idname = "anaplast.preview_solid"
    bl_label = "Preview Thickened Cast"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.anaplast.cast_obj is not None

    def execute(self, context):
        from ..utils import mesh as mu
        p = context.scene.anaplast
        old = bpy.data.objects.get("Cast_Solid_Preview")
        if old:
            mu.delete_object(old)
        roi = mu.world_bbox(p.prosthesis_obj) if p.prosthesis_obj else None
        solid, temp = mu.solid_copy(p.cast_obj, p.scan_thickness, p.scan_flip, name="Cast_Solid_Preview", roi=roi, max_faces=p.boolean_max_faces)
        if not temp:
            rep(self, {'INFO'}, "Cast is already a closed solid — nothing to thicken")
        else:
            rep(self, {'INFO'}, "Cast_Solid_Preview made — if it grows toward you, tick Flip and run again")
        return {'FINISHED'}


class ANAPLAST_OT_fidelity_report(bpy.types.Operator):
    """Face count and edge length (= resolution) of every case object, so you can see exactly where detail changes"""
    bl_idname = "anaplast.fidelity_report"
    bl_label = "Fidelity Report"

    def execute(self, context):
        from ..utils import mesh as mu
        p = context.scene.anaplast
        rows = []
        seen = set()
        named = [("Face scan", p.face_scan_obj), ("Cast", p.cast_obj), ("CBCT", p.cbct_obj), ("Sculpt", p.prosthesis_obj)]
        for label, o in named:
            if o is not None and o.name not in seen:
                seen.add(o.name); rows.append((label, o))
        for name in ("Mirror_FaceScan", "Shell_Prosthesis", "Prosthesis_Fitted", "Prosthesis_Hollow", "SSM_Fit", "Surgical_Guide", "Bar_Assembly",
                     "Mold_Block", "Mold_A", "Mold_B", "Mold_C", "Mold_D"):
            o = bpy.data.objects.get(name)
            if o is not None and o.name not in seen:
                seen.add(o.name); rows.append((name, o))
        lines = []
        for label, o in rows:
            n, mean, lo, hi = mu.fidelity(o)
            lines.append(f"{label:<18} {n:>10,} faces   edge {mean:.3f} mm  (5–95%: {lo:.3f}–{hi:.3f})   {'smooth' if o.data.polygons and o.data.polygons[0].use_smooth else 'flat'}")
        text = "\n".join(lines) if lines else "No case objects yet"
        txt = bpy.data.texts.get("Anaplast_Fidelity") or bpy.data.texts.new("Anaplast_Fidelity")
        txt.clear(); txt.write(text + "\n")
        def draw(self_, ctx):
            for ln in lines or ["No case objects yet"]:
                self_.layout.label(text=ln)
        if not bpy.app.background and getattr(context, "window", None) is not None:
            context.window_manager.popup_menu(draw, title="Fidelity — smaller edge = finer detail", icon='INFO')
        rep(self, {'INFO'}, "Fidelity report shown (also saved as text 'Anaplast_Fidelity')")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_case_setup)
    bpy.utils.register_class(ANAPLAST_OT_preview_solid)
    bpy.utils.register_class(ANAPLAST_OT_fidelity_report)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_fidelity_report)
    bpy.utils.unregister_class(ANAPLAST_OT_preview_solid)
    bpy.utils.unregister_class(ANAPLAST_OT_case_setup)
