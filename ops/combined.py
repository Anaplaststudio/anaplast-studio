"""Export the face scan and the prototype as ONE mesh (visual composite, not a union)."""
import os
import bpy
from ..utils import mesh as mu
from .report import rep


class ANAPLAST_OT_export_combined(bpy.types.Operator):
    """Export face scan + prototype together as one STL (for viewing / sharing; the two are not merged)"""
    bl_idname = "anaplast.export_combined"
    bl_label = "Export Face + Sculpt STL"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype
        p = context.scene.anaplast
        return p.face_scan_obj is not None and current_prototype(p) is not None

    def invoke(self, context, event):
        p = context.scene.anaplast
        out = bpy.path.abspath(p.export_dir) if p.export_dir else ""
        if out and os.path.isdir(out):
            self.directory = out
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .shell import current_prototype
        p = context.scene.anaplast
        proto = current_prototype(p)
        if "anaplast_home" in proto:
            bpy.ops.anaplast.lift_prototype()
        out = self.directory or bpy.path.abspath(p.export_dir)
        os.makedirs(out, exist_ok=True)
        import bmesh
        bm = bmesh.new()
        for o in (p.face_scan_obj, proto):
            tmp = bmesh.new(); tmp.from_mesh(o.data); tmp.transform(o.matrix_world)
            me_tmp = bpy.data.meshes.new("anaplast_join_tmp"); tmp.to_mesh(me_tmp); tmp.free()
            bm.from_mesh(me_tmp); bpy.data.meshes.remove(me_tmp)
        me = bpy.data.meshes.new("Face_plus_Prosthesis"); bm.to_mesh(me); bm.free()
        joined = bpy.data.objects.new("Face_plus_Prosthesis", me)
        context.scene.collection.objects.link(joined)
        mu.clear_old(out, f"{mu.case_tag(p)}_face_plus_prosthesis", (".stl", ".obj"))
        fn = mu.export_mesh(context, joined, os.path.join(out, f"{mu.case_tag(p)}_face_plus_prosthesis"), p.export_format)
        nf = len(joined.data.polygons)
        mu.delete_object(joined)
        rep(self, {'INFO'}, f"Exported {os.path.basename(fn)} ({nf:,} faces, {os.path.getsize(fn) / 1e6:.0f} MB)")
        return {'FINISHED'}


class ANAPLAST_OT_save_case(bpy.types.Operator):
    """Save the .blend as <Last_First_Date>.blend in the export folder"""
    bl_idname = "anaplast.save_case"
    bl_label = "Save Case (.blend)"

    def execute(self, context):
        p = context.scene.anaplast
        out = bpy.path.abspath(p.export_dir) if p.export_dir else os.path.expanduser("~")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{mu.case_tag(p)}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
        rep(self, {'INFO'}, f"Saved {path}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_export_combined)
    bpy.utils.register_class(ANAPLAST_OT_save_case)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_save_case)
    bpy.utils.unregister_class(ANAPLAST_OT_export_combined)
