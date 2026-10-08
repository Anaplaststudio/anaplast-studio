"""Import a scan file (STL / OBJ / PLY) straight into a case slot."""
import os
import bpy
from ..utils import mesh as mu
from .report import rep

SCAN_COLLECTION = "Anaplast_Scans"
SLOTS = [
    ("face_scan_obj", "Scan", "FaceScan"),
    ("cast_obj", "Cast / defect scan", "Cast"),
    ("cbct_obj", "CBCT bone", "CBCT"),
    ("prosthesis_obj", "Sculpt", "Sculpt"),
]


class ANAPLAST_OT_import_slot(bpy.types.Operator):
    """Import an STL / OBJ / PLY file and assign it to this slot (1 file unit = 1 mm)"""
    bl_idname = "anaplast.import_slot"
    bl_label = "Import"
    bl_options = {'REGISTER', 'UNDO'}

    slot: bpy.props.EnumProperty(items=[(k, n, "") for k, n, _ in SLOTS])
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.stl;*.obj;*.ply", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ext = os.path.splitext(self.filepath)[1].lower()
        before = set(bpy.data.objects)
        try:
            if ext == ".stl":
                bpy.ops.wm.stl_import(filepath=self.filepath, use_scene_unit=False, global_scale=1.0)
            elif ext == ".obj":
                bpy.ops.wm.obj_import(filepath=self.filepath, use_split_objects=False, use_split_groups=False)
            elif ext == ".ply":
                bpy.ops.wm.ply_import(filepath=self.filepath, use_scene_unit=False, global_scale=1.0)
            else:
                rep(self, {'ERROR'}, f"Unsupported file type: {ext}")
                return {'CANCELLED'}
        except RuntimeError as e:
            rep(self, {'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}
        new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
        if not new:
            rep(self, {'ERROR'}, "File imported no mesh")
            return {'CANCELLED'}
        obj = max(new, key=lambda o: len(o.data.polygons))
        for extra in new:
            if extra is not obj:
                mu.delete_object(extra)
        base = dict((k, b) for k, _, b in SLOTS)[self.slot]
        obj.name = base
        obj.data.name = base
        mu.link_only(obj, mu.get_collection(context.scene, SCAN_COLLECTION))
        setattr(context.scene.anaplast, self.slot, obj)
        n, nonman, boundary, loose = mu.mesh_stats(obj)
        p = context.scene.anaplast
        if p.solid_on_import and boundary and self.slot != "prosthesis_obj":
            mu.make_solid_twin(obj, p.scan_thickness, p.scan_flip, mu.get_collection(context.scene, SCAN_COLLECTION))
        if self.slot == "prosthesis_obj":
            obj['anaplast_part']='SCULPT'
            p.crop_target = obj
            p.source_prosthesis = obj
        if self.slot == "cast_obj":
            obj.color = (0.93, 0.70, 0.25, 1.0)          # cast is always amber
        elif self.slot == "face_scan_obj":
            obj.color = (0.85, 0.72, 0.62, 1.0)
        mu.frame_view(context, obj)
        note = "" if not (nonman or boundary) else f" — NOTE {nonman} non-manifold / {boundary} open edges (fine for scans, must be fixed for a prosthesis)"
        rep(self, {'INFO'}, f"{base}: {n} faces{note}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_import_slot)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_import_slot)
