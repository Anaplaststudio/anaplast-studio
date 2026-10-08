import os
import bpy
from ..utils import mesh as mu
from .report import rep
from .mold_split import mold_parts


class ANAPLAST_OT_export_check(bpy.types.Operator):
    """Tool 8 — manifold check every mold part, then export one STL per part"""
    bl_idname = "anaplast.export_check"
    bl_label = "Check & Export STL"
    bl_options = {'REGISTER'}

    what: bpy.props.EnumProperty(items=[("MOLD", "Mold parts", ""), ("SHELL", "Prototype shell", ""), ("HOLLOW", "Hollow fitted sculpt", ""), ("FITTED", "Fitted sculpt (solid)", "")], default="MOLD")
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        p = context.scene.anaplast
        out = bpy.path.abspath(p.export_dir) if p.export_dir else ""
        if out and os.path.isdir(os.path.dirname(out.rstrip("/\\")) or out):
            self.directory = out
            return self.execute(context)
        context.window_manager.fileselect_add(self)       # unsaved file or no folder set: ask
        return {'RUNNING_MODAL'}

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype
        return current_prototype(context.scene.anaplast) is not None or any(bpy.data.objects.get(n) for n in ("Mold_A", "Mold_Base", "Mold_Cap", "Prosthesis_Hollow", "Prosthesis_Fitted"))

    def execute(self, context):
        from .explode import ensure_collapsed
        ensure_collapsed(context.scene)
        if self.what=='MOLD' and context.scene.get('anaplast_insert_state'):
            from . import mold_inserts
            try:
                st=mold_inserts.state(context.scene);mold_inserts.checked_hosts(context.scene,st)
                if st.get('settings')!=mold_inserts.settings_signature(context.scene):
                    raise RuntimeError('Build / update inserts before exporting their changed settings')
            except RuntimeError as exc:
                rep(self,{'ERROR'},str(exc));return {'CANCELLED'}
        p = context.scene.anaplast
        col = mu.get_collection(context.scene)
        out = self.directory or bpy.path.abspath(p.export_dir)
        if not out:
            rep(self, {'ERROR'}, "Choose an export folder first (Export folder field)")
            return {'CANCELLED'}
        p.export_dir = out
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            rep(self, {'ERROR'}, f"Cannot create folder {out}: {e}")
            return {'CANCELLED'}

        from .shell import shell_name
        single = {"SHELL": (shell_name(p), "Build Prototype"), "HOLLOW": ("Prosthesis_Hollow", "Hollow Fitted Sculpt"),
                  "FITTED": ("Prosthesis_Fitted", "Fit Sculpt to Tissue")}
        if self.what in single:
            name, how = single[self.what]
            obj = bpy.data.objects.get(name)
            if obj is None:
                rep(self, {'ERROR'}, f"No {name} yet — run {how}")
                return {'CANCELLED'}
            parts = [obj]
            if "anaplast_home" in obj:
                from .explode import seat_object
                seat_object(obj)
        else:
            parts = mold_parts(col)
            auto = [o for o in (bpy.data.objects.get("Mold_Base"), bpy.data.objects.get("Mold_Cap")) if o is not None]
            parts = auto + [o for o in parts if o not in auto]
            parts += [o for o in context.scene.objects if o.type=='MESH' and o.get('insert_uid') and o not in parts and (not o.get('mold_set') or o.get('mold_set')==context.scene.mold_workflow.active_id)]
            parts=[o for o in parts if not o.get('mold_stale')]
            if not parts:
                rep(self, {'ERROR'}, "No mold parts yet — build one first (Two-part mold from this view)")
                return {'CANCELLED'}
        failed, warned = [], []
        # small holes (a few open edges from later editing) are filled rather than blocking the export
        import bmesh as _bmesh
        for o in parts:
            n0, nonman0, open0, loose0 = mu.mesh_stats(o)
            if 0 < open0 <= 200:
                bm = _bmesh.new(); bm.from_mesh(o.data)
                try:
                    _bmesh.ops.holes_fill(bm, edges=[ed for ed in bm.edges if ed.is_boundary], sides=0)
                except Exception:
                    pass
                _bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                bm.to_mesh(o.data); bm.free(); o.data.update()
                n1, nonman1, open1, loose1 = mu.mesh_stats(o)
                warned.append(f"{o.name}: {open0} open edges filled before export ({open1} left)")
        for part in parts:
            n, nonman, boundary, loose = mu.mesh_stats(part)
            if nonman or boundary or loose:
                b0, b1 = mu.repair_nonmanifold(part)
                n, nonman, boundary, loose = mu.mesh_stats(part)
            if boundary or loose or nonman > 20:
                failed.append(f"{part.name}: {nonman} non-manifold, {boundary} open, {loose} loose")
            elif nonman:
                warned.append(f"{part.name}: {nonman} tiny non-manifold edges left (slicers cope)")
        if failed:
            rep(self, {'ERROR'}, "Export blocked — " + " | ".join(failed))
            return {'CANCELLED'}

        stem_prefix = f"{mu.case_tag(p)}_{p.prosthesis_type.lower()}_"
        mu.clear_old(out, stem_prefix + ("Mold_" if self.what == 'MOLD' else parts[0].name), (".stl", ".obj"))
        for part in parts:
            fn = os.path.join(out, f"{mu.case_tag(p)}_{p.prosthesis_type.lower()}_{part.name}")
            mu.export_mesh(context, part, fn, p.export_format)
        if warned:
            rep(self, {'WARNING'}, f"Exported {len(parts)} to {out} — " + " | ".join(warned))
        else:
            rep(self, {'INFO'}, f"Exported {len(parts)} parts to {out}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_export_check)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_export_check)
