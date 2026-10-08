"""Findings / bug report — everything I need to see what happened on your machine, in one text file."""
import os
import json
import sys
import time
import traceback
import platform
from collections import deque
import bpy

LOG = deque(maxlen=400)


def _stamp():
    return time.strftime("%H:%M:%S")


def install_logging():
    """Wrap every ANAPLAST operator so its messages and any Python error are kept in LOG."""
    for name in dir(bpy.types):
        if not name.startswith("ANAPLAST_OT_"):
            continue
        cls = getattr(bpy.types, name)
        if getattr(cls, "_anaplast_logged", False) or not hasattr(cls, "execute"):
            continue
        orig_exec = cls.execute

        def make_exec(orig, label):
            def execute(self, context):
                LOG.append(f"{_stamp()}  RUN   {label}")
                try:
                    r = orig(self, context)
                    LOG.append(f"{_stamp()}  END   {label} -> {sorted(r) if r else r}")
                    return r
                except Exception:
                    LOG.append(f"{_stamp()}  ERROR {label}\n" + traceback.format_exc())
                    raise
            return execute

        cls.execute = make_exec(orig_exec, cls.bl_idname)
        cls._anaplast_logged = True


def rep(self, type_, message):
    """Operator message that is also kept in the findings log."""
    LOG.append(f"{_stamp()}  {','.join(sorted(type_)):<7} {getattr(self, 'bl_idname', '?')}: {message}")
    return self.report(type_, message)


class ANAPLAST_OT_bug_report(bpy.types.Operator):
    """Write a findings file (versions, all settings, every case object with face counts, the last 400 messages and errors) to send back with your feedback"""
    bl_idname = "anaplast.bug_report"
    bl_label = "Write Findings Report"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.txt", options={'HIDDEN'})

    def invoke(self, context, event):
        p = context.scene.anaplast
        base = bpy.path.abspath(p.export_dir) if p.export_dir else os.path.expanduser("~")
        from ..utils import mesh as mu
        self.filepath = os.path.join(base, f"{mu.case_tag(p)}_report.txt")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not getattr(self, "filepath", ""):
            p0 = context.scene.anaplast
            from ..utils import mesh as mu0
            base0 = bpy.path.abspath(p0.export_dir) if p0.export_dir else os.path.expanduser("~")
            self.filepath = os.path.join(base0, f"{mu0.case_tag(p0)}_report.txt")
        from ..utils import mesh as mu
        p = context.scene.anaplast
        lines = []
        w = lines.append
        try:
            from .. import __package__ as pkg
        except Exception:
            pkg = "b4d_anaplastology"
        ver = "?"
        try:
            import tomllib
            man = os.path.join(os.path.dirname(os.path.dirname(__file__)), "blender_manifest.toml")
            with open(man, "rb") as f:
                ver = tomllib.load(f).get("version", "?")
        except Exception:
            pass
        w("ANAPLAST STUDIO — FINDINGS REPORT")
        w(f"date        {time.strftime('%Y-%m-%d %H:%M')}")
        w(f"add-on      {ver}")
        w(f"blender     {bpy.app.version_string}  ({bpy.app.build_platform.decode() if isinstance(bpy.app.build_platform, bytes) else bpy.app.build_platform})")
        w(f"python      {sys.version.split()[0]}   os {platform.platform()}")
        w(f"file        {bpy.data.filepath or '(unsaved)'}")
        w(f"units       {context.scene.unit_settings.length_unit} scale {context.scene.unit_settings.scale_length}")
        w("")
        w("SETTINGS")
        for key in sorted(p.bl_rna.properties.keys()):
            if key in ("rna_type",):
                continue
            try:
                val = getattr(p, key)
                if hasattr(val, "name"):
                    val = f"<{val.name}>"
                w(f"  {key:<24} {val}")
            except Exception:
                pass
        w("")
        if hasattr(context.scene,'mold_inserts'):
            from . import mold_inserts
            w('INTERCHANGEABLE INSERTS')
            setup=context.scene.mold_inserts
            w('  Selected setup: '+json.dumps({key:getattr(setup,key) for key in ('new_kind','target_seat','use_two_seats','second_seat','two_seat_layout')}))
            w('  Settings: '+mold_inserts.settings_signature(context.scene))
            w('  Status: '+context.scene.mold_inserts.report)
            for item in context.scene.mold_inserts.items:
                if item.report:w(f'  {item.name} last successful build: {item.report}')
            w('')
        w("OBJECTS (faces, mean edge mm, open edges, non-manifold)")
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            try:
                n, mean, lo, hi = mu.fidelity(o)
                nf, nonman, boundary, loose = mu.mesh_stats(o)
                loc = o.matrix_world.translation
                bl, bh = mu.world_bbox(o)
                attrs = [a.name for a in o.data.attributes if "mask" in a.name.lower() or a.name.startswith("anaplast")]
                mods = [m.type for m in o.modifiers]
                extra = (" attrs " + ",".join(attrs) if attrs else "") + (" mods " + ",".join(mods) if mods else "")
                w(f"  {o.name:<28} {n:>10,}  edge {mean:.3f}  open {boundary:<6} nonman {nonman:<6} at ({loc.x:.0f},{loc.y:.0f},{loc.z:.0f}) bbox x {bl.x:.0f}..{bh.x:.0f} {'hidden' if o.hide_get() else ''}{extra}")
            except Exception as e:
                w(f"  {o.name:<28} (stats failed: {e})")
        w("")
        w("MOLD")
        if hasattr(context.scene,'mold_workflow'):
            workflow=context.scene.mold_workflow
            for key in ('active_id','base_body','wedge_body','cap_body','cap_support','base_wall','wedge_wall','cap_wall','report'):
                w(f'  Workflow {key}: {getattr(workflow,key)}')
            w('  Last attempt: '+context.scene.get('anaplast_last_mold_attempt','not recorded'))
        try:
            from . import mold_volume
            fill = mold_volume.read(context.scene, validate=True)
            w(f"  Silicone cavity: {fill['cavity_ml']:.3f} mL; ring/channels: {fill['channels_ml']:.3f} mL; total: {fill['total_ml']:.3f} mL")
            mass = mold_volume.weight_estimate(context.scene, fill)
            if mass['density_g_ml'] is not None:
                w(f"  Estimated {mass['material']} weight: cavity {mass['cavity_g']:.3f} g; ring/channels {mass['channels_g']:.3f} g; total {mass['total_g']:.3f} g")
                w(f"  Density {mass['density_g_ml']:.6f} g/mL: {mass['density_basis']}")
            else:w(f"  {mass['material']}: weight unavailable; enter supplier or measured density")
            w('  Weight excludes inserts, additives, cure-volume changes and mixing waste.')
            w(f"  Material / density source: {mass['density_source']}")
            w("  Construction void estimate; excludes shell backs, bolt bores, pry slots and handling waste.")
        except Exception as exc:
            w(f"  Silicone volume: {exc}")
        try:
            import bmesh as _bmesh
            steps = json.loads(getattr(p, "mold_steps", "[]") or "[]")
            for i, st in enumerate(steps, 1):
                w(f"  {i:2d}. {'ok  ' if st.get('ok') else 'FAIL'} {st['step']:<40s} {st.get('s', 0):6.1f}s  {st.get('info', '')}")
            if not steps:
                w("  (no build recorded)")
            for nm in ("Mold_Base", "Mold_Cap"):
                o = bpy.data.objects.get(nm)
                if o is None:
                    continue
                n, nonman, open_e, loose = mu.mesh_stats(o)
                bm = _bmesh.new(); bm.from_mesh(o.data)
                vol = abs(bm.calc_volume()) if not open_e else float('nan')
                seen = set(); pieces = 0
                for f in bm.faces:
                    if f in seen:
                        continue
                    pieces += 1; stack = [f]; seen.add(f)
                    while stack:
                        g = stack.pop()
                        for e_ in g.edges:
                            for nf in e_.link_faces:
                                if nf not in seen:
                                    seen.add(nf); stack.append(nf)
                bm.free()
                up = o.get("mold_up"); lo, hi = mu.world_bbox(o); ext_ = hi - lo
                w(f"  {nm}: {n:,} faces, {pieces} piece(s), open {open_e}, non-manifold {nonman}, volume {vol:,.0f} mm3, "
                  f"bbox {ext_.x:.0f} x {ext_.y:.0f} x {ext_.z:.0f} mm, opening axis {tuple(round(c, 2) for c in up) if up else '?'}, "
                  f"mean edge {mu.fidelity(o)[1]:.3f} mm")
        except Exception as e:
            w(f"  mold section failed: {e}")
        w("")
        w("ADD-ONS / EXTENSIONS")
        try:
            for k in sorted(bpy.context.preferences.addons.keys()):
                w(f"  {k}")
        except Exception:
            pass
        import sys as _sys
        mp = sorted(m for m in _sys.modules if m == "mpfb" or m.endswith(".mpfb"))
        try:
            from .brushes import read_index, lib_file
            w("  Anaplast brushes: " + (", ".join(read_index()) or "none") + (" (file present)" if os.path.exists(lib_file()) else " (no library file)"))
        except Exception:
            pass
        w("  MPFB modules loaded: " + (", ".join(mp) if mp else "none"))
        w("")
        w("LOG (oldest first)")
        for entry in LOG:
            w("  " + entry.replace("\n", "\n    "))
        text = "\n".join(lines) + "\n"
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            self.report({'ERROR'}, f"Could not write report: {e}")
            return {'CANCELLED'}
        try:
            shot = os.path.splitext(self.filepath)[0] + ".png"
            if not bpy.app.background:
                bpy.ops.screen.screenshot(filepath=shot)
        except Exception:
            pass
        self.report({'INFO'}, f"Findings written to {self.filepath} (+ screenshot if possible) — send that file")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_bug_report)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_bug_report)
