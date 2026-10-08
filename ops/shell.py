"""Pathway A — print a thin prototype shell of the sculpt, trimmed to the tissue."""
import bpy
import bmesh
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep


def extrude_direction(p, patch):
    from mathutils import Vector, Matrix
    fixed = {"NEG_X": (-1, 0, 0), "POS_X": (1, 0, 0), "NEG_Y": (0, -1, 0), "NEG_Z": (0, 0, -1), "POS_Z": (0, 0, 1)}
    if p.extrude_dir in fixed:
        return Vector(fixed[p.extrude_dir])
    return -mu.average_normal(patch)            # into the head


def resolve_trim(self, p):
    """The tissue surface to subtract: the chosen slot, else the other one (with a note), else None."""
    want = p.cast_obj if p.shell_trim == 'CAST' else p.face_scan_obj
    other = p.face_scan_obj if p.shell_trim == 'CAST' else p.cast_obj
    if want is not None:
        return want
    if other is not None:
        rep(self, {'WARNING'}, f"{'Cast' if p.shell_trim == 'CAST' else 'Face scan'} slot is empty — subtracting {other.name} instead")
        return other
    rep(self, {'ERROR'}, "Nothing to subtract — import a Cast / stone or a Face scan in the Case panel")
    return None


class ANAPLAST_OT_make_shell(bpy.types.Operator):
    """Thicken the sculpt surface inward to a shell and trim it against the cast or face scan"""
    bl_idname = "anaplast.make_shell"
    bl_label = "Make Prototype Shell"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.prosthesis_obj is not None

    def execute(self, context):
        from ..utils.prototype import build
        p = context.scene.anaplast
        trim = resolve_trim(self, p)
        if trim is None:
            return {'CANCELLED'}
        src = source_surface(p)
        if src is trim:
            rep(self, {'ERROR'}, 'Choose different Sculpt and fitting scan objects')
            return {'CANCELLED'}
        col = mu.get_collection(context.scene, "Anaplast_Prototype")
        name = shell_name(p)
        old = bpy.data.objects.get(name)
        try:
            shell = build(src, trim, p, col, name + '_building')
        except Exception as exc:
            rep(self, {'ERROR'}, str(exc))
            return {'CANCELLED'}
        # Replace only after the new mesh passes all checks.
        if old is not None:
            mu.delete_object(old)
        shell.name = name
        shell.data.name = name
        p.source_prosthesis = src
        rep(self, {'INFO'}, f"Prototype: {len(shell.data.polygons):,} faces; {shell['prototype_detail_method']}; closed, no detected intersections")
        if shell.get('prototype_detail_limited', 0):
            rep(self, {'WARNING'}, 'Some local detail or fitting adjustments were limited to prevent intersecting surfaces; inspect the fit and thickness maps')
        return {'FINISHED'}


class ANAPLAST_OT_adapt_margins(bpy.types.Operator):
    """Seat the open margins of the Sculpt onto the tissue (skin clearance at the edge, blending inward over the band). Undo-able"""
    bl_idname = "anaplast.adapt_margins"
    bl_label = "Adapt Margins to Tissue"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.prosthesis_obj is not None and (p.cast_obj or p.face_scan_obj)

    def execute(self, context):
        p = context.scene.anaplast
        trim = resolve_trim(self, p)
        if trim is None:
            return {'CANCELLED'}
        from .sculpt_surface import target,activate
        obj=target(context)
        if obj is None or obj != source_surface(p) or obj in (p.face_scan_obj,p.cast_obj,p.cbct_obj) or obj.get('anaplast_part')=='MIRROR':
            rep(self,{'ERROR'},'Choose the cropped Sculpt as Working surface; scan references are preserved')
            return {'CANCELLED'}
        if obj.modifiers:
            rep(self,{'ERROR'},'Apply subdivisions/modifiers before adapting margins')
            return {'CANCELLED'}
        n,nonman,boundary,loose=mu.mesh_stats(obj)
        if not boundary:
            rep(self,{'ERROR'},'Sculpt has no open edge; adapt margins before solid remeshing')
            return {'CANCELLED'}
        activate(context,obj)
        try:moved=mu.adapt_margin(obj,trim,p.scan_flip,p.margin_band,p.shell_clearance)
        except Exception as e:
            rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'INFO'},f'Margins adapted: {moved:,} vertices moved; {obj.get("margin_added_vertices",0):,} added automatically')
        return {'FINISHED'}


GENERATED = ("Prosthesis_Fitted", "Prosthesis_Hollow", "Shell_Prosthesis", "Prototype_Cast", "Prototype_FaceScan")


def shell_name(p):
    """One prototype per tissue surface, so a cast build never overwrites a face-scan build."""
    return "Prototype_Cast" if p.shell_trim == 'CAST' else "Prototype_FaceScan"


def current_prototype(p):
    return bpy.data.objects.get(shell_name(p)) or bpy.data.objects.get("Shell_Prosthesis")


def source_surface(p):
    """The open surface to build from: the mirrored/cropped/SSM surface, never a previous build."""
    o = p.prosthesis_obj
    if o is not None and o.name not in GENERATED:
        return o
    return p.source_prosthesis or o


class ANAPLAST_OT_build_prototype(bpy.types.Operator):
    """Build a detail-preserving Prototype, validate it, and select it"""
    bl_idname = "anaplast.build_prototype"
    bl_label = "Build Prototype"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return source_surface(p) is not None and (p.cast_obj or p.face_scan_obj)

    def execute(self, context):
        p = context.scene.anaplast
        src = source_surface(p)
        p.source_prosthesis = src
        p.prosthesis_obj = src
        # Detail preservation is automatic, including when opening older cases.
        p.prototype_kind = 'CAP'
        try:
            r = bpy.ops.anaplast.make_shell()
        except RuntimeError as exc:
            rep(self, {'ERROR'}, str(exc))
            return {'CANCELLED'}
        if 'FINISHED' not in r:
            return {'CANCELLED'}
        out = bpy.data.objects.get(shell_name(p))
        p.prosthesis_obj = src                       # keep the slot on the source so rebuilding just works
        if out is not None and src is not None:
            src.hide_set(True)                       # the mirrored source is in the way once the prototype exists

        if out is not None:
            from .scene_helpers import organize
            organize(context.scene, preserve_visibility=True)
            for o in context.view_layer.objects:
                o.select_set(False)
            out.select_set(True)
            context.view_layer.objects.active = out
            mu.frame_view(context, out)
        return {'FINISHED'} if 'FINISHED' in r else {'CANCELLED'}


class ANAPLAST_OT_export_prototype(bpy.types.Operator):
    """Export the prototype you chose above as STL"""
    bl_idname = "anaplast.export_prototype"
    bl_label = "Export Prototype STL"

    def execute(self, context):
        return bpy.ops.anaplast.export_check('INVOKE_DEFAULT', what='SHELL')


class ANAPLAST_OT_skin_microtexture(bpy.types.Operator):
    """Final step: add multi-scale skin relief (cell network, pores, variation) to the OUTER skin of the built prototype. Undo-able"""
    bl_idname = "anaplast.skin_microtexture"
    bl_label = "Add Skin Micro-texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return current_prototype(p) is not None and source_surface(p) is not None

    def execute(self, context):
        p = context.scene.anaplast
        obj = current_prototype(p)
        src = source_surface(p)
        if "anaplast_home" in obj:
            bpy.ops.anaplast.lift_prototype()
        n, rms = mu.skin_microtexture(obj, src, strength=p.skin_strength, preset=p.skin_preset, subdivide=True)
        rep(self, {'INFO'}, f"Skin relief on {n:,} outer vertices ({p.skin_preset.lower()}), rms {rms * 1000:.0f} µm. Now {len(obj.data.polygons):,} faces")
        return {'FINISHED'}


class ANAPLAST_OT_resnap_fitting(bpy.types.Operator):
    """Put the fitting surface back exactly on the tissue (plus the relief gap) — run after sculpting"""
    bl_idname = "anaplast.resnap_fitting"
    bl_label = "Restore Fitting Surface"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return current_prototype(p) is not None and (p.cast_obj or p.face_scan_obj)

    def execute(self, context):
        p = context.scene.anaplast
        trim = resolve_trim(self, p)
        if trim is None:
            return {'CANCELLED'}
        obj = current_prototype(p)
        if obj.data.attributes.get("anaplast_outer") is None:
            rep(self, {'ERROR'}, "Build the prototype with 'Keep detail' on first — the outer/inner split comes from that step")
            return {'CANCELLED'}
        moved, worst = mu.resnap_fitting(obj, trim, p.scan_flip, p.fit_offset)
        rep(self, {'INFO'}, f"Fitting surface restored: {moved:,} vertices corrected, largest correction {worst:.2f} mm")
        return {'FINISHED'}


class ANAPLAST_OT_thickness_map(bpy.types.Operator):
    """Colour the prototype by wall thickness: red = at or below the edge thickness, green = full wall, blue = thicker. Scan stays visible underneath"""
    bl_idname = "anaplast.thickness_map"
    bl_label = "Show Thickness"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return current_prototype(context.scene.anaplast) is not None

    def execute(self, context):
        from .align import set_viewport_color
        p = context.scene.anaplast
        obj = current_prototype(p)
        mn, med, mx, frac = mu.thickness_map(obj, thin=p.thick_min, full=p.thick_max)
        set_viewport_color(context, 'VERTEX')
        rep(self, {'INFO'}, f"Thickness: min {mn:.2f}  median {med:.2f}  max {mx:.2f} mm; {frac:.0%} at or below {p.thick_min:.2f} mm (red)")
        return {'FINISHED'}


class ANAPLAST_OT_lift_prototype(bpy.types.Operator):
    """Lift the prototype 20 mm off the tissue to inspect its fitting side, or seat it back exactly"""
    bl_idname = "anaplast.lift_prototype"
    bl_label = "Lift / Seat Prototype"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return current_prototype(context.scene.anaplast) is not None

    def execute(self, context):
        from mathutils import Vector
        p = context.scene.anaplast
        obj = current_prototype(p)
        if "anaplast_home" in obj:
            from .explode import seat_object
            seat_object(obj)
            rep(self, {'INFO'}, f"{obj.name} seated")
            return {'FINISHED'}
        src = source_surface(p)
        direction = mu.average_normal(src) if src is not None else Vector((0, 1, 0))
        obj["anaplast_home"] = [list(r) for r in obj.matrix_world]
        obj.matrix_world = Matrix.Translation(direction * 20.0) @ obj.matrix_world
        rep(self, {'INFO'}, f"{obj.name} lifted 20 mm — click again to seat")
        return {'FINISHED'}


class ANAPLAST_OT_fit_to_tissue(bpy.types.Operator):
    """Pathway B prep — subtract the cast/face scan from a closed sculpt so its base conforms to the tissue"""
    bl_idname = "anaplast.fit_to_tissue"
    bl_label = "Fit Sculpt to Tissue"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.prosthesis_obj is not None and (p.cast_obj or p.face_scan_obj)

    def execute(self, context):
        p = context.scene.anaplast
        trim = resolve_trim(self, p)
        if trim is None:
            return {'CANCELLED'}
        if trim is p.prosthesis_obj:
            rep(self, {'ERROR'}, "Sculpt and the surface to subtract are the same object")
            return {'CANCELLED'}
        src = source_surface(p)
        p.source_prosthesis = src
        n, nonman, boundary, loose = mu.mesh_stats(src)
        col = mu.get_collection(context.scene, "Anaplast_Prototype")
        old = bpy.data.objects.get("Prosthesis_Fitted")
        if old:
            mu.delete_object(old)
        fit = mu.duplicate_object(src, "Prosthesis_Fitted", col)
        from .sculpt_prepare import remove_backing
        remove_backing(fit)
        deep = mu.deep_inside_fraction(fit, trim, p.scan_flip, 3.0)
        if deep > 0.5:
            mu.delete_object(fit)
            rep(self, {'ERROR'}, f"{deep:.0%} of the prosthesis is more than 3 mm INSIDE the tissue — check Flip / alignment / slots")
            return {'CANCELLED'}
        if boundary and p.build_method == 'VOXEL':
            r = mu.voxel_fit(fit, trim, p.scan_flip, wall=None, clearance=p.shell_clearance,
                             band=p.margin_band if p.adapt_margins else 0.0, voxel=p.voxel_size,
                             scan_thickness=p.scan_thickness, solver=p.boolean_solver)
            p.prosthesis_obj = fit
            if r["open"] or r["nonmanifold"]:
                rep(self, {'WARNING'}, f"Fitted: {r['faces']} faces, {r['open']} open / {r['nonmanifold']} non-manifold — try a smaller voxel or Advanced → Boolean (prism)")
            else:
                rep(self, {'INFO'}, f"Prosthesis_Fitted: {r['faces']} faces, closed (extract → dynamesh {p.voxel_size:.2f} mm → boolean)")
            return {'FINISHED'}
        elif boundary and p.build_method == 'BOOLEAN':
            r = mu.boolean_fit(fit, trim, p.scan_flip, extrude_direction(p, fit), wall=None,
                               clearance=p.shell_clearance, band=p.margin_band if p.adapt_margins else 0.0,
                               max_faces=p.boolean_max_faces, solver=p.boolean_solver)
            p.prosthesis_obj = fit
            if r["open"] or r["nonmanifold"]:
                rep(self, {'WARNING'}, f"Fitted: {r['faces']} faces; margin cleanup removed {r['removed']} fins, closed {r['holes']} holes; still {r['open']} open / {r['nonmanifold']} non-manifold — try Extrude direction override")
            else:
                rep(self, {'INFO'}, f"Prosthesis_Fitted: {r['faces']} faces, closed (booleans). Margin cleanup removed {r['removed']} fins, closed {r['holes']} holes")
            return {'FINISHED'}
        elif boundary:
            if p.adapt_margins and p.margin_band > 0:
                mu.adapt_margin(fit, trim, p.scan_flip, p.margin_band, p.shell_clearance)
            bm = bmesh.new(); bm.from_mesh(fit.data); mu.clean_margin(bm); bm.to_mesh(fit.data); bm.free(); fit.data.update()
            moved, nf = mu.build_fitted_solid(fit, trim, p.scan_flip, clearance=p.shell_clearance, gap=0.1, wall=None)
            mu.cleanup_mesh(fit)
        else:
            solid, temp = mu.solid_copy(trim, p.scan_thickness, p.scan_flip, roi=mu.world_bbox(fit), max_faces=p.boolean_max_faces)
            mu.apply_boolean(fit, solid, 'DIFFERENCE', solver=p.boolean_solver)
            if temp:
                mu.delete_object(solid)
            mu.cleanup_mesh(fit)
            bm = bmesh.new(); bm.from_mesh(fit.data)
            mu.keep_largest_island(bm)
            bm.to_mesh(fit.data); bm.free(); fit.data.update()
            mu.heal_small_holes(fit)
        p.prosthesis_obj = fit
        rep(self, {'INFO'}, "Prosthesis_Fitted created and set as Sculpt — continue to the mold block")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_make_shell)
    bpy.utils.register_class(ANAPLAST_OT_fit_to_tissue)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_fit_to_tissue)
    bpy.utils.unregister_class(ANAPLAST_OT_make_shell)


class ANAPLAST_OT_hollow_fitted(bpy.types.Operator):
    """Pathway B — hollow the fitted sculpt to a thin wall (outer skin AND tissue-side surface kept), with drain holes on the tissue side"""
    bl_idname = "anaplast.hollow_fitted"
    bl_label = "Hollow Fitted Sculpt"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Prosthesis_Fitted") is not None

    def execute(self, context):
        from mathutils import Vector
        p = context.scene.anaplast
        src = bpy.data.objects["Prosthesis_Fitted"]
        col = mu.get_collection(context.scene, "Anaplast_Prototype")
        old = bpy.data.objects.get("Prosthesis_Hollow")
        if old:
            mu.delete_object(old)
        hollow = mu.duplicate_object(src, "Prosthesis_Hollow", col)
        mu.apply_solidify(hollow, p.shell_thickness)          # closed solid → closed hollow with inner cavity
        mu.cleanup_mesh(hollow)
        # drain holes: two Ø3 mm cylinders through the tissue side (posterior = -Y in the case frame)
        me = src.data
        mw = src.matrix_world
        pts = [mw @ v.co for v in me.vertices]
        posterior = sorted(pts, key=lambda q: q.y)[: max(2, len(pts) // 50)]
        holes = 0
        if len(posterior) >= 2:
            a = posterior[0]
            b = max(posterior, key=lambda q: (q - a).length)     # the two farthest-apart posterior points
            lo, hi = mu.world_bbox(hollow)
            for c in (a, b):
                cyl = mu.make_cylinder("anaplast_drain", Vector((c.x, c.y, c.z)), p.drain_diameter * 0.5, (hi.y - lo.y) * 0.6, col, segments=24)
                cyl.rotation_euler = (1.5707963, 0.0, 0.0)          # axis along Y
                cyl.location = Vector((c.x, c.y, c.z))
                mu.apply_boolean(hollow, cyl, 'DIFFERENCE')
                mu.delete_object(cyl)
                holes += 1
            mu.cleanup_mesh(hollow)
        hollow["anaplast_part"] = "HOLLOW"
        n, nonman, boundary, loose = mu.mesh_stats(hollow)
        rep(self, {'INFO'}, f"Prosthesis_Hollow: {n} faces, {p.shell_thickness:.1f} mm walls, {holes} drain holes Ø{p.drain_diameter:.1f} mm on the tissue side")
        return {'FINISHED'}


_hollow = (ANAPLAST_OT_hollow_fitted, ANAPLAST_OT_adapt_margins, ANAPLAST_OT_build_prototype, ANAPLAST_OT_export_prototype, ANAPLAST_OT_thickness_map, ANAPLAST_OT_lift_prototype, ANAPLAST_OT_skin_microtexture, ANAPLAST_OT_resnap_fitting)
_old_reg, _old_unreg = register, unregister


def register():
    _old_reg()
    for c in _hollow:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_hollow):
        bpy.utils.unregister_class(c)
    _old_unreg()
