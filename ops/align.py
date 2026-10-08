"""Alignment — paired-point rigid fit (Kabsch) followed by ICP refinement.
Lightweight, context-free. The ICP module remains the heavy-duty option."""
import numpy as np
import bpy
import re
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from .report import rep

AL_COLLECTION = "Anaplast_Align"


def _point_index(obj, prefix):
    match = re.fullmatch(re.escape(prefix) + r"(\d+)(?:\.\d+)?", obj.name)
    return int(match.group(1)) if match else None


def _points(prefix, scene=None):
    # Scene organization moves markers into Helpers. Collection membership and
    # visibility must not determine whether an alignment point exists.
    scene = scene if scene is not None else bpy.context.scene
    return sorted((o for o in scene.objects if _point_index(o, prefix) is not None),
                  key=lambda o: (_point_index(o, prefix), o.name))


def _show_points(scene, show):
    from .scene_helpers import set_hidden
    for obj in _points('AL_S_', scene) + _points('AL_T_', scene):
        obj['alignment_visible'] = show
        set_hidden(obj, scene, not show)


def kabsch(P, Q):
    """Rigid transform (R, t) minimising |R P + t - Q|. P, Q: (n,3) numpy."""
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def to_matrix(R, t):
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = float(R[i, j])
        M[i][3] = float(t[i])
    return M


def apply_transform(scene, moving, M):
    moving.matrix_world = M @ moving.matrix_world
    for o in _points("AL_S_", scene):
        o.matrix_world = M @ o.matrix_world


def world_verts(obj, max_n=4000, seed=0):
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if n > max_n:
        rng = np.random.default_rng(seed)
        co = co[rng.choice(n, max_n, replace=False)]
    mw = np.array(obj.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


class ANAPLAST_OT_add_align_point(bpy.types.Operator):
    """Add a paired alignment point at the 3D cursor — source points on the moving object, target points on the fixed one, in the same order"""
    bl_idname = "anaplast.add_align_point"
    bl_label = "Add Align Point"
    bl_options = {'REGISTER', 'UNDO'}

    role: bpy.props.EnumProperty(items=[("S", "Source (moving)", ""), ("T", "Target (fixed)", "")])

    def execute(self, context):
        col = mu.get_collection(context.scene, AL_COLLECTION)
        prefix = f"AL_{self.role}_"
        idx = max((_point_index(o, prefix) for o in _points(prefix, context.scene)), default=0) + 1
        rgb = mu.PALETTE[(idx - 1) % len(mu.PALETTE)]
        name = f"{prefix}{idx}"
        # A different scene can already own this name; never replace its marker.
        suffix = 1
        while name in bpy.data.objects:
            name = f"{prefix}{idx}.{suffix:03d}"
            suffix += 1
        o = mu.make_marker(name, context.scene.cursor.location.copy(), rgb, col,
                           'SPHERE' if self.role == 'S' else 'CUBE', 1.2)
        _show_points(context.scene, True)
        rep(self, {'INFO'}, f"Pair {idx}: {'source' if self.role == 'S' else 'target'} placed")
        return {'FINISHED'}


class ANAPLAST_OT_clear_align_points(bpy.types.Operator):
    """Remove all alignment points"""
    bl_idname = "anaplast.clear_align_points"
    bl_label = "Clear Points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for o in _points("AL_S_") + _points("AL_T_"):
            bpy.data.objects.remove(o)
        return {'FINISHED'}


class ANAPLAST_OT_align_points(bpy.types.Operator):
    """Rigidly move the source object so its points land on the target points (needs ≥3 pairs)"""
    bl_idname = "anaplast.align_points"
    bl_label = "Align by Points"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.align_moving is not None and p.align_target is not None

    def execute(self, context):
        S, T = _points("AL_S_"), _points("AL_T_")
        n = min(len(S), len(T))
        if n < 3:
            rep(self, {'ERROR'}, f"Need at least 3 source/target pairs (have {len(S)}/{len(T)})")
            return {'CANCELLED'}
        P = np.array([o.matrix_world.translation for o in S[:n]], dtype=np.float64)
        Q = np.array([o.matrix_world.translation for o in T[:n]], dtype=np.float64)
        R, t = kabsch(P, Q)
        apply_transform(context.scene, context.scene.anaplast.align_moving, to_matrix(R, t))
        _show_points(context.scene, False)
        res = np.sqrt((((P @ R.T + t) - Q) ** 2).sum(axis=1).mean())
        rep(self, {'INFO'}, f"Aligned on {n} pairs, RMS {res:.2f} mm — now Refine (ICP)")
        return {'FINISHED'}


class ANAPLAST_OT_align_icp(bpy.types.Operator):
    """Refine the fit by iterative closest point against the target surface"""
    bl_idname = "anaplast.align_icp"
    bl_label = "Refine (ICP)"
    bl_options = {'REGISTER', 'UNDO'}

    iterations: bpy.props.IntProperty(name="Iterations", default=30, min=1, max=200)
    max_dist: bpy.props.FloatProperty(name="Max pair distance (mm)", default=5.0, min=0.1,
                                      description="Pairs farther apart than this are ignored — keeps defect regions from biasing the fit")

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.align_moving is not None and p.align_target is not None and p.align_moving != p.align_target

    def execute(self, context):
        p = context.scene.anaplast
        moving, target = p.align_moving, p.align_target
        dg = context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(target, dg)
        t_mw = target.matrix_world
        t_inv = t_mw.inverted()
        pts = world_verts(moving)
        if p.favour_on:
            centre = np.array(p.favour_centre)
            d = np.linalg.norm(pts - centre, axis=1)
            w_sel = d <= p.icp_region
            if w_sel.sum() >= 50:
                pts = pts[w_sel]                       # only the favoured neighbourhood drives the fit
        total = Matrix.Identity(4)
        rms = float('nan')
        for it in range(self.iterations):
            Q, keep = [], []
            for i, q in enumerate(pts):
                loc, nrm, idx, dist = bvh.find_nearest(t_inv @ Vector(q))
                if loc is None:
                    continue
                w = t_mw @ loc
                if (w - Vector(q)).length <= self.max_dist:
                    Q.append((w.x, w.y, w.z))
                    keep.append(i)
            if len(keep) < 10:
                rep(self, {'ERROR'}, "Too few close pairs — run Align by Points first or raise max distance")
                return {'CANCELLED'}
            P, Q = pts[keep], np.array(Q)
            R, t = kabsch(P, Q)
            pts = pts @ R.T + t
            total = to_matrix(R, t) @ total
            rms_new = float(np.sqrt(((pts[keep] - Q) ** 2).sum(axis=1).mean()))
            if abs(rms - rms_new) < 1e-4:
                rms = rms_new
                break
            rms = rms_new
        apply_transform(context.scene, moving, total)
        p.last_rms = rms
        _show_points(context.scene, False)
        note = f", favouring {p.icp_region:.0f} mm around the spot you clicked" if p.favour_on else ""
        rep(self, {'INFO'}, f"ICP done: RMS {rms:.3f} mm on {len(keep)} points{note} — Deviation button colours the result")
        return {'FINISHED'}


def colour_by_distance(context, obj, target, range_mm, attr_name="deviation", only_inner=False, *, object_matrix=None):
    """Colour obj by distance to target (blue 0 → green half → red ≥ range). Returns the distance array."""
    dg = context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(target, dg)
    t_mw, t_inv = target.matrix_world, target.matrix_world.inverted()
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    mw = np.array(object_matrix if object_matrix is not None else obj.matrix_world)
    world = co @ mw[:3, :3].T + mw[:3, 3]
    d = np.empty(n)
    for i, q in enumerate(world):
        loc, nrm, idx, dist = bvh.find_nearest(t_inv @ Vector(q))
        d[i] = (t_mw @ loc - Vector(q)).length if loc is not None else range_mm
    t = np.clip(d / range_mm, 0.0, 1.0)
    r = np.clip(2 * t - 1, 0, 1); g = 1 - np.abs(2 * t - 1); b = np.clip(1 - 2 * t, 0, 1)
    if only_inner:
        # outer skin is not a fit surface: grey it out so the fitting side reads alone
        outer_attr = me.attributes.get("anaplast_outer")
        if outer_attr is not None:
            outer = np.empty(n); outer_attr.data.foreach_get("value", outer)
            grey = outer > 0.5
            r[grey] = g[grey] = b[grey] = 0.75
    attr = me.color_attributes.get(attr_name) or me.color_attributes.new(attr_name, 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", np.stack([r, g, b, np.ones(n)], axis=1).ravel())
    me.color_attributes.active_color = attr
    mat = bpy.data.materials.get("Anaplast_" + attr_name)
    if mat is None:
        mat = bpy.data.materials.new("Anaplast_" + attr_name); mat.use_nodes = True
        nt = mat.node_tree
        a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_name = attr_name
        nt.links.new(a.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    if not me.materials:
        me.materials.append(mat)
    else:
        me.materials[0] = mat
    set_viewport_color(context, 'VERTEX')
    return d


class ANAPLAST_OT_fit_map(bpy.types.Operator):
    """Check the Prototype in its seated position, even while lifted: blue = touching, green = half range, red = range or more"""
    bl_idname = "anaplast.fit_map"
    bl_label = "Fit colours"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype
        p = context.scene.anaplast
        return current_prototype(p) is not None and (p.face_scan_obj or p.cast_obj) is not None

    def execute(self, context):
        from .shell import current_prototype, resolve_trim
        p = context.scene.anaplast
        proto = current_prototype(p)
        trim = resolve_trim(self, p)
        if trim is None:
            return {'CANCELLED'}
        from .explode import seated_matrix
        d = colour_by_distance(context, proto, trim, p.fit_range, attr_name="fit", only_inner=True,
                               object_matrix=seated_matrix(proto))
        outer_attr = proto.data.attributes.get("anaplast_outer")
        if outer_attr is not None:
            outer = np.empty(len(proto.data.vertices)); outer_attr.data.foreach_get("value", outer)
            d = d[outer <= 0.5]
        rep(self, {'INFO'}, f"Seated fit: median gap {np.median(d):.2f}  p90 {np.percentile(d, 90):.2f}  max {d.max():.2f} mm (blue = touching, red ≥ {p.fit_range:.1f} mm)")
        return {'FINISHED'}


class ANAPLAST_OT_deviation_map(bpy.types.Operator):
    """Colour the moving object by distance to the target: blue = 0, green = half range, red = range or more"""
    bl_idname = "anaplast.deviation_map"
    bl_label = "Show Deviation Map"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.align_moving is not None and p.align_target is not None and p.align_moving != p.align_target

    def execute(self, context):
        p = context.scene.anaplast
        self.range_mm = p.deviation_range
        moving, target = p.align_moving, p.align_target
        dg = context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(target, dg)
        t_mw, t_inv = target.matrix_world, target.matrix_world.inverted()
        me = moving.data
        n = len(me.vertices)
        if n == 0:
            rep(self, {'ERROR'}, "Moving object has no geometry")
            return {'CANCELLED'}
        co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        mw = np.array(moving.matrix_world)
        world = co @ mw[:3, :3].T + mw[:3, 3]
        d = np.empty(n)
        for i, q in enumerate(world):
            loc, nrm, idx, dist = bvh.find_nearest(t_inv @ Vector(q))
            d[i] = (t_mw @ loc - Vector(q)).length if loc is not None else self.range_mm
        t = np.clip(d / self.range_mm, 0.0, 1.0)
        r = np.clip(2 * t - 1, 0, 1); g = 1 - np.abs(2 * t - 1); b = np.clip(1 - 2 * t, 0, 1)
        attr = me.color_attributes.get("deviation") or me.color_attributes.new("deviation", 'FLOAT_COLOR', 'POINT')
        rgba = np.stack([r, g, b, np.ones(n)], axis=1).ravel()
        attr.data.foreach_set("color", rgba)
        me.color_attributes.active_color = attr
        mat = bpy.data.materials.get("Anaplast_Deviation")
        if mat is None:
            mat = bpy.data.materials.new("Anaplast_Deviation"); mat.use_nodes = True
            nt = mat.node_tree
            a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_name = "deviation"
            nt.links.new(a.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        if not me.materials:
            me.materials.append(mat)
        else:
            me.materials[0] = mat
        p.last_rms = float(np.sqrt((d ** 2).mean()))
        set_viewport_color(context, 'VERTEX')
        rep(self, {'INFO'}, f"Deviation: mean {d.mean():.2f}  p90 {np.percentile(d, 90):.2f}  max {d.max():.2f} mm")
        return {'FINISHED'}


def set_viewport_color(context, mode):
    """Switch every 3D viewport to Solid shading with the given colour source ('VERTEX' = Attribute)."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            sh = area.spaces.active.shading
            sh.type = 'SOLID'
            sh.color_type = mode


class ANAPLAST_OT_deviation_reset(bpy.types.Operator):
    """Return the viewport to normal material colours"""
    bl_idname = "anaplast.deviation_reset"
    bl_label = "Normal Colours"

    def execute(self, context):
        set_viewport_color(context, 'MATERIAL')
        return {'FINISHED'}


class ANAPLAST_OT_place_align_points(bpy.types.Operator):
    """Click pairs directly on the surfaces: left-click on the MOVING object, then the matching spot on the TARGET, and repeat. Backspace undoes, Enter finishes, Esc cancels"""
    bl_idname = "anaplast.place_align_points"
    bl_label = "Place Pairs (click on the models)"
    bl_options = {'REGISTER', 'UNDO'}

    _n = 0

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.align_moving is not None and p.align_target is not None and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        self._n = 0
        _show_points(context.scene, True)
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        p = context.scene.anaplast
        want = "SOURCE point on " + p.align_moving.name if self._n % 2 == 0 else "TARGET point on " + p.align_target.name
        context.area.header_text_set(f"Click the {want} · BACKSPACE undo · ENTER finish · ESC cancel")

    def _hit(self, context, event, obj):
        from bpy_extras import view3d_utils
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        w, wn, o = mu.ray_cast_clipped(context, origin, direction, obj=obj)
        return w

    def modal(self, context, event):
        p = context.scene.anaplast
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'NUMPAD_PERIOD'} or (event.type == 'LEFTMOUSE' and event.alt):
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            role = 'S' if self._n % 2 == 0 else 'T'
            obj = p.align_moving if role == 'S' else p.align_target
            hit = self._hit(context, event, obj)
            if hit is None:
                return {'RUNNING_MODAL'}
            context.scene.cursor.location = hit
            bpy.ops.anaplast.add_align_point(role=role)
            self._n += 1
            self._header(context)
            return {'RUNNING_MODAL'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._n > 0:
            role = 'S' if (self._n - 1) % 2 == 0 else 'T'
            pts = _points(f"AL_{role}_")
            if pts:
                mu.delete_object(pts[-1])
            self._n -= 1
            self._header(context)
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._finish(context)
            ns, nt = len(_points("AL_S_")), len(_points("AL_T_"))
            rep(self, {'INFO'}, f"{min(ns, nt)} pairs placed — now Align by Points, then Refine (ICP)")
            return {'FINISHED'}
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


class ANAPLAST_OT_favour_here(bpy.types.Operator):
    """Click one spot on the MOVING object: the refinement then fits that neighbourhood first and lets the rest follow"""
    bl_idname = "anaplast.favour_here"
    bl_label = "Favour a spot (click)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.align_moving is not None and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set(f"Click the spot on {context.scene.anaplast.align_moving.name} that matters most · ESC cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        p = context.scene.anaplast
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            from bpy_extras import view3d_utils
            obj = p.align_moving
            region, rv3d = context.region, context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            w, wn, o = mu.ray_cast_clipped(context, origin, direction, obj=obj)
            if w is not None:
                p.favour_centre = w[:]
                p.favour_on = True
                col = mu.get_collection(context.scene, AL_COLLECTION)
                mu.make_marker("AL_FAVOUR", Vector(p.favour_centre), mu.PALETTE[5], col, 'SPHERE', 2.5)
                self._finish(context)
                rep(self, {'INFO'}, f"Favouring {p.icp_region:.0f} mm around that spot — now Refine (ICP)")
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


class ANAPLAST_OT_favour_clear(bpy.types.Operator):
    """Fit the whole overlap again"""
    bl_idname = "anaplast.favour_clear"
    bl_label = "Clear"

    def execute(self, context):
        context.scene.anaplast.favour_on = False
        o = bpy.data.objects.get("AL_FAVOUR")
        if o:
            mu.delete_object(o)
        return {'FINISHED'}


_classes = (ANAPLAST_OT_fit_map, ANAPLAST_OT_favour_here, ANAPLAST_OT_favour_clear, ANAPLAST_OT_place_align_points, ANAPLAST_OT_add_align_point, ANAPLAST_OT_clear_align_points,
            ANAPLAST_OT_align_points, ANAPLAST_OT_align_icp, ANAPLAST_OT_deviation_map,
            ANAPLAST_OT_deviation_reset)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
