"""Fit a sphere to a dome (eyelid dome, orbit, any curved patch) and place a ball of that size there —
for ocular positioning or any spherical reference."""
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep


def fit_sphere(points):
    """Algebraic least-squares sphere fit. points (n,3) → centre (3,), radius, rms residual mm."""
    P = np.asarray(points, dtype=np.float64)
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + (c ** 2).sum(), 1e-12)))
    resid = np.linalg.norm(P - c, axis=1) - r
    return c, r, float(np.sqrt((resid ** 2).mean()))


def selected_world_points(context):
    """Selected vertices (Edit Mode) or masked vertices (Sculpt Mode / stored mask) of the active object."""
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return None, []
    mw = obj.matrix_world
    if context.mode == 'EDIT_MESH':
        bm = bmesh.from_edit_mesh(obj.data)
        pts = [mw @ v.co for v in bm.verts if v.select]
        return obj, pts
    if context.mode == 'SCULPT':
        bpy.ops.object.mode_set(mode='OBJECT')
    me = obj.data
    for a in me.attributes:
        if "mask" in a.name.lower() and a.domain == 'POINT' and a.data_type == 'FLOAT':
            m = np.empty(len(me.vertices)); a.data.foreach_get("value", m)
            idx = np.nonzero(m > 0.5)[0]
            return obj, [mw @ me.vertices[int(i)].co for i in idx]
    pts = [mw @ v.co for v in me.vertices if v.select]
    return obj, pts


def place_ball(ball, diameter_override=0.0, sink=0.0):
    """Position/size the ball from the fit stored on it: tangent to the dome for an override, sunk along the dome axis."""
    c = Vector(ball["fit_centre"]); r = float(ball["fit_radius"]); n = Vector(ball["fit_normal"])
    mean_pt = c + n * r
    if diameter_override > 0:
        r = diameter_override * 0.5
        c = mean_pt - n * r
    c = c - n * sink
    ball.location = c
    ball.scale = (r / float(ball["mesh_radius"]),) * 3


def current_ball(context):
    p = context.scene.anaplast
    return p.ball_obj if (p.ball_obj is not None and p.ball_obj.name in bpy.data.objects) else None


def ball_update(self, context):
    b = current_ball(context)
    if b is not None and "fit_centre" in b:
        place_ball(b, self.ball_diameter, self.ball_sink)


class ANAPLAST_OT_fit_sphere(bpy.types.Operator):
    """Select (Edit Mode) or mask-paint (Sculpt Mode) a dome-shaped patch, then click: the ball that best fits the dome is placed there (or re-fitted if one exists). Diameter override / Sink in then adjust that same ball"""
    bl_idname = "anaplast.fit_sphere"
    bl_label = "Fit a ball into the dome"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.anaplast
        obj, pts = selected_world_points(context)
        if obj is None or len(pts) < 12:
            rep(self, {'ERROR'}, "Select or mask-paint the dome first (at least a dozen vertices)")
            return {'CANCELLED'}
        c, r, rms = fit_sphere([q[:] for q in pts])
        centre = Vector(c.tolist())
        mean_pt = Vector(np.mean([q[:] for q in pts], axis=0).tolist())
        n = (mean_pt - centre).normalized()
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        ball = current_ball(context)
        if ball is None:
            col = mu.get_collection(context.scene, "Anaplast_Spheres")
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
            for f in bm.faces:
                f.smooth = True
            ball = mu.new_mesh_object("Ball", bm, col)
            ball.color = (0.35, 0.55, 0.85, 1.0)
            ball["anaplast_part"] = "BALL"
            ball["mesh_radius"] = 1.0
            p.ball_obj = ball
        ball["fit_centre"] = list(centre); ball["fit_radius"] = r; ball["fit_normal"] = list(n); ball["fit_rms"] = rms
        place_ball(ball, p.ball_diameter, p.ball_sink)
        context.view_layer.objects.active = obj
        shown = p.ball_diameter if p.ball_diameter > 0 else 2 * r
        rep(self, {'INFO'}, f"Ball: Ø {shown:.2f} mm (dome fits Ø {2 * r:.2f} at {rms:.2f} mm rms over {len(pts):,} vertices) — adjust with Diameter override / Sink in")
        return {'FINISHED'}


class ANAPLAST_OT_ball_new(bpy.types.Operator):
    """Start a second ball (the current one is kept); the next Fit creates a new one"""
    bl_idname = "anaplast.ball_new"
    bl_label = "New ball"

    def execute(self, context):
        p = context.scene.anaplast
        b = current_ball(context)
        if b is not None:
            b.name = f"Ball_{len([o for o in bpy.data.objects if o.name.startswith('Ball')])}"
        p.ball_obj = None
        p.ball_diameter = 0.0; p.ball_sink = 0.0
        return {'FINISHED'}


class ANAPLAST_OT_ball_to_prosthesis(bpy.types.Operator):
    """Make the selected ball the Sculpt slot (e.g. an ocular to set into an orbital prototype)"""
    bl_idname = "anaplast.ball_to_slot"
    bl_label = "Use ball as ocular"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.name.startswith("Ball")

    def execute(self, context):
        context.scene.anaplast.ocular_obj = context.active_object
        rep(self, {'INFO'}, f"{context.active_object.name} set as the ocular")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_fit_sphere)
    bpy.utils.register_class(ANAPLAST_OT_ball_new)
    bpy.utils.register_class(ANAPLAST_OT_ball_to_prosthesis)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_ball_to_prosthesis)
    bpy.utils.unregister_class(ANAPLAST_OT_ball_new)
    bpy.utils.unregister_class(ANAPLAST_OT_fit_sphere)
