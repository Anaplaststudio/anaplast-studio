"""Contact check between the two mold halves in their closed position: colour the cap's underside by its
relation to the base — green touching, blue gap, red interpenetration — and report the numbers."""
import numpy as np
import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from .report import rep


class ANAPLAST_OT_mold_contact(bpy.types.Operator):
    """Show where the cap and base touch (green), have a gap (blue) or run into each other (red), measured with the mold closed. Reports contact %, largest gap and deepest interpenetration"""
    bl_idname = "anaplast.mold_contact"
    bl_label = "Check contact"
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: bpy.props.FloatProperty(name="Touching within (mm)", default=0.15, min=0.01, soft_max=1.0)
    gap_range: bpy.props.FloatProperty(name="Blue at gap (mm)", default=1.0, min=0.1, soft_max=5.0)

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Mold_Cap") is not None and bpy.data.objects.get("Mold_Base") is not None

    def execute(self, context):
        from .explode import HOME
        p = context.scene.anaplast
        cap, base = bpy.data.objects["Mold_Cap"], bpy.data.objects["Mold_Base"]
        # measure in the closed position, whatever the current display state
        cap_mat = Matrix([list(r) for r in cap[HOME + "_mat"]]) if (HOME + "_mat") in cap else cap.matrix_world.copy()
        dg = context.evaluated_depsgraph_get()
        tree = BVHTree.FromObject(base, dg)
        b_inv = base.matrix_world.inverted()
        me = cap.data
        n = len(me.vertices)
        co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        nrm = np.empty(n * 3); me.vertices.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
        M = np.array(cap_mat); W = co @ M[:3, :3].T + M[:3, 3]
        up = Vector(cap.get("mold_up", (0, 0, 1)))
        Nw = nrm @ M[:3, :3].T
        # the cavity (the prosthesis negative) faces the prosthesis, not the base: leave it out of the statistics
        from .shell import source_surface
        sculpt = source_surface(p)
        s_tree = BVHTree.FromObject(sculpt, dg) if sculpt is not None else None
        s_inv = sculpt.matrix_world.inverted() if sculpt is not None else None
        signed = np.full(n, np.nan)
        cavity = np.zeros(n, dtype=bool)
        for i in range(n):
            if Nw[i] @ np.array(up) > -0.2:
                continue                                                 # only the cap's underside faces the base
            if s_tree is not None:
                sl, sn, si, sd = s_tree.find_nearest(s_inv @ Vector(W[i]), 0.6)
                if sl is not None:
                    cavity[i] = True
                    continue
            loc, bn, idx, dist = tree.find_nearest(b_inv @ Vector(W[i]), 3.0)
            if loc is None:
                continue
            w_loc = base.matrix_world @ loc
            inside = (Vector(W[i]) - w_loc).dot(base.matrix_world.to_3x3() @ bn) < 0
            signed[i] = -dist if inside else dist
        valid = ~np.isnan(signed)
        r = np.zeros(n); g = np.zeros(n); b = np.zeros(n)
        grey = 0.7
        r[:] = g[:] = b[:] = grey
        touch = valid & (np.abs(signed) <= self.tolerance)
        gap = valid & (signed > self.tolerance)
        pen = valid & (signed < -self.tolerance)
        g[touch] = 0.9; r[touch] = 0.1; b[touch] = 0.2
        t = np.clip(signed[gap] / self.gap_range, 0, 1)
        r[gap] = 0.2 * (1 - t); g[gap] = 0.6 * (1 - t) + 0.2 * t; b[gap] = 0.5 + 0.5 * t
        r[pen] = 1.0; g[pen] = 0.1; b[pen] = 0.1
        r[cavity] = 0.85; g[cavity] = 0.72; b[cavity] = 0.62                    # the negative itself: skin tone
        attr = me.color_attributes.get("contact") or me.color_attributes.new("contact", 'FLOAT_COLOR', 'POINT')
        attr.data.foreach_set("color", np.stack([r, g, b, np.ones(n)], axis=1).ravel())
        me.color_attributes.active_color = attr
        mat = bpy.data.materials.get("Anaplast_contact")
        if mat is None:
            mat = bpy.data.materials.new("Anaplast_contact"); mat.use_nodes = True
            nt = mat.node_tree; a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_name = "contact"
            nt.links.new(a.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        if not me.materials:
            me.materials.append(mat)
        else:
            me.materials[0] = mat
        try:
            from .align import set_viewport_color
            set_viewport_color(context, 'VERTEX')
        except Exception:
            pass
        nv = int(valid.sum())
        if nv == 0:
            rep(self, {'ERROR'}, "The cap's underside does not face the base — collapse the mold first?")
            return {'CANCELLED'}
        worst = float(-signed[pen].min()) if pen.any() else 0.0
        big_gap = float(signed[gap].max()) if gap.any() else 0.0
        rep(self, {'WARNING' if pen.any() else 'INFO'},
            f"Contact (cavity excluded): {100 * touch.sum() / nv:.1f}% of the cap's underside touches the base (±{self.tolerance:.2f} mm), "
            f"{100 * gap.sum() / nv:.1f}% has a gap (largest {big_gap:.2f} mm), {100 * pen.sum() / nv:.1f}% interpenetrates"
            + (f" — deepest {worst:.2f} mm (RED)" if pen.any() else "") + ". Green touch, blue gap, red overlap")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_mold_contact)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_contact)
