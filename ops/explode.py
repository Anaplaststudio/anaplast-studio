import bpy
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .mold_split import mold_parts

HOME = "anaplast_home"


def seated_matrix(obj):
    """Read either saved lift format without moving the displayed object."""
    if HOME + "_mat" in obj:
        return Matrix([list(row) for row in obj[HOME + "_mat"]])
    if HOME in obj:
        home = obj[HOME]
        if len(home) == 4:
            return Matrix([list(row) for row in home])
        # Older exploded objects store their local location only.
        local = obj.matrix_basis.copy(); local.translation = Vector(home)
        return obj.matrix_world @ obj.matrix_basis.inverted_safe() @ local
    return obj.matrix_world.copy()


def seat_object(obj):
    obj.matrix_world = seated_matrix(obj)
    for key in (HOME, HOME + "_mat"):
        if key in obj:del obj[key]


def collapse(scene):
    for o in scene.objects:
        if HOME in o or HOME + "_mat" in o:seat_object(o)
    scene.anaplast.exploded = False


def ensure_collapsed(scene):
    if scene.anaplast.exploded:
        collapse(scene)


class ANAPLAST_OT_explode_view(bpy.types.Operator):
    """Push mold parts apart for inspection, or pull them back together (positions are restored exactly)"""
    bl_idname = "anaplast.explode_view"
    bl_label = "Explode / Collapse"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(bpy.data.objects.get(n) is not None for n in ("Mold_A","Mold_Cap","Mold_Wedge"))

    def execute(self, context):
        scene = context.scene
        p = scene.anaplast
        if p.exploded:
            collapse(scene)
            return {'FINISHED'}
        col = mu.get_collection(scene)
        cap, base = bpy.data.objects.get("Mold_Cap"), bpy.data.objects.get("Mold_Base")
        wedge=bpy.data.objects.get("Mold_Wedge")
        if cap is None and base is not None and wedge is not None:
            from mathutils import Matrix
            right=Vector(base['mold_right']);lo,hi=mu.world_bbox(base)
            wedge[HOME+"_mat"]=[list(row) for row in wedge.matrix_world]
            wedge.matrix_world=Matrix.Translation(right*(abs((hi-lo).dot(right))+8.))@wedge.matrix_world
            p.exploded=True;return {'FINISHED'}
        if cap is not None and base is not None and "mold_up" in cap:
            # automatic mold: open it like a book — the cap turns over and lies beside the base, so both
            # inner surfaces face you at once
            from mathutils import Matrix
            up = Vector(cap["mold_up"]); right = Vector(cap.get("mold_right", (1, 0, 0)))
            cap[HOME + "_mat"] = [list(r) for r in cap.matrix_world]
            base[HOME] = list(base.location)
            lo, hi = mu.world_bbox(base)
            width = abs((hi - lo).dot(right))
            c = (mu.world_bbox(cap)[0] + mu.world_bbox(cap)[1]) * 0.5
            # hinge on the right-hand edge of the base: rotate 180° about `up × right` (the front-back axis)
            hinge_axis = up.cross(right).normalized()
            R = Matrix.Rotation(3.14159265, 4, hinge_axis)
            wedge=bpy.data.objects.get("Mold_Wedge")
            cap.matrix_world = Matrix.Translation(c + right * (width + 8.0)*(2 if wedge is not None else 1)) @ R @ Matrix.Translation(-c) @ cap.matrix_world
            if wedge is not None:
                wedge[HOME+"_mat"]=[list(row) for row in wedge.matrix_world]
                wedge.matrix_world=Matrix.Translation(right*(width+8.))@wedge.matrix_world
            p.exploded = True
            return {'FINISHED'}
        parts = mold_parts(col)
        if not parts:
            self.report({'ERROR'}, "No mold parts found")
            return {'CANCELLED'}
        centers = []
        for o in parts:
            lo, hi = mu.world_bbox(o)
            centers.append((lo + hi) * 0.5)
        assembly = sum(centers, Vector()) / len(centers)
        for o, c in zip(parts, centers):
            o[HOME] = list(o.location)
            o.location = o.location + (c - assembly) * p.explode_factor
        p.exploded = True
        return {'FINISHED'}


class ANAPLAST_OT_mold_flip_cap(bpy.types.Operator):
    """Turn the cap over (180° about a horizontal axis) so its underside — the negative of the sculpt — faces you; click again to put it back"""
    bl_idname = "anaplast.mold_flip_cap"
    bl_label = "Flip cap over"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Mold_Cap") is not None

    def execute(self, context):
        from mathutils import Matrix
        cap = bpy.data.objects["Mold_Cap"]
        up = Vector(cap.get("mold_up", (0, 0, 1)))
        right = Vector(cap.get("mold_right", (1, 0, 0)))
        lo, hi = mu.world_bbox(cap)
        c = (lo + hi) * 0.5
        R = Matrix.Rotation(3.14159265, 4, right)
        cap.matrix_world = Matrix.Translation(c) @ R @ Matrix.Translation(-c) @ cap.matrix_world
        cap["flipped"] = not cap.get("flipped", False)
        self.report({'INFO'}, "Cap turned over — you are looking at the negative" if cap["flipped"] else "Cap back in place")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_explode_view)
    bpy.utils.register_class(ANAPLAST_OT_mold_flip_cap)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_flip_cap)
    bpy.utils.unregister_class(ANAPLAST_OT_explode_view)
