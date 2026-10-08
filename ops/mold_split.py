import string
import bpy
import bmesh
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep

PLANE_PREFIX = "Split_Plane_"


def split_planes(col):
    return sorted((o for o in col.objects if o.name.startswith(PLANE_PREFIX)), key=lambda o: o.name)


def mold_parts(col):
    return sorted((o for o in col.objects
                   if o.name.startswith("Mold_") and not o.name.startswith("Mold_Block")
                   and o.get("anaplast_part") not in (None, "BLOCK")), key=lambda o: o.name)


class ANAPLAST_OT_add_split_plane(bpy.types.Operator):
    """Tool 3 — add a split plane at the sculpt centre; move/rotate it, then run Split"""
    bl_idname = "anaplast.add_split_plane"
    bl_label = "Add Split Plane"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(name="Normal", items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")], default="Z")

    @classmethod
    def poll(cls, context):
        return context.scene.anaplast.prosthesis_obj is not None

    def execute(self, context):
        p = context.scene.anaplast
        col = mu.get_collection(context.scene)
        lo, hi = mu.world_bbox(p.prosthesis_obj)
        center = (lo + hi) * 0.5
        size = (hi - lo).length + 4 * p.block_offset

        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size * 0.5)
        idx = len(split_planes(col)) + 1
        plane = mu.new_mesh_object(f"{PLANE_PREFIX}{idx}", bm, col)
        rot = {'X': Matrix.Rotation(1.5707963, 4, 'Y'),
               'Y': Matrix.Rotation(-1.5707963, 4, 'X'),
               'Z': Matrix.Identity(4)}[self.axis]
        plane.matrix_world = Matrix.Translation(center) @ rot
        plane.display_type = 'WIRE'
        plane["anaplast_part"] = "PLANE"
        return {'FINISHED'}


class ANAPLAST_OT_mold_split(bpy.types.Operator):
    """Tool 3 — split the mold block by every Split_Plane_* into parts Mold_A, Mold_B, …"""
    bl_idname = "anaplast.mold_split"
    bl_label = "Split Mold"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Mold_Block") is not None

    def execute(self, context):
        col = mu.get_collection(context.scene)
        block = bpy.data.objects["Mold_Block"]
        planes = split_planes(col)
        if not planes:
            rep(self, {'ERROR'}, "Add at least one split plane first")
            return {'CANCELLED'}

        for o in mold_parts(col):
            mu.delete_object(o)

        lo, hi = mu.world_bbox(block)
        big = (hi - lo).length * 4.0

        parts = [mu.duplicate_object(block, "Mold_tmp0", col)]
        parts[0]["anaplast_sides"] = ""
        for plane in planes:
            # half-space cutter: cube on the +normal side of the plane
            cutter = mu.make_box("anaplast_cutter", Vector((-big, -big, 0.0)), Vector((big, big, big)), col)
            cutter.matrix_world = plane.matrix_world.copy()
            next_parts = []
            for part in parts:
                pos = mu.duplicate_object(part, part.name + "p", col)
                mu.apply_boolean(pos, cutter, 'INTERSECT')
                neg = mu.duplicate_object(part, part.name + "n", col)
                mu.apply_boolean(neg, cutter, 'DIFFERENCE')
                sides = part.get("anaplast_sides", "")
                kept = []
                for cand, tag in ((pos, "+"), (neg, "-")):
                    if len(cand.data.polygons) == 0:
                        mu.delete_object(cand)
                    else:
                        cand["anaplast_sides"] = sides + f"{plane.name}:{tag};"
                        kept.append(cand)
                mu.delete_object(part)
                next_parts.extend(kept)
            mu.delete_object(cutter)
            parts = next_parts

        letters = string.ascii_uppercase
        for i, part in enumerate(parts):
            part["anaplast_part"] = "MOLD"
            part["mold_index"] = i + 1
            part["open_dir"] = list(axis) if 'axis' in dir() else [0.0, 0.0, 1.0]
            part.name = f"Mold_{letters[i]}"
            part.data.name = part.name
            part["anaplast_part"] = letters[i]
            mu.cleanup_mesh(part)
        block.hide_set(True)
        block.hide_render = True
        rep(self, {'INFO'}, f"Split into {len(parts)} parts")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_add_split_plane)
    bpy.utils.register_class(ANAPLAST_OT_mold_split)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_split)
    bpy.utils.unregister_class(ANAPLAST_OT_add_split_plane)
