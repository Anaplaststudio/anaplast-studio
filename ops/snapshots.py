"""Standard-view snapshots of the case: R90, R45, front, L45, L90, back, top-45, bottom-45."""
import os
import math
import bpy
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep

VIEWS = [   # name, direction FROM which the camera looks (patient: +X right, +Y anterior, +Z up)
    ("right_90",  Vector((1, 0, 0))),
    ("right_45",  Vector((1, 1, 0)).normalized()),
    ("front",     Vector((0, 1, 0))),
    ("left_45",   Vector((-1, 1, 0)).normalized()),
    ("left_90",   Vector((-1, 0, 0))),
    ("back",      Vector((0, -1, 0))),
    ("top_45",    Vector((0, 1, 1)).normalized()),
    ("bottom_45", Vector((0, 1, -1)).normalized()),
]
HELPER_PREFIXES = ("LM_", "AL_", "MG_", "Margin_Preview", "Crop_Box", "Split_Plane_", "Frame_")


def case_bounds(scene):
    p = scene.anaplast
    from .shell import current_prototype
    objs = [o for o in (p.face_scan_obj, current_prototype(p)) if o is not None]
    if not objs:
        objs = [o for o in scene.objects if o.type == 'MESH' and not o.name.startswith(HELPER_PREFIXES)]
    if not objs:
        return None, None
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for o in objs:
        a, b = mu.world_bbox(o)
        lo = Vector((min(lo.x, a.x), min(lo.y, a.y), min(lo.z, a.z)))
        hi = Vector((max(hi.x, b.x), max(hi.y, b.y), max(hi.z, b.z)))
    return (lo + hi) * 0.5, (hi - lo).length * 0.5


def side_by_side(path_a, path_b, out_path):
    """Compose two rendered PNGs of equal size into one image, left | right."""
    import numpy as np
    ia = bpy.data.images.load(path_a); ib = bpy.data.images.load(path_b)
    w, h = ia.size
    A = np.empty(w * h * 4, dtype=np.float32); ia.pixels.foreach_get(A); A = A.reshape(h, w, 4)
    B = np.empty(w * h * 4, dtype=np.float32); ib.pixels.foreach_get(B); B = B.reshape(h, w, 4)
    gap = np.ones((h, 24, 4), dtype=np.float32)
    C = np.concatenate([A, gap, B], axis=1)
    out = bpy.data.images.new("anaplast_compare", C.shape[1], h, alpha=True)
    out.pixels.foreach_set(C.ravel())
    out.filepath_raw = out_path; out.file_format = 'PNG'; out.save()
    for im in (ia, ib, out):
        bpy.data.images.remove(im)


class ANAPLAST_OT_snapshots(bpy.types.Operator):
    """Save eight standard-view images of the case (right 90/45, front, left 45/90, back, top and bottom 45) to the export folder"""
    bl_idname = "anaplast.snapshots"
    bl_label = "Snapshots (8 views)"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    focus_prototype: bpy.props.BoolProperty(name="Frame the prototype only", default=False)
    mode: bpy.props.EnumProperty(items=[("WITH", "Face + sculpt", ""), ("WITHOUT", "Face only", ""), ("PROTO", "Sculpt only", ""),
                                        ("COMPARE", "Side by side (without | with)", "")], default="WITH")

    def invoke(self, context, event):
        p = context.scene.anaplast
        out = bpy.path.abspath(p.export_dir) if p.export_dir else ""
        if out and os.path.isdir(out):
            self.directory = os.path.join(out, "snapshots")
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        p = scene.anaplast
        out = self.directory or os.path.join(bpy.path.abspath(p.export_dir) or os.path.expanduser("~"), "snapshots")
        os.makedirs(out, exist_ok=True)
        if self.mode == 'PROTO':
            self.focus_prototype = True
        if self.focus_prototype:
            from .shell import current_prototype
            proto = current_prototype(p)
            if proto is None:
                rep(self, {'ERROR'}, "No prototype built yet")
                return {'CANCELLED'}
            lo, hi = mu.world_bbox(proto); centre, radius = (lo + hi) * 0.5, (hi - lo).length * 0.5
        else:
            centre, radius = case_bounds(scene)
        if centre is None:
            rep(self, {'ERROR'}, "Nothing to photograph — import a scan first")
            return {'CANCELLED'}
        cam_data = bpy.data.cameras.get("Anaplast_Cam") or bpy.data.cameras.new("Anaplast_Cam")
        cam = bpy.data.objects.get("Anaplast_Cam") or bpy.data.objects.new("Anaplast_Cam", cam_data)
        if cam.name not in scene.collection.objects and not any(cam.name in c.objects for c in bpy.data.collections):
            scene.collection.objects.link(cam)
        cam_data.lens = 85.0
        cam_data.sensor_width = 36.0
        cam_data.clip_end = 100000.0
        fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
        dist = radius / math.tan(fov / 2) * 1.15
        # render settings: solid/workbench, white background, 1920x1440
        prev = dict(engine=scene.render.engine, camera=scene.camera, path=scene.render.filepath,
                    rx=scene.render.resolution_x, ry=scene.render.resolution_y, pct=scene.render.resolution_percentage,
                    fmt=scene.render.image_settings.file_format, film=scene.render.film_transparent)
        scene.render.engine = 'BLENDER_WORKBENCH'
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 1920, 1440, 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.film_transparent = False
        sh = scene.display.shading
        sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_specular_highlight = True
        sh.show_cavity = True; sh.cavity_type = 'BOTH'
        sh.background_type = 'VIEWPORT'
        try:
            sh.background_color = (1.0, 1.0, 1.0)
        except Exception:
            pass
        from .shell import current_prototype
        proto = current_prototype(p)
        keep_with = {o for o in (p.face_scan_obj, proto) if o is not None}
        keep_without = {o for o in (p.face_scan_obj,) if o is not None}
        if self.mode == 'PROTO':
            keep_with = {o for o in (proto,) if o is not None}
        hidden = []
        for o in scene.objects:
            if o not in keep_with and not o.hide_render:
                o.hide_render = True; hidden.append(o)
        proto_was_hidden = proto.hide_render if proto is not None else False
        scene.camera = cam
        written = []
        tag = mu.case_tag(p)
        mu.clear_old(out, tag + "_", (".png",))
        def shoot(path):
            scene.render.filepath = path
            if not bpy.app.background:
                bpy.ops.render.render(write_still=True)
            written.append(path)
        try:
            for name, d in VIEWS:
                cam.location = centre + d * dist
                forward = (centre - cam.location).normalized()
                cam.rotation_euler = forward.to_track_quat('-Z', 'Y').to_euler()   # camera looks down its -Z
                if self.mode in ('WITH', 'PROTO'):
                    shoot(os.path.join(out, f"{tag}_{name}.png"))
                elif self.mode == 'WITHOUT':
                    if proto is not None: proto.hide_render = True
                    shoot(os.path.join(out, f"{tag}_{name}_noprosthesis.png"))
                    if proto is not None: proto.hide_render = proto_was_hidden
                else:   # COMPARE
                    if proto is not None: proto.hide_render = True
                    a = os.path.join(out, f"{tag}_{name}_noprosthesis.png"); shoot(a)
                    if proto is not None: proto.hide_render = proto_was_hidden
                    b = os.path.join(out, f"{tag}_{name}.png"); shoot(b)
                    if not bpy.app.background:
                        side_by_side(a, b, os.path.join(out, f"{tag}_{name}_compare.png"))
        finally:
            for o in hidden:
                o.hide_render = False
            scene.render.engine = prev["engine"]; scene.camera = prev["camera"]; scene.render.filepath = prev["path"]
            scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = prev["rx"], prev["ry"], prev["pct"]
            scene.render.image_settings.file_format = prev["fmt"]; scene.render.film_transparent = prev["film"]
        rep(self, {'INFO'}, f"{len(written)} snapshots saved to {out}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_snapshots)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_snapshots)
