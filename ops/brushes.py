"""Your own brush library: save the brush you have tuned (size, strength, alpha, stroke) into the module's
brush file. It is registered as a Blender asset library, so the brushes show up together under an
'Anaplast' catalog in Sculpt Mode's brush shelf, and the buttons here activate them directly."""
import os
import json
import uuid
import bpy
from ..utils import mesh as mu
from .report import rep

LIB_NAME = "Anaplast brushes"
CATALOG = "Anaplast"


def lib_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "brushes")


def lib_file():
    return os.path.join(lib_dir(), "anaplast_brushes.blend")


def index_file():
    return os.path.join(lib_dir(), "brushes.json")


def read_index():
    try:
        with open(index_file(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def write_index(names):
    os.makedirs(lib_dir(), exist_ok=True)
    with open(index_file(), "w", encoding="utf-8") as f:
        json.dump(sorted(set(names)), f, indent=1)


def ensure_library(context):
    """Register the module's brushes folder as an asset library (once) and write the catalog file."""
    prefs = context.preferences
    d = lib_dir()
    os.makedirs(d, exist_ok=True)
    cats = os.path.join(d, "blender_assets.cats.txt")
    if not os.path.exists(cats):
        with open(cats, "w", encoding="utf-8") as f:
            f.write("# This is an Anthropic-made catalog file for the Anaplast brush library\nVERSION 1\n\n")
            f.write(f"{uuid.uuid4()}:{CATALOG}:{CATALOG}\n")
    for lib in prefs.filepaths.asset_libraries:
        if os.path.normpath(lib.path) == os.path.normpath(d):
            return lib
    try:
        bpy.ops.preferences.asset_library_add(directory=d)
        for lib in prefs.filepaths.asset_libraries:
            if os.path.normpath(lib.path) == os.path.normpath(d):
                lib.name = LIB_NAME
                return lib
    except Exception:
        pass
    return None


def catalog_id():
    cats = os.path.join(lib_dir(), "blender_assets.cats.txt")
    try:
        with open(cats, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith(("#", "VERSION")):
                    return line.split(":")[0].strip()
    except Exception:
        pass
    return ""


class ANAPLAST_OT_brush_save(bpy.types.Operator):
    """Save the brush you are using right now (with its size, strength, alpha and stroke settings) into your Anaplast brush library"""
    bl_idname = "anaplast.brush_save"
    bl_label = "Save current brush"

    name: bpy.props.StringProperty(name="Brush name", default="My skin brush")

    @classmethod
    def poll(cls, context):
        return context.mode == 'SCULPT' and context.tool_settings.sculpt.brush is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        src = context.tool_settings.sculpt.brush
        name = self.name.strip() or src.name
        br = src.copy()
        br.name = name
        br.use_fake_user = True
        try:
            br.asset_mark()
            br.asset_data.catalog_id = catalog_id()
            br.asset_data.description = "Anaplast custom brush"
        except Exception:
            pass
        ensure_library(context)
        path = lib_file()
        # merge with brushes already in the library: prefer the copy already in this session (loading a
        # second copy would rename it '.001'), load from the file only what the session doesn't have
        existing = {}
        wanted = [n for n in read_index() if n != name]
        missing = [n for n in wanted if bpy.data.brushes.get(n) is None]
        if missing and os.path.exists(path):
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                data_to.brushes = [b for b in data_from.brushes if b in missing]
        for n in wanted:
            b = bpy.data.brushes.get(n)
            if b is not None:
                existing[n] = b
        blocks = set(existing.values()) | {br}
        tex_blocks = {b.texture for b in blocks if b.texture is not None}
        img_blocks = {t.image for t in tex_blocks if getattr(t, "image", None) is not None}
        bpy.data.libraries.write(path, blocks | tex_blocks | img_blocks, fake_user=True, compress=True)
        names = read_index() + [name]
        write_index(names)
        rep(self, {'INFO'}, f"'{name}' saved to the Anaplast brush library ({len(set(names))} brushes). It appears under the Anaplast catalog in the brush shelf")
        return {'FINISHED'}


class ANAPLAST_OT_brush_use(bpy.types.Operator):
    """Activate one of your saved brushes in Sculpt Mode"""
    bl_idname = "anaplast.brush_use"
    bl_label = "Use brush"

    name: bpy.props.StringProperty()

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        from .sculpt_surface import target, activate
        obj = target(context)
        if obj is None:
            rep(self, {'ERROR'}, "Nothing to sculpt in the chosen target")
            return {'CANCELLED'}
        br = bpy.data.brushes.get(self.name)
        if br is None and os.path.exists(lib_file()):
            with bpy.data.libraries.load(lib_file(), link=False) as (data_from, data_to):
                data_to.brushes = [b for b in data_from.brushes if b == self.name]
            br = bpy.data.brushes.get(self.name)
        if br is None:
            rep(self, {'ERROR'}, f"Brush '{self.name}' not found in the library")
            return {'CANCELLED'}
        activate(context, obj)
        bpy.ops.object.mode_set(mode='SCULPT')
        activated = False
        try:
            lib = ensure_library(context)
            bpy.ops.brush.asset_activate(asset_library_type='CUSTOM', asset_library_identifier=(lib.name if lib else LIB_NAME),
                                         relative_asset_identifier=os.path.basename(lib_file()) + "/Brush/" + self.name)
            activated = True
        except Exception:
            try:
                context.tool_settings.sculpt.brush = br
                activated = True
            except Exception:
                pass
        rep(self, {'INFO'}, f"Brush '{self.name}' on {obj.name}" if activated else f"Loaded '{self.name}' — pick it from the Anaplast catalog in the brush shelf")
        return {'FINISHED'}


class ANAPLAST_OT_brush_remove(bpy.types.Operator):
    """Remove a saved brush from the library"""
    bl_idname = "anaplast.brush_remove"
    bl_label = "Remove"

    name: bpy.props.StringProperty()

    def execute(self, context):
        names = [n for n in read_index() if n != self.name]
        write_index(names)
        path = lib_file()
        if os.path.exists(path):
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                data_to.brushes = [b for b in data_from.brushes if b != self.name]
            blocks = {b for b in data_to.brushes if b is not None}
            tex_blocks = {b.texture for b in blocks if b.texture is not None}
            img_blocks = {t.image for t in tex_blocks if getattr(t, "image", None) is not None}
            if blocks:
                bpy.data.libraries.write(path, blocks | tex_blocks | img_blocks, fake_user=True, compress=True)
            else:
                os.remove(path)
        rep(self, {'INFO'}, f"'{self.name}' removed")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_brush_save, ANAPLAST_OT_brush_use, ANAPLAST_OT_brush_remove)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
