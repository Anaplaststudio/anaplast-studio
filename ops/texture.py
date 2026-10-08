"""Skin texture — three geometric approaches. Only geometry survives into a mold,
so every tool here moves vertices; none is a render-only material trick."""
import os
import bpy
import bmesh
import numpy as np
from ..utils import mesh as mu
from .report import rep


def _densify(obj, levels):
    if levels <= 0:
        return
    def setup(m):
        m.levels = levels
        m.render_levels = levels
        m.subdivision_type = 'SIMPLE'
    mu.apply_modifier(obj, 'SUBSURF', setup)


def _target(context):
    from .sculpt_surface import target
    return target(context)


class ANAPLAST_OT_texture_enhance(bpy.types.Operator):
    """1 · Accentuate the scan's own detail: high-pass the surface and add it back with gain (geometric unsharp mask)"""
    bl_idname = "anaplast.texture_enhance"
    bl_label = "Enhance Scan Detail"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def execute(self, context):
        p = context.scene.anaplast
        obj = _target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        me = obj.data
        n = len(me.vertices)
        orig = np.empty(n * 3); me.vertices.foreach_get("co", orig); orig = orig.reshape(-1, 3)
        normals = np.empty(n * 3); me.vertices.foreach_get("normal", normals); normals = normals.reshape(-1, 3)
        # volume-preserving Laplacian smooth on a throwaway copy = the "low-pass" version of the surface
        tmp = mu.duplicate_object(obj, "anaplast_tmp_smooth", obj.users_collection[0])
        def setup(m):
            m.iterations = p.tex_radius
            m.lambda_factor = 1.0
            m.lambda_border = 0.0
            m.use_volume_preserve = True
            m.use_normalized = True
        mu.apply_modifier(tmp, 'LAPLACIANSMOOTH', setup)
        s1 = np.empty(n * 3); tmp.data.vertices.foreach_get("co", s1); s1 = s1.reshape(-1, 3)
        mu.apply_modifier(tmp, 'LAPLACIANSMOOTH', setup)        # smooth the smoothed copy once more
        s2 = np.empty(n * 3); tmp.data.vertices.foreach_get("co", s2); s2 = s2.reshape(-1, 3)
        mu.delete_object(tmp)
        # (orig - s1) holds detail + curvature shrink; (s1 - s2) holds almost only the shrink.
        # Their difference isolates the fine detail without flattening or inflating the form.
        along = ((orig - 2 * s1 + s2) * normals).sum(axis=1, keepdims=True)
        detail = normals * along
        new = orig + detail * p.tex_gain
        me.vertices.foreach_set("co", new.ravel())
        me.update()
        amp = np.linalg.norm(detail, axis=1)
        rep(self, {'INFO'}, f"Detail ×{1 + p.tex_gain:.1f}: median detail {np.median(amp):.3f} mm, max {amp.max():.2f} mm")
        return {'FINISHED'}


class ANAPLAST_OT_texture_procedural(bpy.types.Operator):
    """2 · Add procedural pores and fine creases as real displacement"""
    bl_idname = "anaplast.texture_procedural"
    bl_label = "Add Procedural Skin"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def execute(self, context):
        p = context.scene.anaplast
        obj = _target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        _densify(obj, p.tex_subdiv)

        pores = bpy.data.textures.get("Anaplast_Pores") or bpy.data.textures.new("Anaplast_Pores", 'VORONOI')
        pores.noise_scale = p.tex_pore_size
        pores.noise_intensity = 1.0
        pores.color_mode = 'INTENSITY'
        pores.distance_metric = 'DISTANCE'

        creases = bpy.data.textures.get("Anaplast_Creases") or bpy.data.textures.new("Anaplast_Creases", 'CLOUDS')
        creases.noise_scale = p.tex_crease_size
        creases.noise_depth = 3
        creases.noise_basis = 'ORIGINAL_PERLIN'

        def disp(tex, strength):
            def setup(m):
                m.texture = tex
                m.texture_coords = 'GLOBAL'
                m.direction = 'NORMAL'
                m.mid_level = 0.5
                m.strength = strength
            mu.apply_modifier(obj, 'DISPLACE', setup)

        disp(pores, -p.tex_pore_depth)      # negative: pores are depressions
        disp(creases, p.tex_crease_depth)
        rep(self, {'INFO'}, f"Procedural skin: pores {p.tex_pore_size:.2f} mm / {p.tex_pore_depth:.2f} mm deep, creases {p.tex_crease_size:.1f} mm")
        return {'FINISHED'}


class ANAPLAST_OT_texture_skin(bpy.types.Operator):
    """Layered skin micro-relief: polygonal micro-plateaus with groove network, pores, fine creases along a chosen direction, micro-roughness — all as real geometry"""
    bl_idname = "anaplast.texture_skin"
    bl_label = "Add Realistic Skin"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def execute(self, context):
        import math
        from mathutils import Euler
        p = context.scene.anaplast
        obj = _target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        _densify(obj, p.tex_subdiv)
        # an empty gives every layer its own scale/orientation (crease lines run along one direction)
        frame = bpy.data.objects.get("Anaplast_TexFrame")
        if frame is None:
            frame = bpy.data.objects.new("Anaplast_TexFrame", None)
            context.scene.collection.objects.link(frame)
            frame.hide_viewport = True
        frame.rotation_euler = Euler((0.0, 0.0, math.radians(p.skin_line_angle)))
        s = p.skin_scale
        # 1 · micro-plateau network: Voronoi F2-F1, plateaus raised, grooves between (the primary skin pattern)
        net = bpy.data.textures.get("Anaplast_SkinNet") or bpy.data.textures.new("Anaplast_SkinNet", 'VORONOI')
        net.noise_scale = 0.9 * s; net.distance_metric = 'DISTANCE'; net.color_mode = 'INTENSITY'
        net.weight_1 = -1.0; net.weight_2 = 1.0; net.weight_3 = 0.0; net.weight_4 = 0.0; net.noise_intensity = 1.0
        # 2 · secondary, larger cells (skin islands) — softer
        isl = bpy.data.textures.get("Anaplast_SkinIslands") or bpy.data.textures.new("Anaplast_SkinIslands", 'VORONOI')
        isl.noise_scale = 2.2 * s; isl.distance_metric = 'DISTANCE'; isl.color_mode = 'INTENSITY'
        isl.weight_1 = -1.0; isl.weight_2 = 1.0; isl.weight_3 = 0.0; isl.weight_4 = 0.0
        # 3 · pores: sparse sharp F1 depressions
        pores = bpy.data.textures.get("Anaplast_SkinPores") or bpy.data.textures.new("Anaplast_SkinPores", 'VORONOI')
        pores.noise_scale = 1.4 * s; pores.distance_metric = 'DISTANCE_SQUARED'; pores.color_mode = 'INTENSITY'
        pores.weight_1 = 1.0; pores.weight_2 = 0.0; pores.weight_3 = 0.0; pores.weight_4 = 0.0; pores.noise_intensity = 1.6
        # 4 · fine creases along the line direction (anisotropic clouds), 5 · micro roughness
        cre = bpy.data.textures.get("Anaplast_SkinCreases") or bpy.data.textures.new("Anaplast_SkinCreases", 'CLOUDS')
        cre.noise_scale = 3.0 * s; cre.noise_depth = 2; cre.noise_basis = 'ORIGINAL_PERLIN'
        mic = bpy.data.textures.get("Anaplast_SkinMicro") or bpy.data.textures.new("Anaplast_SkinMicro", 'CLOUDS')
        mic.noise_scale = 0.25 * s; mic.noise_depth = 1

        def disp(tex, strength, mid, aniso=None):
            def setup(m):
                m.texture = tex
                m.direction = 'NORMAL'
                m.mid_level = mid
                m.strength = strength
                if aniso is not None:
                    frame.scale = aniso
                    m.texture_coords = 'OBJECT'
                    m.texture_coords_object = frame
                else:
                    m.texture_coords = 'GLOBAL'
            mu.apply_modifier(obj, 'DISPLACE', setup)
        d = p.skin_depth
        disp(net, +0.55 * d, 0.0)                       # plateaus up, grooves stay
        disp(isl, +0.35 * d, 0.0)
        disp(pores, -0.9 * d, 0.85)                     # only the deepest F1 wells go down = sparse pores
        disp(cre, +0.6 * d, 0.5, aniso=(1.0, 0.18, 1.0))   # stretched → crease lines
        disp(mic, +0.15 * d, 0.5)
        rep(self, {'INFO'}, f"Realistic skin: scale {s:.2f}, depth {d:.2f} mm, lines at {p.skin_line_angle:.0f}°. Undo (Ctrl+Z) and adjust to taste")
        return {'FINISHED'}


class ANAPLAST_OT_texture_image(bpy.types.Operator):
    """3 · Displace by a grayscale skin alpha image (Texturing.xyz, free alpha packs, or your own photo of skin)"""
    bl_idname = "anaplast.texture_image"
    bl_label = "Displace by Skin Alpha"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.exr", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        p = context.scene.anaplast
        obj = _target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            rep(self, {'ERROR'}, f"Could not load image: {e}")
            return {'CANCELLED'}
        _densify(obj, p.tex_subdiv)

        # UV: smart project so the alpha wraps the surface without a manual unwrap
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02, scale_to_bounds=True)
        bpy.ops.object.mode_set(mode='OBJECT')

        tex = bpy.data.textures.get("Anaplast_Alpha") or bpy.data.textures.new("Anaplast_Alpha", 'IMAGE')
        tex.image = img
        tex.extension = 'REPEAT'
        tex.repeat_x = tex.repeat_y = max(1, int(p.tex_alpha_tiles))
        tex.use_interpolation = True

        def setup(m):
            m.texture = tex
            m.texture_coords = 'UV'
            m.uv_layer = obj.data.uv_layers.active.name
            m.direction = 'NORMAL'
            m.mid_level = 0.5
            m.strength = p.tex_alpha_depth
        mu.apply_modifier(obj, 'DISPLACE', setup)
        rep(self, {'INFO'}, f"Alpha applied: {os.path.basename(self.filepath)}, {p.tex_alpha_tiles} tiles, ±{p.tex_alpha_depth/2:.2f} mm")
        return {'FINISHED'}


class ANAPLAST_OT_skin_brush(bpy.types.Operator):
    """Enter Sculpt Mode on the Sculpt with a pore/wrinkle brush: strokes stamp real geometry, like ZBrush alphas"""
    bl_idname = "anaplast.skin_brush"
    bl_label = "Skin Brush (Sculpt)"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.exr", options={'HIDDEN'})
    use_image: bpy.props.BoolProperty(name="Pick an alpha image", default=True)

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def invoke(self, context, event):
        if self.use_image:
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}
        return self.execute(context)

    def execute(self, context):
        p = context.scene.anaplast
        obj = _target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        if self.use_image and self.filepath:
            try:
                img = bpy.data.images.load(self.filepath, check_existing=True)
            except RuntimeError as e:
                rep(self, {'ERROR'}, f"Could not load image: {e}")
                return {'CANCELLED'}
            tex = bpy.data.textures.get("Anaplast_BrushAlpha") or bpy.data.textures.new("Anaplast_BrushAlpha", 'IMAGE')
            tex.image = img
        else:
            tex = bpy.data.textures.get("Anaplast_BrushPores") or bpy.data.textures.new("Anaplast_BrushPores", 'VORONOI')
            tex.noise_scale = p.tex_pore_size
        _densify(obj, p.tex_subdiv)
        br = bpy.data.brushes.get("Anaplast Skin") or bpy.data.brushes.new("Anaplast Skin", mode='SCULPT')
        try:
            br.sculpt_tool = 'DRAW'
        except Exception:
            pass
        br.strength = p.brush_strength
        br.texture = tex
        br.texture_slot.map_mode = 'VIEW_PLANE'
        br.texture_slot.scale = (p.tex_alpha_tiles, p.tex_alpha_tiles, p.tex_alpha_tiles)
        br.use_accumulate = False
        try:
            br.direction = 'SUBTRACT'            # pores go IN (enum depends on the tool; Ctrl inverts anyway)
        except TypeError:
            pass
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        try:
            bpy.ops.object.mode_set(mode='SCULPT')
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.draw")
        except Exception:
            pass
        try:
            context.tool_settings.sculpt.brush = br
        except Exception:
            pass
        rep(self, {'INFO'}, "Sculpt Mode: brush 'Anaplast Skin' — paint pores/wrinkles; F = size, Shift+F = strength, Ctrl = push out. Tab returns to Object Mode")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_texture_enhance, ANAPLAST_OT_texture_procedural, ANAPLAST_OT_texture_skin, ANAPLAST_OT_texture_image, ANAPLAST_OT_skin_brush)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
