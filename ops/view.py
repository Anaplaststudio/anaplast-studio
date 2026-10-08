"""Viewport display and ZBrush-style sculpt brushes."""
import bpy
from ..utils import mesh as mu
from .report import rep

MODES = [
    ("NORMAL", "Solid", "Plain studio shading, object colours"),
    ("CLAY", "Clay", "Warm clay matcap with cavity — ZBrush look"),
    ("PLASTER", "Plaster", "White plaster–marble matcap with cavity"),
    ("XRAY", "X-ray all", "Make the entire viewport transparent; use Working surface opacity for individual objects"),
    ("SCAN_COLOR", "Scan Color", "Show image textures or vertex colors from the scan"),
]
MATCAPS = {"CLAY": ("clay_brown.exr", "clay_studio.exr"), "METAL": ("metal_shiny.exr", "metal_carpaint.exr", "check_normal+y.exr"),
           "PLASTER": ("basic_1.exr", "pearl.exr", "jade.exr")}
SURFACES = [("SCAN", "Scan"), ("CAST", "Cast"), ("PROTO", "Prototype"), ("SOURCE", "Sculpt")]
SWATCHES = [("skin", (0.85, 0.72, 0.62)), ("stone", (0.88, 0.88, 0.86)), ("silver", (0.72, 0.74, 0.78)),
            ("blue", (0.35, 0.55, 0.85)), ("green", (0.45, 0.75, 0.50)), ("amber", (0.93, 0.70, 0.25))]
BRUSHES = [        # key, label, (legacy tool id, essentials asset name)
    ("DRAW", "Standard", ('builtin_brush.draw', "Draw")),
    ("SMOOTH", "Smooth", ('builtin_brush.smooth', "Smooth")),
    ("GRAB", "Move", ('builtin_brush.grab', "Grab")),
    ("CLAY", "Clay", ('builtin_brush.clay_strips', "Clay Strips")),
    ("INFLATE", "Inflate", ('builtin_brush.inflate', "Inflate/Deflate")),
    ("FLATTEN", "Flatten", ('builtin_brush.flatten', "Flatten/Contrast")),
    ("PINCH", "Pinch", ('builtin_brush.pinch', "Pinch/Magnify")),
    ("MASK", "Mask", ('builtin_brush.mask', "Mask")),
]


def activate_brush(context, key):
    """Blender ≤4.2: tool ids. 4.3+/5.x: brushes are assets — activate from the Essentials library."""
    legacy, asset = dict((a, t) for a, _, t in BRUSHES)[key]
    try:
        bpy.ops.wm.tool_set_by_id(name="builtin.brush")
    except Exception:
        pass
    for fname in ("essentials_brushes-mesh_sculpt.blend", "essentials_sculpt.blend"):
        try:
            bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS', asset_library_identifier="",
                                         relative_asset_identifier=f"brushes/{fname}/Brush/{asset}")
            return True
        except Exception:
            continue
    try:
        bpy.ops.wm.tool_set_by_id(name=legacy)
        return True
    except Exception:
        return False


def scan_color_type(obj):
    from .surface_appearance import color_source
    return color_source(obj) if obj else None


def object_opacity_get(obj):return float(obj.color[3])


def object_opacity_set(obj,value):
    obj.color[3]=value


def object_opacity_update(obj,context):
    from .surface_appearance import update
    update(obj,context)


class ANAPLAST_OT_display_mode(bpy.types.Operator):
    """Set how the viewport shades everything: plain, clay matcap with cavity (best for skin relief), or x-ray"""
    bl_idname = "anaplast.display_mode"
    bl_label = "Display"

    mode: bpy.props.EnumProperty(items=[(a, b, c) for a, b, c in MODES], default="NORMAL")

    def execute(self, context):
        color_type='TEXTURE' if context.scene.get('b4d_object_appearance') else 'OBJECT'
        if self.mode=='SCAN_COLOR':
            from .sculpt_surface import target
            obj=target(context)
            if not obj or not scan_color_type(obj):
                rep(self,{'WARNING'},'No scan colors on the chosen Working surface');return {'CANCELLED'}
            obj.b4d_scan_color=True;color_type='TEXTURE'
        context.scene.anaplast.show_texture=self.mode=='SCAN_COLOR'
        for area in context.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            sh = area.spaces.active.shading
            sh.type = 'SOLID'
            sh.show_xray = (self.mode == 'XRAY')
            if self.mode in MATCAPS:
                sh.light = 'MATCAP'; sh.color_type = color_type; sh.show_cavity = True; sh.cavity_type = 'BOTH'
                for name in MATCAPS[self.mode]:
                    try:
                        sh.studio_light = name
                        break
                    except Exception:
                        continue
            else:
                sh.light = 'STUDIO'; sh.color_type = color_type; sh.show_cavity = False
        return {'FINISHED'}


class ANAPLAST_OT_toggle_visible(bpy.types.Operator):
    """Show or hide the face scan, the cast/stone or the prototype"""
    bl_idname = "anaplast.toggle_visible"
    bl_label = "Show / hide"

    which: bpy.props.EnumProperty(items=[("SCAN", "Scan", ""), ("CAST", "Cast", ""), ("PROTO", "Prototype", ""), ("SOURCE", "Sculpt", "")], default="SCAN")

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj = {"SCAN": p.face_scan_obj, "CAST": p.cast_obj, "PROTO": current_prototype(p), "SOURCE": source_surface(p)}[self.which]
        if obj is None:
            rep(self, {'ERROR'}, "Nothing in that slot")
            return {'CANCELLED'}
        obj.hide_set(not obj.hide_get())
        return {'FINISHED'}


class ANAPLAST_OT_surface_colour(bpy.types.Operator):
    """Give this surface its own viewport colour (shading Color is set to Object so the colours show)"""
    bl_idname = "anaplast.surface_colour"
    bl_label = "Colour"

    which: bpy.props.EnumProperty(items=[(a, b, "") for a, b in SURFACES], default="PROTO")
    swatch: bpy.props.EnumProperty(items=[(a, a.capitalize(), "") for a, _ in SWATCHES], default="skin")

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj = {"SCAN": p.face_scan_obj, "CAST": p.cast_obj, "PROTO": current_prototype(p), "SOURCE": source_surface(p)}[self.which]
        if obj is None:
            rep(self, {'ERROR'}, "Nothing in that slot")
            return {'CANCELLED'}
        obj.b4d_scan_color=False
        obj.b4d_surface_tint=dict(SWATCHES)[self.swatch]
        return {'FINISHED'}


def _apply_section(context):
    p = context.scene.anaplast
    if bpy.app.background:
        return
    axis = {"SAG": (1.0, 0.0, 0.0), "COR": (0.0, 1.0, 0.0), "AXI": (0.0, 0.0, 1.0)}[p.section_axis]
    sign = -1.0 if p.section_flip else 1.0
    n = tuple(sign * a for a in axis)
    # plane: keep the half-space where n·x + d > 0
    plane = (n[0], n[1], n[2], -sign * p.section_pos)
    areas = []
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                areas.append(area)
    done = 0
    for area in areas:
        space = area.spaces.active
        for region in area.regions:
            rv3d = getattr(region, "data", None)
            targets = [rv3d] if rv3d is not None else []
            if not targets and getattr(space, "region_3d", None) is not None:
                targets = [space.region_3d]
            for t in targets:
                if not hasattr(t, "use_clip_planes"):
                    continue
                if p.section_on:
                    t.clip_planes = [plane] * 6
                    t.use_clip_planes = True
                else:
                    t.use_clip_planes = False
                done += 1
        area.tag_redraw()
    return done


def section_update(self, context):
    _apply_section(context)


class ANAPLAST_OT_section(bpy.types.Operator):
    """Hide everything in front of a sagittal / coronal / axial plane, so you can look inside the sculpt and the face. Slide Position to move the cut"""
    bl_idname = "anaplast.section"
    bl_label = "Cut-away"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.anaplast
        p.section_on = not p.section_on
        n = _apply_section(context) or 0
        if p.section_on and n == 0:
            p.section_on = False
            rep(self, {'ERROR'}, "Could not clip this viewport — hover the 3D view and try again")
            return {'CANCELLED'}
        rep(self, {'INFO'}, f"Cut-away ON ({p.section_axis}) — slide Position; everything in front of the plane is hidden" if p.section_on else "Cut-away OFF")
        return {'FINISHED'}


class ANAPLAST_OT_clear_masks(bpy.types.Operator):
    """Remove the sculpt mask from the chosen working surface"""
    bl_idname = "anaplast.clear_masks"
    bl_label = "Clear surface mask"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .sculpt_surface import target
        p = context.scene.anaplast
        obj = target(context)
        objs = [obj] if obj is not None else []
        n = 0
        for o in objs:
            a = o.data.attributes.get(".sculpt_mask")
            if a:
                o.data.attributes.remove(a); n += 1
        rep(self, {'INFO'}, f"Masks cleared on {n} object(s)" if n else "No masks found")
        return {'FINISHED'}


class ANAPLAST_OT_sculpt(bpy.types.Operator):
    """Enter Sculpt Mode on the chosen object with a ZBrush-style brush (F = size, Shift+F = strength, Ctrl = invert, Shift = smooth)"""
    bl_idname = "anaplast.sculpt"
    bl_label = "Sculpt"

    brush: bpy.props.EnumProperty(items=[(a, b, "") for a, b, _ in BRUSHES], default="DRAW")

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        from .sculpt_surface import target, activate
        obj = target(context)
        if obj is None:
            rep(self, {'ERROR'}, f"Nothing to sculpt in '{p.sculpt_target}'")
            return {'CANCELLED'}
        activate(context, obj)
        locked = 0
        bpy.ops.object.mode_set(mode='SCULPT')
        sd = getattr(context, "space_data", None)
        if sd is not None and hasattr(sd, "overlay"):
            sd.overlay.show_sculpt_mask = True
        ok = activate_brush(context, self.brush)
        try:
            br = context.tool_settings.sculpt.brush
            if br is not None:
                br.strength = p.brush_strength
                br.use_frontface = p.brush_frontface
        except Exception:
            pass
        if p.brush_frontface:
            for b in bpy.data.brushes:                   # and every other sculpt brush the user may pick from the shelf
                try:
                    if b.use_paint_sculpt:
                        b.use_frontface = True
                except Exception:
                    pass
        label = dict((a, b) for a, b, _ in BRUSHES)[self.brush]
        note = f"; fitting surface locked ({locked:,} verts masked)" if locked else ""
        rep(self, {'INFO'}, f"{label} brush on {obj.name}{note} — F size, Shift+F strength, Ctrl inverts, Shift smooths, Tab back to Object Mode"
            if ok else f"Sculpt Mode on {obj.name}; pick the {label} brush from the toolbar")
        return {'FINISHED'}


class ANAPLAST_OT_sculpt_exit(bpy.types.Operator):
    """Back to Object Mode"""
    bl_idname = "anaplast.sculpt_exit"
    bl_label = "Done sculpting"

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


_classes = (ANAPLAST_OT_display_mode, ANAPLAST_OT_toggle_visible, ANAPLAST_OT_surface_colour, ANAPLAST_OT_section, ANAPLAST_OT_clear_masks, ANAPLAST_OT_sculpt, ANAPLAST_OT_sculpt_exit)


def register():
    from .surface_appearance import update,tint_get,tint_set
    bpy.types.Object.b4d_scan_color=bpy.props.BoolProperty(name='Use scan color',default=False,update=update,description="Show this surface's scan texture or vertex colors instead of its tint")
    bpy.types.Object.b4d_surface_tint=bpy.props.FloatVectorProperty(name='Tint',size=3,subtype='COLOR',min=0,max=1,get=tint_get,set=tint_set,update=update)

    bpy.types.Object.b4d_surface_opacity=bpy.props.FloatProperty(name='Opacity',min=0,max=1,subtype='FACTOR',get=object_opacity_get,set=object_opacity_set,update=object_opacity_update,
        description='Opacity of this surface only in Solid, Clay or Plaster view; disables global X-ray')
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    del bpy.types.Object.b4d_scan_color
    del bpy.types.Object.b4d_surface_tint
    del bpy.types.Object.b4d_surface_opacity
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
