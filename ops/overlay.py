"""Crosshair guides that follow the mouse, fit-to-screen helpers, texture display toggle."""
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from ..utils import mesh as mu
from .report import rep

_handle=None
_mouse=[0,0]
_guide_area=None
_guide_window=None
_guide_visible=False
_guide_generation=0


def disable_crosshair(context):
    global _handle,_guide_visible,_guide_generation
    _guide_generation+=1;_guide_visible=False
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle,'WINDOW');_handle=None
    for scene in bpy.data.scenes:
        if hasattr(scene,'anaplast'):scene.anaplast.crosshair=False
    if context and context.screen:
        for area in context.screen.areas:
            if area.type=='VIEW_3D':area.tag_redraw()


def guide_coordinates(area,event):
    from .viewport_tools import visible_rect
    region=next((r for r in area.regions if r.type=='WINDOW'),None)
    if region is None:return None
    x,y=event.mouse_x-region.x,event.mouse_y-region.y
    left,right,bottom,top=visible_rect(area,region)
    return (x,y,left,right,bottom,top) if left<=x<=right and bottom<=y<=top else None


def _draw():
    context=bpy.context
    if not _guide_visible or not context.area or context.area.as_pointer()!=_guide_area:return
    if not context.window or context.window.as_pointer()!=_guide_window:return
    from .viewport_tools import visible_rect
    left,right,bottom,top=visible_rect(context.area,context.region)
    x,y=_mouse;shader=gpu.shader.from_builtin('UNIFORM_COLOR')
    coords=[(left,y),(right,y),(x,bottom),(x,top)]
    oldblend=gpu.state.blend_get()
    try:
        gpu.state.blend_set('ALPHA');shader.bind()
        shader.uniform_float('color',(.95,.65,.2,.65))
        batch_for_shader(shader,'LINES',{'pos':coords}).draw(shader)
    finally:gpu.state.blend_set(oldblend)


class ANAPLAST_OT_crosshair(bpy.types.Operator):
    """Horizontal and vertical lines follow the mouse; click again to turn off"""
    bl_idname='anaplast.crosshair';bl_label='Mouse guides'
    def modal(self,context,event):
        global _guide_visible
        if self.generation!=_guide_generation:return {'CANCELLED'}
        if not context.scene.anaplast.crosshair or _handle is None:
            disable_crosshair(context);return {'CANCELLED'}
        area=next((a for a in context.window.screen.areas if a.as_pointer()==_guide_area),None)
        if area is None or area.type!='VIEW_3D':
            disable_crosshair(context);return {'CANCELLED'}
        if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE','WINDOW_DEACTIVATE'}:
            position=guide_coordinates(area,event) if event.type!='WINDOW_DEACTIVATE' else None
            _guide_visible=position is not None
            if position:_mouse[:]=position[:2]
            area.tag_redraw()
        return {'PASS_THROUGH'}
    def invoke(self,context,event):
        global _handle,_guide_area,_guide_window,_guide_visible
        if _handle is not None:
            disable_crosshair(context);return {'FINISHED'}
        area=clipping_area(context)
        if not area:return {'CANCELLED'}
        _guide_area=area.as_pointer();_guide_window=context.window.as_pointer()
        position=guide_coordinates(area,event);_guide_visible=position is not None
        if position:_mouse[:]=position[:2]
        _handle=bpy.types.SpaceView3D.draw_handler_add(_draw,(),'WINDOW','POST_PIXEL')
        self.generation=_guide_generation
        context.scene.anaplast.crosshair=True
        context.window_manager.modal_handler_add(self);area.tag_redraw()
        return {'RUNNING_MODAL'}
    def cancel(self,context):
        if self.generation==_guide_generation:disable_crosshair(context)


@bpy.app.handlers.persistent
def _reset_guides_on_load(_):disable_crosshair(bpy.context)


PRESETS = [("FRONT", "Front", 0.0, 0.0), ("LEFT", "Left 45°", -45.0, 0.0), ("RIGHT", "Right 45°", 45.0, 0.0),
           ("BACK", "Behind", 180.0, 0.0), ("RAKING", "Raking", 100.0, 0.0)]


def apply_light(context):
    """Point the solid-mode studio light where the user asked."""
    if bpy.app.background:
        return 0
    p = context.scene.anaplast
    n = 0
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        sh = area.spaces.active.shading
        if sh.type == 'SOLID' and sh.light == 'MATCAP':
            continue                             # matcaps carry their own light; leave the user's shading alone
        try:
            sh.use_world_space_lighting = True   # otherwise the light follows the camera and never moves
        except Exception:
            pass
        try:
            sh.studiolight_rotate_z = ((p.light_rot + p.light_zero + 180.0) % 360.0 - 180.0) * 3.14159265 / 180.0
        except Exception:
            pass

        for attr, val in (("studiolight_intensity", 1.0),):
            try:
                setattr(sh, attr, val)
            except Exception:
                pass
        area.tag_redraw()
        n += 1
    return n


class ANAPLAST_OT_light_preset(bpy.types.Operator):
    """Put the viewport light where you want it: front, 45° left or right, top, or raking across the surface"""
    bl_idname = "anaplast.light_preset"
    bl_label = "Light"

    preset: bpy.props.EnumProperty(items=[(a, b, "") for a, b, _r, _t in PRESETS], default="FRONT")

    def execute(self, context):
        p = context.scene.anaplast
        rot, tilt = dict((a, (r, t)) for a, _b, r, t in PRESETS)[self.preset]
        p.light_rot, p.light_tilt = rot, tilt
        apply_light(context)
        rep(self, {'INFO'}, f"Light: {dict((a, b) for a, b, _r, _t in PRESETS)[self.preset]} ({p.light_rot:.0f}°)" + (" — note: matcap shading ignores light direction" if any(a.type == 'VIEW_3D' and a.spaces.active.shading.light == 'MATCAP' for a in context.screen.areas) else ""))
        return {'FINISHED'}


class ANAPLAST_OT_light_calibrate(bpy.types.Operator):
    """Drag the light until it clearly shines onto the patient's face, then click this: that position becomes 'Front' and the presets follow from it"""
    bl_idname = "anaplast.light_calibrate"
    bl_label = "This is front"

    def execute(self, context):
        p = context.scene.anaplast
        p.light_zero = ((p.light_zero + p.light_rot + 180.0) % 360.0) - 180.0
        p.light_rot = 0.0
        apply_light(context)
        rep(self, {'INFO'}, f"Front calibrated (offset {p.light_zero:.0f}°). Presets now: Front 0, L/R ±45, Behind 180, Raking 100")
        return {'FINISHED'}


class ANAPLAST_OT_light_drag(bpy.types.Operator):
    """Move the light with the mouse: drag left/right to swing it around the case. Click to keep, Esc to revert"""
    bl_idname = "anaplast.light_drag"
    bl_label = "Drag light"

    _start_x = 0
    _start_rot = 0.0

    def invoke(self, context, event):
        p = context.scene.anaplast
        self._start_x = event.mouse_x
        self._start_rot = p.light_rot
        apply_light(context)
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Drag left / right to move the light · CLICK to keep · ESC to revert")
        context.window.cursor_modal_set('MOVE_X')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        p = context.scene.anaplast
        if event.type == 'MOUSEMOVE':
            delta = (event.mouse_x - self._start_x) * 0.5          # half a degree per pixel
            p.light_rot = ((self._start_rot + delta + 180.0) % 360.0) - 180.0
            apply_light(context)
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RET'} and event.value == 'PRESS':
            self._finish(context)
            rep(self, {'INFO'}, f"Light at {p.light_rot:.0f}°")
            return {'FINISHED'}
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            p.light_rot = self._start_rot
            apply_light(context)
            self._finish(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


class ANAPLAST_OT_light_orbit(bpy.types.Operator):
    """Sweep the light slowly around the case so every fold and margin catches raking light in turn. Click to stop"""
    bl_idname = "anaplast.light_orbit"
    bl_label = "Orbit light"

    _timer = None

    def invoke(self, context, event):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        context.area.header_text_set("Light orbiting · CLICK or ESC to stop")
        apply_light(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        p = context.scene.anaplast
        if event.type == 'TIMER':
            p.light_rot = ((p.light_rot + 2.0 + 180.0) % 360.0) - 180.0
            apply_light(context)
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'ESC', 'RET'} and event.value == 'PRESS':
            context.window_manager.event_timer_remove(self._timer)
            context.area.header_text_set(None)
            rep(self, {'INFO'}, f"Light stopped at {p.light_rot:.0f}°")
            return {'FINISHED'}
        return {'PASS_THROUGH'}


_clip_shading = {}


def clipping_area(context):
    if context.area and context.area.type=='VIEW_3D':return context.area
    return next((a for a in context.screen.areas if a.type=='VIEW_3D'),None)


class ANAPLAST_OT_clip_region(bpy.types.Operator):
    """Draw a clipping box in Solid view; invoke again to clear and restore shading"""
    bl_idname = "anaplast.clip_region"
    bl_label = "Clip region / clear (Alt+B)"

    def invoke(self, context, event):
        area=clipping_area(context)
        if area is None:return {'CANCELLED'}
        region=context.region if context.area==area and context.region and context.region.type=='WINDOW' else next((r for r in area.regions if r.type=='WINDOW'),None)
        if region is None:return {'CANCELLED'}
        rv3d=context.region_data if context.area==area and context.region==region and context.region_data else area.spaces.active.region_3d
        shading=area.spaces.active.shading;key=area.as_pointer()
        if rv3d.use_clip_planes or key in _clip_shading:
            rv3d.use_clip_planes=False
            previous=_clip_shading.pop(key,None)
            if previous:shading.type=previous
            area.tag_redraw();rep(self,{'INFO'},'Clipping cleared; previous shading restored')
            return {'FINISHED'}
        previous=_clip_shading.get(key,shading.type)
        _clip_shading[key]=previous;shading.type='SOLID'
        try:
            with context.temp_override(area=area,region=region):
                result=bpy.ops.view3d.clip_border('INVOKE_DEFAULT')
            if 'CANCELLED' in result:
                shading.type=_clip_shading.pop(key,previous)
                return {'CANCELLED'}
        except RuntimeError as error:
            shading.type=_clip_shading.pop(key,previous)
            rep(self,{'ERROR'},str(error));return {'CANCELLED'}
        rep(self,{'INFO'},'Draw the region to keep. Alt+B or this button clears it and restores shading.')
        return {'FINISHED'}


class ANAPLAST_OT_grid_front(bpy.types.Operator):
    """Overlay a view-aligned millimetre grid at the view center without changing X-ray"""
    bl_idname = "anaplast.grid_front"
    bl_label = "Millimetre grid"

    def execute(self, context):
        p = context.scene.anaplast
        p.grid_front = not p.grid_front
        from .viewport_tools import register_grid
        register_grid()
        for area in context.screen.areas:
            if area.type=='VIEW_3D':area.tag_redraw()
        rep(self, {'INFO'}, "Millimetre grid on" if p.grid_front else "Millimetre grid off")
        return {'FINISHED'}


class ANAPLAST_OT_polyframe(bpy.types.Operator):
    """Show the mesh polygons as a wire overlay on top of the shaded surface (ZBrush PolyF), or hide it again"""
    bl_idname = "anaplast.polyframe"
    bl_label = "Polyframe"

    def execute(self, context):
        p = context.scene.anaplast
        p.polyframe = not p.polyframe
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                ov = area.spaces.active.overlay
                ov.show_wireframes = p.polyframe
                try:
                    ov.wireframe_threshold = 1.0
                    ov.wireframe_opacity = 0.6
                except Exception:
                    pass
                area.tag_redraw()
        rep(self, {'INFO'}, "Polyframe on" if p.polyframe else "Polyframe off")
        return {'FINISHED'}


class ANAPLAST_OT_stats_overlay(bpy.types.Operator):
    """Show Blender's scene statistics (verts / faces / objects) in the viewport corner"""
    bl_idname = "anaplast.stats_overlay"
    bl_label = "Statistics overlay"

    def execute(self, context):
        on = None
        for area in context.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            ov = area.spaces.active.overlay
            if on is None:
                on = not ov.show_stats
            ov.show_stats = on
            ov.show_text = True
            area.tag_redraw()
        rep(self, {'INFO'}, f"Statistics overlay {'on' if on else 'off'}")
        return {'FINISHED'}


class ANAPLAST_OT_fit_view(bpy.types.Operator):
    """Fit the view to everything in the case, or to the selected object"""
    bl_idname = "anaplast.fit_view"
    bl_label = "Fit to screen"

    selected_only: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .viewport_tools import fit,case_targets
        objects=context.selected_objects if self.selected_only else case_targets(context)
        try:fit(context,objects)
        except RuntimeError as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}



class ANAPLAST_OT_texture_view(bpy.types.Operator):
    """Show the scan's own colour/texture (from the OBJ's material) or go back to flat surface colours"""
    bl_idname = "anaplast.texture_view"
    bl_label = "Scan colour"

    def execute(self, context):
        return bpy.ops.anaplast.display_mode(mode='NORMAL' if context.scene.anaplast.show_texture else 'SCAN_COLOR')


class ANAPLAST_OT_no_auto_perspective(bpy.types.Operator):
    """Stop Blender switching to perspective when you zoom (turns off Auto Perspective in Preferences)"""
    bl_idname = "anaplast.no_auto_perspective"
    bl_label = "Lock orthographic"

    def execute(self, context):
        try:
            context.preferences.inputs.use_auto_perspective = False
        except Exception:
            pass
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                rv = area.spaces.active.region_3d
                if rv.view_perspective == 'PERSP':
                    rv.view_perspective = 'ORTHO'
        rep(self, {'INFO'}, "Auto Perspective off — zooming keeps the view orthographic")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_clip_region, ANAPLAST_OT_polyframe, ANAPLAST_OT_grid_front, ANAPLAST_OT_crosshair, ANAPLAST_OT_light_drag, ANAPLAST_OT_light_orbit, ANAPLAST_OT_stats_overlay, ANAPLAST_OT_fit_view, ANAPLAST_OT_texture_view, ANAPLAST_OT_no_auto_perspective)


def register():
    bpy.app.handlers.load_post.append(_reset_guides_on_load)
    from .viewport_tools import register_grid
    register_grid()
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    disable_crosshair(bpy.context)
    if _reset_guides_on_load in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(_reset_guides_on_load)
    from .viewport_tools import unregister_grid
    unregister_grid()
    global _handle
    if _handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        except Exception:
            pass
        _handle = None
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
