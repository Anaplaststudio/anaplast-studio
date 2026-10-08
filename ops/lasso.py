"""Smooth lasso: draw around the region; the path is smoothed as you go (shaky hands welcome) and the mask
is built from the smoothed curve. Works on the Crop target; Keep / Remove then act on the mask as usual."""
import numpy as np
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep

_handle = None
_raw = []
_smooth = []


def smooth_closed_path(pts, amount_px, spacing=3.0):
    """Resample a 2D path to even spacing, then average it around a window `amount_px` wide (closed loop)."""
    P = np.asarray(pts, dtype=np.float64)
    if len(P) < 3:
        return P
    # close, then resample evenly along the loop
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    L = seg.sum()
    if L < 1e-6:
        return P
    n = max(24, int(L / spacing))
    t = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, L, n, endpoint=False)
    R = np.column_stack([np.interp(s, t, Q[:, 0]), np.interp(s, t, Q[:, 1])])
    # circular moving average, window in samples
    k = min(n//2-1,int(max(0, round(amount_px / spacing))))
    if k > 0:
        w = 2 * k + 1
        Rp = np.vstack([R[-k:], R, R[:k]])
        kern = np.ones(w) / w
        R = np.column_stack([np.convolve(Rp[:, 0], kern, mode='valid'), np.convolve(Rp[:, 1], kern, mode='valid')])
    return R


def _draw():
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1.0)
        if len(_raw) >= 2:
            b = batch_for_shader(shader, 'LINE_STRIP', {"pos": [tuple(q) for q in _raw]})
            shader.bind(); shader.uniform_float("color", (1.0, 1.0, 1.0, 0.25)); b.draw(shader)
        if len(_smooth) >= 3:
            gpu.state.line_width_set(2.5)
            loop = [tuple(q) for q in _smooth] + [tuple(_smooth[0])]
            b = batch_for_shader(shader, 'LINE_STRIP', {"pos": loop})
            shader.bind(); shader.uniform_float("color", (1.0, 0.45, 0.15, 0.95)); b.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
    except Exception:
        pass


def points_in_polygon(px, py, poly):
    """Even-odd test, vectorised over points, looped over polygon edges."""
    inside = np.zeros(len(px), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % n]
        cond = ((y0 > py) != (y1 > py))
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
        inside ^= cond & (px < xint)
    return inside


def shape_outline(points,shape,smoothing=0.):
    if shape=='LASSO':return smooth_closed_path(points,smoothing)
    if len(points)<2:return np.empty((0,2))
    a,b=np.asarray(points[0],float),np.asarray(points[-1],float)
    lo,hi=np.minimum(a,b),np.maximum(a,b)
    if np.any(hi-lo<2):return np.empty((0,2))
    if shape=='RECTANGLE':return np.array([lo,(hi[0],lo[1]),hi,(lo[0],hi[1])])
    angles=np.linspace(0,2*np.pi,128,endpoint=False)
    return (lo+hi)/2+np.column_stack((np.cos(angles),np.sin(angles)))*(hi-lo)/2


def projected_selection(co,matrix,width,height,poly):
    if len(poly)<3:return np.zeros(len(co),dtype=bool)
    hom=np.column_stack((co,np.ones(len(co))))@np.asarray(matrix).T
    valid=hom[:,3]>1e-9;safe=np.where(valid,hom[:,3],1.)
    ndc=hom[:,:3]/safe[:,None];valid&=(ndc[:,2]>=-1)&(ndc[:,2]<=1)
    px=(ndc[:,0]+1)*width*.5;py=(ndc[:,1]+1)*height*.5
    inside=np.zeros(len(co),dtype=bool)
    inside[valid]=points_in_polygon(px[valid],py[valid],poly)
    return inside


class ANAPLAST_OT_smooth_lasso(bpy.types.Operator):
    """Draw a lasso, oval or rectangle; release to preview, then Keep or Remove"""
    bl_idname = "anaplast.smooth_lasso"
    bl_label = "Draw crop region"
    bl_options = {'REGISTER', 'UNDO'}

    _drawing = False

    @classmethod
    def poll(cls, context):
        return mu.crop_object(context.scene.anaplast) is not None and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        global _handle, _raw, _smooth
        if _handle is not None:return {'CANCELLED'}
        from .crop import clear_preview
        clear_preview(context)
        p = context.scene.anaplast
        obj = mu.crop_object(p)
        if context.mode != 'OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        self._target=obj
        self._region=next(r for r in context.area.regions if r.type=='WINDOW')
        self._rv3d=context.space_data.region_3d
        self._shape=p.crop_shape
        context.view_layer.update()
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.hide_set(False); obj.select_set(True); context.view_layer.objects.active = obj
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        _raw, _smooth = [], []
        self._drawing = False
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set(f"{self._shape.title()} crop on {obj.name}: drag a region · release to preview · Esc cancels")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _raw, _smooth
        p = context.scene.anaplast
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'} and not self._drawing:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not (self._region.x<=event.mouse_x<self._region.x+self._region.width and self._region.y<=event.mouse_y<self._region.y+self._region.height):
                return {'RUNNING_MODAL'}
            self._drawing = True
            _raw = [(event.mouse_x-self._region.x, event.mouse_y-self._region.y)]
            _smooth = []
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self._drawing:
            point=(event.mouse_x-self._region.x,event.mouse_y-self._region.y)
            if self._shape=='LASSO':_raw.append(point)
            else:_raw[:]=[_raw[0],point]
            _smooth=shape_outline(_raw,self._shape,p.lasso_smooth).tolist()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._drawing:
            self._drawing = False
            point=(event.mouse_x-self._region.x,event.mouse_y-self._region.y)
            if self._shape!='LASSO':_raw[:]=[_raw[0],point]
            if len(_raw) < (3 if self._shape=='LASSO' else 2):
                self._finish(context)
                return {'CANCELLED'}
            poly = shape_outline(_raw,self._shape,p.lasso_smooth)
            try:n = self._apply_mask(context, poly)
            except Exception as exc:
                self._finish(context);rep(self,{'ERROR'},str(exc));return {'CANCELLED'}
            self._finish(context)
            if n == 0:
                rep(self, {'WARNING'}, "The loop enclosed no vertices — draw around the region you want to keep")
                return {'CANCELLED'}
            rep(self, {'INFO'}, f"Marked {n:,} vertices — choose Keep or Remove")
            return {'FINISHED'}
        if event.type in {'ESC','RIGHTMOUSE'}:
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _apply_mask(self, context, poly):
        p=context.scene.anaplast;source=self._target
        dg=context.evaluated_depsgraph_get()
        me=bpy.data.meshes.new_from_object(source.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
        co=np.empty(len(me.vertices)*3);me.vertices.foreach_get('co',co)
        inside=projected_selection(co.reshape(-1,3),self._rv3d.perspective_matrix@source.matrix_world,
                                   self._region.width,self._region.height,poly)
        if not inside.any():
            bpy.data.meshes.remove(me);return 0
        obj=bpy.data.objects.new('Crop_Preview',me);context.scene.collection.objects.link(obj)
        obj.matrix_world=source.matrix_world.copy();obj['anaplast_construction']=True
        obj.color=source.color
        obj['crop_source']=source.name;obj['crop_source_hidden']=source.hide_get()
        obj['crop_view_direction']=list(self._rv3d.view_rotation@Vector((0,0,-1)))
        obj['crop_projection']=[list(row) for row in self._rv3d.perspective_matrix@source.matrix_world]
        obj['crop_outline_ndc']=[[2.*float(x)/self._region.width-1.,2.*float(y)/self._region.height-1.] for x,y in poly]
        obj['crop_axis_local']=list(source.matrix_world.to_3x3().inverted()@Vector(obj['crop_view_direction']))
        p.crop_preview_obj=obj
        source.hide_set(True)
        for o in context.selected_objects:o.select_set(False)
        obj.select_set(True);context.view_layer.objects.active=obj
        attr=me.attributes.get('.sculpt_mask') or me.attributes.new('.sculpt_mask','FLOAT','POINT')
        attr.data.foreach_set('value',inside.astype(np.float32));me.update()
        bpy.ops.object.mode_set(mode='SCULPT')
        context.space_data.overlay.show_sculpt_mask=True
        context.space_data.overlay.sculpt_mode_mask_opacity=.8
        return int(inside.sum())

    def _finish(self, context):
        global _handle, _raw, _smooth
        if _handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW'); _handle = None
        _raw, _smooth = [], []
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()
        context.area.tag_redraw()


def register():
    bpy.utils.register_class(ANAPLAST_OT_smooth_lasso)


def unregister():
    global _handle
    if _handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        except Exception:
            pass
        _handle = None
    bpy.utils.unregister_class(ANAPLAST_OT_smooth_lasso)
