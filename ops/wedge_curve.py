"""Freehand, editable smooth contact curves on an unchanged Sculpt."""
import math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.geometry import interpolate_bezier
from ..utils import mesh as mu


def simplify(points,tolerance=.35):
    if len(points)<3:return points
    a,b=points[0],points[-1];d=b-a
    distances=[(p-(a+d*max(0,min(1,(p-a).dot(d)/max(d.length_squared,1e-12))))).length for p in points]
    i=max(range(len(points)),key=distances.__getitem__)
    if distances[i]<=tolerance:return [a,b]
    return simplify(points[:i+1],tolerance)[:-1]+simplify(points[i:],tolerance)


def create(context,points,frame,source):
    from .wedge_marking import geometry_signature
    if (points[-1]-points[0]).length<.3:points=points[:-1]
    # Simplify two arcs separately to keep a stable closed outline.
    middle=len(points)//2
    controls=simplify(points[:middle+1])[:-1]+simplify(points[middle:]+[points[0]])[:-1]
    if len(controls)<6:controls=points[::max(1,len(points)//8)]
    cu=bpy.data.curves.new('Wedge contact boundary','CURVE');cu.dimensions='3D';cu.resolution_u=16;cu.bevel_depth=.06;cu.bevel_resolution=2
    sp=cu.splines.new('BEZIER');sp.bezier_points.add(len(controls)-1);sp.use_cyclic_u=True
    for point,co in zip(sp.bezier_points,controls):point.co=co;point.handle_left_type='AUTO';point.handle_right_type='AUTO'
    obj=bpy.data.objects.new('Wedge_Contact_Boundary',cu);mu.get_collection(context.scene).objects.link(obj)
    obj['wedge_contact_curve']=True;obj['anaplast_construction']=True;obj['source_signature']=geometry_signature(source)
    obj['trace_up']=list(frame[0]);obj['trace_right']=list(frame[1]);obj['curve_source']=source.name
    obj.show_in_front=True;obj.hide_render=True;obj.color=(1,.4,.02,1)
    mod=obj.modifiers.new('Follow Sculpt','SHRINKWRAP');mod.target=source;mod.wrap_method='NEAREST_SURFACEPOINT';mod.offset=.02
    p=context.scene.anaplast;old=p.wedge_trace;p.wedge_trace=obj;p.wedge_selection='TRACE'
    if old and old!=obj:old.hide_set(True);old.hide_render=True;old.name='Previous contact boundary'
    return obj


def sample(trace,source):
    from .mold_auricular import world_tree
    from .wedge_marking import geometry_signature
    if trace.get('source_signature') and trace['source_signature']!=geometry_signature(source):raise RuntimeError('The Sculpt changed. Draw its contact boundary again')
    tree=world_tree(source);result=[]
    for spline in trace.data.splines:
        if spline.type!='BEZIER':result.extend(trace.matrix_world@Vector(v.co[:3]) for v in spline.points);continue
        controls=list(spline.bezier_points)
        for a,b in zip(controls,controls[1:]+controls[:1]):
            count=max(4,min(64,math.ceil((a.co-b.co).length/.2)+1))
            for q in interpolate_bezier(a.co,a.handle_right,b.handle_left,b.co,count)[:-1]:
                world=trace.matrix_world@q;hit=tree.find_nearest(world)
                if hit[0] is None or hit[3]>2:raise RuntimeError('A boundary point is too far from the Sculpt. Move it closer before building')
                result.append(hit[0])
    return result


def preview(context):
    from . import wedge_marking as wm,mold_auricular as aw
    p=context.scene.anaplast;source=aw.reference(p);trace=p.wedge_trace
    if trace is None:raise RuntimeError('Draw the contact boundary first')
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    p.wedge_selection='TRACE';_,raw=wm.selection_masks(p,(Vector(trace['trace_up']),Vector(trace['trace_right']),Vector(trace['trace_up']).cross(Vector(trace['trace_right']))))
    obj=wm.ensure_marks(context);wm.write_values(obj,raw);p.wedge_selection='TRACE';trace.hide_set(False)
    p.wedge_report='Orange is the Sculpt surface assigned to the wedge. Edit the curve or draw it again to change the contact.'


class ANAPLAST_OT_wedge_curve(bpy.types.Operator):
    bl_idname='anaplast.wedge_curve';bl_label='Draw smooth contact boundary';bl_options={'REGISTER','UNDO'}
    action:bpy.props.EnumProperty(items=[('DRAW','Draw smooth boundary',''),('EDIT','Edit control points',''),('PREVIEW','Preview contact area','')])
    def invoke(self,context,event):
        if self.action!='DRAW':return self.execute(context)
        from . import mold_auricular as aw,wedge_marking as wm
        self.source=aw.reference(context.scene.anaplast)
        if self.source is None:self.report({'ERROR'},'Choose the Sculpt first');return {'CANCELLED'}
        wm.ensure_marks(context)
        self.region=next(r for r in context.area.regions if r.type=='WINDOW');self.rv=context.space_data.region_3d
        self.tree=aw.world_tree(self.source);self.points=[];self.drawing=False;self.frame=None
        self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_line,(), 'WINDOW','POST_VIEW')
        context.area.header_text_set('Drag on Sculpt to draw a closed contact line. Enter: finish. Backspace: redraw. Esc: cancel.')
        context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self);return {'RUNNING_MODAL'}
    def draw_line(self):
        if len(self.points)<2:return
        import gpu
        from gpu_extras.batch import batch_for_shader
        shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(1,.4,.02,1))
        batch_for_shader(shader,'LINE_STRIP',{'pos':self.points+[self.points[0]]}).draw(shader)
    def finish(self,context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle,'WINDOW');context.area.header_text_set(None);context.window.cursor_modal_restore();context.area.tag_redraw()
    def modal(self,context,event):
        from bpy_extras import view3d_utils as vu
        if event.type in {'ESC','RIGHTMOUSE'}:
            from .wedge_marking import restore_view
            self.finish(context);restore_view(context);return {'CANCELLED'}
        if event.type=='BACK_SPACE' and event.value=='PRESS':self.points=[]
        if event.type=='LEFTMOUSE':
            self.drawing=event.value=='PRESS'
            if self.drawing and not self.points:
                q=self.rv.view_rotation;self.frame=(q@Vector((0,0,1)),q@Vector((1,0,0)))
        if self.drawing and event.type in {'LEFTMOUSE','MOUSEMOVE'}:
            xy=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
            if 0<=xy[0]<self.region.width and 0<=xy[1]<self.region.height:
                origin=vu.region_2d_to_origin_3d(self.region,self.rv,xy);direction=vu.region_2d_to_vector_3d(self.region,self.rv,xy)
                hit=self.tree.ray_cast(origin,direction)[0]
                if hit is not None and (not self.points or (hit-self.points[-1]).length>.15):self.points.append(hit)
        if event.type in {'RET','NUMPAD_ENTER'} and event.value=='PRESS':
            if len(self.points)<8:self.report({'WARNING'},'Draw a closed area with a longer line');return {'RUNNING_MODAL'}
            try:create(context,self.points,self.frame,self.source);preview(context)
            except Exception as exc:self.report({'ERROR'},str(exc));self.finish(context);return {'CANCELLED'}
            self.finish(context);return {'FINISHED'}
        context.area.tag_redraw()
        return {'RUNNING_MODAL'} if event.type in {'LEFTMOUSE','BACK_SPACE'} or (event.type=='MOUSEMOVE' and self.drawing) else {'PASS_THROUGH'}
    def execute(self,context):
        try:
            if self.action=='PREVIEW':preview(context)
            elif self.action=='EDIT':
                obj=context.scene.anaplast.wedge_trace
                if obj is None:raise RuntimeError('Draw the contact boundary first')
                if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
                for other in context.selected_objects:other.select_set(False)
                obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj
                bpy.ops.object.mode_set(mode='EDIT')
            else:raise RuntimeError('Draw the boundary in the 3D view')
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


def register():bpy.utils.register_class(ANAPLAST_OT_wedge_curve)
def unregister():bpy.utils.unregister_class(ANAPLAST_OT_wedge_curve)
