"""Editable rear-ear marks and refined surface boundaries; anatomy stays unchanged."""
import hashlib, json, heapq, math
import bpy
import numpy as np
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep
from .lasso import points_in_polygon, smooth_closed_path


def geometry_signature(obj):
    co=np.empty(len(obj.data.vertices)*3,np.float32);obj.data.vertices.foreach_get('co',co)
    loops=np.empty(len(obj.data.loops),np.int32);obj.data.loops.foreach_get('vertex_index',loops)
    return hashlib.sha256(co.tobytes()+loops.tobytes()+np.asarray(obj.matrix_world,np.float32).tobytes()).hexdigest()


def values_for(obj):
    if obj.get('ocular_edit_copy'):
        from .ocular_marking import values_for as ocular_values
        return ocular_values(obj)
    values=np.zeros(len(obj.data.vertices),np.float32);attr=obj.data.attributes.get('.sculpt_mask')
    if attr:attr.data.foreach_get('value',values)
    return values


def write_values(obj,values):
    if obj.get('ocular_edit_copy'):
        from .ocular_marking import write_values as ocular_write
        return ocular_write(obj,values)
    attr=obj.data.attributes.get('.sculpt_mask') or obj.data.attributes.new('.sculpt_mask','FLOAT','POINT')
    attr.data.foreach_set('value',np.asarray(values,np.float32));obj.data.update()
    obj.pop('wedge_boundary_ready',None)
    for border in bpy.data.objects:
        if border.get('wedge_border_preview'):border.hide_set(True)
    show_colors(obj,values)


def show_colors(obj,values,missing=None):
    if obj.get('ocular_edit_copy'):
        from .ocular_marking import show_colors as ocular_colors
        return ocular_colors(obj)
    colors=np.tile([.48,.53,.58,1.],(len(values),1)).astype(np.float32)
    weight=(np.asarray(values)>.5).astype(float)[:,None]
    colors[:,:3]=colors[:,:3]*(1-weight)+np.array([1.,.22,.035])*weight
    if missing is not None:
        colors[np.asarray(values)>.5]=[.08,.75,.24,1.]
        colors[missing]=[1.,.025,.65,1.]
    attr=obj.data.color_attributes.get('Wedge_Marks') or obj.data.color_attributes.new(name='Wedge_Marks',type='FLOAT_COLOR',domain='POINT')
    attr.data.foreach_set('color',colors.ravel());obj.data.color_attributes.active_color=attr;obj.data.update()


def focus_marks(context,obj):
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    from .explode import ensure_collapsed
    ensure_collapsed(context.scene)
    if 'wedge_visibility_before_marking' not in context.scene:
        context.scene['wedge_visibility_before_marking']=json.dumps({o.name:o.hide_get() for o in context.view_layer.objects})
    for other in context.view_layer.objects:
        other.select_set(False)
        if other.type=='MESH':other.hide_set(other!=obj)
    obj.hide_set(False);obj.hide_render=True;obj.select_set(True);context.view_layer.objects.active=obj
    if context.area and context.area.type=='VIEW_3D':
        sd=context.space_data
        if 'wedge_shading_before_marking' not in context.scene:context.scene['wedge_shading_before_marking']=json.dumps({'type':sd.shading.type,'color_type':sd.shading.color_type})
        sd.shading.type='SOLID';sd.shading.color_type='VERTEX';sd.overlay.show_sculpt_mask=True
    context.view_layer.update()


def restore_view(context):
    stored=context.scene.get('wedge_visibility_before_marking')
    if stored:
        for name,hidden in json.loads(stored).items():
            obj=bpy.data.objects.get(name)
            if obj:obj.hide_set(hidden)
        del context.scene['wedge_visibility_before_marking']
    for obj in context.view_layer.objects:
        if obj.get('wedge_edit_copy') or obj.get('wedge_border_preview'):obj.hide_set(True)
    if context.area and context.area.type=='VIEW_3D':
        previous=json.loads(context.scene.get('wedge_shading_before_marking','{}'))
        context.space_data.shading.type=previous.get('type','SOLID');context.space_data.shading.color_type=previous.get('color_type','OBJECT')
    context.scene.pop('wedge_shading_before_marking',None)


def ensure_marks(context):
    from . import mold_auricular as aw
    p=context.scene.anaplast;source=aw.reference(p)
    if source is None:raise RuntimeError('Assign the ear sculpt first')
    signature=geometry_signature(source);obj=p.wedge_mark_obj
    if obj and obj.get('wedge_source_signature')!=signature:
        raise RuntimeError('The ear changed since marking. Choose New marking copy to start from the current sculpt')
    if obj is None:
        donor=p.wedge_suggestion if p.wedge_selection=='AUTO' else source
        if donor is None:donor=source
        obj=mu.duplicate_object(source,'Wedge_Marked_Area',mu.get_collection(context.scene))
        obj['wedge_edit_copy']=True;obj['wedge_source_signature']=signature;obj['anaplast_part']='WEDGE_MARKS'
        obj.hide_render=True;p.wedge_mark_obj=obj
        if len(donor.data.vertices)==len(source.data.vertices):write_values(obj,values_for(donor))
        else:write_values(obj,np.zeros(len(source.data.vertices),np.float32))
    p.wedge_selection='MARK';focus_marks(context,obj);show_colors(obj,values_for(obj))
    return obj


def surface_graph(obj):
    co=np.empty(len(obj.data.vertices)*3,np.float64);obj.data.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    m=np.asarray(obj.matrix_world);co=co@m[:3,:3].T+m[:3,3]
    edges=np.empty(len(obj.data.edges)*2,np.int32);obj.data.edges.foreach_get('vertices',edges);edges=edges.reshape(-1,2)
    lengths=np.linalg.norm(co[edges[:,0]]-co[edges[:,1]],axis=1)
    neighbors=[[] for _ in co]
    for (a,b),length in zip(edges,lengths):neighbors[a].append((int(b),float(length)));neighbors[b].append((int(a),float(length)))
    return co,edges,lengths,neighbors


def brush_indices(co,neighbors,seeds,hit,radius):
    """Geodesic brush footprint: never jump through the thin ear to another sheet."""
    distance={int(i):float(np.linalg.norm(co[i]-hit)) for i in seeds};heap=[(d,i) for i,d in distance.items() if d<=radius];heapq.heapify(heap)
    found=[]
    while heap:
        d,index=heapq.heappop(heap)
        if d>distance[index]:continue
        found.append(index)
        for other,length in neighbors[index]:
            trial=d+length
            if trial<=radius and trial<distance.get(other,np.inf):distance[other]=trial;heapq.heappush(heap,(trial,other))
    return np.asarray(found,np.int32)


def boundary_lines(obj,raw):
    """Trace the exact 0.5 mask contour across each source triangle.

    Segments meet on shared mesh edges. The contour lies on the untouched
    mesh and displays a crisp line, rather than a feathered colour band.
    """
    values=np.asarray(raw,float);obj.data.calc_loop_triangles();co=np.asarray([obj.matrix_world@v.co for v in obj.data.vertices]);segments=[]
    for tri in obj.data.loop_triangles:
        ids=list(tri.vertices);crossings=[]
        for a,b in zip(ids,ids[1:]+ids[:1]):
            if (values[a]>.5)!=(values[b]>.5):
                t=(.5-values[a])/(values[b]-values[a]);crossings.append(co[a]*(1-t)+co[b]*t)
        if len(crossings)==2:segments.append(crossings)
    return segments


def preview_border(context,obj,raw):
    name='Wedge_Refined_Border'
    for old in list(bpy.data.objects):
        if old.get('wedge_border_preview'):
            data=old.data;bpy.data.objects.remove(old,do_unlink=True)
            if data.users==0:bpy.data.curves.remove(data)
    curves=bpy.data.curves.new(name,'CURVE');curves.dimensions='3D';curves.resolution_u=1;curves.bevel_depth=.035;curves.bevel_resolution=0
    for segment in boundary_lines(obj,raw):
        spline=curves.splines.new('POLY');spline.points.add(1)
        for point,co in zip(spline.points,segment):point.co=(*co,1.)
    line=bpy.data.objects.new(name,curves);mu.get_collection(context.scene).objects.link(line);line.color=(1.,.7,.03,1.);line.hide_render=True;line['wedge_border_preview']=True
    return line


def selection_masks(p,frame):
    from . import mold_auricular as aw
    source=aw.reference(p)
    if source is None:raise RuntimeError('Assign the ear sculpt first')
    if p.wedge_selection=='TRACE':
        trace=p.wedge_trace
        if not trace or trace.type!='CURVE':raise RuntimeError('Trace the wedge boundary first')
        from .wedge_curve import sample
        boundary=sample(trace,source)
        if len(boundary)<6:raise RuntimeError('Trace at least six boundary points')
        up=Vector(trace.get('trace_up',list(frame[0]))).normalized();right=Vector(trace.get('trace_right',list(frame[1]))).normalized();side=up.cross(right).normalized()
        polygon=np.asarray([[q.dot(right),q.dot(side)] for q in boundary]);co=[source.matrix_world@v.co for v in source.data.vertices]
        projected=np.asarray([[q.dot(right),q.dot(side)] for q in co]);inside=points_in_polygon(projected[:,0],projected[:,1],polygon)
        tree=aw.world_tree(source);raw=np.zeros(len(co),np.float32)
        for i in np.flatnonzero(inside):
            if tree.ray_cast(co[i]+up*.02,up,400)[0] is None:raw[i]=1.
        obj=source
    else:
        obj=p.wedge_mark_obj if p.wedge_selection=='MARK' else p.wedge_suggestion if p.wedge_selection=='AUTO' else source
        if obj is None:raise RuntimeError('Choose Mark / edit area or Suggest Undercuts first')
        if obj.get('wedge_edit_copy') and obj.get('wedge_source_signature')!=geometry_signature(source):raise RuntimeError('The ear changed since marking; make a new marking copy')
        raw=values_for(obj)
    if np.count_nonzero(raw>.5)<6:raise RuntimeError('Mark at least six vertices under the helix')
    return obj,raw


def selection_data(p,frame):
    from .wedge_border import refined_border
    if p.wedge_selection=='TRACE' and p.wedge_trace and p.wedge_trace.get('wedge_contact_curve'):
        from .wedge_curve import sample
        obj,raw=selection_masks(p,frame);u,r,f=frame
        coords=[obj.matrix_world@v.co for v in obj.data.vertices if raw[v.index]>.5]+sample(p.wedge_trace,obj)
        co=np.asarray(coords);points=np.c_[co@np.asarray(r),co@np.asarray(f),co@np.asarray(u)]
        return points,points,{'mode':'DRAWN_SMOOTH_CURVE','contour_samples':len(coords)-int((raw>.5).sum()),'required_vertices_retained':int((raw>.5).sum())}
    obj,raw=selection_masks(p,frame);fair,stats=refined_border(obj,raw,p.wedge_border_detail)
    co=np.asarray([obj.matrix_world@v.co for v in obj.data.vertices]);u,r,f=frame
    local=np.c_[co@np.asarray(r),co@np.asarray(f),co@np.asarray(u)]
    # Include the continuous contour, rather than ending at a stair-step ring
    # of selected mesh vertices.
    segments=boundary_lines(obj,fair) if p.wedge_border_detail>0 else [];border=np.asarray(segments).reshape(-1,3) if segments else np.empty((0,3))
    boundary=np.c_[border@np.asarray(r),border@np.asarray(f),border@np.asarray(u)]
    points=np.vstack([local[fair>.5],boundary])
    stats['contour_samples']=len(boundary)
    return points,points,stats


class ANAPLAST_OT_wedge_mark_start(bpy.types.Operator):
    """Mark the wedge contact on a separate ear copy; source geometry and source mask are preserved"""
    bl_idname='anaplast.wedge_mark_start';bl_label='Mark / edit area';bl_options={'REGISTER','UNDO'}
    fresh:bpy.props.BoolProperty(default=False,options={'HIDDEN'})
    def execute(self,context):
        try:
            if self.fresh:
                # Keep old work available in the scene instead of deleting it.
                old=context.scene.anaplast.wedge_mark_obj
                if old:old.name='Wedge_Marks_Previous';old.hide_set(True)
                context.scene.anaplast.wedge_mark_obj=None
            ensure_marks(context)
            context.scene.anaplast.wedge_report='Orange is the wedge area. Paint or lasso to add; erase to remove. Rotate the ear to reach under the helix.'
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class ANAPLAST_OT_wedge_mark_action(bpy.types.Operator):
    bl_idname='anaplast.wedge_mark_action';bl_label='Wedge marking';bl_options={'REGISTER','UNDO'}
    action:bpy.props.EnumProperty(items=[('BORDER','Preview refined border',''),('CLEAR','Clear marks',''),('VIEW','Return to mold',''),('COVERAGE','Check marked coverage','')])
    def execute(self,context):
        try:
            if self.action=='VIEW':restore_view(context);return {'FINISHED'}
            curve_mode=self.action=='COVERAGE' and context.scene.anaplast.wedge_selection=='TRACE'
            if curve_mode:
                from .wedge_curve import preview
                preview(context)
            obj=ensure_marks(context);raw=values_for(obj)
            if curve_mode:context.scene.anaplast.wedge_selection='TRACE'
            if self.action=='CLEAR':write_values(obj,np.zeros_like(raw));context.scene.anaplast.wedge_report='Marks cleared on the copy. Paint the wedge area.'
            elif self.action=='BORDER':
                from .wedge_border import refined_border
                fair,stats=refined_border(obj,raw,context.scene.anaplast.wedge_border_detail);show_colors(obj,fair);preview_border(context,obj,fair);obj['wedge_boundary_ready']=json.dumps(stats)
                context.scene.anaplast.wedge_report=f"Refined under-helix border shown. Build Wedge uses this outline. {stats.get('removed_mark_vertices',0)} edge vertices removed and {stats.get('added_mark_vertices',0)} added; raw paint retained."
            else:
                from . import mold_auricular as aw
                wedge=bpy.data.objects.get('Mold_Wedge')
                if wedge is None:raise RuntimeError('Build the wedge before checking coverage')
                from .wedge_border import refined_border
                if not curve_mode:raw,_=refined_border(obj,raw,context.scene.anaplast.wedge_border_detail)
                tree=aw.world_tree(wedge);chosen=np.flatnonzero(raw>.5);missing=np.zeros(len(raw),bool)
                if not len(chosen):raise RuntimeError('Mark the wedge area before checking its coverage')
                for i in chosen:missing[i]=tree.find_nearest(obj.matrix_world@obj.data.vertices[int(i)].co)[3]>.1
                show_colors(obj,raw,missing);n=int(missing.sum())
                context.scene.anaplast.wedge_report=f'{n:,} of {len(chosen):,} refined region vertices are over 0.1 mm from the wedge. Magenta = uncovered; green = covered. Rotate to inspect.'
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class ANAPLAST_OT_wedge_mark_tool(bpy.types.Operator):
    """Paint or lasso visible ear surface. Ctrl reverses add/erase; Enter or right click finishes; Esc cancels"""
    bl_idname='anaplast.wedge_mark_tool';bl_label='Mark wedge surface';bl_options={'REGISTER','UNDO'}
    target:bpy.props.EnumProperty(items=[('WEDGE','Wedge',''),('COMPONENT','Components',''),('OCULAR','Ocular','')],default='WEDGE')
    erase:bpy.props.BoolProperty(default=False)
    lasso:bpy.props.BoolProperty(default=False)
    @classmethod
    def poll(cls,context):return context.area is not None and context.area.type=='VIEW_3D'
    def invoke(self,context,event):
        try:
            from . import component_paint
            from . import ocular_marking
            self.obj=ocular_marking.ensure_marks(context) if self.target=='OCULAR' else (component_paint.ensure_marks(context) if self.target=='COMPONENT' else ensure_marks(context));self.values=values_for(self.obj);self.original=self.values.copy();self.undo=[]
            self.co,self.edges,self.lengths,self.neighbors=surface_graph(self.obj)
            self.region=next(r for r in context.area.regions if r.type=='WINDOW');self.rv=context.space_data.region_3d;self.area=context.area
            from .mold_auricular import world_tree
            self.tree=world_tree(self.obj);self.drag=False;self.path=[];self.cursor=None;self.previous=None
            self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_cursor,(),'WINDOW','POST_PIXEL')
            context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self)
            context.area.header_text_set('Drag to '+('erase' if self.erase else 'mark')+' | Ctrl reverses | Rotate with middle mouse | Ctrl Z undo stroke | Enter/right click finish | Esc cancel')
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'RUNNING_MODAL'}
    def radius(self,context):
        if self.target=='OCULAR':return context.scene.ocular_production.mark_radius
        return context.scene.anaplast.sub_mark_radius if self.target=='COMPONENT' else context.scene.anaplast.wedge_mark_radius
    def point(self,event):return (event.mouse_x-self.region.x,event.mouse_y-self.region.y)
    def hit(self,xy):
        from bpy_extras import view3d_utils as v
        origin=v.region_2d_to_origin_3d(self.region,self.rv,xy);direction=v.region_2d_to_vector_3d(self.region,self.rv,xy)
        # Object ray-cast returns the original polygon index; world BVH is
        # used for visibility checks and is built only once per tool session.
        inv=self.obj.matrix_world.inverted();ok,q,n,face=self.obj.ray_cast(inv@origin,(inv.to_3x3()@direction).normalized())
        return (self.obj.matrix_world@q,face) if ok else (None,None)
    def paint(self,xy,erase):
        q,face=self.hit(xy)
        if q is None:return
        radius=self.radius(bpy.context)
        ids=brush_indices(self.co,self.neighbors,self.obj.data.polygons[face].vertices,np.asarray(q),radius)
        self.values[ids]=0. if erase else 1.
    def apply_lasso(self,erase):
        if len(self.path)<3:return
        poly=smooth_closed_path(self.path,5.)
        hom=np.c_[self.co,np.ones(len(self.co))]@np.asarray(self.rv.perspective_matrix).T;w=hom[:,3];valid=w>1e-9
        px=(hom[:,0]/np.where(valid,w,1)+1)*self.region.width*.5;py=(hom[:,1]/np.where(valid,w,1)+1)*self.region.height*.5
        candidates=np.flatnonzero(valid&points_in_polygon(px,py,poly));direction=self.rv.view_rotation@Vector((0,0,1))
        eye=self.rv.view_matrix.inverted().translation
        for i in candidates:
            q=Vector(self.co[i]);toward=(eye-q).normalized() if self.rv.is_perspective else direction
            travel=(eye-q).length if self.rv.is_perspective else 1000.
            if self.tree.ray_cast(q+toward*.01,toward,travel)[0] is None:self.values[i]=0. if erase else 1.
    def draw_cursor(self):
        if bpy.context.area!=self.area:return
        import gpu
        from gpu_extras.batch import batch_for_shader
        shader=gpu.shader.from_builtin('UNIFORM_COLOR');line=[]
        if self.lasso and self.path:line=self.path+[self.path[0]]
        elif self.cursor:
            q,_=self.hit(self.cursor)
            if q is not None:
                from bpy_extras import view3d_utils as v
                right=self.rv.view_rotation@Vector((1,0,0));a=v.location_3d_to_region_2d(self.region,self.rv,q);b=v.location_3d_to_region_2d(self.region,self.rv,q+right*self.radius(bpy.context))
                if a is not None and b is not None:
                    radius=(b-a).length;line=[(a.x+radius*math.cos(t),a.y+radius*math.sin(t)) for t in np.linspace(0,2*math.pi,65)]
        if len(line)>1:
            shader.bind();shader.uniform_float('color',(1.,.35,.06,1.));batch_for_shader(shader,'LINE_STRIP',{'pos':line}).draw(shader)
    def finish(self,context,cancel=False):
        if cancel:write_values(self.obj,self.original)
        bpy.types.SpaceView3D.draw_handler_remove(self.handle,'WINDOW');context.window.cursor_modal_restore();self.area.header_text_set(None);self.area.tag_redraw()
    def modal(self,context,event):
        if event.type=='B' and event.alt and not event.ctrl and event.value=='PRESS' and not self.drag:
            self.finish(context)
            bpy.ops.anaplast.clip_region('INVOKE_DEFAULT')
            return {'FINISHED'}
        if event.type=='ESC':self.finish(context,True);return {'CANCELLED'}
        if event.type in {'RET','NUMPAD_ENTER','RIGHTMOUSE'} and event.value=='PRESS':self.finish(context);return {'FINISHED'}
        if event.type in {'MIDDLEMOUSE','WHEELUPMOUSE','WHEELDOWNMOUSE','TRACKPADPAN','TRACKPADZOOM','NDOF_MOTION'} and not self.drag:return {'PASS_THROUGH'}
        if event.type=='Z' and event.ctrl and event.value=='PRESS' and self.undo:
            self.values=self.undo.pop();self.drag=False;self.path=[];write_values(self.obj,self.values);self.area.tag_redraw();return {'RUNNING_MODAL'}
        xy=self.point(event);self.cursor=xy
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            if not (0<=xy[0]<self.region.width and 0<=xy[1]<self.region.height):return {'RUNNING_MODAL'}
            self.undo.append(self.values.copy());self.undo=self.undo[-30:];self.drag=True;self.path=[xy];self.previous=xy
            if not self.lasso:self.paint(xy,self.erase!=event.ctrl);write_values(self.obj,self.values)
        elif event.type=='MOUSEMOVE' and self.drag:
            if self.lasso:self.path.append(xy)
            else:
                start=np.asarray(self.previous);end=np.asarray(xy);steps=min(80,max(1,int(np.linalg.norm(end-start)/3.)))
                for t in np.linspace(0.,1.,steps+1)[1:]:self.paint(tuple(start+(end-start)*t),self.erase!=event.ctrl)
                write_values(self.obj,self.values)
            self.previous=xy
        elif event.type=='LEFTMOUSE' and event.value=='RELEASE' and self.drag:
            if self.lasso:self.apply_lasso(self.erase!=event.ctrl);write_values(self.obj,self.values)
            self.drag=False;self.path=[]
        self.area.tag_redraw();return {'RUNNING_MODAL'}


_classes=(ANAPLAST_OT_wedge_mark_start,ANAPLAST_OT_wedge_mark_action,ANAPLAST_OT_wedge_mark_tool)
def register():
    for cls in _classes:bpy.utils.register_class(cls)
def unregister():
    for cls in reversed(_classes):bpy.utils.unregister_class(cls)

