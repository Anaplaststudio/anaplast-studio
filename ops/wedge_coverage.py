"""Automatic shortest rim connections with extra coverage toward the cap."""
import math,json
import bpy
import numpy as np
from mathutils import Matrix,Vector
from . import mold_auricular as aw
from .mold_auto import dist_to_polygon
from ..utils import mesh as mu
from .report import rep


def nearest_rim(q,outline):
    a=np.asarray(outline);edge=np.roll(a,-1,axis=0)-a
    t=np.clip(np.sum((q-a)*edge,axis=1)/np.maximum(np.sum(edge*edge,axis=1),1e-12),0,1)
    feet=a+edge*t[:,None];return feet[np.argmin(np.linalg.norm(feet-q,axis=1))]


def rim_geometry(outline):
    outer=np.asarray(outline,float);edges=np.roll(outer,-1,axis=0)-outer;length=np.linalg.norm(edges,axis=1);starts=np.r_[0.,np.cumsum(length)]
    return outer,edges,length,starts


def shift_rim(q,outline,distance,radial):
    outer,edges,length,starts=rim_geometry(outline);total=starts[-1]
    if abs(distance)>=total*.25:raise RuntimeError('The end distance is too large for this mold')
    t=np.clip(np.sum((q-outer)*edges,axis=1)/np.maximum(length*length,1e-12),0,1);feet=outer+t[:,None]*edges;i=int(np.argmin(np.linalg.norm(feet-q,axis=1)));start=starts[i]+t[i]*length[i]
    def at(station):
        station%=total;j=min(len(outer)-1,int(np.searchsorted(starts,station,side='right')-1));return outer[j]+edges[j]*((station-starts[j])/max(length[j],1e-12))
    direction=1. if (at(start+.1)-q)@radial<(at(start-.1)-q)@radial else -1.
    return at(start+direction*distance)


def resolve(p,points,outline,radial,tangent,allow_pending=False):
    original,connections=aw.shortest_end_outline(points,outline,radial,tangent)
    changed=False;result=np.asarray(outline);inside=points[:,:2].mean(axis=0)+radial*8.
    for index,c in enumerate(connections,1):
        end=np.asarray(c['inner_xy']);nearest=np.asarray(c['outer_xy']);mode=getattr(p,f'wedge_end{index}_mode');distance=getattr(p,f'wedge_end{index}_extra')
        if mode=='POINT':
            stored=json.loads(getattr(p,f'wedge_end{index}_point') or '[]')
            if len(stored)!=2:
                if not allow_pending:raise RuntimeError(f'Pick end {index} on the outer rim, or switch it to Distance')
                chosen=nearest
            else:chosen=nearest_rim(np.asarray(stored),outline)
        else:chosen=shift_rim(nearest,outline,distance,radial)
        changed|=np.linalg.norm(chosen-nearest)>1e-6
        line=chosen-end
        if np.linalg.norm(line)<1.:raise RuntimeError(f'End {index} is too close to the marked ear')
        cross=lambda q:float(line[0]*(q[1]-end[1])-line[1]*(q[0]-end[0]))
        sign=1. if cross(inside)>=0 else -1.;clipped=[]
        for a,b in zip(result,np.roll(result,-1,axis=0)):
            da,db=cross(a)*sign,cross(b)*sign
            if da>=-1e-8:clipped.append(a)
            if (da>=0)!=(db>=0):clipped.append(a+(b-a)*da/(da-db))
        result=np.asarray(clipped)
        if len(result)<3:raise RuntimeError('These endpoints leave no usable wedge region; move them apart')
        c.update(shortest_outer_xy=nearest.tolist(),shortest_distance_mm=c['distance_mm'],outer_xy=chosen.tolist(),distance_mm=float(np.linalg.norm(chosen-end)),end_mode=mode,extra_rim_mm=float(distance) if mode=='DISTANCE' else None)
    if not changed:return original,connections
    # Keep the marked ear and both selected exits in the construction envelope.
    # This supports independent endpoints without a freehand base polygon.
    poly=aw.convex_hull_2d(np.vstack([result,points[:,:2],[c['outer_xy'] for c in connections]]))
    return poly,connections


def delete_preview(obj):
    data=obj.data;mu.delete_object(obj)
    if data is not None and isinstance(data,bpy.types.Curve) and data.users==0:bpy.data.curves.remove(data)


def preview(context,poly,base,frame,ends=(),tree=None):
    for o in list(context.scene.objects):
        if o.get('wedge_coverage_preview'):delete_preview(o)
    if len(poly)<2:return
    tree=tree or aw.world_tree(base);u,r,f=frame;level=float(base['mold_plane_h'])
    def point(xy):
        q=r*float(xy[0])+f*float(xy[1]);hit=tree.ray_cast(q+u*(level+150),-u,400)[0]
        return (hit if hit is not None else q+u*level)+u*.12
    curve=bpy.data.curves.new('Wedge_base_boundary','CURVE');curve.dimensions='3D';curve.bevel_depth=.12;curve.bevel_resolution=1
    path=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        for t in np.linspace(0,1,max(2,int(np.linalg.norm(np.asarray(b)-a)/1.)),endpoint=False):path.append(point(np.asarray(a)*(1-t)+np.asarray(b)*t))
    sp=curve.splines.new('POLY');sp.points.add(len(path)-1)
    for v,q in zip(sp.points,path):v.co=(*q,1.)
    sp.use_cyclic_u=True;obj=bpy.data.objects.new('Wedge_base_boundary',curve);context.scene.collection.objects.link(obj);obj['wedge_coverage_preview']=True;obj.color=(1.,.65,.03,1);obj.show_in_front=True;obj.hide_render=True
    for i,q in enumerate(ends):
        obj=bpy.data.objects.new(f'Wedge end {i+1}',None);context.scene.collection.objects.link(obj);obj.location=point(q);obj.empty_display_type='SPHERE';obj.empty_display_size=1.5;obj.show_in_front=True;obj.show_name=True;obj['wedge_coverage_preview']=True


class ANAPLAST_OT_wedge_coverage(bpy.types.Operator):
    bl_idname='anaplast.wedge_coverage';bl_label='Show automatic base boundary';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return bpy.data.objects.get('Mold_Base') is not None
    def execute(self,context):
        try:
            from .explode import ensure_collapsed
            ensure_collapsed(context.scene);p=context.scene.anaplast;base=bpy.data.objects['Mold_Base'];frame=aw.frame_for(base)
            points=aw.selected_points(p,frame);outline=aw.offset_polygon(list(base['mold_hull']),float(base['mold_skin'])+float(base['mold_land']))
            direction=points[:,:2].mean(axis=0)-np.asarray(base['mold_hull']).mean(axis=0);angle=math.atan2(direction[1],direction[0]) if p.wedge_auto_exit else p.wedge_exit_angle
            poly,connections=resolve(p,points,outline,np.array([math.cos(angle),math.sin(angle)]),np.array([-math.sin(angle),math.cos(angle)]),allow_pending=True)
            preview(context,poly,base,frame,[c['outer_xy'] for c in connections])
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class ANAPLAST_OT_wedge_endpoint(bpy.types.Operator):
    bl_idname='anaplast.wedge_endpoint';bl_label='Pick wedge endpoint';bl_options={'REGISTER','UNDO'}
    end:bpy.props.IntProperty(default=1,min=1,max=2)
    @classmethod
    def poll(cls,context):return context.area is not None and context.area.type=='VIEW_3D' and bpy.data.objects.get('Mold_Base') is not None
    def invoke(self,context,event):
        try:
            from .explode import ensure_collapsed
            ensure_collapsed(context.scene);self.area=context.area;self.region=next(r for r in self.area.regions if r.type=='WINDOW');self.rv=context.space_data.region_3d
            self.base=bpy.data.objects['Mold_Base'];self.frame=aw.frame_for(self.base);self.outline=aw.offset_polygon(list(self.base['mold_hull']),float(self.base['mold_skin'])+float(self.base['mold_land']))
            self.old_view=(self.rv.view_rotation.copy(),self.rv.view_location.copy(),self.rv.view_distance,self.rv.view_perspective)
            context.scene['wedge_coverage_visibility']=json.dumps({o.name:o.hide_get() for o in context.view_layer.objects})
            u,r,f=self.frame;corners=[self.base.matrix_world@Vector(v) for v in self.base.bound_box]
            self.rv.view_rotation=Matrix((r,f,u)).transposed().to_quaternion();self.rv.view_location=sum(corners,Vector())/8;self.rv.view_distance=max((a-b).length for a in corners for b in corners)*1.1;self.rv.view_perspective='ORTHO'
            for obj in context.view_layer.objects:
                if obj.type=='MESH':obj.hide_set(obj!=self.base)
            self.area.header_text_set(f'Click outer rim for end {self.end} | H: prosthesis | Esc: cancel')
            bpy.ops.anaplast.wedge_coverage();context.window_manager.modal_handler_add(self)
        except Exception as e:
            if hasattr(self,'old_view'):self.finish(context)
            rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'RUNNING_MODAL'}
    def finish(self,context):
        for name,hidden in json.loads(context.scene.pop('wedge_coverage_visibility','{}')).items():
            obj=bpy.data.objects.get(name)
            if obj:obj.hide_set(hidden)
        self.rv.view_rotation,self.rv.view_location,self.rv.view_distance,self.rv.view_perspective=self.old_view
        self.area.header_text_set(None);self.area.tag_redraw()
    def hit(self,event):
        from bpy_extras import view3d_utils as vu
        xy=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
        if not (0<=xy[0]<self.region.width and 0<=xy[1]<self.region.height):return None
        origin=vu.region_2d_to_origin_3d(self.region,self.rv,xy);direction=vu.region_2d_to_vector_3d(self.region,self.rv,xy);u,r,f=self.frame;den=direction.dot(u)
        if abs(den)<1e-6:return None
        q=origin+direction*((float(self.base['mold_plane_h'])-origin.dot(u))/den);return np.array([q.dot(r),q.dot(f)])
    def accept(self,context,q):
        p=context.scene.anaplast;point_name=f'wedge_end{self.end}_point';mode_name=f'wedge_end{self.end}_mode';old=(getattr(p,point_name),getattr(p,mode_name))
        setattr(p,point_name,json.dumps(nearest_rim(q,self.outline).tolist()));setattr(p,mode_name,'POINT')
        try:
            points=aw.selected_points(p,self.frame);direction=points[:,:2].mean(axis=0)-np.asarray(self.base['mold_hull']).mean(axis=0);angle=math.atan2(direction[1],direction[0]) if p.wedge_auto_exit else p.wedge_exit_angle
            poly,con=resolve(p,points,self.outline,np.array([math.cos(angle),math.sin(angle)]),np.array([-math.sin(angle),math.cos(angle)]),allow_pending=True);preview(context,poly,self.base,self.frame,[c['outer_xy'] for c in con])
        except Exception:setattr(p,point_name,old[0]);setattr(p,mode_name,old[1]);raise
    def modal(self,context,event):
        if event.type=='H' and event.value=='PRESS':bpy.ops.anaplast.substructure_view(target='WEDGE');return {'RUNNING_MODAL'}
        if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':self.finish(context);return {'CANCELLED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            q=self.hit(event)
            if q is None:return {'RUNNING_MODAL'}
            if np.linalg.norm(q-nearest_rim(q,self.outline))>5.:rep(self,{'WARNING'},'Click within 5 mm of the outside mold edge');return {'RUNNING_MODAL'}
            try:self.accept(context,q)
            except Exception as e:rep(self,{'WARNING'},str(e));return {'RUNNING_MODAL'}
            self.finish(context);return {'FINISHED'}
        if event.type in {'MIDDLEMOUSE','WHEELUPMOUSE','WHEELDOWNMOUSE'}:return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


def register():
    for cls in (ANAPLAST_OT_wedge_coverage,ANAPLAST_OT_wedge_endpoint):bpy.utils.register_class(cls)
def unregister():
    for cls in (ANAPLAST_OT_wedge_endpoint,ANAPLAST_OT_wedge_coverage):bpy.utils.unregister_class(cls)
